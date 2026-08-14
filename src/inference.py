"""
inference_fno.py — Auto-regressive inference for TFNOFlood (or any trained model).

Rolls out the full water-depth sequence from an initial condition by feeding
predicted depth back as input at each 1-hr timestep.

Patch-based inference with Hann-window blending ensures seamless stitching
of the 512×512 patches back into the full 1080×936 domain.

Usage
-----
python inference_fno.py \\
    --ckpt_path  logs/tfno_.../checkpoints/best.ckpt \\
    --model_type fno \\
    --data_dir   /home/hl1138/TFNO/data \\
    --out_path   /home/hl1138/TFNO/predictions/pred_depth.tif

# evaluate against ground truth and save per-timestep metrics
python inference_fno.py \\
    --ckpt_path  logs/tfno_.../checkpoints/best.ckpt \\
    --model_type fno \\
    --data_dir   /home/hl1138/TFNO/data \\
    --out_path   pred_depth.tif \\
    --eval
"""

import argparse
import random
import re
from pathlib import Path

import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_loader import (
    NORM_DEM, NORM_MAN, NORM_SLOPE, NORM_RAIN, NORM_DEPTH, _read, _patch_offsets
)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #

def _set_deterministic(seed: int = 42) -> None:
    """Fix all sources of randomness so every run produces identical output."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False   # don't auto-tune kernel choice
    torch.backends.cudnn.deterministic = True    # force deterministic cuDNN ops
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Blending helpers
# --------------------------------------------------------------------------- #

def _hann2d(patch_size: int) -> torch.Tensor:
    """2D Hann window of shape (1, P, P) for smooth overlap blending."""
    w = torch.hann_window(patch_size, periodic=False)
    return (w.unsqueeze(0) * w.unsqueeze(1)).unsqueeze(0)   # (1, P, P)


# --------------------------------------------------------------------------- #
# Static / dynamic data loaders
# --------------------------------------------------------------------------- #

def load_static(static_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (static (4,H,W), land_mask (1,H,W))."""
    dem      = _read(static_dir / 'dem.tif')           / NORM_DEM
    manning  = _read(static_dir / 'manning_coef.tif')  / NORM_MAN
    pervious = _read(static_dir / 'pervious_cover.tif')
    slope    = _read(static_dir / 'slope.tif')         / NORM_SLOPE
    static   = torch.from_numpy(np.stack([dem, manning, pervious, slope], axis=0))

    mask = torch.from_numpy(_read(static_dir / 'land_mask.tif')[None])
    return static, mask


def load_rain_sequence(data_dir: Path) -> list[torch.Tensor]:
    """Return list of (1, H, W) tensors, one per timestep (normalised)."""
    files = sorted((data_dir / 'pcpout_timesteps').glob('pcpout_hr????.00.tif'))
    return [
        torch.from_numpy(_read(f)[None] / NORM_RAIN) for f in files
    ]


def load_depth_sequence(data_dir: Path) -> list[torch.Tensor]:
    """Return GT depth sequence as list of (1, H, W) tensors in metres."""
    files = sorted((data_dir / 'depth_timesteps').glob('depth_hr????.00.tif'))
    return [
        torch.from_numpy(_read(f)[None] / 1000.0) for f in files   # mm → m
    ]


# --------------------------------------------------------------------------- #
# Patched inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def patched_forward(
    model:      torch.nn.Module,
    x_full:     torch.Tensor,       # (7, H, W)  normalised, on CPU
    patch_size: int,
    stride:     int,
    device:     torch.device,
    hann:       torch.Tensor,       # (1, P, P) weight map
) -> torch.Tensor:
    """Run model on overlapping patches and blend into a (1, H, W) output."""
    # GNN precomputes edges for a fixed patch size — verify it matches
    if hasattr(model, 'patch_size') and model.patch_size != patch_size:
        raise ValueError(
            f'GNN was built with patch_size={model.patch_size} '
            f'but inference uses patch_size={patch_size}. Pass --patch_size {model.patch_size}.'
        )

    _, H, W = x_full.shape
    pred_acc   = torch.zeros(1, H, W)
    weight_acc = torch.zeros(1, H, W)
    p = patch_size

    for r, c in _patch_offsets(H, W, patch_size, stride):
        patch = x_full[:, r:r+p, c:c+p].unsqueeze(0).to(device)   # (1,7,P,P)
        out   = model(patch).squeeze(0).cpu().clamp(min=0.0)        # (1,P,P)
        pred_acc  [:, r:r+p, c:c+p] += out  * hann
        weight_acc[:, r:r+p, c:c+p] += hann

    return pred_acc / weight_acc.clamp(min=1e-8)


# --------------------------------------------------------------------------- #
# Full auto-regressive rollout
# --------------------------------------------------------------------------- #

def run_inference(
    model:      torch.nn.Module,
    data_dir:   Path,
    static_dir: Path,
    patch_size: int,
    stride:     int,
    device:     torch.device,
    start_step: int      = 0,
    end_step:   int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Roll out the full depth sequence auto-regressively.

    Parameters
    ----------
    data_dir   : simulation directory (must contain depth_timesteps/ and pcpout_timesteps/).
    static_dir : shared static-feature directory (dem.tif, manning_coef.tif, etc.).
    start_step : index of the initial-condition timestep.
                 0 = cold start (zero depth); >0 = seed from GT depth at that step.
    end_step   : last timestep to predict (inclusive, 0-based into rain sequence).
                 Default: last available timestep.

    Returns
    -------
    preds    : list of (1, H, W) float tensors in metres,
               predicting timesteps start_step+1 … end_step
    static   : (4, H, W) static features (for reference)
    land_mask: (1, H, W)
    """
    static, land_mask = load_static(static_dir)
    rain_seq = load_rain_sequence(data_dir)
    T = len(rain_seq)
    if end_step is None:
        end_step = T - 1
    end_step = min(end_step, T - 1)

    if start_step >= end_step:
        raise ValueError(f'start_step ({start_step}) must be < end_step ({end_step})')

    hann = _hann2d(patch_size)

    if start_step == 0:
        depth_t   = torch.zeros_like(rain_seq[0])   # cold start
        rain_prev = torch.zeros_like(rain_seq[0])
    else:
        depth_files = sorted((data_dir / 'depth_timesteps').glob('depth_hr????.00.tif'))
        depth_t   = torch.from_numpy(_read(depth_files[start_step])[None] / NORM_DEPTH)
        rain_prev = rain_seq[start_step - 1]

    predictions = []
    for t in tqdm(range(start_step, end_step), desc='Rolling out', unit='step'):
        rain_t = rain_seq[t]
        x_full = torch.cat([static, rain_prev, rain_t, depth_t], dim=0)  # (7,H,W)
        depth_next = patched_forward(
            model, x_full, patch_size, stride, device, hann
        )                                                                  # (1,H,W) norm
        predictions.append(depth_next * NORM_DEPTH / 1000.0)              # → m
        rain_prev = rain_t
        depth_t   = depth_next

    return predictions, static, land_mask


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #

def evaluate(
    preds:      list[torch.Tensor],
    data_dir:   Path,
    land_mask:  torch.Tensor,
    start_step: int = 0,
) -> dict:
    """Compute per-step and overall land-masked RMSE and MAE (in metres)."""
    gt_seq = load_depth_sequence(data_dir)
    mask   = land_mask.float()
    # preds[0] corresponds to gt_seq[start_step + 1]
    gt_aligned = gt_seq[start_step + 1 : start_step + 1 + len(preds)]

    rmse_list, mae_list = [], []
    for pred, gt in zip(preds, gt_aligned):
        diff  = (pred - gt) * mask
        n     = mask.sum().clamp(min=1)
        rmse_list.append(((diff ** 2).sum() / n).sqrt().item())
        mae_list.append( (diff.abs().sum()  / n).item())

    # Per-pixel max depth along time axis: shape (1, H, W)
    pred_stack = torch.stack(preds,       dim=0)   # (T, 1, H, W)
    gt_stack   = torch.stack(gt_aligned,  dim=0)   # (T, 1, H, W)
    max_depth_pred = pred_stack.max(dim=0).values  # (1, H, W)
    max_depth_gt   = gt_stack.max(dim=0).values    # (1, H, W)

    diff_max = (max_depth_pred - max_depth_gt) * mask
    n        = mask.sum().clamp(min=1)
    max_rmse = ((diff_max ** 2).sum() / n).sqrt().item()
    max_mae  = (diff_max.abs().sum()  / n).item()

    return {
        'rmse_per_step':   rmse_list,
        'mae_per_step':    mae_list,
        'rmse_mean':       float(np.mean(rmse_list)),
        'mae_mean':        float(np.mean(mae_list)),
        'max_depth_pred':  max_depth_pred,   # (1, H, W) tensor
        'max_depth_gt':    max_depth_gt,     # (1, H, W) tensor
        'max_depth_rmse':  max_rmse,
        'max_depth_mae':   max_mae,
    }


# --------------------------------------------------------------------------- #
# Save output GeoTIFF
# --------------------------------------------------------------------------- #

def save_geotiff(
    preds:    list[torch.Tensor],
    data_dir: Path,
    out_path: Path,
) -> None:
    """Save predicted depth sequence as a multi-band GeoTIFF (metres, float32)."""
    ref_file = sorted((data_dir / 'depth_timesteps').glob('depth_hr????.00.tif'))[0]
    with rasterio.open(ref_file) as ref:
        profile = ref.profile.copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile.update(count=len(preds), dtype='float32', nodata=-9999.0)

    with rasterio.open(out_path, 'w', **profile) as dst:
        for i, pred in enumerate(preds):
            band = pred.squeeze(0).numpy().clip(0, 30.0).astype(np.float32)
            dst.write(band, i + 1)
            dst.update_tags(i + 1, description=f'pred depth (m) step {i+1}')

    print(f'Saved {len(preds)}-band GeoTIFF → {out_path}')


# --------------------------------------------------------------------------- #
# GIF export
# --------------------------------------------------------------------------- #

def _to_np(t: torch.Tensor, land: torch.Tensor) -> np.ndarray:
    """Squeeze to (H,W), mask ocean to NaN."""
    arr = t.squeeze().float().numpy()
    arr[land.squeeze().numpy() == 0] = np.nan
    return arr


def save_gif(
    preds:      list[torch.Tensor],
    data_dir:   Path,
    land_mask:  torch.Tensor,
    gif_path:   Path,
    fps:        int   = 10,
    max_depth:  float | None = None,
    max_rain:   float | None = None,
    start_step: int   = 0,
) -> None:
    """
    Save a 4-panel animated GIF: Rainfall | Prediction | Ground Truth | Error.

    Parameters
    ----------
    preds      : list of (1,H,W) tensors in metres
    data_dir   : simulation root (for loading GT depth + rain sequences)
    land_mask  : (1,H,W)  1=land, 0=ocean
    gif_path   : output .gif path
    fps        : frames per second
    max_depth  : colour scale cap in metres (auto from GT peak if None)
    max_rain   : colour scale cap in mm/hr  (auto from sequence peak if None)
    start_step : initial-condition index used in run_inference (for GT alignment)
    """
    gt_seq   = load_depth_sequence(data_dir)    # list of (1,H,W) in metres
    rain_seq = load_rain_sequence(data_dir)     # list of (1,H,W) normalised

    # preds[i] → depth at t = start_step + i + 1
    #            rain at t = start_step + i  (the forcing used at that step)
    gt_seq   = gt_seq  [start_step + 1 : start_step + 1 + len(preds)]
    rain_seq = rain_seq[start_step     : start_step     + len(preds)]

    # Compute shared colour limits
    if max_depth is None:
        max_depth = max(float(_to_np(g, land_mask).max()) for g in gt_seq
                        if not np.all(np.isnan(_to_np(g, land_mask))))
        max_depth = max(max_depth, 1e-3)

    if max_rain is None:
        max_rain = max(float(r.squeeze().max().item() * NORM_RAIN) for r in rain_seq)
        max_rain = max(max_rain, 1.0)

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []

    # time axis: 1-hr steps
    times = np.arange(start_step + 1, start_step + 1 + len(preds)).astype(float)

    for i, (pred, gt, rain) in enumerate(tqdm(
            zip(preds, gt_seq, rain_seq),
            total=len(preds), desc='Rendering GIF', unit='frame')):

        p_np = _to_np(pred, land_mask)
        g_np = _to_np(gt,   land_mask)
        e_np = p_np - g_np
        r_np = rain.squeeze().float().numpy() * NORM_RAIN  # → mm/hr (unmasked)

        err_max = float(np.nanmax(np.abs(e_np)))
        err_max = max(err_max, 1e-3)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.patch.set_facecolor('#1a1a2e')

        panels = [
            (r_np, 'Rainfall',     'YlGnBu', 0,        max_rain,  'mm/hr'),
            (p_np, 'Prediction',   'Blues',  0,        max_depth, 'm'),
            (g_np, 'Ground Truth', 'Blues',  0,        max_depth, 'm'),
            (e_np, 'Error',        'RdBu_r', -err_max, err_max,   'm'),
        ]

        for ax, (data, title, cmap, vmin, vmax, unit) in zip(axes, panels):
            im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                           origin='upper', aspect='equal',
                           interpolation='nearest')
            ax.set_title(title, color='white', fontsize=12, pad=6)
            ax.axis('off')
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(unit, color='white', fontsize=9)
            cb.ax.yaxis.set_tick_params(color='white', labelcolor='white')
            cb.outline.set_edgecolor('white')

        fig.suptitle(
            f't = {times[i]:.2f} hr  (step {i+1}/{len(preds)})',
            color='white', fontsize=13, y=1.01,
        )
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        frames.append(buf.reshape(h, w, 4)[..., :3])
        plt.close(fig)

    imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
    print(f'Saved {len(frames)}-frame GIF → {gif_path}')


# --------------------------------------------------------------------------- #
# Max-depth map plot
# --------------------------------------------------------------------------- #

def save_max_depth_plot(
    max_depth_gt:   torch.Tensor,   # (1, H, W)
    max_depth_pred: torch.Tensor,   # (1, H, W)
    land_mask:      torch.Tensor,   # (1, H, W)
    out_path:       Path,
    data_dir:       Path | None = None,
) -> None:
    """Save a 3-panel image and (optionally) GeoTIFFs for GT and pred max depth."""
    gt_np   = _to_np(max_depth_gt,   land_mask)
    pred_np = _to_np(max_depth_pred, land_mask)

    # GeoTIFF export — requires data_dir for the spatial reference
    if data_dir is not None:
        ref_file = sorted((data_dir / 'depth_timesteps').glob('depth_hr????.00.tif'))[0]
        with rasterio.open(ref_file) as ref:
            profile = ref.profile.copy()
        profile.update(count=1, dtype='float32', nodata=-9999.0)

        for arr, suffix in [(gt_np, '_max_depth_gt.tif'), (pred_np, '_max_depth_pred.tif')]:
            tif_path = out_path.with_name(out_path.stem.replace('_max_depth', '') + suffix)
            band = np.where(np.isnan(arr), -9999.0, arr).astype(np.float32)
            with rasterio.open(tif_path, 'w', **profile) as dst:
                dst.write(band, 1)
                dst.update_tags(1, description='temporal max water depth (m)')
            print(f'Saved max-depth GeoTIFF → {tif_path}')
    err_np  = pred_np - gt_np

    vmax     = 5.0   # fixed at 5 m to indicate inundation extent
    err_max  = float(np.nanmax(np.abs(err_np)))
    err_max  = max(err_max, 1e-3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')

    panels = [
        (gt_np,   'GT Max Depth',   'Blues',  0,        vmax,    'm'),
        (pred_np, 'Pred Max Depth', 'Blues',  0,        vmax,    'm'),
        (err_np,  'Error',          'RdBu_r', -err_max, err_max, 'm'),
    ]
    for ax, (data, title, cmap, vmin, vmax_, unit) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax_,
                       origin='upper', aspect='equal', interpolation='nearest')
        ax.set_title(title, color='white', fontsize=13, pad=6)
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(unit, color='white', fontsize=10)
        cb.ax.yaxis.set_tick_params(color='white', labelcolor='white')
        cb.outline.set_edgecolor('white')

    fig.suptitle('Temporal-max water depth', color='white', fontsize=14, y=1.01)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved max-depth plot → {out_path}')


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def _parse_exp_name(ckpt_path: Path) -> dict:
    """
    Extract model hyperparameters from the experiment-name directory embedded
    in the checkpoint path (logs/{exp_name}/version_N/checkpoints/best.ckpt).

    Returns a partial cfg dict (empty if the name doesn't match any pattern).
    """
    # walk up: .../checkpoints/best.ckpt  →  .../version_N  →  .../exp_name
    try:
        exp_name = ckpt_path.parent.parent.parent.name
    except Exception:
        return {}

    # FNO: tfno_h64_l4_m16x16_r0.42_p384_bs16_lr0.001
    m = re.match(r'tfno_h(\d+)_l(\d+)_m(\d+)x(\d+)_r([\d.]+)_p(\d+)', exp_name)
    if m:
        return dict(
            model_type     = 'fno',
            hidden_dim     = int(m.group(1)),
            n_layers       = int(m.group(2)),
            modes1         = int(m.group(3)),
            modes2         = int(m.group(4)),
            rank           = float(m.group(5)),
            patch_size     = int(m.group(6)),
        )

    # CNN: unet_ch64_d4_p384_bs16_lr0.001
    m = re.match(r'unet_ch(\d+)_d(\d+)_p(\d+)', exp_name)
    if m:
        return dict(
            model_type    = 'cnn',
            base_channels = int(m.group(1)),
            depth         = int(m.group(2)),
            patch_size    = int(m.group(3)),
        )

    # GNN: gnn_h128_l6_p384_bs4_lr0.001
    m = re.match(r'gnn_h(\d+)_l(\d+)_p(\d+)', exp_name)
    if m:
        return dict(
            model_type = 'gnn',
            hidden_dim = int(m.group(1)),
            n_layers   = int(m.group(2)),
            patch_size = int(m.group(3)),
        )

    # GNO: gno_h128_l4_nh4_np16_p384_bs4_lr0.001
    m = re.match(r'gno_h(\d+)_l(\d+)_nh(\d+)_np(\d+)_p(\d+)', exp_name)
    if m:
        return dict(
            model_type = 'gno',
            hidden_dim = int(m.group(1)),
            n_layers   = int(m.group(2)),
            num_heads  = int(m.group(3)),
            num_phys   = int(m.group(4)),
            patch_size = int(m.group(5)),
        )

    return {}


def load_model(ckpt_path: Path, model_type: str, args) -> torch.nn.Module:
    """
    Instantiate model and load weights from a Lightning checkpoint.

    Architecture hyperparameters are read from the checkpoint's saved
    model_cfg (written by the training scripts). CLI args are used as
    fallback for old checkpoints that predate model_cfg.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Priority: saved model_cfg  >  parsed experiment name  >  CLI args
    cfg = ckpt.get('hyper_parameters', {}).get('model_cfg', {})
    if cfg:
        print(f'[load_model] config from checkpoint hyper_parameters: {cfg}')
    else:
        cfg = _parse_exp_name(ckpt_path)
        if cfg:
            print(f'[load_model] config parsed from experiment name: {cfg}')
        else:
            print('[load_model] no saved config found — using CLI args only')

    model_type = cfg.get('model_type', model_type)

    def _get(key):
        return cfg.get(key, getattr(args, key, None))

    if model_type == 'fno':
        from fno_model import TFNOFlood
        model = TFNOFlood(
            in_channels    = _get('in_channels'),
            hidden_dim     = _get('hidden_dim'),
            n_layers       = _get('n_layers'),
            modes1         = _get('modes1'),
            modes2         = _get('modes2'),
            rank           = _get('rank'),
            domain_padding = _get('domain_padding'),
        )
    elif model_type == 'cnn':
        from cnn_model import UNetFlood
        model = UNetFlood(
            in_channels   = _get('in_channels'),
            base_channels = _get('base_channels'),
            depth         = _get('depth'),
        )
    elif model_type == 'gnn':
        from gnn_model import GNNFlood
        model = GNNFlood(
            in_channels = _get('in_channels'),
            hidden_dim  = _get('hidden_dim'),
            n_layers    = _get('n_layers'),
            patch_size  = _get('patch_size'),
        )
    elif model_type == 'gno':
        from gno_model import GNOFlood
        model = GNOFlood(
            in_channels  = _get('in_channels'),
            hidden_dim   = _get('hidden_dim'),
            n_layers     = _get('n_layers'),
            patch_size   = _get('patch_size'),
            num_heads    = _get('num_heads'),
            num_phys     = _get('num_phys'),
            global_ratio = _get('global_ratio'),
            global_k     = _get('global_k'),
        )
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    state = {k[len('model.'):]: v
             for k, v in ckpt['state_dict'].items()
             if k.startswith('model.')}
    model.load_state_dict(state)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='Auto-regressive flood depth inference')

    p.add_argument('--ckpt_path',  required=True, help='Lightning checkpoint (.ckpt)')
    p.add_argument('--data_dir',   required=True,
                   help='Simulation directory (must contain depth_timesteps/ and pcpout_timesteps/)')
    p.add_argument('--static_dir', default='/home/hl1138/surrogate/data/parms_bands',
                   help='Shared static-feature directory (dem.tif, manning_coef.tif, …)')
    p.add_argument('--out_dir',    default=None,
                   help='Output directory (default: predictions/<exp_name> derived from ckpt path)')
    p.add_argument('--out_path',   default=None,
                   help='Override output GeoTIFF path (default: auto-named in --out_dir)')
    p.add_argument('--model_type', default='fno', choices=['fno', 'cnn', 'gnn', 'gno'])
    p.add_argument('--start_step', type=int, default=0,
                   help='Initial-condition timestep index (0 = cold start; >0 seeds from GT depth)')
    p.add_argument('--end_step',   type=int, default=None,
                   help='Last timestep to predict, inclusive (default: all available)')
    p.add_argument('--patch_size', type=int, default=512)
    p.add_argument('--stride',     type=int, default=256,
                   help='Patch stride for inference. Must be < patch_size to avoid coverage gaps.')
    p.add_argument('--eval',      action='store_true',
                   help='Compute land-masked RMSE/MAE against ground truth')
    p.add_argument('--gif',       action='store_true',
                   help='Save prediction/GT/error animation (auto-named in --out_dir)')
    p.add_argument('--gif_path',  default=None,
                   help='Override GIF output path (implies --gif)')
    p.add_argument('--fps',       type=int, default=5,
                   help='GIF frames per second (default 10)')
    p.add_argument('--max_depth', type=float, default=None,
                   help='Depth colour-scale cap in metres (default: auto from GT peak)')
    p.add_argument('--max_rain',  type=float, default=None,
                   help='Rainfall colour-scale cap in mm/hr (default: auto from sequence peak)')
    p.add_argument('--device',    default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed',      type=int, default=42,
                   help='Global random seed for reproducibility (default 42)')

    # FNO args
    p.add_argument('--in_channels',    type=int,   default=7)
    p.add_argument('--hidden_dim',     type=int,   default=64)
    p.add_argument('--n_layers',       type=int,   default=4)
    p.add_argument('--modes1',         type=int,   default=16)
    p.add_argument('--modes2',         type=int,   default=16)
    p.add_argument('--rank',           type=float, default=0.42)
    p.add_argument('--domain_padding', type=float, default=0.1)

    # CNN args
    p.add_argument('--base_channels', type=int, default=64)
    p.add_argument('--depth',         type=int, default=4)

    # GNO args
    p.add_argument('--num_heads',    type=int,   default=4)
    p.add_argument('--num_phys',     type=int,   default=8)
    p.add_argument('--global_ratio', type=float, default=0.1)
    p.add_argument('--global_k',     type=int,   default=8)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    args   = parse_args()
    _set_deterministic(args.seed)
    device = torch.device(args.device)

    # Resolve end_step early so it's available for filename generation
    rain_files = sorted((Path(args.data_dir) / 'pcpout_timesteps').glob('pcpout_hr????.00.tif'))
    T = len(rain_files)
    end_step = min(args.end_step if args.end_step is not None else T - 1, T - 1)

    # Auto-name: {sample}_{model_type}_s{start:04d}_e{end:04d}
    sample  = Path(args.data_dir).name
    stem    = f'{sample}_{args.model_type}_s{args.start_step:04d}_e{end_step:04d}'
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        exp_name = Path(args.ckpt_path).parent.parent.parent.name
        out_dir  = Path('predictions') / exp_name
    out_tif = Path(args.out_path) if args.out_path else out_dir / f'{stem}.tif'
    out_gif = (Path(args.gif_path) if args.gif_path
               else out_dir / f'{stem}.gif' if args.gif
               else None)

    print(f'Model    : {args.model_type}  |  checkpoint: {args.ckpt_path}')
    print(f'Data dir : {args.data_dir}')
    print(f'Steps    : {args.start_step} → {end_step}')
    print(f'Device   : {device}')
    print(f'TIF out  : {out_tif}')
    if out_gif:
        print(f'GIF out  : {out_gif}')

    model = load_model(Path(args.ckpt_path), args.model_type, args).to(device)

    preds, _, land_mask = run_inference(
        model      = model,
        data_dir   = Path(args.data_dir),
        static_dir = Path(args.static_dir),
        patch_size = args.patch_size,
        stride     = args.stride,
        device     = device,
        start_step = args.start_step,
        end_step   = end_step,
    )
    print(f'Predicted {len(preds)} steps  '
          f'(hr {args.start_step+1}–{args.start_step+len(preds)})')

    save_geotiff(preds, Path(args.data_dir), out_tif)

    if args.eval:
        metrics = evaluate(preds, Path(args.data_dir), land_mask,
                           start_step=args.start_step)
        print(f'\n=== Evaluation (land pixels only) ===')
        print(f'Mean RMSE : {metrics["rmse_mean"]:.4f} m')
        print(f'Mean MAE  : {metrics["mae_mean"]:.4f} m')
        print(f'Peak RMSE : {max(metrics["rmse_per_step"]):.4f} m  '
              f'(step {np.argmax(metrics["rmse_per_step"])+1})')
        print(f'\n--- Temporal-max depth map (per-pixel max over all timesteps) ---')
        print(f'GT   max depth : {metrics["max_depth_gt"].max().item():.4f} m')
        print(f'Pred max depth : {metrics["max_depth_pred"].max().item():.4f} m')
        print(f'Max-depth RMSE : {metrics["max_depth_rmse"]:.4f} m')
        print(f'Max-depth MAE  : {metrics["max_depth_mae"]:.4f} m')

        max_depth_img = out_dir / f'{stem}_max_depth.png'
        save_max_depth_plot(
            max_depth_gt   = metrics['max_depth_gt'],
            max_depth_pred = metrics['max_depth_pred'],
            land_mask      = land_mask,
            out_path       = max_depth_img,
            data_dir       = Path(args.data_dir),
        )

    if out_gif:
        save_gif(
            preds      = preds,
            data_dir   = Path(args.data_dir),
            land_mask  = land_mask,
            gif_path   = out_gif,
            fps        = args.fps,
            max_depth  = args.max_depth,
            max_rain   = args.max_rain,
            start_step = args.start_step,
        )


if __name__ == '__main__':
    main()
