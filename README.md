# TFNO Flood Depth Surrogate — Delivery Package

A Tucker-factorized Fourier Neural Operator (TFNO) surrogate that predicts pluvial (rainfall-driven) flood water depth over a fixed spatial domain, trained on a library of 2D hydrodynamic simulations. This package contains **only** the core TFNO model, training, and inference code — no CNN/GNN/GNO baselines, no hybrid architectures, no auxiliary-input (EF5 runoff) or cross-domain (Antigua, Barbuda) variants.

## Contents

```
delivery/
└── src/
    ├── fno_model.py     — TFNOFlood: the model itself (wraps neuralop's TFNO)
    ├── data_loader.py   — Dataset/DataLoader: patch tiling, normalisation, train/val/test split
    ├── train_fno.py     — PyTorch Lightning training script (also defines the Lightning module)
    └── inference.py     — Autoregressive rollout, patch-blended full-domain inference
```

## Requirements

Python 3.10, with:

```
torch==2.8.0            (+ matching torchvision/torchaudio if used elsewhere)
pytorch-lightning==2.6.0
neuraloperator==2.0.0
tensorly==0.9.0
tensorly-torch==0.5.0
torchmetrics==1.8.2
rasterio==1.4.4
numpy==2.2.6
matplotlib==3.10.8
imageio==2.37.2
tqdm==4.67.1
```

`torch`/`neuraloperator` need a CUDA build to train at any reasonable speed; inference can run on CPU but will be slow at full-domain resolution.

## Input data format

The code expects two directory trees, read by [src/data_loader.py](src/data_loader.py). An empty skeleton of this layout is included at [data/](data/) — drop your own GeoTIFFs into it following the same structure.

**Static features** (`--static_dir`), one shared set of rasters reused across every sample:
```
parms_bands/
├── dem.tif              — elevation (m), float32
├── manning_coef.tif     — Manning's roughness coefficient, float32
├── pervious_cover.tif   — pervious ground-cover fraction [0,1], float32
├── slope.tif            — terrain slope, float32
└── land_mask.tif        — binary land(1)/ocean(0) mask, uint8
```

**Per-simulation samples** (`--samples_dir`), one subdirectory per simulation:
```
samples/
└── sampleN/
    ├── depth_timesteps/   depth_hr0000.00.tif … depth_hr????.00.tif   (mm, whole-hour only)
    └── pcpout_timesteps/  pcpout_hr0000.00.tif … pcpout_hr????.00.tif (mm/interval, whole-hour only)
```
Only whole-hour files (`*_hr????.00.tif`) are read — any sub-hourly files present are ignored. A sample is only used if it has a matching, non-zero count of depth and rain files.

All rasters must share the same grid (height/width/CRS) as the static features.

## Model — `TFNOFlood` ([src/fno_model.py](src/fno_model.py))

A thin wrapper around `neuralop.models.TFNO`:

- **Input**: `(B, 7, H, W)` — `[DEM, Manning, Pervious, Slope, Rain_{t-1}, Rain_t, Depth_t]`, all channels normalized (see constants below).
- **Output**: `(B, 1, H, W)` — predicted `Depth_{t+1}`.
- **Spectral layers**: `n_modes=(modes1, modes2)` low-frequency Fourier modes retained per spatial dimension (default 16×16); `factorization='tucker'` with `rank` controlling how aggressively the spectral weight tensor is compressed (default training rank `0.1`).
- **Boundary handling**: `domain_padding` fractional zero-padding before the FFT, reducing periodic-boundary aliasing artifacts (this is a non-periodic physical domain).
- **Precision**: the forward pass forces `autocast` off and runs in fp32 — `rfftn` doesn't support bf16/fp16 — then casts the output back to the caller's dtype. This matters if you train under `--precision 16-mixed`/`bf16-mixed`.
- At the training defaults below (`hidden_dim=64, n_layers=4, rank=0.1`), the model is **~291K parameters**.

Normalisation constants (in [src/data_loader.py](src/data_loader.py), must match between training and inference):
```
NORM_DEM   = 400.0
NORM_MAN   = 0.16
NORM_SLOPE = 40.0
NORM_RAIN  = 100.0
NORM_DEPTH = 30000.0   # mm
```

## Training — [src/train_fno.py](src/train_fno.py)

```bash
python src/train_fno.py \
    --samples_dir /path/to/samples \
    --static_dir  /path/to/parms_bands \
    --log_dir     logs \
    --batch_size 8 --max_epochs 100 --n_steps 4 --loss hybrid
```

**Data / patching**: samples are shuffled with a fixed seed (42) and split **by simulation** (default 70/15/15 train/val/test) so no simulation's patches leak across splits. Each raster is tiled into `--patch_size` (default 512) patches at `--stride` (default 568 — non-overlapping on the training grid).

**Autoregressive rollout**: training unrolls `--n_steps` steps (default 4) per batch — at each step the model consumes `[static, Rain_{t-1}, Rain_t, Depth_t]` and predicts `Depth_{t+1}`, which becomes `Depth_t` for the next step; rainfall always advances from ground truth, never from a prediction. By default each step's prediction is `.detach()`-ed before feeding back in (truncated BPTT) to bound memory — pass `--full_bptt` to keep the full multi-step graph.

**Loss functions** (`--loss`, always computed only over land pixels via `land_mask`):

| Name | What it does |
|---|---|
| `mse` | Plain masked MSE over the rollout (default) |
| `peak_weighted_rmse` | RMSE upweighted per-pixel wherever a timestep's depth exceeds that pixel's own temporal mean |
| `hybrid` | `mse + 0.1·(1-SSIM) + 0.5·peak_aware_mse` |
| `hybrid_2` | Flood-focused: `wet_rmse + 0.5·dry_rmse + 0.3·extent_dice + 0.1·(1-SSIM) + 0.5·peak_rmse`, all denormalized to metres before summing (kept in raw normalized units, these terms are 4-5 orders of magnitude smaller than extent/SSIM and get numerically swamped) |
| `hybrid_3` | `hybrid_2` **+ `1.0·global_rmse`** — a magnitude/severity-weighted peak term (see `_global_peak_mse`), added after diagnosing that the top 1% deepest ground-truth pixels were predicted at roughly a third of their true depth (6.09m → 2.17m) despite low aggregate loss: the temporal peak term only catches a pixel spiking relative to its *own* recent history, not a pixel that's simply, persistently deep |
| `hybrid_4` | `hybrid` (not `hybrid_2`) **+ `1.0·global_rmse`** — tests whether the magnitude-peak term alone fixes deep-peak underestimation without `hybrid_2/3`'s wet/dry split and soft-Dice extent term |

An optional physics regularizer (`--lambda_phys`, default 0 = off) penalizes predicted depth *increases* at pixels with no current rainfall.

**Optimization/logging**: AdamW (`lr=1e-3`, `weight_decay=1e-4`) + `ReduceLROnPlateau` (halves LR on `val/loss` plateau, patience 5); gradient clipping at norm 1.0; `EarlyStopping` on `val/loss` (patience 15, `--early_stop 0` disables); `TensorBoardLogger` under `--log_dir`; best + last checkpoints saved, best one evaluated on the held-out test split at the end of `trainer.fit`. If `--exp_name` is omitted it's auto-generated from hyperparameters, e.g. `tfno_fullsamples_h64_l4_m16x16_r0.1_p512_n4_bs8_lr0.001_hybrid`.

## Inference — [src/inference.py](src/inference.py)

```bash
python src/inference.py \
    --ckpt_path  logs/<exp_name>/version_0/checkpoints/best.ckpt \
    --model_type fno \
    --data_dir   /path/to/samples/sampleN \
    --static_dir /path/to/parms_bands \
    --eval --gif
```

Runs full-domain, autoregressive rollout against **one simulation directory at a time**.

- **Patch-tiled + Hann-blended**: the domain is tiled with the trained patch size at a denser, overlapping stride (`--stride` default 256 — 50% overlap, vs. training's non-overlapping 568), and each patch's contribution is weighted by a 2D Hann window before being summed and normalized — this avoids visible seams between patches.
- **Rollout**: `--start_step 0` = cold start (zero initial depth); `>0` seeds from the ground-truth depth at that timestep. Rainfall forcing is always read from the sample's real data — never predicted.
- **Determinism**: `--seed` (default 42) fixes all RNGs and forces deterministic cuDNN kernels.
- **Model loading**: architecture hyperparameters are read from `model_cfg` saved inside the checkpoint (written automatically by `train_fno.py`); if that's missing, it falls back to parsing the experiment name from the checkpoint path, then to CLI args. **Note**: the fallback regex expects a bare `tfno_h...` name and will *not* match the `tfno_fullsamples_...`-prefixed names this script's own `_auto_exp_name()` generates — this only matters for checkpoints missing `model_cfg`.
- **Outputs**: a multi-band GeoTIFF (metres, clamped to [0, 30], one band per predicted step), and optionally (`--eval`) land-masked RMSE/MAE per step plus a temporal-max-depth comparison (GeoTIFF + PNG), and (`--gif`) a 4-panel animated GIF (rainfall / prediction / ground truth / error).

## Known gaps / things to check before reuse

- [src/data_loader.py](src/data_loader.py)'s `_DEFAULT_STATIC_DIR`/`_DEFAULT_SAMPLES_DIR` module constants point at a stale path (`/home/hl1138/TFNO/data/...`) — harmless as long as `--static_dir`/`--samples_dir` are always passed explicitly, which both scripts here do by default.
- `fno_model.py`'s `TFNOFlood.__init__` default `rank=0.42` differs from `train_fno.py`'s CLI default `rank=0.1` — the CLI default is what's actually used in practice; only matters if you instantiate `TFNOFlood` directly without going through the training script.
- This package is deliberately code-only — no trained checkpoints, no sample data are included.
