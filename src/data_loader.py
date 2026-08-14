"""
data_loader.py — Dataset and DataLoader for flood surrogate models.

Static features (DEM, Manning, Pervious, Slope, land_mask) are loaded once
from a single shared directory and reused across all simulation samples.

Each sample in samples_dir/ must have:
    depth_timesteps/   depth_hr????.00.tif  (whole-hour snapshots only)
    pcpout_timesteps/  pcpout_hr????.00.tif  (whole-hour snapshots only)

Samples are split **by simulation** (not by patch) to avoid data leakage.

Each __getitem__ returns a 4-tuple for N-step rollout training:
    x            (7, P, P)       — [DEM, Manning, Pervious, Slope, Rain_{t-1}, Rain_t, Depth_t]
    rain_future  (N-1, 1, P, P)  — Rain_{t+1} … Rain_{t+N-1}  (inputs for steps 2…N)
    depth_future (N,   1, P, P)  — Depth_{t+1} … Depth_{t+N}  (step targets)
    land         (1, P, P)       — land mask (1=land, 0=ocean)
"""

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Normalisation constants
# --------------------------------------------------------------------------- #
NORM_DEM   = 400.0
NORM_MAN   = 0.16
NORM_SLOPE = 40.0
NORM_RAIN  = 100.0
NORM_DEPTH = 30000.0

# Default paths
_DEFAULT_STATIC_DIR  = Path('/home/hl1138/TFNO/data/parms_bands')
_DEFAULT_SAMPLES_DIR = Path('/home/hl1138/TFNO/data/samples')


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = 0.0
        return data


def _patch_offsets(H: int, W: int, patch: int, stride: int) -> list[tuple[int, int]]:
    """All (row, col) top-left corners covering the full H×W grid."""
    def positions(dim: int) -> list[int]:
        pos = list(range(0, dim - patch, stride))
        pos.append(dim - patch)
        return sorted(set(pos))
    return [(r, c) for r in positions(H) for c in positions(W)]


def _load_static(static_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load shared static features → (4, H, W) and land_mask → (1, H, W)."""
    dem      = _read(static_dir / 'dem.tif')           / NORM_DEM
    manning  = _read(static_dir / 'manning_coef.tif')  / NORM_MAN
    pervious = _read(static_dir / 'pervious_cover.tif')
    slope    = _read(static_dir / 'slope.tif')         / NORM_SLOPE
    static   = torch.from_numpy(np.stack([dem, manning, pervious, slope], axis=0))
    mask     = torch.from_numpy(_read(static_dir / 'land_mask.tif')[None])
    return static, mask


def _discover_samples(samples_dir: Path) -> list[Path]:
    """Return sorted list of sample dirs that have already been split."""
    dirs = sorted(
        [d for d in samples_dir.iterdir() if d.is_dir()],
        key=lambda d: (len(d.name), d.name),
    )
    ready = [
        d for d in dirs
        if (d / 'depth_timesteps').exists()
        and len(list((d / 'depth_timesteps').glob('depth_hr????.00.tif'))) > 0
        and (d / 'pcpout_timesteps').exists()
        and len(list((d / 'pcpout_timesteps').glob('pcpout_hr????.00.tif'))) > 0
    ]
    skipped = len(dirs) - len(ready)
    if skipped:
        print(f'[data_loader] {len(ready)}/{len(dirs)} samples ready '
              f'({skipped} skipped — not yet split)')
    return ready


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class FloodDataset(Dataset):
    """
    Parameters
    ----------
    sample_dirs : list of simulation directories.
    static      : (4, H, W) shared static tensor (pre-loaded, normalised).
    land_mask   : (1, H, W) shared land mask tensor.
    patch_size  : spatial size of each square crop.
    stride      : step between patch origins.
    n_steps     : number of autoregressive rollout steps (default 2).
                  Requires n_steps+1 consecutive depth files and n_steps rain files
                  beyond t, so valid t shrinks by n_steps - 2 relative to 2-step.
    """

    def __init__(
        self,
        sample_dirs: list[Path],
        static:      torch.Tensor,
        land_mask:   torch.Tensor,
        patch_size:  int = 512,
        stride:      int = 568,
        n_steps:     int = 2,
    ):
        self.patch_size = patch_size
        self.n_steps    = n_steps
        self.static     = static
        self.land_mask  = land_mask
        self.samples: list[tuple] = []

        H, W = static.shape[1], static.shape[2]
        offsets = _patch_offsets(H, W, patch_size, stride)

        for d in sample_dirs:
            depth_files = sorted((d / 'depth_timesteps').glob('depth_hr????.00.tif'))
            rain_files  = sorted((d / 'pcpout_timesteps').glob('pcpout_hr????.00.tif'))

            if len(depth_files) != len(rain_files):
                print(f'[data_loader] WARNING: skipping {d.name} — '
                      f'depth/rain count mismatch ({len(depth_files)} vs {len(rain_files)})')
                continue

            # valid t: need t-1, t, t+1 … t+n_steps (depth) and t+1 … t+n_steps-1 (rain)
            for t in range(1, len(depth_files) - n_steps):
                for (r, c) in offsets:
                    self.samples.append((rain_files, depth_files, t, r, c))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        rain_files, depth_files, t, r, c = self.samples[idx]
        p = self.patch_size
        N = self.n_steps

        def crop(arr: np.ndarray) -> np.ndarray:
            return arr[r:r+p, c:c+p]

        rain_tm1 = crop(_read(rain_files[t - 1])) / NORM_RAIN
        rain_t   = crop(_read(rain_files[t]))     / NORM_RAIN
        depth_t  = crop(_read(depth_files[t]))    / NORM_DEPTH

        # future rain for steps 2…N: rain_{t+1} … rain_{t+N-1}  → (N-1, P, P)
        rain_fut = np.stack([
            crop(_read(rain_files[t + k])) / NORM_RAIN
            for k in range(1, N)
        ], axis=0) if N > 1 else np.empty((0, p, p), dtype=np.float32)

        # depth targets for steps 1…N: depth_{t+1} … depth_{t+N}  → (N, P, P)
        depth_fut = np.stack([
            crop(_read(depth_files[t + k])) / NORM_DEPTH
            for k in range(1, N + 1)
        ], axis=0)

        dynamic = torch.from_numpy(np.stack([rain_tm1, rain_t, depth_t], axis=0))
        x            = torch.cat([self.static[:, r:r+p, c:c+p], dynamic], dim=0)  # (7,P,P)
        rain_future  = torch.from_numpy(rain_fut [:, None])   # (N-1, 1, P, P)
        depth_future = torch.from_numpy(depth_fut[:, None])   # (N,   1, P, P)
        land         = self.land_mask[:, r:r+p, c:c+p]        # (1,   P, P)
        return x, rain_future, depth_future, land


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #

def build_loaders(
    samples_dir:  str | Path = _DEFAULT_SAMPLES_DIR,
    static_dir:   str | Path = _DEFAULT_STATIC_DIR,
    patch_size:   int   = 512,
    stride:       int   = 568,
    val_split:    float = 0.15,
    test_split:   float = 0.15,
    batch_size:   int   = 4,
    num_workers:  int   = 4,
    seed:         int   = 42,
    n_steps:      int   = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return ``(train_loader, val_loader, test_loader)``.

    Samples are shuffled then split **by simulation** to prevent data leakage:
      train : 1 - val_split - test_split  (default 70%)
      val   : val_split                   (default 15%)
      test  : test_split                  (default 15%)
    """
    samples_dir = Path(samples_dir)
    static_dir  = Path(static_dir)

    # Load static features once, shared across all splits
    static, land_mask = _load_static(static_dir)

    # Discover and shuffle sample directories
    all_dirs = _discover_samples(samples_dir)
    rng      = np.random.default_rng(seed)
    rng.shuffle(all_dirs)

    n       = len(all_dirs)
    n_test  = max(1, int(n * test_split))
    n_val   = max(1, int(n * val_split))
    n_train = n - n_val - n_test

    train_dirs = all_dirs[:n_train]
    val_dirs   = all_dirs[n_train : n_train + n_val]
    test_dirs  = all_dirs[n_train + n_val:]

    print(f'Samples — train: {len(train_dirs)}  val: {len(val_dirs)}  '
          f'test: {len(test_dirs)}  (total: {n})')

    def _make_loader(dirs, shuffle):
        ds = FloodDataset(dirs, static, land_mask,
                          patch_size=patch_size, stride=stride, n_steps=n_steps)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return (
        _make_loader(train_dirs, shuffle=True),
        _make_loader(val_dirs,   shuffle=False),
        _make_loader(test_dirs,  shuffle=False),
    )


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #

if __name__ == '__main__':
    train_loader, val_loader, test_loader = build_loaders(
        batch_size  = 2,
        num_workers = 0,
        patch_size  = 512,
        stride      = 568,
    )

    x, rain_future, depth_future, land = next(iter(train_loader))
    print(f'x            : {tuple(x.shape)}')
    print(f'rain_future  : {tuple(rain_future.shape)}')
    print(f'depth_future : {tuple(depth_future.shape)}')
    print(f'land         : {tuple(land.shape)}  land_frac={land.float().mean():.3f}')
    print(f'Train batches : {len(train_loader)}  ({len(train_loader.dataset)} samples)')
    print(f'Val   batches : {len(val_loader)}  ({len(val_loader.dataset)} samples)')
    print(f'Test  batches : {len(test_loader)}  ({len(test_loader.dataset)} samples)')
