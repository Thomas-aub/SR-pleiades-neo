"""
src/train/dataset.py — Paired HR / LR GeoTIFF tile dataset for SR fine-tuning.
===============================================================================
Reads tiled GeoTIFF pairs produced by build_dataset.py, normalises pixel
values to [0, 1] float32, and applies paired random crops + augmentation.

Design notes
------------
* rasterio datasets are opened inside __getitem__ (not in __init__) so each
  DataLoader worker gets its own independent file handles — required for
  thread safety with num_workers > 0.

* Tile matching is done by relative path: a file at
  ``<hr_root>/WO_.../IMG_..._row0000_col0001.TIF`` is matched with
  ``<lr_root>/WO_.../IMG_..._row0000_col0001.TIF``.

* Random crop samples a position in the *LR* coordinate space; the HR crop
  is derived by multiplying offsets / sizes by the scale factor, ensuring
  perfect spatial correspondence.

* Edge tiles smaller than lr_patch_size are padded with zeros before
  random-crop sampling so every tile can yield at least one patch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError as exc:
    raise ImportError("rasterio is required for the SR dataset.  pip install rasterio") from exc

log = logging.getLogger(__name__)

TIF_EXTENSIONS = frozenset({".tif", ".TIF", ".tiff", ".TIFF"})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _scan_pairs(hr_root: Path, lr_root: Path) -> List[Tuple[Path, Path]]:
    """Return sorted list of (hr_path, lr_path) tuples matched by relative path.

    Files present in HR but missing from LR (or vice-versa) are logged as
    warnings and skipped.
    """
    hr_map: Dict[Path, Path] = {
        f.relative_to(hr_root): f
        for f in sorted(hr_root.rglob("*"))
        if f.suffix in TIF_EXTENSIONS
    }
    lr_map: Dict[Path, Path] = {
        f.relative_to(lr_root): f
        for f in sorted(lr_root.rglob("*"))
        if f.suffix in TIF_EXTENSIONS
    }

    missing_lr = set(hr_map) - set(lr_map)
    missing_hr = set(lr_map) - set(hr_map)
    if missing_lr:
        log.warning("%d HR tiles have no matching LR tile and will be skipped.", len(missing_lr))
    if missing_hr:
        log.warning("%d LR tiles have no matching HR tile and will be ignored.", len(missing_hr))

    matched = sorted(set(hr_map) & set(lr_map))
    log.info("Dataset: %d matched HR/LR tile pairs.", len(matched))
    return [(hr_map[k], lr_map[k]) for k in matched]


def _pad_to_size(arr: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    """Zero-pad a (C, H, W) array so that H ≥ min_h and W ≥ min_w."""
    c, h, w = arr.shape
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    hflip: bool,
    vflip: bool,
    rot90: bool,
    rng:   np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply identical random spatial augmentation to an LR/HR pair.

    All transforms are applied consistently to both arrays so spatial
    correspondence is preserved.

    Parameters
    ----------
    lr, hr : (C, H, W) float32 arrays.
    hflip, vflip, rot90 : Which augmentations are enabled.
    rng : seeded numpy RNG (per-worker, seeded by DataLoader worker_init_fn).
    """
    if hflip and rng.random() < 0.5:
        lr = lr[:, :, ::-1].copy()
        hr = hr[:, :, ::-1].copy()

    if vflip and rng.random() < 0.5:
        lr = lr[:, ::-1, :].copy()
        hr = hr[:, ::-1, :].copy()

    if rot90:
        k = rng.integers(0, 4)   # 0, 1, 2 or 3 × 90°
        if k > 0:
            lr = np.rot90(lr, k, axes=(1, 2)).copy()
            hr = np.rot90(hr, k, axes=(1, 2)).copy()

    return lr, hr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SRTileDataset(Dataset):
    """Paired HR / LR GeoTIFF tile dataset for super-resolution fine-tuning.

    Parameters
    ----------
    hr_root : Directory containing HR tiles (produced by build_dataset.py).
    lr_root : Matching LR tile directory.
    scale : Spatial upscale factor (HR / LR).
    lr_patch_size : Side length of the LR patch returned per sample.
                    HR patch = lr_patch_size * scale.
                    Must be a multiple of the model's window_size.
    augment : Whether to apply random flip / rotation augmentation.
    augmentation_cfg : Dict with keys horizontal_flip, vertical_flip, rot90.
    dtype_max : Divisor used to normalise raw pixel values to [0, 1].
                65535 for uint16 data, 255 for uint8.
    """

    def __init__(
        self,
        hr_root:         Path,
        lr_root:         Path,
        scale:           int,
        lr_patch_size:   int,
        augment:         bool               = False,
        augmentation_cfg: Optional[dict]   = None,
        dtype_max:       float             = 65535.0,
    ) -> None:
        super().__init__()
        self.hr_root       = Path(hr_root)
        self.lr_root       = Path(lr_root)
        self.scale         = scale
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.augment       = augment
        self.dtype_max     = float(dtype_max)
        self.aug_cfg       = augmentation_cfg or {}

        self.pairs = _scan_pairs(self.hr_root, self.lr_root)
        if not self.pairs:
            raise RuntimeError(
                f"No matched HR/LR tile pairs found.\n"
                f"  HR root: {self.hr_root}\n"
                f"  LR root: {self.lr_root}"
            )

        # Per-worker RNG, initialised lazily in _get_rng().
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_rng(self) -> np.random.Generator:
        """Return the per-worker RNG, creating it on first access.

        worker_init_fn (set in scripts/training.py) re-seeds this via
        ``dataset._rng = np.random.default_rng(worker_seed)`` so each
        DataLoader worker produces a distinct random crop sequence.
        """
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def _read_tile(self, path: Path) -> np.ndarray:
        """Open *path* with rasterio and return a (C, H, W) float32 array
        normalised to [0, 1].

        File is opened and closed within this call — no persistent handles
        are stored so the dataset is safe across DataLoader workers.
        """
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
        data /= self.dtype_max
        return data   # (C, H, W), float32, [0, 1]

    def _random_crop(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random crop from a paired LR / HR tile.

        The LR crop position is sampled uniformly; the corresponding HR crop
        is obtained by multiplying offsets and size by the scale factor.
        Both arrays are zero-padded to the required minimum size beforehand
        so edge tiles smaller than lr_patch_size are handled gracefully.
        """
        ps  = self.lr_patch_size
        lr  = _pad_to_size(lr, ps, ps)
        hr  = _pad_to_size(hr, ps * self.scale, ps * self.scale)

        _, lh, lw = lr.shape
        top  = rng.integers(0, lh - ps + 1)
        left = rng.integers(0, lw - ps + 1)

        lr_crop = lr[:, top : top + ps,                  left : left + ps]
        hr_crop = hr[:, top * self.scale : (top  + ps) * self.scale,
                         left * self.scale : (left + ps) * self.scale]
        return lr_crop, hr_crop

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hr_path, lr_path = self.pairs[idx]
        rng = self._get_rng()

        lr = self._read_tile(lr_path)
        hr = self._read_tile(hr_path)

        lr, hr = self._random_crop(lr, hr, rng)

        if self.augment:
            lr, hr = _augment_pair(
                lr, hr,
                hflip = self.aug_cfg.get("horizontal_flip", True),
                vflip = self.aug_cfg.get("vertical_flip",   True),
                rot90 = self.aug_cfg.get("rot90",           True),
                rng   = rng,
            )

        return {
            "lr":   torch.from_numpy(lr.copy()),
            "hr":   torch.from_numpy(hr.copy()),
            "name": hr_path.stem,
        }


# ---------------------------------------------------------------------------
# Worker initialisation (used by DataLoader in scripts/training.py)
# ---------------------------------------------------------------------------

def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker's RNG independently.

    Called by PyTorch before the worker starts processing items.  We derive
    a unique seed from the base torch seed and the worker id so that
    different workers produce uncorrelated random crop sequences even when
    the epoch-level seed is fixed.
    """
    worker_seed = torch.initial_seed() % (2**32)
    # Patch the dataset's RNG on the *worker copy* of the dataset object.
    dataset = torch.utils.data.get_worker_info().dataset
    dataset._rng = np.random.default_rng(worker_seed + worker_id)
