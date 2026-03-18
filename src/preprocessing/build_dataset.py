"""
=============================================================================
build_dataset.py — Dataset Builder for Super-Resolution Training
=============================================================================
Tiles pre-processed HR (pansharpened) and LR (degraded) image pairs into
fixed-size patches and splits them into train / val sets.

Key guarantees
--------------
  * All tiles from the same source TIF go to the same split (train or val).
  * HR and LR tile filenames are identical so downstream dataloaders can
    match them trivially (same relative path under HR/ and LR/).
  * Acquisition groups (WO_* folders) are kept intact: every file under the
    same acquisition directory lands in the same split.

Directory contract
------------------
    <hr_root>/  <WO_dir>/  <stem>.TIF        (pansharpening.py output)
    <lr_root>/  <WO_dir>/  <stem>.TIF        (degrade_pipeline.py output)

    → <output_root>/train/HR/<WO_dir>/<stem>_row????_col????.TIF
    → <output_root>/train/LR/<WO_dir>/<stem>_row????_col????.TIF
    → <output_root>/val/HR/  ...
    → <output_root>/val/LR/  ...

=============================================================================
PARAMETERS  — edit this block
=============================================================================
"""

from __future__ import annotations

# ── Folders ───────────────────────────────────────────────────────────────────
HR_ROOT     = "data/pansharpened/HR"   # Source of pansharpened (high-res) TIFs
LR_ROOT     = "data/pansharpened/LR"   # Source of degraded (low-res) TIFs
OUTPUT_ROOT = "data/processed"         # Destination for tiled dataset

# ── Tiling ────────────────────────────────────────────────────────────────────
# HR tile size in pixels.  LR tile size is inferred as tile_size // scale_factor
# where scale_factor is computed per-pair from the file dimensions.
TILE_SIZE = 512

# ── Train / validation split ──────────────────────────────────────────────────
# Fraction of acquisition groups assigned to the *validation* set.
# Splitting is performed at the acquisition-group level: all files that share
# a WO_* parent directory are assigned to the same split.
VAL_RATIO = 0.2
SEED      = 42

# ── Output options ────────────────────────────────────────────────────────────
# GeoTIFF compression for output tiles.  "none" | "deflate" | "lzw"
COMPRESS  = "deflate"

# Tiles whose valid (non-zero) pixel fraction is below this threshold are
# discarded.  Set to 0.0 to keep all tiles including fully black/nodata ones.
MIN_VALID_FRACTION = 0.1

# When True, re-tile and overwrite any tiles that already exist on disk.
OVERWRITE = False

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

import hashlib
import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: 'tqdm' is required.  pip install tqdm")

try:
    import rasterio
    from rasterio.windows import Window
except ImportError:
    sys.exit("ERROR: 'rasterio' is required.  pip install rasterio")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TIF_EXTENSIONS = frozenset({".tif", ".TIF", ".tiff", ".TIFF"})

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImagePair:
    """A matched HR / LR file pair sharing the same relative path."""
    hr_path: Path
    lr_path: Path
    # Relative path from the HR/LR root, e.g. WO_.../IMG_...TIF
    rel_path: Path


@dataclass
class TilingStats:
    """Accumulated statistics for a single tiling run."""
    pairs_processed: int = 0
    pairs_skipped:   int = 0
    tiles_written:   int = 0
    tiles_dropped:   int = 0

    def __iadd__(self, other: "TilingStats") -> "TilingStats":
        self.pairs_processed += other.pairs_processed
        self.pairs_skipped   += other.pairs_skipped
        self.tiles_written   += other.tiles_written
        self.tiles_dropped   += other.tiles_dropped
        return self


# ---------------------------------------------------------------------------
# Discovery and matching
# ---------------------------------------------------------------------------

def _discover_tifs(root: Path) -> List[Path]:
    """Return all TIF files under *root*, sorted for determinism."""
    return sorted(f for f in root.rglob("*") if f.suffix in TIF_EXTENSIONS)


def discover_pairs(hr_root: Path, lr_root: Path) -> List[ImagePair]:
    """
    Match HR and LR files by their relative path.

    Expects LR files to share the same subfolder structure and filename as
    their HR counterparts (this is the naming convention of degrade_pipeline.py
    — output paths mirror the input structure).
    """
    hr_files: Dict[Path, Path] = {
        f.relative_to(hr_root): f for f in _discover_tifs(hr_root)
    }
    lr_files: Dict[Path, Path] = {
        f.relative_to(lr_root): f for f in _discover_tifs(lr_root)
    }

    hr_keys = set(hr_files)
    lr_keys = set(lr_files)

    unmatched_hr = hr_keys - lr_keys
    unmatched_lr = lr_keys - hr_keys

    if unmatched_hr:
        log.warning(
            "%d HR file(s) have no matching LR counterpart and will be skipped:\n  %s",
            len(unmatched_hr),
            "\n  ".join(str(p) for p in sorted(unmatched_hr)),
        )
    if unmatched_lr:
        log.warning(
            "%d LR file(s) have no matching HR counterpart and will be ignored:\n  %s",
            len(unmatched_lr),
            "\n  ".join(str(p) for p in sorted(unmatched_lr)),
        )

    pairs = [
        ImagePair(hr_path=hr_files[k], lr_path=lr_files[k], rel_path=k)
        for k in sorted(hr_keys & lr_keys)
    ]
    log.info("Matched %d HR/LR pair(s).", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Acquisition-group train/val split
# ---------------------------------------------------------------------------

def split_pairs(
    pairs:     List[ImagePair],
    val_ratio: float,
    seed:      int,
) -> Tuple[List[ImagePair], List[ImagePair]]:
    """
    Assign pairs to train / val ensuring that all files from the same
    acquisition group (WO_* parent directory) stay in the same split.

    Strategy
    --------
    1. Collect the set of unique acquisition groups (immediate parent of each
       rel_path, e.g. "WO_000373512_10_2_...").
    2. Shuffle groups deterministically with *seed*.
    3. Take the last ceil(n_groups * val_ratio) groups as validation.
    4. Map each pair back to its assigned split.
    """
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    # Group pairs by their acquisition directory (first path component).
    groups: Dict[str, List[ImagePair]] = {}
    for pair in pairs:
        group_key = pair.rel_path.parts[0] if len(pair.rel_path.parts) > 1 else "_root_"
        groups.setdefault(group_key, []).append(pair)

    group_names = sorted(groups)  # sort for determinism before shuffle
    rng = random.Random(seed)
    rng.shuffle(group_names)

    n_val    = max(1, math.ceil(len(group_names) * val_ratio)) if val_ratio > 0 else 0
    val_set  = set(group_names[-n_val:]) if n_val else set()
    train_set = set(group_names) - val_set

    train_pairs: List[ImagePair] = []
    val_pairs:   List[ImagePair] = []

    for name in group_names:
        if name in val_set:
            val_pairs.extend(groups[name])
        else:
            train_pairs.extend(groups[name])

    log.info(
        "Split  : %d group(s) → train=%d group(s) / %d file(s) | "
        "val=%d group(s) / %d file(s)",
        len(group_names),
        len(train_set), len(train_pairs),
        len(val_set),   len(val_pairs),
    )
    if val_set:
        log.info("Val groups : %s", ", ".join(sorted(val_set)))

    return train_pairs, val_pairs


# ---------------------------------------------------------------------------
# Scale-factor detection
# ---------------------------------------------------------------------------

def detect_scale_factor(hr_path: Path, lr_path: Path) -> int:
    """
    Infer the integer spatial scale factor between an HR and LR file pair.

    Reads only the file headers (no pixel data).  Raises ValueError if the
    dimensions are inconsistent (non-integer ratio).
    """
    with rasterio.open(hr_path) as hr_src, rasterio.open(lr_path) as lr_src:
        hr_w, hr_h = hr_src.width, hr_src.height
        lr_w, lr_h = lr_src.width, lr_src.height

    if lr_w == 0 or lr_h == 0:
        raise ValueError(f"LR image has zero dimension: {lr_path}")

    scale_x = hr_w / lr_w
    scale_y = hr_h / lr_h

    if abs(scale_x - scale_y) > 0.05:
        raise ValueError(
            f"Asymmetric scale factors (x={scale_x:.3f}, y={scale_y:.3f}) "
            f"for pair {hr_path.name}"
        )

    scale = round(scale_x)
    if abs(scale_x - scale) > 0.05:
        raise ValueError(
            f"Non-integer scale factor ({scale_x:.3f}) for pair {hr_path.name}"
        )

    return scale


# ---------------------------------------------------------------------------
# Tiling primitives
# ---------------------------------------------------------------------------

def _iter_tile_windows(
    width:     int,
    height:    int,
    tile_size: int,
) -> Tuple[int, int, Window]:
    """
    Yield (row_idx, col_idx, window) for a non-overlapping tile grid.

    The last column / row may be narrower than tile_size when the image
    dimensions are not evenly divisible.
    """
    n_rows = math.ceil(height / tile_size)
    n_cols = math.ceil(width  / tile_size)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            x_off = col_idx * tile_size
            y_off = row_idx * tile_size
            w     = min(tile_size, width  - x_off)
            h     = min(tile_size, height - y_off)
            yield row_idx, col_idx, Window(x_off, y_off, w, h)


def _is_valid_tile(
    data:               np.ndarray,
    min_valid_fraction: float,
) -> bool:
    """Return True if the fraction of non-zero pixels meets the threshold."""
    if min_valid_fraction <= 0.0:
        return True
    valid = np.count_nonzero(data)
    total = data.size
    return (valid / total) >= min_valid_fraction


def _write_tile(
    data:      np.ndarray,
    out_path:  Path,
    profile:   dict,
    window:    Window,
    compress:  str,
) -> None:
    """Write a (C, H, W) array as a GeoTIFF tile."""

    def _block_size(pixels: int, preferred: int = 256) -> int:
        """Return the largest multiple of 16 that is ≤ min(preferred, pixels).

        GDAL/GeoTIFF requires block dimensions to be multiples of 16.
        Edge tiles are narrower than the preferred block size, so we snap
        down to the nearest valid value (minimum 16).
        """
        cap = min(preferred, pixels)
        snapped = (cap // 16) * 16
        return max(snapped, 16)

    tile_profile = profile.copy()
    tile_profile.update(
        width    = window.width,
        height   = window.height,
        compress = None if compress.lower() == "none" else compress,
        tiled    = True,
        blockxsize = _block_size(window.width),
        blockysize = _block_size(window.height),
    )
    # Remove BigTIFF flag for small tiles — avoids unnecessary overhead.
    tile_profile.pop("BIGTIFF", None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **tile_profile) as dst:
        dst.write(data)


# ---------------------------------------------------------------------------
# Per-pair tiling
# ---------------------------------------------------------------------------

def tile_pair(
    pair:               ImagePair,
    split_hr_root:      Path,
    split_lr_root:      Path,
    tile_size:          int,
    compress:           str,
    min_valid_fraction: float,
    overwrite:          bool,
) -> TilingStats:
    """
    Tile one HR/LR pair and write output tiles to *split_hr_root* / *split_lr_root*.

    The LR tile window is derived from the HR tile window by dividing all
    offsets and dimensions by the detected integer scale factor.

    Output tile naming convention:
        {original_stem}_row{row:04d}_col{col:04d}.TIF
    """
    stats = TilingStats()

    try:
        scale = detect_scale_factor(pair.hr_path, pair.lr_path)
    except ValueError as exc:
        log.error("Skipping pair %s : %s", pair.rel_path, exc)
        stats.pairs_skipped += 1
        return stats

    lr_tile_size = max(1, tile_size // scale)

    # Derive output directory relative to each split root, preserving the
    # WO acquisition subfolder (parent of the file within HR/LR root).
    tile_subdir = pair.rel_path.parent  # e.g. "WO_000373512_10_2_..."

    out_hr_dir = split_hr_root / tile_subdir
    out_lr_dir = split_lr_root / tile_subdir

    try:
        with rasterio.open(pair.hr_path) as hr_src, \
             rasterio.open(pair.lr_path) as lr_src:

            hr_profile = hr_src.meta.copy()
            lr_profile = lr_src.meta.copy()
            stem       = pair.hr_path.stem

            n_tiles = (
                math.ceil(hr_src.height / tile_size)
                * math.ceil(hr_src.width  / tile_size)
            )
            tqdm.write(
                f"  {pair.rel_path}  "
                f"[HR {hr_src.width}×{hr_src.height}px  "
                f"LR {lr_src.width}×{lr_src.height}px  "
                f"scale={scale}  ~{n_tiles} tile(s)]"
            )

            for row_idx, col_idx, hr_win in _iter_tile_windows(
                hr_src.width, hr_src.height, tile_size
            ):
                tile_name = f"{stem}_row{row_idx:04d}_col{col_idx:04d}.TIF"
                hr_out    = out_hr_dir / tile_name
                lr_out    = out_lr_dir / tile_name

                if not overwrite and hr_out.exists() and lr_out.exists():
                    stats.tiles_written += 1  # count as already done
                    continue

                hr_data = hr_src.read(window=hr_win)

                if not _is_valid_tile(hr_data, min_valid_fraction):
                    stats.tiles_dropped += 1
                    continue

                # Corresponding LR window: divide HR pixel coordinates by scale.
                lr_win = Window(
                    col_off = hr_win.col_off // scale,
                    row_off = hr_win.row_off // scale,
                    width   = hr_win.width   // scale,
                    height  = hr_win.height  // scale,
                )

                # Guard against rounding producing a zero-size LR window.
                if lr_win.width < 1 or lr_win.height < 1:
                    log.debug(
                        "Skipping tile (%d,%d) of %s — LR window too small after scaling.",
                        row_idx, col_idx, stem,
                    )
                    stats.tiles_dropped += 1
                    continue

                lr_data = lr_src.read(window=lr_win)

                # Update spatial transform for HR tile.
                hr_tile_profile = hr_profile.copy()
                hr_tile_profile["transform"] = rasterio.windows.transform(
                    hr_win, hr_src.transform
                )

                # Update spatial transform for LR tile.
                lr_tile_profile = lr_profile.copy()
                lr_tile_profile["transform"] = rasterio.windows.transform(
                    lr_win, lr_src.transform
                )

                _write_tile(hr_data, hr_out, hr_tile_profile, hr_win, compress)
                _write_tile(lr_data, lr_out, lr_tile_profile, lr_win, compress)

                stats.tiles_written += 1

    except Exception as exc:
        log.error("FAILED  %s : %s", pair.rel_path, exc, exc_info=True)
        stats.pairs_skipped += 1
        return stats

    stats.pairs_processed += 1
    return stats


# ---------------------------------------------------------------------------
# Split-level driver
# ---------------------------------------------------------------------------

def tile_split(
    pairs:              List[ImagePair],
    split_name:         str,
    output_root:        Path,
    tile_size:          int,
    compress:           str,
    min_valid_fraction: float,
    overwrite:          bool,
) -> TilingStats:
    """Tile all pairs assigned to a split and write results under *output_root*."""
    split_hr_root = output_root / split_name / "HR"
    split_lr_root = output_root / split_name / "LR"
    split_hr_root.mkdir(parents=True, exist_ok=True)
    split_lr_root.mkdir(parents=True, exist_ok=True)

    log.info("─── %s  (%d pair(s)) ────────────────", split_name.upper(), len(pairs))
    total = TilingStats()

    with tqdm(pairs, desc=split_name, unit="pair", colour="cyan") as bar:
        for pair in bar:
            bar.set_postfix_str(pair.rel_path.name[:55], refresh=True)
            stats = tile_pair(
                pair               = pair,
                split_hr_root      = split_hr_root,
                split_lr_root      = split_lr_root,
                tile_size          = tile_size,
                compress           = compress,
                min_valid_fraction = min_valid_fraction,
                overwrite          = overwrite,
            )
            total += stats
            bar.set_postfix(
                ok      = total.pairs_processed,
                skip    = total.pairs_skipped,
                tiles   = total.tiles_written,
                dropped = total.tiles_dropped,
                refresh = True,
            )

    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    hr_root     = Path(HR_ROOT).resolve()
    lr_root     = Path(LR_ROOT).resolve()
    output_root = Path(OUTPUT_ROOT).resolve()

    # ── Pre-flight checks ──────────────────────────────────────────────────────
    for label, path in (("HR_ROOT", hr_root), ("LR_ROOT", lr_root)):
        if not path.exists():
            sys.exit(f"ERROR: {label} does not exist: {path}")

    log.info("HR root    : %s", hr_root)
    log.info("LR root    : %s", lr_root)
    log.info("Output     : %s", output_root)
    log.info(
        "Tile size  : %d px (HR)  |  VAL_RATIO=%.2f  |  SEED=%d",
        TILE_SIZE, VAL_RATIO, SEED,
    )
    log.info(
        "Compress   : %s  |  min_valid=%.2f  |  overwrite=%s",
        COMPRESS, MIN_VALID_FRACTION, OVERWRITE,
    )

    # ── Discover and match pairs ───────────────────────────────────────────────
    pairs = discover_pairs(hr_root, lr_root)
    if not pairs:
        sys.exit("ERROR: No matched HR/LR pairs found. Check HR_ROOT and LR_ROOT.")

    # ── Train / val split ──────────────────────────────────────────────────────
    train_pairs, val_pairs = split_pairs(pairs, VAL_RATIO, SEED)

    # ── Tile each split ────────────────────────────────────────────────────────
    grand_total = TilingStats()

    for split_name, split_pairs_ in (("train", train_pairs), ("val", val_pairs)):
        if not split_pairs_:
            log.warning("No pairs assigned to '%s' — skipping.", split_name)
            continue
        stats = tile_split(
            pairs              = split_pairs_,
            split_name         = split_name,
            output_root        = output_root,
            tile_size          = TILE_SIZE,
            compress           = COMPRESS,
            min_valid_fraction = MIN_VALID_FRACTION,
            overwrite          = OVERWRITE,
        )
        log.info(
            "[%s] pairs=%d  skipped=%d  tiles_written=%d  tiles_dropped=%d",
            split_name, stats.pairs_processed, stats.pairs_skipped,
            stats.tiles_written, stats.tiles_dropped,
        )
        grand_total += stats

    # ── Summary ────────────────────────────────────────────────────────────────
    separator = "─" * 60
    tqdm.write(f"\n{separator}")
    tqdm.write(
        f"  Done"
        f"  │  pairs ✓ {grand_total.pairs_processed}"
        f"  │  pairs ✗ {grand_total.pairs_skipped}"
        f"  │  tiles written {grand_total.tiles_written}"
        f"  │  tiles dropped {grand_total.tiles_dropped}"
    )
    tqdm.write(separator)


if __name__ == "__main__":
    main()