"""
=============================================================================
Pansharpening Script for Pléiades NEO Imagery
=============================================================================
Matches RGB (MS-FS) and PAN image pairs based on shared filename tokens,
then applies Brovey pansharpening to produce high-resolution color images.

Matching logic uses three shared tokens extracted from filenames:
  - STD_<token>   e.g. STD_202309210716008
  - ORT_<token>   e.g. ORT_PWOI
  - Last code     e.g. R1C1

Output images are saved as GeoTIFF in the `pansharpened/` subfolder.
=============================================================================
PARAMETERS
=============================================================================
"""

# ── Input / output ────────────────────────────────────────────────────────────
INPUT_FOLDER      = "raw/"          # Folder that contains all source .TIF files
OUTPUT_FOLDER     = "HR"  # Subfolder for pansharpened results

# ── File-matching patterns (substrings that distinguish each modality) ────────
RGB_MARKER        = "_RGB_"      # Substring present in RGB/multispectral filenames
PAN_MARKER        = "_P_"        # Substring present in PAN filenames
TIF_EXTENSIONS    = (".tif", ".TIF", ".tiff", ".TIFF")  # Accepted extensions

# ── Pansharpening method ──────────────────────────────────────────────────────
# Supported: "brovey" | "hsv" | "simple_mean"
PANSHARPEN_METHOD = "brovey"

# ── Resampling algorithm used when upscaling the RGB to PAN resolution ────────
# Any rasterio / GDAL resampling name:
#   "nearest" | "bilinear" | "cubic" | "cubic_spline" | "lanczos"
RESAMPLE_ALGO     = "lanczos"

# ── Output options ────────────────────────────────────────────────────────────
OUTPUT_DTYPE      = "uint16"     # Output pixel depth: "uint8" or "uint16"
OUTPUT_SUFFIX     = "_PANSHARP"  # Appended to the RGB filename stem
COMPRESS          = "none"        # GeoTIFF compression: "lzw" | "deflate" | "none"

# ── GPU acceleration ─────────────────────────────────────────────────────────
# Set to True to use CuPy (NVIDIA CUDA) for the pansharpening math.
USE_GPU           = True

# ── Memory management ────────────────────────────────────────────────────────
# Height (in PAN pixels) of each processing strip.
CHUNK_ROWS        = 2048

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL         = "INFO"       # "DEBUG" | "INFO" | "WARNING" | "ERROR"




# =============================================================================
# END OF PARAMETERS
# =============================================================================

import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("pip install tqdm")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import Window, transform as window_transform
except ImportError:
    sys.exit("pip install rasterio")

# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------

GPU_AVAILABLE = False
xp = np

if USE_GPU:
    try:
        import cupy as cp
        cp.zeros(1)
        xp = cp
        GPU_AVAILABLE = True
    except Exception:
        print("GPU disabled — CuPy unavailable")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resampling map
# ---------------------------------------------------------------------------

RESAMPLE_MAP = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "cubic_spline": Resampling.cubic_spline,
    "lanczos": Resampling.lanczos,
}

# ---------------------------------------------------------------------------
# Filename token extraction
# ---------------------------------------------------------------------------

def extract_match_key(filename: str):

    std = re.search(r"STD_([^_]+)", filename)
    ort = re.search(r"ORT_([^_]+)", filename)
    rc = re.search(r"(R\d+C\d+)", filename)

    if not (std and ort and rc):
        return None

    return std.group(1), ort.group(1), rc.group(1)


# ---------------------------------------------------------------------------
# Discover pairs
# ---------------------------------------------------------------------------

def discover_pairs(root: Path):

    rgb_files = {}
    pan_files = {}

    files = [f for f in root.rglob("*") if f.suffix in TIF_EXTENSIONS]

    for f in tqdm(files, desc="Scanning files"):

        key = extract_match_key(f.name)
        if key is None:
            continue

        if RGB_MARKER in f.name:
            rgb_files[key] = f

        elif PAN_MARKER in f.name:
            pan_files[key] = f

    pairs = []

    for k, rgb in rgb_files.items():
        if k in pan_files:
            pairs.append((rgb, pan_files[k]))

    return pairs


# ---------------------------------------------------------------------------
# Pansharpen methods
# ---------------------------------------------------------------------------

def pansharpen_brovey(rgb, pan):

    rgb_sum = xp.sum(rgb, axis=0, keepdims=True)
    rgb_sum = xp.where(rgb_sum == 0, 1, rgb_sum)

    return (rgb / rgb_sum) * pan


def pansharpen_simple_mean(rgb, pan):

    return (rgb + pan) * 0.5


PANSHARPEN_FUNCS = {
    "brovey": pansharpen_brovey,
    "simple_mean": pansharpen_simple_mean,
}


# ---------------------------------------------------------------------------
# Resample helper
# ---------------------------------------------------------------------------

def resample_strip(src, dst_transform, dst_crs, dst_rows, dst_cols, bands, resampling):

    dst = np.empty((len(bands), dst_rows, dst_cols), dtype=np.float32)

    for i, b in enumerate(bands):

        reproject(
            source=rasterio.band(src, b),
            destination=dst[i],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    return dst


# ---------------------------------------------------------------------------
# Process pair
# ---------------------------------------------------------------------------

def process_pair(rgb_path, pan_path, output_root, input_root):

    pansharpen_fn = PANSHARPEN_FUNCS[PANSHARPEN_METHOD]
    resampling = RESAMPLE_MAP[RESAMPLE_ALGO]

    rel = rgb_path.parent.relative_to(input_root)
    out_dir = output_root / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (rgb_path.stem + OUTPUT_SUFFIX + ".TIF")

    try:

        with rasterio.open(pan_path) as pan_src, rasterio.open(rgb_path) as rgb_src:

            pan_rows, pan_cols = pan_src.height, pan_src.width
            pan_transform = pan_src.transform
            pan_crs = pan_src.crs

            pan_max = pan_src.read(1, masked=True).max()
            rgb_max = rgb_src.read([1,2,3], masked=True).max()

            pan_max = pan_max if pan_max else 1
            rgb_max = rgb_max if rgb_max else 1

            profile = pan_src.profile.copy()
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)

            profile.update(
                count=3,
                dtype=OUTPUT_DTYPE,
                compress=None if COMPRESS == "none" else COMPRESS,
                tiled=True,
                blockxsize=512,
                blockysize=512,
            )

            dtype_max = np.iinfo(np.dtype(OUTPUT_DTYPE)).max

            n_strips = (pan_rows + CHUNK_ROWS - 1) // CHUNK_ROWS

            with rasterio.open(out_path, "w", **profile) as dst:

                for strip in tqdm(range(n_strips), leave=False):

                    row0 = strip * CHUNK_ROWS
                    row1 = min(row0 + CHUNK_ROWS, pan_rows)
                    h = row1 - row0

                    window = Window(0, row0, pan_cols, h)

                    transform = window_transform(window, pan_transform)

                    pan_cpu = pan_src.read(1, window=window).astype(np.float32)

                    rgb_cpu = resample_strip(
                        rgb_src,
                        transform,
                        pan_crs,
                        h,
                        pan_cols,
                        [1,2,3],
                        resampling,
                    )

                    if GPU_AVAILABLE:

                        pan = cp.asarray(pan_cpu)
                        rgb = cp.asarray(rgb_cpu)

                    else:

                        pan = pan_cpu
                        rgb = rgb_cpu

                    pan_norm = pan / pan_max
                    rgb_norm = rgb / rgb_max

                    pan_norm = pan_norm[xp.newaxis,:,:]

                    sharp = pansharpen_fn(rgb_norm, pan_norm)

                    sharp = xp.clip(sharp * dtype_max,0,dtype_max).astype(OUTPUT_DTYPE)

                    if GPU_AVAILABLE:
                        sharp = cp.asnumpy(sharp)

                    dst.write(sharp, window=window)

        return out_path

    except Exception as e:

        log.error(f"FAILED {rgb_path.name} : {e}")

        if out_path.exists():
            out_path.unlink()

        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    input_root = Path(INPUT_FOLDER).resolve()
    output_root = input_root / OUTPUT_FOLDER

    output_root.mkdir(exist_ok=True)

    pairs = discover_pairs(input_root)

    if not pairs:
        print("No pairs found")
        return

    success = 0
    failed = 0

    for rgb, pan in tqdm(pairs, desc="Pansharpening"):

        r = process_pair(rgb, pan, output_root, input_root)

        if r:
            success += 1
        else:
            failed += 1

    print()
    print("Finished")
    print("Success:", success)
    print("Failed :", failed)


if __name__ == "__main__":
    main()