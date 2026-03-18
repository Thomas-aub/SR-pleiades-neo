"""
tiling.py — Tile geometry and radiometric normalisation for large GeoTIFF processing.
=====================================================================================

Two responsibilities:

  1. Geometry — tile grid iteration, padded read windows, overlap cropping.
     Mirrors ImageTiler conventions (tiler.py):
       stride  = tile_size - overlap
       n_cols  = ceil(W / stride)
       x_off   = col_idx * stride
       Window(x_off, y_off, win_w, win_h)
       rasterio.windows.transform(window, src_transform)   (= src.window_transform)

     Extension over tiler.py: overlap is applied *symmetrically* (leading +
     trailing).  Trailing overlap alone, as in tiler.py, protects only the
     downstream edge; FFT-based ops (mtf_blur, spectral_misalign) also
     produce wrap-around artefacts on the leading edge without this addition.

  2. Radiometric normalisation — global percentile stretch + gamma correction.
     Ported directly from tiler.py (ImageTiler.tile_image Steps 1 & 2):
       • Stretch params are computed once from a thumbnail so every tile
         in a mosaic shares a consistent colour rendering.
       • Per-tile: normalise → gamma → scale to uint8 [1, 254].

Public API
----------
    TileConfig(tile_size, overlap)
    Tile                                   – frozen geometry dataclass
    iter_tiles(W, H, transform, cfg)       – generator, row-major order
    crop_tile(arr, tile, scale)            – strip overlap → data region
    output_window(tile, scale)             – rasterio.Window for write
    tile_count(W, H, cfg)                 – total tile count

    StretchParams(lo, hi)                  – per-band stretch values
    compute_stretch_params(src, ...)       – thumbnail percentile calc (tiler.py §1)
    apply_stretch_gamma(bands, params, γ) – normalise + gamma → uint8 (tiler.py §2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


# =============================================================================
# Part 1 — Tile geometry
# =============================================================================

@dataclass(frozen=True)
class TileConfig:
    """Tiling hyperparameters — field names mirror TilerConfig (tiler.py).

    Attributes
    ----------
    tile_size : Read-window side in pixels (= stride + overlap).
                stride = tile_size - overlap  (same formula as tiler.py).
    overlap   : Pixels of overlap on each side.  Applied both as the trailing
                overlap (identical to tiler.py) and as a leading overlap added
                by iter_tiles for FFT boundary protection.
                Must satisfy: 0 ≤ overlap < tile_size // 2.
    """
    tile_size: int
    overlap:   int

    def __post_init__(self) -> None:
        if self.tile_size < 1:
            raise ValueError(f"tile_size must be ≥ 1, got {self.tile_size}")
        if not (0 <= self.overlap < self.tile_size // 2):
            raise ValueError(
                f"overlap ({self.overlap}) must be in [0, tile_size // 2) "
                f"= [0, {self.tile_size // 2})"
            )

    @property
    def stride(self) -> int:
        """Non-overlapping step between tile origins (tile_size - overlap)."""
        return self.tile_size - self.overlap


@dataclass(frozen=True)
class Tile:
    """Geometry of one tile in the tiling grid.

    Attributes
    ----------
    row, col      : Zero-based grid indices (row-major).
    data_window   : Canonical (no-overlap) tile extent in source pixels.
                    Defines which pixels this tile *owns* for writing.
    read_window   : data_window expanded symmetrically by overlap on all sides,
                    clipped to image bounds.  Pass to src.read(window=...).
    src_transform : Affine anchored at read_window's top-left pixel.
                    Equivalent to src.window_transform(read_window).
    """
    row:           int
    col:           int
    data_window:   Window
    read_window:   Window
    src_transform: Affine


def iter_tiles(
    src_width:     int,
    src_height:    int,
    src_transform: Affine,
    config:        TileConfig,
) -> Generator[Tile, None, None]:
    """Yield Tile objects covering the full source image in row-major order.

    Grid layout mirrors tiler.py exactly:
        stride = tile_size - overlap
        n_cols = ceil(W / stride)
        n_rows = ceil(H / stride)
        x_off  = col_idx * stride
        y_off  = row_idx * stride
    """
    stride = config.stride
    n_cols = math.ceil(src_width  / stride)
    n_rows = math.ceil(src_height / stride)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):

            # ── Canonical data region (mirrors tiler.py window) ───────────────
            x_off  = col_idx * stride
            y_off  = row_idx * stride
            data_w = min(stride, src_width  - x_off)
            data_h = min(stride, src_height - y_off)
            data_window = Window(x_off, y_off, data_w, data_h)

            # ── Symmetric read window: extend by overlap on all four sides ─────
            # Trailing overlap matches tiler.py (reads up to x_off + tile_size).
            # Leading overlap is additional and provides FFT boundary protection.
            read_x  = max(0,          x_off  - config.overlap)
            read_y  = max(0,          y_off  - config.overlap)
            read_x2 = min(src_width,  x_off  + data_w + config.overlap)
            read_y2 = min(src_height, y_off  + data_h + config.overlap)
            read_window = Window(read_x, read_y, read_x2 - read_x, read_y2 - read_y)

            # ── Affine at read_window top-left (= src.window_transform) ────────
            tile_transform = window_transform(read_window, src_transform)

            yield Tile(
                row           = row_idx,
                col           = col_idx,
                data_window   = data_window,
                read_window   = read_window,
                src_transform = tile_transform,
            )


def crop_tile(arr: np.ndarray, tile: Tile, scale: int = 1) -> np.ndarray:
    """Crop a processed tile array back to its canonical data region.

    Strips the leading overlap (read_window − data_window origin delta) and
    trailing overlap, in output-pixel coordinates (source pixels ÷ scale).

    Parameters
    ----------
    arr   : 2-D processed array (H × W) in output pixels.
    tile  : The Tile whose geometry defines how much to crop.
    scale : Total integer spatial downscale applied by the pipeline.
    """
    lead_col = (tile.data_window.col_off - tile.read_window.col_off) // scale
    lead_row = (tile.data_window.row_off - tile.read_window.row_off) // scale
    data_w   = tile.data_window.width  // scale
    data_h   = tile.data_window.height // scale
    return arr[lead_row : lead_row + data_h, lead_col : lead_col + data_w]


def output_window(tile: Tile, scale: int = 1) -> Window:
    """Return the rasterio Window for writing a processed tile into the output.

    Derived from data_window divided by the pipeline's spatial scale factor.
    """
    dw = tile.data_window
    return Window(
        col_off = dw.col_off // scale,
        row_off = dw.row_off // scale,
        width   = dw.width   // scale,
        height  = dw.height  // scale,
    )


def tile_count(src_width: int, src_height: int, config: TileConfig) -> int:
    """Total number of tiles iter_tiles will yield."""
    s = config.stride
    return math.ceil(src_width / s) * math.ceil(src_height / s)


# =============================================================================
# Part 2 — Radiometric normalisation  (ported from tiler.py)
# =============================================================================

@dataclass(frozen=True)
class StretchParams:
    """Per-band linear stretch values derived from a global thumbnail.

    Attributes
    ----------
    lo : Raw pixel value mapped to 0.0 (lower percentile of valid pixels).
    hi : Raw pixel value mapped to 1.0 (upper percentile of valid pixels).
    """
    lo: float
    hi: float


def compute_stretch_params(
    src:            rasterio.DatasetReader,
    bands_cfg:      Optional[List[int]] = None,
    min_percentile: float = 1.0,
    max_percentile: float = 99.0,
) -> List[StretchParams]:
    """Compute global stretch parameters from a thumbnail — mirrors tiler.py §1.

    Reads a downsampled overview (max 1024 px on the longest side) so the
    cost is negligible even for 30 000 px images.  Zero-valued pixels are
    excluded (treated as nodata / black padding), matching tiler.py behaviour.

    Computing once before the tile loop ensures every tile in a mosaic shares
    the same colour rendering — preventing per-tile contrast drift.

    Parameters
    ----------
    src            : Open rasterio dataset.
    bands_cfg      : 1-based band indices to use.  None → auto (first 3, or
                     replicate single band to RGB — same as tiler.py).
    min_percentile : Lower percentile for lo value.
    max_percentile : Upper percentile for hi value.

    Returns
    -------
    One StretchParams per output band (always 3 after band selection).
    """
    W, H      = src.width, src.height
    n_bands   = src.count
    thumb_scale = 1024.0 / max(W, H)
    out_shape   = (
        n_bands,
        max(int(H * thumb_scale), 1),
        max(int(W * thumb_scale), 1),
    )

    thumbnail = src.read(out_shape=out_shape, resampling=Resampling.bilinear)
    rgb_thumb = _select_bands(thumbnail, n_bands, bands_cfg)

    params: List[StretchParams] = []
    for i in range(rgb_thumb.shape[0]):
        band         = rgb_thumb[i].astype(np.float32)
        valid_pixels = band[band > 0]           # exclude zero padding (tiler.py)

        if len(valid_pixels) == 0:
            params.append(StretchParams(lo=0.0, hi=1.0))
        else:
            lo = float(np.percentile(valid_pixels, min_percentile))
            hi = float(np.percentile(valid_pixels, max_percentile))
            if hi <= lo:
                hi = lo + 1.0
            params.append(StretchParams(lo=lo, hi=hi))

    del thumbnail, rgb_thumb
    return params


def apply_stretch_gamma(
    bands:   np.ndarray,             # (n_bands, H, W) — any numeric dtype
    params:  List[StretchParams],
    gamma:   float = 0.6,
    bands_cfg: Optional[List[int]] = None,
) -> np.ndarray:                     # (3, H, W) uint8
    """Normalise, apply gamma, and scale to uint8 — mirrors tiler.py §2.

    Steps applied per band (identical to ImageTiler.tile_image Step 2):
        1. Normalise to [0, 1] using pre-computed StretchParams.
        2. Clip to [0, 1].
        3. Apply gamma curve: s = s ** gamma.
        4. Scale to uint8 [1, 254]  (avoids exact-black nodata / exact-white).

    Parameters
    ----------
    bands     : Source tile data, shape (n_bands, H, W).
    params    : One StretchParams per output band (from compute_stretch_params).
    gamma     : Gamma exponent  (<1 brightens midtones, typical default 0.6).
    bands_cfg : 1-based band selection (passed to _select_bands).  None = auto.

    Returns
    -------
    uint8 array of shape (3, H, W).
    """
    n_src   = bands.shape[0]
    rgb     = _select_bands(bands, n_src, bands_cfg)
    out     = np.empty(rgb.shape, dtype=np.float32)

    for i in range(rgb.shape[0]):
        lo, hi     = params[i].lo, params[i].hi
        s          = (rgb[i].astype(np.float32) - lo) / (hi - lo)
        s          = np.clip(s, 0.0, 1.0)
        s          = np.power(s, gamma)                 # gamma correction
        out[i]     = np.clip(s * 255.0, 1.0, 254.0)    # [1, 254] as in tiler.py

    return out.astype(np.uint8)


def pad_tile(arr: np.ndarray, tile_size: int) -> np.ndarray:
    """Zero-pad a (C, H, W) array to (C, tile_size, tile_size).

    Identical to tiler.py _pad_tile — edge tiles are zero-padded to a uniform
    size so every output tile on disk has the same dimensions.
    """
    pad_h = tile_size - arr.shape[1]
    pad_w = tile_size - arr.shape[2]
    if pad_h == 0 and pad_w == 0:
        return arr
    return np.pad(arr, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)


# ---------------------------------------------------------------------------
# Internal helpers  (ported from tiler.py, kept private)
# ---------------------------------------------------------------------------

def _select_bands(
    data:       np.ndarray,
    n_src_bands: int,
    bands_cfg:  Optional[List[int]],
) -> np.ndarray:
    """Select / replicate source bands into a 3-channel array.

    Mirrors tiler.py _select_bands exactly:
      - bands_cfg not None → use specified 1-based indices.
      - n_src_bands ≥ 3   → take first three.
      - panchromatic       → replicate single band × 3.
    """
    if bands_cfg is not None:
        for b in bands_cfg:
            if b < 1 or b > n_src_bands:
                raise ValueError(
                    f"Band {b} out of range for a {n_src_bands}-band image."
                )
        selected = data[[b - 1 for b in bands_cfg]]
    elif n_src_bands >= 3:
        selected = data[:3]
    else:
        selected = data[:1]

    if selected.shape[0] == 1:
        selected = np.repeat(selected, 3, axis=0)   # pan → RGB

    return selected  # (3, H, W)