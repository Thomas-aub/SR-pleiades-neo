"""
src/inference/predict.py — SwinIR super-resolution inference engine.
=====================================================================
Loads a SwinIR checkpoint, runs tiled inference on an LR GeoTIFF, and
writes a geographically registered SR GeoTIFF to the output directory.

Key design decisions
--------------------
Tiled inference with overlap blending
    Large images cannot fit in a single forward pass.  We split the LR input
    into overlapping tiles, run the model on each, then blend the SR outputs
    in the overlap zone using a linear (ramp) weight mask.  This eliminates
    the hard seam artefacts that appear with simple non-overlapping tiling.

Correct GeoTIFF output
    The output profile is built from scratch (same approach as build_dataset)
    to guarantee PHOTOMETRIC=RGB, explicit INTERLEAVE, and accurate STATISTICS
    tags.  The spatial transform is updated so the SR pixel spacing is
    input_pixel_size / upscale_factor, keeping the image geo-registered.

No float overflow
    All intermediate accumulation uses float32.  The final result is clipped
    to [0, 1] before scaling to the output dtype.

Public API
----------
    load_model(cfg, device)                          → nn.Module
    run_inference(cfg, model, device)                → Path (output file)
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

try:
    import rasterio
    import rasterio.transform
    from rasterio.windows import Window
except ImportError as exc:
    raise ImportError("rasterio is required.  pip install rasterio") from exc

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg, device: torch.device) -> nn.Module:
    """Load SwinIR from a checkpoint and move it to *device*.

    Supports both:
    - Fine-tuned checkpoints (key ``"model"``) produced by trainer.py.
    - Original KAIR/SwinIR pretrained checkpoints (key ``"params"``).

    The key to look for is controlled by ``cfg.model.checkpoint_key``.
    """
    import sys

    kair_root = Path(cfg.kair_root).resolve()
    if not kair_root.exists():
        raise FileNotFoundError(
            f"KAIR root not found: {kair_root}\n"
            "Clone it with:  git clone https://github.com/cszn/KAIR"
        )

    sys.path.insert(0, str(kair_root))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="torch.meshgrid.*indexing",
                                    category=UserWarning)
            warnings.filterwarnings("ignore", message="Importing from timm.models.layers",
                                    category=FutureWarning)
            from models.network_swinir import SwinIR  # noqa: PLC0415
    finally:
        sys.path.remove(str(kair_root))

    m = cfg.model
    model = SwinIR(
        upscale         = int(m.upscale),
        in_chans        = int(m.in_chans),
        img_size        = int(m.img_size),
        window_size     = int(m.window_size),
        img_range       = float(m.img_range),
        depths          = list(m.depths),
        embed_dim       = int(m.embed_dim),
        num_heads       = list(m.num_heads),
        mlp_ratio       = float(m.mlp_ratio),
        upsampler       = str(m.upsampler),
        resi_connection = str(m.resi_connection),
    )

    ckpt_path = Path(cfg.model.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Set model.checkpoint_path in inference.yaml."
        )

    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    key = str(cfg.model.checkpoint_key) if cfg.model.checkpoint_key else None
    if isinstance(ckpt, dict):
        # Try the configured key first, then common fallbacks.
        for candidate in ([key] if key else []) + ["model", "params", "params_ema"]:
            if candidate and candidate in ckpt:
                state_dict = ckpt[candidate]
                log.info("Using state_dict key: '%s'", candidate)
                break
        else:
            # Assume the checkpoint IS the state dict.
            state_dict = ckpt
            log.info("Checkpoint has no recognised key — using as flat state dict.")
    else:
        state_dict = ckpt

    # Strip DataParallel prefix if present.
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log.warning("Missing keys (%d): %s …", len(missing), missing[:3])
    if unexpected:
        log.warning("Unexpected keys (%d): %s …", len(unexpected), unexpected[:3])

    model = model.to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    log.info("SwinIR loaded — %s parameters", f"{total:,}")
    return model


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _build_weight_ramp(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
    """Build a (H, W) float32 weight mask with linear ramps in the overlap zone.

    Interior pixels have weight 1.0.  Pixels within *overlap* of any edge
    ramp linearly from 0 to 1, so when adjacent tiles are blended the
    weighted average reconstructs a seamless image.
    """
    w_y = np.ones(tile_h, dtype=np.float32)
    w_x = np.ones(tile_w, dtype=np.float32)

    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        if overlap <= tile_h:
            w_y[:overlap]  = ramp
            w_y[-overlap:] = ramp[::-1]
        if overlap <= tile_w:
            w_x[:overlap]  = ramp
            w_x[-overlap:] = ramp[::-1]

    return np.outer(w_y, w_x)   # (H, W)


def _pad_lr_to_window(
    lr:          np.ndarray,    # (C, H, W)
    window_size: int,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad *lr* so H and W are multiples of *window_size*.

    Returns the padded array and (pad_h, pad_w) needed to unpad the SR output.
    """
    _, h, w = lr.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        lr = np.pad(lr, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
    return lr, (pad_h, pad_w)


# ---------------------------------------------------------------------------
# Core inference loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def _infer_tile(
    model:       nn.Module,
    lr_tile:     np.ndarray,   # (C, H, W) float32, [0, 1]
    window_size: int,
    device:      torch.device,
    use_fp16:    bool,          # kept in signature for API compat; ignored (see note)
) -> np.ndarray:                # (C, H*scale, W*scale) float32, [0, 1]
    """Run SwinIR on a single (C, H, W) LR tile and return the SR result.

    Why fp16 is disabled
    --------------------
    SwinIR's shifted-window self-attention accumulates large intermediate
    activation tensors.  With fp16 the softmax denominator overflows to Inf
    for inputs larger than ~128 px (LR), which propagates NaN through the
    entire output — producing a fully black tile after uint16 cast.
    Float32 avoids the overflow with negligible throughput cost on Ampere+
    GPUs (typically < 15 % slower than fp16 for this model size).
    """
    lr_padded, (pad_h, pad_w) = _pad_lr_to_window(lr_tile, window_size)

    # Always float32 — autocast disabled regardless of use_fp16 setting.
    tensor = torch.from_numpy(lr_padded).unsqueeze(0).float().to(device)

    with torch.autocast(device_type=device.type, enabled=False):
        sr_tensor = model(tensor)

    sr = sr_tensor.squeeze(0).float().cpu().numpy()  # (C, H*s, W*s)

    # Unpad: remove the SR-space padding (pad × scale).
    scale  = sr.shape[1] // lr_padded.shape[1]
    sr_h   = sr.shape[1] - pad_h * scale
    sr_w   = sr.shape[2] - pad_w * scale
    sr     = sr[:, :sr_h, :sr_w]

    return np.clip(sr, 0.0, 1.0)


# ---------------------------------------------------------------------------
# GeoTIFF output helpers
# ---------------------------------------------------------------------------

def _photometric(n_bands: int) -> str:
    return "rgb" if n_bands in (3, 4) else "minisblack"


def _block_size(pixels: int, preferred: int = 256) -> int:
    cap     = min(preferred, pixels)
    snapped = (cap // 16) * 16
    return max(snapped, 16)


def _build_sr_profile(
    src_profile: dict,
    src_width:   int,
    src_height:  int,
    scale:       int,
    compress:    str,
    output_dtype: str,
) -> dict:
    """Build a clean output GeoTIFF profile for the SR image."""
    sr_w = src_width  * scale
    sr_h = src_height * scale

    # Scale the pixel spacing in the affine transform.
    src_t = src_profile["transform"]
    sr_transform = rasterio.transform.from_origin(
        src_t.c, src_t.f,
        abs(src_t.a) / scale,
        abs(src_t.e) / scale,
    )

    in_dtype  = src_profile["dtype"]
    out_dtype = in_dtype if output_dtype == "same" else output_dtype

    return {
        "driver":      "GTiff",
        "dtype":       out_dtype,
        "count":       src_profile["count"],
        "width":       sr_w,
        "height":      sr_h,
        "crs":         src_profile.get("crs"),
        "transform":   sr_transform,
        "compress":    None if compress.lower() == "none" else compress,
        "tiled":       True,
        "blockxsize":  _block_size(sr_w),
        "blockysize":  _block_size(sr_h),
        "interleave":  "band",
        "photometric": _photometric(src_profile["count"]),
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_inference(cfg, model: nn.Module, device: torch.device) -> Path:
    """Run tiled SR inference on the configured input LR GeoTIFF.

    Parameters
    ----------
    cfg    : DotDict configuration (from load_config).
    model  : Loaded, eval-mode SwinIR model on *device*.
    device : Compute device.

    Returns
    -------
    Path to the written SR GeoTIFF.
    """
    input_path = Path(cfg.io.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + "_SR.TIF")

    scale       = int(cfg.model.upscale)
    window_size = int(cfg.model.window_size)
    dtype_max   = float(cfg.io.dtype_max)
    tile_size   = int(cfg.tiling.tile_size)
    overlap     = int(cfg.tiling.overlap)
    use_fp16    = bool(cfg.misc.use_fp16) and device.type == "cuda"
    compress    = str(cfg.io.compress)
    output_dtype = str(cfg.io.output_dtype)

    log.info("Input  : %s", input_path)
    log.info("Output : %s", output_path)

    with rasterio.open(input_path) as src:
        src_profile = src.meta.copy()
        lr_full     = src.read().astype(np.float32) / dtype_max   # (C, H, W) [0,1]
        n_bands, lr_h, lr_w = lr_full.shape

    log.info(
        "LR image: %d bands  %d×%d px → SR %d×%d px  (scale=%d)",
        n_bands, lr_w, lr_h, lr_w * scale, lr_h * scale, scale,
    )

    sr_h = lr_h * scale
    sr_w = lr_w * scale

    # ── Tiled accumulation buffers ─────────────────────────────────────────────
    sr_acc  = np.zeros((n_bands, sr_h, sr_w), dtype=np.float64)
    wt_acc  = np.zeros((sr_h, sr_w),          dtype=np.float64)

    # Pre-compute SR-space weight ramp template (will be sliced per tile).
    sr_tile_size = tile_size * scale
    sr_overlap   = overlap   * scale
    weight_full  = _build_weight_ramp(sr_tile_size, sr_tile_size, sr_overlap)

    stride      = tile_size - overlap
    n_rows_t    = max(1, (lr_h + stride - 1) // stride)
    n_cols_t    = max(1, (lr_w + stride - 1) // stride)
    total_tiles = n_rows_t * n_cols_t
    tile_idx    = 0

    log.info(
        "Tiling: %d×%d tiles (tile_size=%d  overlap=%d  stride=%d)",
        n_rows_t, n_cols_t, tile_size, overlap, stride,
    )

    for row_t in range(n_rows_t):
        for col_t in range(n_cols_t):
            tile_idx += 1

            # LR tile window (clipped to image bounds).
            lr_x0 = col_t * stride
            lr_y0 = row_t * stride
            lr_x1 = min(lr_x0 + tile_size, lr_w)
            lr_y1 = min(lr_y0 + tile_size, lr_h)

            lr_tile = lr_full[:, lr_y0:lr_y1, lr_x0:lr_x1]  # (C, th, tw)
            th, tw  = lr_tile.shape[1], lr_tile.shape[2]

            sr_tile = _infer_tile(model, lr_tile, window_size, device, use_fp16)
            # sr_tile is (C, th*scale, tw*scale)

            # SR accumulation window.
            sr_x0 = lr_x0 * scale
            sr_y0 = lr_y0 * scale
            sr_x1 = sr_x0 + tw * scale
            sr_y1 = sr_y0 + th * scale

            # Slice the weight mask to match the actual (possibly smaller) tile.
            w = weight_full[:th * scale, :tw * scale]

            sr_acc[:, sr_y0:sr_y1, sr_x0:sr_x1] += sr_tile * w[np.newaxis]
            wt_acc[    sr_y0:sr_y1, sr_x0:sr_x1] += w

            if tile_idx % max(1, total_tiles // 10) == 0 or tile_idx == total_tiles:
                log.info(
                    "  tile %d / %d  (row=%d  col=%d)",
                    tile_idx, total_tiles, row_t, col_t,
                )

    # ── Normalise by accumulated weights ──────────────────────────────────────
    wt_acc = np.maximum(wt_acc, 1e-8)
    sr_out = (sr_acc / wt_acc[np.newaxis]).astype(np.float32)
    sr_out = np.clip(sr_out, 0.0, 1.0)

    # ── NaN safety — must come before integer cast ────────────────────────────
    # Residual NaN (fully-nodata tile, or degenerate model output) would cast
    # to 0 silently under NumPy, producing a black image with no error.
    # Replace with 0 and warn so the user knows something is off.
    nan_count = int(np.isnan(sr_out).sum())
    if nan_count:
        log.warning(
            "%d NaN value(s) in SR output — replacing with 0.  "
            "This likely means the input tile is a nodata region.  "
            "If the whole image is black, check that your checkpoint "
            "was trained on the same dtype_max (%g) as this input.",
            nan_count, dtype_max,
        )
        sr_out = np.nan_to_num(sr_out, nan=0.0)

    # ── Scale to output dtype and write GeoTIFF ────────────────────────────────
    out_profile  = _build_sr_profile(
        src_profile, lr_w, lr_h, scale, compress, output_dtype,
    )
    np_out_dtype = np.dtype(out_profile["dtype"])
    if np.issubdtype(np_out_dtype, np.integer):
        info      = np.iinfo(np_out_dtype)
        sr_scaled = (sr_out * info.max).clip(info.min, info.max).astype(np_out_dtype)
    else:
        sr_scaled = sr_out.astype(np_out_dtype)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(sr_scaled)
        # Embed per-band statistics so QGIS stretches correctly.
        for b_idx in range(n_bands):
            plane = sr_scaled[b_idx].astype(np.float64)
            valid = plane.ravel()
            valid = valid[np.isfinite(valid)]
            if valid.size:
                dst.update_tags(b_idx + 1,
                    STATISTICS_MINIMUM = float(valid.min()),
                    STATISTICS_MAXIMUM = float(valid.max()),
                    STATISTICS_MEAN    = float(valid.mean()),
                    STATISTICS_STDDEV  = float(valid.std()),
                )

    log.info("SR image written: %s  (%d×%d px)", output_path, sr_w, sr_h)
    return output_path