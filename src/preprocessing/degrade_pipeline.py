"""
=============================================================================
Sensor Degradation Pipeline for Pansharpened GeoTIFF Imagery  [GPU + Tiling]
=============================================================================
Core degradation logic.  Can be run standalone (main) or imported as a
library by build_dataset.py:

    from degrade_pipeline import run_pipeline, SpatialState

Tiling is handled externally via tiling.py so this module only owns the
per-tile GPU processing.

Output files mirror INPUT_FOLDER's subfolder structure under OUTPUT_FOLDER,
keeping original filenames.  When run standalone, degraded full-resolution
images are written (no gamma / uint8 conversion — that is build_dataset.py's
responsibility).

Depends on tiling.py (same directory or PYTHONPATH).

=============================================================================
PARAMETERS  –  edit this block
=============================================================================
"""

# ── Folders ───────────────────────────────────────────────────────────────────
INPUT_FOLDER  = "/home/thomas/Documents/img_to_pan/raw/Panshaprened"
OUTPUT_FOLDER = "/home/thomas/Documents/img_to_pan/dataset/train/raw"

# ── GPU ───────────────────────────────────────────────────────────────────────
GPU_ENABLED = True
GPU_DEVICE  = 0

# ── Tiling ────────────────────────────────────────────────────────────────────
TILE_SIZE    = 4096   # stride + overlap  (peak mem ≈ (tile_size+overlap)² × bands × 8 B)
TILE_OVERLAP = 64

# ── Degradation pipeline ──────────────────────────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  mtf_blur          – anisotropic Gaussian MTF (FFT, normalised conv.)    │
# │    mtf_nyquist_x/y   MTF @ Nyquist per axis, in (0,1). Smaller = blur.   │
# │                                                                          │
# │  downsample        – integer spatial downscale via rasterio reproject.   │
# │    ⚠ CPU round-trip per tile (rasterio has no GPU reproject).            │
# │    scale (int)  resampling ("average"|"bilinear"|"cubic"|…)              │
# │                                                                          │
# │  spectral_misalign – two-component inter-band misregistration (FFT).     │
# │    global_shift_px [dy,dx]  PAN–MS co-reg error shared by all bands.     │
# │    per_band_sigma_px        per-band jitter σ ~ N(0,σ).                  │
# │                                                                          │
# │  add_noise         – zero-mean Gaussian noise on valid pixels.           │
# │    sigma (float)  seed (int)                                             │
# └──────────────────────────────────────────────────────────────────────────┘
PIPELINE = [
    {"op": "mtf_blur",   "mtf_nyquist_x": 0.15, "mtf_nyquist_y": 0.15},
    {"op": "downsample", "scale": 2, "resampling": "average"},
    {"op": "spectral_misalign", "global_shift_px": [0.3, 0.2],
     "per_band_sigma_px": 0.15, "seed": 42},
    {"op": "add_noise",  "sigma": 2.0, "seed": 42},
]

# ── Output options (standalone mode only) ─────────────────────────────────────
OUTPUT_DTYPE   = None
COMPRESS       = "none"
OVERWRITE      = False
TIF_EXTENSIONS = (".tif", ".TIF", ".tiff", ".TIFF")

LOG_LEVEL = "INFO"

# =============================================================================
# END OF PARAMETERS
# =============================================================================

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: 'tqdm' is required.  pip install tqdm")

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ImportError:
    sys.exit("ERROR: 'rasterio' is required.  pip install rasterio")

try:
    from src.preprocessing.tiling import TileConfig, Tile, crop_tile, iter_tiles, output_window, tile_count
except ImportError:
    sys.exit("ERROR: 'tiling.py' not found.  Place it in the same directory.")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------

def _init_backend(enabled: bool, device: int) -> Any:
    if not enabled:
        log.info("Backend  : NumPy (GPU_ENABLED=False)")
        return np
    try:
        import cupy as cp
        cp.cuda.Device(device).use()
        props = cp.cuda.runtime.getDeviceProperties(device)
        name  = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        log.info("Backend  : CuPy  [device %d — %s]", device, name)
        return cp
    except Exception as exc:
        log.warning("CuPy unavailable (%s) — falling back to NumPy.", exc)
        return np


xp = _init_backend(GPU_ENABLED, GPU_DEVICE)


def _to_device(arr: np.ndarray) -> Any:
    return xp.asarray(arr) if xp is not np else arr


def _to_host(arr: Any) -> np.ndarray:
    return xp.asnumpy(arr) if xp is not np else arr


# ---------------------------------------------------------------------------
# Spatial state  — public; imported by build_dataset.py
# ---------------------------------------------------------------------------

@dataclass
class SpatialState:
    """Spatial metadata for one tile or image, threaded through the pipeline."""
    width:     int
    height:    int
    transform: Any   # rasterio.Affine
    crs:       Any   # rasterio.CRS


RESAMPLE_MAP = {
    "nearest":  Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic":    Resampling.cubic,
    "lanczos":  Resampling.lanczos,
    "average":  Resampling.average,
}

# ---------------------------------------------------------------------------
# Frequency-domain primitives  (device-native)
# ---------------------------------------------------------------------------

def _freq_grids(height: int, width: int) -> Tuple[Any, Any]:
    return xp.meshgrid(xp.fft.fftfreq(height), xp.fft.fftfreq(width), indexing="ij")


def _fft_phase_shift(band: Any, dy: float, dx: float) -> Any:
    """Translate *band* by (dy, dx) pixels via the FFT shift theorem."""
    Fy, Fx = _freq_grids(*band.shape)
    phase  = xp.exp(xp.asarray(-2j * math.pi) * (Fy * dy + Fx * dx))
    return xp.real(xp.fft.ifft2(xp.fft.fft2(band) * phase))


def _normalised_fft_convolve(band: Any, nodata_mask: Any, H: Any) -> Any:
    """Apply H in frequency domain with normalised convolution (no nodata bleed)."""
    valid  = (~nodata_mask).astype(xp.float64)
    filled = xp.where(~nodata_mask, band, xp.float64(0.0))
    bd = xp.real(xp.fft.ifft2(xp.fft.fft2(filled) * H))
    bw = xp.real(xp.fft.ifft2(xp.fft.fft2(valid)  * H))
    return xp.where(bw > 1e-6, bd / bw, band)


# ---------------------------------------------------------------------------
# Pipeline operations — bandwise
# ---------------------------------------------------------------------------

def op_mtf_blur(band: Any, nodata_mask: Any, mtf_nyquist_x: float, mtf_nyquist_y: float) -> Any:
    """
    Anisotropic Gaussian MTF blur in the frequency domain.

    H(fx,fy) = exp(−σx²π²fx²) · exp(−σy²π²fy²)
    σ² = −4 ln(MTF_nyq) / π²   (inverted Gaussian MTF at Nyquist f=0.5 cy/px)
    """
    def _s2(q: float) -> float:
        return -4.0 * math.log(max(1e-9, min(float(q), 1.0 - 1e-9))) / math.pi**2

    Fy, Fx = _freq_grids(*band.shape)
    H = (
        xp.exp(xp.float64(-_s2(mtf_nyquist_x) * math.pi**2) * Fx**2)
        * xp.exp(xp.float64(-_s2(mtf_nyquist_y) * math.pi**2) * Fy**2)
    )
    return _normalised_fft_convolve(band, nodata_mask, H)


def op_add_noise(band: Any, nodata_mask: Any, sigma: float, rng: np.random.Generator) -> Any:
    """Zero-mean Gaussian noise on valid pixels only (CPU-generated for reproducibility)."""
    noise               = _to_device(rng.normal(0.0, sigma, band.shape))
    result              = band + noise
    result[nodata_mask] = band[nodata_mask]
    return result


# ---------------------------------------------------------------------------
# Pipeline operations — stack-level
# ---------------------------------------------------------------------------

def op_downsample(
    bands: List[Any], masks: List[Any], state: SpatialState,
    scale: int, resampling: str,
) -> Tuple[List[Any], List[Any], SpatialState]:
    """
    Integer spatial downscale via rasterio reproject.

    ⚠ CPU round-trip per tile — rasterio has no GPU reproject path.
    The nodata mask is downsampled with nearest-neighbour to keep boundaries crisp.
    """
    algo = RESAMPLE_MAP.get(resampling.lower())
    if algo is None:
        raise ValueError(f"Unknown resampling '{resampling}'. Choose from: {list(RESAMPLE_MAP)}")

    dst_w = max(1, state.width  // scale)
    dst_h = max(1, state.height // scale)
    dst_t = rasterio.transform.from_origin(
        state.transform.c, state.transform.f,
        abs(state.transform.a) * (state.width  / dst_w),
        abs(state.transform.e) * (state.height / dst_h),
    )
    new_state = SpatialState(width=dst_w, height=dst_h, transform=dst_t, crs=state.crs)

    def _reproj(arr_cpu: np.ndarray, interp: Resampling) -> np.ndarray:
        out = np.zeros((dst_h, dst_w), dtype=np.float64)
        reproject(
            source=arr_cpu, destination=out,
            src_transform=state.transform, src_crs=state.crs,
            dst_transform=dst_t,           dst_crs=state.crs,
            resampling=interp,
        )
        return out

    out_bands, out_masks = [], []
    for band, mask in zip(bands, masks):
        out_bands.append(_to_device(_reproj(_to_host(band), algo)))
        out_masks.append(_to_device(
            _reproj(_to_host(mask).astype(np.float64), Resampling.nearest) > 0.5
        ))
    return out_bands, out_masks, new_state


def op_spectral_misalign(
    bands: List[Any], masks: List[Any],
    global_shift_px: List[float], per_band_sigma_px: float,
    rng: np.random.Generator,
) -> Tuple[List[Any], List[Any]]:
    """
    Two-component inter-band misregistration via FFT phase shifts.

    1. Global PAN–MS co-reg error: shared (dy, dx) for all bands.
    2. Per-band jitter: independent Δ ~ N(0, per_band_sigma_px) per band.

    Shifts are drawn on the CPU (seeded) for backend-independent reproducibility.
    """
    gdy, gdx = float(global_shift_px[0]), float(global_shift_px[1])
    out_bands, out_masks = [], []
    for i, (band, mask) in enumerate(zip(bands, masks)):
        dy = gdy + rng.normal(0.0, per_band_sigma_px)
        dx = gdx + rng.normal(0.0, per_band_sigma_px)
        out_bands.append(_fft_phase_shift(band, dy, dx))
        out_masks.append(_fft_phase_shift(mask.astype(xp.float64), dy, dx) > 0.5)
        log.debug("    band %d  shift (dy=%.4f, dx=%.4f)", i + 1, dy, dx)
    return out_bands, out_masks


# ---------------------------------------------------------------------------
# Pipeline runner  — public; imported by build_dataset.py
# ---------------------------------------------------------------------------

def run_pipeline(
    bands:    List[np.ndarray],   # host float64, one array per band
    nodata:   Optional[float],
    state:    SpatialState,
    pipeline: List[Dict],
) -> Tuple[List[np.ndarray], SpatialState]:
    """
    Transfer bands to device, execute *pipeline* sequentially, return to host.

    Bandwise ops → mapped over (band, mask) pairs independently.
    Stack ops    → receive and return the full band list; may mutate SpatialState.

    Parameters
    ----------
    bands    : Host float64 arrays, one per band.
    nodata   : Nodata sentinel value (or None).
    state    : Spatial metadata for the input tile/image.
    pipeline : List of op dicts as defined in PIPELINE.

    Returns
    -------
    Processed host arrays and final SpatialState (dimensions may have changed).
    """
    dev_bands = [_to_device(b) for b in bands]
    dev_masks = [
        _to_device(b == nodata if nodata is not None else np.zeros(b.shape, dtype=bool))
        for b in bands
    ]

    for step in pipeline:
        name = step["op"]

        if name == "mtf_blur":
            nx, ny    = float(step["mtf_nyquist_x"]), float(step["mtf_nyquist_y"])
            dev_bands = [op_mtf_blur(b, m, nx, ny) for b, m in zip(dev_bands, dev_masks)]
            log.debug("  → mtf_blur  MTF_nyq=(%.3f, %.3f)", nx, ny)

        elif name == "downsample":
            dev_bands, dev_masks, state = op_downsample(
                dev_bands, dev_masks, state,
                scale=int(step["scale"]),
                resampling=step.get("resampling", "average"),
            )
            log.debug("  → downsample ×%d → %d×%d", step["scale"], state.width, state.height)

        elif name == "spectral_misalign":
            dev_bands, dev_masks = op_spectral_misalign(
                dev_bands, dev_masks,
                global_shift_px   = step.get("global_shift_px",   [0.0, 0.0]),
                per_band_sigma_px = float(step.get("per_band_sigma_px", 0.0)),
                rng               = np.random.default_rng(step.get("seed")),
            )

        elif name == "add_noise":
            rng       = np.random.default_rng(step.get("seed"))
            dev_bands = [op_add_noise(b, m, float(step["sigma"]), rng)
                         for b, m in zip(dev_bands, dev_masks)]
            log.debug("  → add_noise  σ=%.2f", step["sigma"])

        else:
            raise ValueError(
                f"Unknown pipeline op '{name}'. "
                f"Supported: mtf_blur, downsample, spectral_misalign, add_noise"
            )

    return [_to_host(b) for b in dev_bands], state


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def discover_tifs(root: Path) -> List[Path]:
    return [
        f for f in sorted(root.rglob("*"))
        if f.is_file() and f.suffix in TIF_EXTENSIONS
    ]


def build_output_path(src: Path, in_root: Path, out_root: Path) -> Path:
    return out_root / src.relative_to(in_root)


def _native_gsd_metres(transform: Any, crs: Any, height: int) -> float:
    px = abs(transform.a)
    if crs and crs.is_geographic:
        lat  = transform.f + height / 2 * transform.e
        px  *= 111_320.0 * math.cos(math.radians(lat))
    return px


def _pipeline_spatial_scale(pipeline: List[Dict]) -> int:
    scale = 1
    for step in pipeline:
        if step["op"] == "downsample":
            scale *= int(step["scale"])
    return scale


# ---------------------------------------------------------------------------
# Per-file entry point (standalone mode)
# ---------------------------------------------------------------------------

def process_image(
    src_path: Path, input_root: Path, out_root: Path,
    out_dtype: Optional[str], compress: str, tile_cfg: TileConfig,
) -> Optional[Path]:
    """
    Tile *src_path*, degrade each tile through PIPELINE, and write the
    assembled result to the mirrored path under *out_root*.

    This writes float/original-dtype output without gamma conversion.
    For uint8 + gamma output, use build_dataset.py instead.
    """
    out_path = build_output_path(src_path, input_root, out_root)
    if out_path.exists() and not OVERWRITE:
        tqdm.write(f"  ⟳  Already exists, skipping: {src_path.name}")
        return None

    spatial_scale = _pipeline_spatial_scale(PIPELINE)

    try:
        with rasterio.open(src_path) as src:
            src_transform = src.transform
            src_crs       = src.crs
            src_nodata    = src.nodata
            src_dtype     = src.dtypes[0]
            n_bands       = src.count
            src_meta      = src.meta.copy()

            n_tiles    = tile_count(src.width, src.height, tile_cfg)
            out_w      = src.width  // spatial_scale
            out_h      = src.height // spatial_scale
            out_t      = rasterio.transform.from_origin(
                src_transform.c, src_transform.f,
                abs(src_transform.a) * spatial_scale,
                abs(src_transform.e) * spatial_scale,
            )

            tqdm.write(
                f"  {src_path.name}  [{n_bands}b  "
                f"{src.width}×{src.height}px  →  {n_tiles} tile(s)]"
            )

            eff_dtype  = out_dtype or src_dtype
            is_int     = np.issubdtype(np.dtype(eff_dtype), np.integer)
            dtype_info = np.iinfo(np.dtype(eff_dtype)) if is_int else None

            def _cast(arr):
                if dtype_info:
                    arr = np.clip(arr, dtype_info.min, dtype_info.max)
                return arr.astype(eff_dtype)

            src_meta.update(
                width=out_w, height=out_h, transform=out_t,
                dtype=eff_dtype,
                compress=compress if compress.lower() != "none" else None,
                BIGTIFF="IF_SAFER",
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(out_path, "w", **src_meta) as dst:
                for tile in tqdm(
                    iter_tiles(src.width, src.height, src_transform, tile_cfg),
                    total=n_tiles, desc="  tiles", unit="tile",
                    leave=False, colour="green",
                ):
                    tile_bands = [
                        src.read(i, window=tile.read_window).astype(np.float64)
                        for i in range(1, n_bands + 1)
                    ]
                    tile_state = SpatialState(
                        width=tile.read_window.width, height=tile.read_window.height,
                        transform=tile.src_transform, crs=src_crs,
                    )
                    tile_bands, _ = run_pipeline(tile_bands, src_nodata, tile_state, PIPELINE)

                    write_win = output_window(tile, spatial_scale)
                    for band_idx, band in enumerate(tile_bands, start=1):
                        cropped   = crop_tile(band, tile, spatial_scale)
                        actual_win = rasterio.windows.Window(
                            write_win.col_off, write_win.row_off,
                            cropped.shape[1], cropped.shape[0],
                        )
                        dst.write(_cast(cropped), band_idx, window=actual_win)

        tqdm.write(f"    → {out_w}×{out_h}px  ✓")
        return out_path

    except Exception as exc:
        tqdm.write(f"  ✗  FAILED – {src_path.name}: {exc}")
        log.debug("Exception details:", exc_info=True)
        if out_path.exists():
            out_path.unlink()
        return None


# ---------------------------------------------------------------------------
# Entry point (standalone)
# ---------------------------------------------------------------------------

def _log_pipeline() -> None:
    steps = "  →  ".join(
        f"{s['op']}({', '.join(f'{k}={v}' for k, v in s.items() if k != 'op')})"
        for s in PIPELINE
    )
    log.info("Pipeline : %s", steps)


def main() -> None:
    input_root  = Path(INPUT_FOLDER).resolve()
    output_root = Path(OUTPUT_FOLDER).resolve()

    if not input_root.exists():
        sys.exit(f"ERROR: INPUT_FOLDER does not exist: {input_root}")
    if input_root == output_root:
        sys.exit("ERROR: INPUT_FOLDER and OUTPUT_FOLDER must differ.")

    tile_cfg = TileConfig(tile_size=TILE_SIZE, overlap=TILE_OVERLAP)

    log.info("Input    : %s", input_root)
    log.info("Output   : %s", output_root)
    _log_pipeline()
    log.info("Tiling   : tile_size=%d  overlap=%d  stride=%d",
             tile_cfg.tile_size, tile_cfg.overlap, tile_cfg.stride)
    log.info("Compress : %s  |  Overwrite : %s", COMPRESS, OVERWRITE)

    tif_files = discover_tifs(input_root)
    if not tif_files:
        log.warning("No TIF files found under %s", input_root)
        return

    log.info("Found %d file(s) to process.", len(tif_files))
    success = skipped = 0

    with tqdm(tif_files, desc="Images", unit="img", colour="cyan") as bar:
        for src_path in bar:
            bar.set_postfix_str(src_path.name[:50], refresh=True)
            result = process_image(src_path, input_root, output_root, OUTPUT_DTYPE, COMPRESS, tile_cfg)
            if result:
                success += 1
            else:
                skipped += 1
            bar.set_postfix(done=success, skipped=skipped, refresh=True)

    tqdm.write(f"\n{'─'*60}")
    tqdm.write(f"  Finished  │  ✓ {success}  │  ⟳/✗ {skipped}  │  {len(tif_files)} total")
    tqdm.write(f"{'─'*60}")


if __name__ == "__main__":
    main()