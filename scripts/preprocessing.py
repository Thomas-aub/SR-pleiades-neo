"""
preprocessing.py — Pléiades NEO SR Dataset Preprocessing Pipeline
==================================================================
Single entry point for the full three-stage preprocessing pipeline.
All parameters are read from a YAML configuration file.

Stages
------
    1 · pansharpening   data/raw/         →  data/pansharpened/HR/
    2 · degradation     pansharpened/HR/  →  data/pansharpened/LR/
    3 · tiling          HR/ + LR/         →  data/processed/train|val/

Usage
-----
    # Full pipeline
    python preprocessing.py

    # Custom config
    python preprocessing.py --config path/to/config.yaml

    # Run specific stages only
    python preprocessing.py --stages degradation tiling

    # Start from a specific stage (run it and all subsequent ones)
    python preprocessing.py --from-stage tiling

    # Validate config without processing any data
    python preprocessing.py --dry-run

    # Override overwrite flag for all stages
    python preprocessing.py --stages tiling --overwrite

Dependencies
------------
    pip install pyyaml tqdm rasterio numpy
    # GPU acceleration (optional):
    pip install cupy-cuda12x   # match your CUDA version

Architecture
------------
Each stage is implemented as a standalone function that receives explicit
config parameters — no module global mutation.  Pansharpening and
degradation modules are imported locally within their stage functions so
their globals can be patched in a controlled, isolated scope without
affecting other imports or concurrent use.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings

# ---------------------------------------------------------------------------
# Path bootstrap — must run before ANY other import so that flat modules
# (pansharpening, degrade_pipeline, build_dataset, tiling) are findable
# regardless of the working directory the script is launched from.
#
# Layout handled:
#   scripts/preprocessing.py  →  project root = scripts/../
#   preprocessing.py (root)   →  project root = ./
# ---------------------------------------------------------------------------
import pathlib as _pathlib
_SCRIPT_DIR   = _pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = (
    _SCRIPT_DIR.parent       # scripts/preprocessing.py
    if _SCRIPT_DIR.name == "scripts"
    else _SCRIPT_DIR         # preprocessing.py at project root
)
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
del _pathlib   # keep namespace clean
from dataclasses import dataclass, field
from pathlib import Path

# Re-expose _PROJECT_ROOT as a Path object for use in this module.
# (_PROJECT_ROOT was bootstrapped above using pathlib directly to avoid
# a forward-reference to Path before it was imported.)
_PROJECT_ROOT = Path(_PROJECT_ROOT)
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Hard dependency checks — fail fast with a clear message
# ---------------------------------------------------------------------------

try:
    import yaml
except ImportError:
    sys.exit("ERROR: pyyaml is required.  pip install pyyaml")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: tqdm is required.  pip install tqdm")

try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is required.  pip install numpy")



# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass(frozen=True)
class DataPaths:
    raw_root:        Path
    pansharpened_hr: Path
    pansharpened_lr: Path
    processed:       Path


@dataclass(frozen=True)
class PansharpeningConfig:
    method:        str  = "brovey"
    resample_algo: str  = "bilinear"
    output_dtype:  str  = "uint16"
    output_suffix: str  = "_PANSHARP"
    compress:      str  = "none"
    use_gpu:       bool = True
    chunk_rows:    int  = 2048


@dataclass(frozen=True)
class DegradationConfig:
    use_gpu:      bool       = True
    gpu_device:   int        = 0
    tile_size:    int        = 4096
    tile_overlap: int        = 64
    compress:     str        = "none"
    overwrite:    bool       = False
    pipeline:     List[Dict] = field(default_factory=list)

    def __hash__(self) -> int:
        return id(self)


@dataclass(frozen=True)
class TilingConfig:
    tile_size:            int   = 512
    val_ratio:            float = 0.2
    seed:                 int   = 42
    compress:             str   = "deflate"
    min_valid_fraction:   float = 0.1
    overwrite:            bool  = False
    stats_min_percentile: float = 1.0
    stats_max_percentile: float = 99.0


@dataclass(frozen=True)
class LoggingConfig:
    level: str           = "INFO"
    file:  Optional[str] = None


@dataclass
class PipelineConfig:
    data:          DataPaths
    pansharpening: PansharpeningConfig
    degradation:   DegradationConfig
    tiling:        TilingConfig
    logging:       LoggingConfig


# =============================================================================
# YAML loading and validation
# =============================================================================

class ConfigError(Exception):
    """Raised when the configuration file is invalid."""


def _get(mapping: Dict, key: str, default: Any, section: str) -> Any:
    """Return mapping[key] with a typed default, logging missing keys."""
    return mapping.get(key, default)


def _require(mapping: Dict, key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"[{section}] Required key '{key}' is missing.")
    return mapping[key]


def load_config(path: Path) -> PipelineConfig:
    """Parse, validate, and return a typed PipelineConfig from a YAML file.

    Raises
    ------
    FileNotFoundError : Config file not found.
    ConfigError       : Required key missing or value invalid.
    yaml.YAMLError    : YAML syntax error.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: Dict = yaml.safe_load(fh) or {}

    # ── data ──────────────────────────────────────────────────────────────────
    sec = _require(raw, "data", "root")
    data = DataPaths(
        raw_root        = Path(_require(sec, "raw_root",        "data")),
        pansharpened_hr = Path(_require(sec, "pansharpened_hr", "data")),
        pansharpened_lr = Path(_require(sec, "pansharpened_lr", "data")),
        processed       = Path(_require(sec, "processed",       "data")),
    )

    # ── pansharpening ─────────────────────────────────────────────────────────
    sec = raw.get("pansharpening", {})
    ps  = PansharpeningConfig(
        method        = str(_get(sec, "method",        "brovey",    "pansharpening")),
        resample_algo = str(_get(sec, "resample_algo", "bilinear",  "pansharpening")),
        output_dtype  = str(_get(sec, "output_dtype",  "uint16",    "pansharpening")),
        output_suffix = str(_get(sec, "output_suffix", "_PANSHARP", "pansharpening")),
        compress      = str(_get(sec, "compress",      "none",      "pansharpening")),
        use_gpu       = bool(_get(sec, "use_gpu",      True,        "pansharpening")),
        chunk_rows    = int(_get(sec, "chunk_rows",    2048,        "pansharpening")),
    )
    _validate_pansharpening(ps)

    # ── degradation ───────────────────────────────────────────────────────────
    sec      = raw.get("degradation", {})
    pipeline = sec.get("pipeline", [])
    if not pipeline:
        raise ConfigError(
            "[degradation] 'pipeline' must contain at least one operation."
        )
    dg = DegradationConfig(
        use_gpu      = bool(_get(sec, "use_gpu",      True,   "degradation")),
        gpu_device   = int(_get(sec, "gpu_device",    0,      "degradation")),
        tile_size    = int(_get(sec, "tile_size",     4096,   "degradation")),
        tile_overlap = int(_get(sec, "tile_overlap",  64,     "degradation")),
        compress     = str(_get(sec, "compress",      "none", "degradation")),
        overwrite    = bool(_get(sec, "overwrite",    False,  "degradation")),
        pipeline     = [dict(step) for step in pipeline],
    )
    _validate_degradation(dg)

    # ── tiling ────────────────────────────────────────────────────────────────
    sec = raw.get("tiling", {})
    tl  = TilingConfig(
        tile_size            = int(_get(sec, "tile_size",            512,     "tiling")),
        val_ratio            = float(_get(sec, "val_ratio",          0.2,     "tiling")),
        seed                 = int(_get(sec, "seed",                 42,      "tiling")),
        compress             = str(_get(sec, "compress",             "deflate","tiling")),
        min_valid_fraction   = float(_get(sec, "min_valid_fraction", 0.1,     "tiling")),
        overwrite            = bool(_get(sec, "overwrite",           False,   "tiling")),
        stats_min_percentile = float(_get(sec, "stats_min_percentile", 1.0,  "tiling")),
        stats_max_percentile = float(_get(sec, "stats_max_percentile", 99.0, "tiling")),
    )
    _validate_tiling(tl)

    # ── logging ───────────────────────────────────────────────────────────────
    sec    = raw.get("logging", {})
    log_cfg = LoggingConfig(
        level = str(_get(sec, "level", "INFO", "logging")).upper(),
        file  = _get(sec, "file", None, "logging"),
    )

    return PipelineConfig(
        data          = data,
        pansharpening = ps,
        degradation   = dg,
        tiling        = tl,
        logging       = log_cfg,
    )


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def _validate_pansharpening(cfg: PansharpeningConfig) -> None:
    _choices(cfg.method,        {"brovey", "simple_mean"},                      "pansharpening.method")
    _choices(cfg.resample_algo, {"nearest", "bilinear", "cubic", "cubic_spline", "lanczos"},
             "pansharpening.resample_algo")
    _choices(cfg.output_dtype,  {"uint8", "uint16"},                            "pansharpening.output_dtype")
    _choices(cfg.compress,      {"none", "lzw", "deflate"},                     "pansharpening.compress")
    if cfg.chunk_rows < 1:
        raise ConfigError("pansharpening.chunk_rows must be ≥ 1.")


def _validate_degradation(cfg: DegradationConfig) -> None:
    _choices(cfg.compress, {"none", "lzw", "deflate"}, "degradation.compress")
    valid_ops = {"mtf_blur", "downsample", "spectral_misalign", "add_noise"}
    for i, step in enumerate(cfg.pipeline):
        op = step.get("op")
        if op not in valid_ops:
            raise ConfigError(
                f"degradation.pipeline[{i}]: unknown op '{op}'. "
                f"Valid: {sorted(valid_ops)}."
            )
    if cfg.tile_size < 1:
        raise ConfigError("degradation.tile_size must be ≥ 1.")


def _validate_tiling(cfg: TilingConfig) -> None:
    _choices(cfg.compress, {"none", "lzw", "deflate"}, "tiling.compress")
    if cfg.tile_size < 16:
        raise ConfigError("tiling.tile_size must be ≥ 16.")
    if not (0.0 <= cfg.val_ratio < 1.0):
        raise ConfigError("tiling.val_ratio must be in [0, 1).")
    if not (0.0 <= cfg.min_valid_fraction <= 1.0):
        raise ConfigError("tiling.min_valid_fraction must be in [0, 1].")
    if not (0.0 <= cfg.stats_min_percentile < cfg.stats_max_percentile <= 100.0):
        raise ConfigError(
            "tiling: stats_min_percentile must be < stats_max_percentile, "
            "both in [0, 100]."
        )


def _choices(value: str, valid: set, key: str) -> None:
    if value not in valid:
        raise ConfigError(
            f"{key}: '{value}' is not valid.  Choose from: {sorted(valid)}."
        )


# =============================================================================
# Logging setup
# =============================================================================

def configure_logging(cfg: LoggingConfig) -> logging.Logger:
    """Configure the root logger and return the pipeline logger."""
    numeric_level = getattr(logging, cfg.level, logging.INFO)
    formatter     = logging.Formatter(
        fmt     = "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if cfg.file:
        log_path = Path(cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)

    return logging.getLogger("preprocessing")


# =============================================================================
# GPU utilities
# =============================================================================

def _probe_gpu(device_idx: int = 0) -> Tuple[bool, str]:
    """Return (available, description_string) for CUDA device *device_idx*."""
    try:
        import cupy as cp  # noqa: PLC0415
        cp.cuda.Device(device_idx).use()
        props = cp.cuda.runtime.getDeviceProperties(device_idx)
        name  = props["name"]
        if isinstance(name, bytes):
            name = name.decode()
        mem   = props["totalGlobalMem"] / (1 << 30)
        return True, f"{name}  ({mem:.1f} GiB)"
    except Exception as exc:
        return False, str(exc)


# =============================================================================
# Helper utilities
# =============================================================================

def _elapsed(start: float) -> str:
    secs = time.perf_counter() - start
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _banner(log: logging.Logger, title: str) -> None:
    sep = "═" * 62
    log.info(sep)
    log.info("  %s", title)
    log.info(sep)


def _discover_tifs(root: Path) -> List[Path]:
    ext = {".tif", ".TIF", ".tiff", ".TIFF"}
    return sorted(f for f in root.rglob("*") if f.suffix in ext)


# =============================================================================
# Stage 1 — Pansharpening
# =============================================================================

def run_pansharpening(
    cfg: PipelineConfig,
    log: logging.Logger,
    overwrite: bool = False,
) -> None:
    """Fuse each PAN + MS pair under data.raw_root into an HR GeoTIFF.

    Each output file is placed in data.pansharpened_hr, mirroring the
    WO_* subfolder structure of the raw input.

    GPU backend
    -----------
    When pansharpening.use_gpu is true, CuPy is used for the Brovey
    arithmetic.  The fallback to NumPy is automatic and silent; a warning
    is logged so users know which backend is active.
    """
    _banner(log, "Stage 1 · Pansharpening")

    try:
        import src.preprocessing.pansharpening as ps_mod  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "Cannot import 'pansharpening'.  "
            "Ensure pansharpening.py is in the working directory."
        )

    ps = cfg.pansharpening
    input_root  = cfg.data.raw_root.resolve()
    output_root = cfg.data.pansharpened_hr.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(
            f"[pansharpening] raw_root not found: {input_root}"
        )

    # ── GPU backend ──────────────────────────────────────────────────────────
    ps_mod.USE_GPU = ps.use_gpu
    ps_mod.xp      = __import__("numpy")
    ps_mod.GPU_AVAILABLE = False

    if ps.use_gpu:
        available, info = _probe_gpu(0)
        if available:
            import cupy as cp  # noqa: PLC0415
            ps_mod.xp            = cp
            ps_mod.GPU_AVAILABLE = True
            log.info("GPU (pansharpening) : %s", info)
        else:
            log.warning(
                "GPU requested for pansharpening but CuPy unavailable (%s) "
                "— using NumPy.", info,
            )
    else:
        log.info("Backend (pansharpening) : NumPy (CPU)")

    # ── Patch module globals ──────────────────────────────────────────────────
    ps_mod.PANSHARPEN_METHOD = ps.method
    ps_mod.RESAMPLE_ALGO     = ps.resample_algo
    ps_mod.OUTPUT_DTYPE      = ps.output_dtype
    ps_mod.OUTPUT_SUFFIX     = ps.output_suffix
    ps_mod.COMPRESS          = ps.compress
    ps_mod.CHUNK_ROWS        = ps.chunk_rows

    log.info("Input   : %s", input_root)
    log.info("Output  : %s", output_root)
    log.info(
        "Settings: method=%s  resample=%s  dtype=%s  compress=%s  chunks=%d",
        ps.method, ps.resample_algo, ps.output_dtype, ps.compress, ps.chunk_rows,
    )

    # ── Discover pairs and filter already-done if not overwriting ────────────
    pairs = ps_mod.discover_pairs(input_root)
    if not pairs:
        log.warning("No RGB/PAN pairs found under %s — nothing to do.", input_root)
        return

    if not overwrite:
        pending = []
        skipped = 0
        for rgb, pan in pairs:
            rel     = rgb.parent.relative_to(input_root)
            out_path = output_root / rel / (rgb.stem + ps.output_suffix + ".TIF")
            if out_path.exists():
                skipped += 1
            else:
                pending.append((rgb, pan))
        if skipped:
            log.info(
                "%d pair(s) already exist — skipping (use --overwrite to redo).",
                skipped,
            )
        pairs = pending

    if not pairs:
        log.info("All pairs already pansharpened.  Nothing to do.")
        return

    log.info("Processing %d pair(s).", len(pairs))
    success = failed = 0

    for rgb, pan in tqdm(pairs, desc="Pansharpening", unit="pair"):
        result = ps_mod.process_pair(rgb, pan, output_root, input_root)
        if result:
            success += 1
        else:
            failed += 1

    log.info(
        "Pansharpening complete — ✓ %d  ✗ %d  (total %d)",
        success, failed, success + failed,
    )
    if failed:
        log.warning("%d file(s) failed.  Check logs above for details.", failed)


# =============================================================================
# Stage 2 — Sensor Degradation
# =============================================================================

def run_degradation(
    cfg: PipelineConfig,
    log: logging.Logger,
    overwrite: bool = False,
) -> None:
    """Apply the degradation pipeline to every HR GeoTIFF in pansharpened_hr.

    Each file is processed tile-by-tile on the GPU (or CPU fallback) using
    the ops listed in degradation.pipeline.  Output files mirror the input
    subfolder structure under pansharpened_lr.

    The --overwrite flag (or degradation.overwrite in the YAML) controls
    whether already-degraded files are re-processed.
    """
    _banner(log, "Stage 2 · Sensor Degradation")

    try:
        import src.preprocessing.degrade_pipeline as dp_mod  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "Cannot import 'degrade_pipeline'.  "
            "Ensure degrade_pipeline.py is in the working directory."
        )

    dg = cfg.degradation
    effective_overwrite = overwrite or dg.overwrite

    input_root  = cfg.data.pansharpened_hr.resolve()
    output_root = cfg.data.pansharpened_lr.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(
            f"[degradation] pansharpened_hr not found: {input_root}\n"
            "Run the pansharpening stage first."
        )

    # ── GPU backend ──────────────────────────────────────────────────────────
    dp_mod.GPU_ENABLED = dg.use_gpu
    dp_mod.GPU_DEVICE  = dg.gpu_device
    dp_mod.xp          = dp_mod._init_backend(dg.use_gpu, dg.gpu_device)

    if dg.use_gpu:
        available, info = _probe_gpu(dg.gpu_device)
        if available:
            log.info("GPU (degradation) : %s [device %d]", info, dg.gpu_device)
        else:
            log.warning(
                "GPU requested for degradation but CuPy unavailable (%s) "
                "— using NumPy.", info,
            )
    else:
        log.info("Backend (degradation) : NumPy (CPU)")

    # ── Patch module globals ──────────────────────────────────────────────────
    dp_mod.PIPELINE  = dg.pipeline
    dp_mod.OVERWRITE = effective_overwrite
    dp_mod.COMPRESS  = dg.compress

    log.info("Input   : %s", input_root)
    log.info("Output  : %s", output_root)
    log.info(
        "Settings: tile=%d  overlap=%d  compress=%s  overwrite=%s",
        dg.tile_size, dg.tile_overlap, dg.compress, effective_overwrite,
    )

    steps_str = "  →  ".join(
        f"{s['op']}({', '.join(f'{k}={v}' for k, v in s.items() if k != 'op')})"
        for s in dg.pipeline
    )
    log.info("Pipeline: %s", steps_str)

    from src.preprocessing.tiling import TileConfig  # noqa: PLC0415
    tile_cfg  = TileConfig(tile_size=dg.tile_size, overlap=dg.tile_overlap)
    tif_files = dp_mod.discover_tifs(input_root)

    if not tif_files:
        log.warning("No TIF files found under %s — nothing to do.", input_root)
        return

    log.info("Found %d file(s) to degrade.", len(tif_files))
    success = skipped = 0

    with tqdm(tif_files, desc="Degrading", unit="img") as bar:
        for src_path in bar:
            bar.set_postfix_str(src_path.name[:55], refresh=True)
            result = dp_mod.process_image(
                src_path   = src_path,
                input_root = input_root,
                out_root   = output_root,
                out_dtype  = None,
                compress   = dg.compress,
                tile_cfg   = tile_cfg,
            )
            if result:
                success += 1
            else:
                skipped += 1

    log.info(
        "Degradation complete — ✓ %d  ⟳/✗ %d  (total %d)",
        success, skipped, len(tif_files),
    )


# =============================================================================
# Stage 3 — Dataset Tiling
# =============================================================================

def run_tiling(
    cfg: PipelineConfig,
    log: logging.Logger,
    overwrite: bool = False,
) -> None:
    """Tile HR/LR pairs into fixed-size patches and split into train/val.

    Statistics strategy (mirrors tiler.py)
    ----------------------------------------
    Before the per-tile loop, a thumbnail of the full source image is read
    and used to compute per-band percentile statistics (p1 / p99 of valid
    pixels, excluding zeros).  These global statistics are embedded as
    STATISTICS_* tags in every output tile so that QGIS and other GIS
    tools apply a consistent display stretch across all tiles from the
    same acquisition.  Adjacent tiles over water and sand look
    photometrically consistent — matching the full-resolution image.
    """
    _banner(log, "Stage 3 · Dataset Tiling")

    try:
        import src.preprocessing.build_dataset as bd_mod  # noqa: PLC0415
    except ImportError:
        raise ImportError(
            "Cannot import 'build_dataset'.  "
            "Ensure build_dataset.py is in the working directory."
        )

    tl = cfg.tiling
    effective_overwrite = overwrite or tl.overwrite

    hr_root     = cfg.data.pansharpened_hr.resolve()
    lr_root     = cfg.data.pansharpened_lr.resolve()
    output_root = cfg.data.processed.resolve()

    for label, path in (
        ("pansharpened_hr", hr_root),
        ("pansharpened_lr", lr_root),
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"[tiling] '{label}' not found: {path}\n"
                "Ensure the pansharpening and degradation stages have run."
            )

    log.info("HR root : %s", hr_root)
    log.info("LR root : %s", lr_root)
    log.info("Output  : %s", output_root)
    log.info(
        "Settings: tile=%d  val=%.0f%%  seed=%d  compress=%s  "
        "min_valid=%.0f%%  stats_p=[%.1f, %.1f]  overwrite=%s",
        tl.tile_size, tl.val_ratio * 100, tl.seed, tl.compress,
        tl.min_valid_fraction * 100,
        tl.stats_min_percentile, tl.stats_max_percentile,
        effective_overwrite,
    )

    # ── Patch module globals that tile_split / tile_pair read ─────────────────
    bd_mod.STATS_MIN_PERCENTILE = tl.stats_min_percentile
    bd_mod.STATS_MAX_PERCENTILE = tl.stats_max_percentile

    pairs = bd_mod.discover_pairs(hr_root, lr_root)
    if not pairs:
        log.warning("No matched HR/LR pairs found — nothing to tile.")
        return

    train_pairs, val_pairs = bd_mod.split_pairs(pairs, tl.val_ratio, tl.seed)

    from src.preprocessing.build_dataset import TilingStats  # noqa: PLC0415
    grand_total = TilingStats()

    for split_name, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        if not split_pairs:
            log.warning("No pairs assigned to '%s' — skipping.", split_name)
            continue

        stats = bd_mod.tile_split(
            pairs              = split_pairs,
            split_name         = split_name,
            output_root        = output_root,
            tile_size          = tl.tile_size,
            compress           = tl.compress,
            min_valid_fraction = tl.min_valid_fraction,
            overwrite          = effective_overwrite,
        )
        log.info(
            "[%s] pairs=%d  skipped=%d  tiles_written=%d  tiles_dropped=%d",
            split_name,
            stats.pairs_processed,
            stats.pairs_skipped,
            stats.tiles_written,
            stats.tiles_dropped,
        )
        grand_total += stats

    log.info(
        "Tiling complete — pairs ✓ %d  ✗ %d  |  tiles written %d  dropped %d",
        grand_total.pairs_processed,
        grand_total.pairs_skipped,
        grand_total.tiles_written,
        grand_total.tiles_dropped,
    )


# =============================================================================
# Dry-run: print resolved configuration
# =============================================================================

def print_config(cfg: PipelineConfig, log: logging.Logger) -> None:
    """Log the fully resolved configuration without running any stage."""
    _banner(log, "Resolved configuration (dry-run)")

    d = cfg.data
    log.info("[data]")
    log.info("  raw_root        : %s", d.raw_root.resolve())
    log.info("  pansharpened_hr : %s", d.pansharpened_hr.resolve())
    log.info("  pansharpened_lr : %s", d.pansharpened_lr.resolve())
    log.info("  processed       : %s", d.processed.resolve())

    ps = cfg.pansharpening
    log.info("[pansharpening]")
    log.info(
        "  method=%s  resample=%s  dtype=%s  compress=%s  "
        "gpu=%s  chunk_rows=%d",
        ps.method, ps.resample_algo, ps.output_dtype,
        ps.compress, ps.use_gpu, ps.chunk_rows,
    )

    dg = cfg.degradation
    log.info("[degradation]")
    log.info(
        "  gpu=%s  device=%d  tile_size=%d  overlap=%d  "
        "compress=%s  overwrite=%s",
        dg.use_gpu, dg.gpu_device, dg.tile_size,
        dg.tile_overlap, dg.compress, dg.overwrite,
    )
    for i, step in enumerate(dg.pipeline):
        op_str = "  ".join(f"{k}={v}" for k, v in step.items())
        log.info("  pipeline[%d] : %s", i, op_str)

    tl = cfg.tiling
    log.info("[tiling]")
    log.info(
        "  tile_size=%d  val_ratio=%.0f%%  seed=%d  compress=%s  "
        "min_valid=%.0f%%  overwrite=%s",
        tl.tile_size, tl.val_ratio * 100, tl.seed,
        tl.compress, tl.min_valid_fraction * 100, tl.overwrite,
    )
    log.info(
        "  stats_percentiles=[%.1f, %.1f]",
        tl.stats_min_percentile, tl.stats_max_percentile,
    )

    # GPU probe
    log.info("[gpu]")
    available, info = _probe_gpu(dg.gpu_device)
    status = "available" if available else "unavailable"
    log.info("  device %d : %s — %s", dg.gpu_device, status, info)


# =============================================================================
# CLI
# =============================================================================

_STAGE_ORDER: Sequence[str] = ("pansharpening", "degradation", "tiling")

_STAGE_RUNNERS: Dict[str, Any] = {
    "pansharpening": run_pansharpening,
    "degradation":   run_degradation,
    "tiling":        run_tiling,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog            = "preprocessing.py",
        description     = (
            "Pléiades NEO SR preprocessing pipeline: "
            "pansharpening → degradation → tiling."
        ),
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
        epilog          = (
            "Examples:\n"
            "  python preprocessing.py\n"
            "  python preprocessing.py --stages degradation tiling\n"
            "  python preprocessing.py --from-stage tiling\n"
            "  python preprocessing.py --dry-run\n"
            "  python preprocessing.py --stages tiling --overwrite"
        ),
    )
    parser.add_argument(
        "--config", "-c",
        type    = Path,
        default = _PROJECT_ROOT / "configs" / "preprocessing.yaml",
        metavar = "PATH",
        help    = "YAML configuration file.",
    )

    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--stages",
        nargs   = "+",
        choices = list(_STAGE_ORDER),
        metavar = "STAGE",
        help    = (
            "Explicit list of stages to run.  "
            "Valid: pansharpening, degradation, tiling.  "
            "Stages run in the order given."
        ),
    )
    stage_group.add_argument(
        "--from-stage",
        choices = list(_STAGE_ORDER),
        metavar = "STAGE",
        dest    = "from_stage",
        help    = (
            "Start from this stage and run all subsequent ones.  "
            "E.g. --from-stage tiling runs only the tiling stage."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action  = "store_true",
        default = False,
        help    = (
            "Override the per-stage overwrite settings from the config.  "
            "Forces all selected stages to re-process existing outputs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Validate config, print resolved settings, then exit without processing.",
    )
    return parser


def _resolve_stages(args: argparse.Namespace) -> List[str]:
    """Return the ordered list of stages to execute."""
    if args.stages:
        # Explicit list — honour user order.
        return list(args.stages)
    if args.from_stage:
        # Start from a given stage and run all subsequent ones.
        idx = list(_STAGE_ORDER).index(args.from_stage)
        return list(_STAGE_ORDER[idx:])
    # Default: full pipeline.
    return list(_STAGE_ORDER)


# =============================================================================
# Entry point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    # ── Load and validate config ──────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except (ConfigError, yaml.YAMLError) as exc:
        print(f"ERROR in config '{args.config}': {exc}", file=sys.stderr)
        sys.exit(1)

    log = configure_logging(cfg.logging)
    log.info("Config  : %s", args.config.resolve())

    stages = _resolve_stages(args)
    log.info("Stages  : %s", " → ".join(stages))
    if args.overwrite:
        log.info("Overwrite: forced ON via --overwrite flag.")

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        print_config(cfg, log)
        log.info("Dry-run complete — no files written.")
        return

    # ── Create output directories ─────────────────────────────────────────────
    for directory in (
        cfg.data.pansharpened_hr,
        cfg.data.pansharpened_lr,
        cfg.data.processed,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # ── Run stages ────────────────────────────────────────────────────────────
    pipeline_start  = time.perf_counter()
    stage_timings: List[Tuple[str, float, bool]] = []   # (name, seconds, success)

    for stage_name in stages:
        stage_start = time.perf_counter()
        log.info("")
        log.info("▶  Starting  %s", stage_name)
        success = True

        try:
            _STAGE_RUNNERS[stage_name](cfg, log, overwrite=args.overwrite)
        except Exception as exc:
            log.error(
                "Stage '%s' FAILED after %s:\n  %s",
                stage_name, _elapsed(stage_start), exc,
                exc_info=True,
            )
            success = False
            stage_timings.append((stage_name, time.perf_counter() - stage_start, False))
            # Abort remaining stages — outputs are incomplete.
            sys.exit(1)

        elapsed = time.perf_counter() - stage_start
        log.info("✓  Finished  %s  in %s", stage_name, _elapsed(stage_start))
        stage_timings.append((stage_name, elapsed, True))

    # ── Summary ───────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - pipeline_start
    sep = "═" * 62

    print(f"\n{sep}", flush=True)
    print(f"  Pipeline complete in {_elapsed(pipeline_start)}", flush=True)
    print(f"  {'Stage':<20} {'Status':<10} {'Time':>8}", flush=True)
    print(f"  {'─'*20} {'─'*10} {'─'*8}", flush=True)
    for name, elapsed, ok in stage_timings:
        status = "✓ OK" if ok else "✗ FAILED"
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        if h:
            t_str = f"{h}h {m:02d}m {s:02d}s"
        elif m:
            t_str = f"{m}m {s:02d}s"
        else:
            t_str = f"{s}s"
        print(f"  {name:<20} {status:<10} {t_str:>8}", flush=True)
    print(sep, flush=True)


if __name__ == "__main__":
    main()