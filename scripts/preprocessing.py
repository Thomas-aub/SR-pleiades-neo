"""
=============================================================================
preprocessing.py — Full SR Dataset Preprocessing Pipeline
=============================================================================
Single entry point that runs the three-stage pipeline:

    Stage 1 · pansharpening   raw PAN + MS  →  data/pansharpened/HR/
    Stage 2 · degradation     HR TIFs       →  data/pansharpened/LR/
    Stage 3 · tiling          HR + LR TIFs  →  data/processed/train|val/

All parameters are read from a YAML configuration file (default:
``preprocessing.yaml`` in the working directory).  The Python scripts
for each stage (pansharpening.py, degrade_pipeline.py, build_dataset.py)
must be importable from the same directory or via PYTHONPATH.

Usage
-----
    # Run the full pipeline with the default config:
    python preprocessing.py

    # Point to a custom config file:
    python preprocessing.py --config path/to/my_config.yaml

    # Run only specific stages:
    python preprocessing.py --stages pansharpening degradation

    # Validate config and print the resolved settings without running:
    python preprocessing.py --dry-run

Dependencies
------------
    pip install pyyaml tqdm rasterio numpy
    # Optional GPU acceleration:
    pip install cupy-cuda12x   # adjust to your CUDA version

=============================================================================
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Optional dependency guard — PyYAML
# ---------------------------------------------------------------------------

try:
    import yaml
except ImportError:
    sys.exit(
        "ERROR: 'pyyaml' is required.  pip install pyyaml"
    )

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: 'tqdm' is required.  pip install tqdm")


# =============================================================================
# Configuration dataclasses
# =============================================================================

@dataclass
class DataPaths:
    """Resolved filesystem paths for all pipeline inputs and outputs."""
    raw_root:        Path
    pansharpened_hr: Path
    pansharpened_lr: Path
    processed:       Path


@dataclass
class PansharpeningConfig:
    method:        str   = "brovey"
    resample_algo: str   = "bilinear"
    output_dtype:  str   = "uint16"
    output_suffix: str   = "_PANSHARP"
    compress:      str   = "none"
    use_gpu:       bool  = True
    chunk_rows:    int   = 2048


@dataclass
class DegradationConfig:
    use_gpu:      bool        = True
    gpu_device:   int         = 0
    tile_size:    int         = 4096
    tile_overlap: int         = 64
    compress:     str         = "none"
    overwrite:    bool        = False
    pipeline:     List[Dict]  = field(default_factory=list)


@dataclass
class TilingConfig:
    tile_size:          int   = 512
    val_ratio:          float = 0.2
    seed:               int   = 42
    compress:           str   = "deflate"
    min_valid_fraction: float = 0.1
    overwrite:          bool  = False


@dataclass
class LoggingConfig:
    level: str            = "INFO"
    file:  Optional[str]  = None


@dataclass
class PipelineConfig:
    """Top-level configuration object parsed from the YAML file."""
    data:           DataPaths
    pansharpening:  PansharpeningConfig
    degradation:    DegradationConfig
    tiling:         TilingConfig
    logging:        LoggingConfig


# =============================================================================
# YAML loading and validation
# =============================================================================

def _require(mapping: Dict, key: str, context: str) -> Any:
    """Return ``mapping[key]`` or raise a descriptive ``KeyError``."""
    if key not in mapping:
        raise KeyError(f"[{context}] Missing required key: '{key}'")
    return mapping[key]


def load_config(path: Path) -> PipelineConfig:
    """
    Parse and validate the YAML configuration file.

    Parameters
    ----------
    path : Absolute or relative path to the YAML config file.

    Returns
    -------
    Fully validated ``PipelineConfig`` instance.

    Raises
    ------
    FileNotFoundError   : Config file not found.
    KeyError            : Required section or key missing.
    ValueError          : Invalid parameter value.
    yaml.YAMLError      : YAML syntax error.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: Dict = yaml.safe_load(fh) or {}

    # ── data ──────────────────────────────────────────────────────────────────
    data_raw = _require(raw, "data", "data")
    data = DataPaths(
        raw_root        = Path(_require(data_raw, "raw_root",        "data")),
        pansharpened_hr = Path(_require(data_raw, "pansharpened_hr", "data")),
        pansharpened_lr = Path(_require(data_raw, "pansharpened_lr", "data")),
        processed       = Path(_require(data_raw, "processed",       "data")),
    )

    # ── pansharpening ─────────────────────────────────────────────────────────
    ps_raw = raw.get("pansharpening", {})
    pansharpening = PansharpeningConfig(
        method        = str(ps_raw.get("method",        PansharpeningConfig.method)),
        resample_algo = str(ps_raw.get("resample_algo", PansharpeningConfig.resample_algo)),
        output_dtype  = str(ps_raw.get("output_dtype",  PansharpeningConfig.output_dtype)),
        output_suffix = str(ps_raw.get("output_suffix", PansharpeningConfig.output_suffix)),
        compress      = str(ps_raw.get("compress",      PansharpeningConfig.compress)),
        use_gpu       = bool(ps_raw.get("use_gpu",      PansharpeningConfig.use_gpu)),
        chunk_rows    = int(ps_raw.get("chunk_rows",    PansharpeningConfig.chunk_rows)),
    )
    _validate_pansharpening(pansharpening)

    # ── degradation ───────────────────────────────────────────────────────────
    dg_raw = raw.get("degradation", {})
    pipeline_steps: List[Dict] = dg_raw.get("pipeline", [])
    if not pipeline_steps:
        raise ValueError("[degradation] 'pipeline' must contain at least one operation.")
    degradation = DegradationConfig(
        use_gpu      = bool(dg_raw.get("use_gpu",      DegradationConfig.use_gpu)),
        gpu_device   = int(dg_raw.get("gpu_device",    DegradationConfig.gpu_device)),
        tile_size    = int(dg_raw.get("tile_size",     DegradationConfig.tile_size)),
        tile_overlap = int(dg_raw.get("tile_overlap",  DegradationConfig.tile_overlap)),
        compress     = str(dg_raw.get("compress",      DegradationConfig.compress)),
        overwrite    = bool(dg_raw.get("overwrite",    DegradationConfig.overwrite)),
        pipeline     = [dict(step) for step in pipeline_steps],
    )
    _validate_degradation(degradation)

    # ── tiling ────────────────────────────────────────────────────────────────
    tl_raw = raw.get("tiling", {})
    tiling = TilingConfig(
        tile_size          = int(tl_raw.get("tile_size",          TilingConfig.tile_size)),
        val_ratio          = float(tl_raw.get("val_ratio",        TilingConfig.val_ratio)),
        seed               = int(tl_raw.get("seed",               TilingConfig.seed)),
        compress           = str(tl_raw.get("compress",           TilingConfig.compress)),
        min_valid_fraction = float(tl_raw.get("min_valid_fraction", TilingConfig.min_valid_fraction)),
        overwrite          = bool(tl_raw.get("overwrite",         TilingConfig.overwrite)),
    )
    _validate_tiling(tiling)

    # ── logging ───────────────────────────────────────────────────────────────
    lg_raw = raw.get("logging", {})
    log_cfg = LoggingConfig(
        level = str(lg_raw.get("level", LoggingConfig.level)).upper(),
        file  = lg_raw.get("file", None),
    )

    return PipelineConfig(
        data          = data,
        pansharpening = pansharpening,
        degradation   = degradation,
        tiling        = tiling,
        logging       = log_cfg,
    )


def _validate_pansharpening(cfg: PansharpeningConfig) -> None:
    valid_methods  = {"brovey", "simple_mean"}
    valid_algos    = {"nearest", "bilinear", "cubic", "cubic_spline", "lanczos"}
    valid_dtypes   = {"uint8", "uint16"}
    valid_compress = {"none", "lzw", "deflate"}

    if cfg.method not in valid_methods:
        raise ValueError(f"[pansharpening] method must be one of {valid_methods}, got '{cfg.method}'")
    if cfg.resample_algo not in valid_algos:
        raise ValueError(f"[pansharpening] resample_algo must be one of {valid_algos}")
    if cfg.output_dtype not in valid_dtypes:
        raise ValueError(f"[pansharpening] output_dtype must be one of {valid_dtypes}")
    if cfg.compress not in valid_compress:
        raise ValueError(f"[pansharpening] compress must be one of {valid_compress}")
    if cfg.chunk_rows < 1:
        raise ValueError(f"[pansharpening] chunk_rows must be ≥ 1, got {cfg.chunk_rows}")


def _validate_degradation(cfg: DegradationConfig) -> None:
    valid_compress = {"none", "lzw", "deflate"}
    valid_ops      = {"mtf_blur", "downsample", "spectral_misalign", "add_noise"}

    if cfg.tile_size < 1:
        raise ValueError(f"[degradation] tile_size must be ≥ 1, got {cfg.tile_size}")
    if cfg.compress not in valid_compress:
        raise ValueError(f"[degradation] compress must be one of {valid_compress}")
    for i, step in enumerate(cfg.pipeline):
        op = step.get("op")
        if op not in valid_ops:
            raise ValueError(
                f"[degradation.pipeline[{i}]] Unknown op '{op}'. Valid: {valid_ops}"
            )


def _validate_tiling(cfg: TilingConfig) -> None:
    if cfg.tile_size < 16:
        raise ValueError(f"[tiling] tile_size must be ≥ 16, got {cfg.tile_size}")
    if not (0.0 <= cfg.val_ratio < 1.0):
        raise ValueError(f"[tiling] val_ratio must be in [0, 1), got {cfg.val_ratio}")
    if not (0.0 <= cfg.min_valid_fraction <= 1.0):
        raise ValueError(
            f"[tiling] min_valid_fraction must be in [0, 1], got {cfg.min_valid_fraction}"
        )


# =============================================================================
# Logging setup
# =============================================================================

def configure_logging(cfg: LoggingConfig) -> logging.Logger:
    """
    Configure the root logger with console + optional file handlers.

    Returns the module-level logger for ``preprocessing.py``.
    """
    numeric_level = getattr(logging, cfg.level, logging.INFO)
    formatter     = logging.Formatter(
        fmt     = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt = "%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional file handler
    if cfg.file:
        log_path = Path(cfg.file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return logging.getLogger("preprocessing")


# =============================================================================
# Stage helpers
# =============================================================================

def _banner(log: logging.Logger, title: str) -> None:
    separator = "═" * 60
    log.info(separator)
    log.info("  %s", title)
    log.info(separator)


def _elapsed(start: float) -> str:
    secs = time.perf_counter() - start
    mins, secs = divmod(int(secs), 60)
    return f"{mins:02d}m {secs:02d}s"


# =============================================================================
# Stage 1 — Pansharpening
# =============================================================================

def run_pansharpening(cfg: PipelineConfig, log: logging.Logger) -> None:
    """
    Import pansharpening.py, inject config parameters, and process all
    RGB/PAN pairs found under data.raw_root.

    The module's ``discover_pairs`` and ``process_pair`` public functions
    are called directly; module-level parameters are patched before each
    call so that no globals from the default parameter block are used.
    """
    _banner(log, "Stage 1 · Pansharpening")

    try:
        import src.preprocessing.pansharpening as ps_mod
    except ImportError:
        log.error("Cannot import 'pansharpening'.  Ensure pansharpening.py is on PYTHONPATH.")
        raise

    # ── Patch module globals ───────────────────────────────────────────────────
    ps = cfg.pansharpening
    ps_mod.PANSHARPEN_METHOD = ps.method
    ps_mod.RESAMPLE_ALGO     = ps.resample_algo
    ps_mod.OUTPUT_DTYPE      = ps.output_dtype
    ps_mod.OUTPUT_SUFFIX     = ps.output_suffix
    ps_mod.COMPRESS          = ps.compress
    ps_mod.CHUNK_ROWS        = ps.chunk_rows
    ps_mod.USE_GPU           = ps.use_gpu

    # Re-initialise the GPU backend with the (possibly patched) USE_GPU flag.
    import numpy as np
    ps_mod.xp            = np
    ps_mod.GPU_AVAILABLE = False
    if ps.use_gpu:
        try:
            import cupy as cp
            cp.zeros(1)
            ps_mod.xp            = cp
            ps_mod.GPU_AVAILABLE = True
            log.info("Pansharpening backend : CuPy (GPU)")
        except Exception as exc:
            log.warning("CuPy unavailable (%s) — falling back to NumPy.", exc)
    else:
        log.info("Pansharpening backend : NumPy (CPU)")

    input_root  = cfg.data.raw_root.resolve()
    output_root = cfg.data.pansharpened_hr.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    log.info("Input  : %s", input_root)
    log.info("Output : %s", output_root)

    pairs = ps_mod.discover_pairs(input_root)
    if not pairs:
        log.warning("No RGB/PAN pairs found under %s — skipping stage.", input_root)
        return

    log.info("Found %d pair(s) to pansharpen.", len(pairs))

    success = failed = 0
    for rgb_path, pan_path in tqdm(pairs, desc="Pansharpening", unit="pair"):
        result = ps_mod.process_pair(rgb_path, pan_path, output_root, input_root)
        if result:
            success += 1
        else:
            failed += 1

    log.info("Pansharpening complete — ✓ %d  ✗ %d", success, failed)


# =============================================================================
# Stage 2 — Sensor degradation
# =============================================================================

def run_degradation(cfg: PipelineConfig, log: logging.Logger) -> None:
    """
    Import degrade_pipeline.py, inject config parameters, and degrade all
    TIF files found under data.pansharpened_hr.

    The GPU backend (xp) is re-initialised after patching GPU_ENABLED so
    that the correct NumPy/CuPy module is used for all subsequent math.
    """
    _banner(log, "Stage 2 · Sensor Degradation")

    try:
        import degrade_pipeline as dp_mod
    except ImportError:
        log.error(
            "Cannot import 'degrade_pipeline'.  Ensure degrade_pipeline.py is on PYTHONPATH."
        )
        raise

    # ── Patch module globals ───────────────────────────────────────────────────
    dg = cfg.degradation
    dp_mod.GPU_ENABLED = dg.use_gpu
    dp_mod.GPU_DEVICE  = dg.gpu_device
    dp_mod.PIPELINE    = dg.pipeline
    dp_mod.OVERWRITE   = dg.overwrite
    dp_mod.COMPRESS    = dg.compress

    # Re-initialise the GPU backend so xp reflects the patched GPU_ENABLED.
    dp_mod.xp = dp_mod._init_backend(dg.use_gpu, dg.gpu_device)

    input_root  = cfg.data.pansharpened_hr.resolve()
    output_root = cfg.data.pansharpened_lr.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        raise FileNotFoundError(
            f"[degradation] HR root not found: {input_root}. "
            "Run the pansharpening stage first."
        )

    log.info("Input  : %s", input_root)
    log.info("Output : %s", output_root)

    from src.preprocessing.tiling import TileConfig
    tile_cfg = TileConfig(tile_size=dg.tile_size, overlap=dg.tile_overlap)

    tif_files = dp_mod.discover_tifs(input_root)
    if not tif_files:
        log.warning("No TIF files found under %s — skipping stage.", input_root)
        return

    log.info("Found %d file(s) to degrade.", len(tif_files))

    success = skipped = 0
    with tqdm(tif_files, desc="Degrading", unit="img") as bar:
        for src_path in bar:
            bar.set_postfix_str(src_path.name[:50], refresh=True)
            result = dp_mod.process_image(
                src_path   = src_path,
                input_root = input_root,
                out_root   = output_root,
                out_dtype  = None,       # preserve source dtype
                compress   = dg.compress,
                tile_cfg   = tile_cfg,
            )
            if result:
                success += 1
            else:
                skipped += 1

    log.info("Degradation complete — ✓ %d  ⟳/✗ %d", success, skipped)


# =============================================================================
# Stage 3 — Dataset tiling
# =============================================================================

def run_tiling(cfg: PipelineConfig, log: logging.Logger) -> None:
    """
    Import build_dataset.py, inject config parameters, discover matched
    HR/LR pairs, split into train/val by acquisition group, and tile.
    """
    _banner(log, "Stage 3 · Dataset Tiling")

    try:
        import build_dataset as bd_mod
    except ImportError:
        log.error(
            "Cannot import 'build_dataset'.  Ensure build_dataset.py is on PYTHONPATH."
        )
        raise

    tl = cfg.tiling
    hr_root     = cfg.data.pansharpened_hr.resolve()
    lr_root     = cfg.data.pansharpened_lr.resolve()
    output_root = cfg.data.processed.resolve()

    for label, path in (("pansharpened_hr", hr_root), ("pansharpened_lr", lr_root)):
        if not path.exists():
            raise FileNotFoundError(
                f"[tiling] '{label}' root not found: {path}. "
                "Ensure the pansharpening and degradation stages have completed."
            )

    log.info("HR root : %s", hr_root)
    log.info("LR root : %s", lr_root)
    log.info("Output  : %s", output_root)

    pairs = bd_mod.discover_pairs(hr_root, lr_root)
    if not pairs:
        log.warning("No matched HR/LR pairs found — skipping stage.")
        return

    train_pairs, val_pairs = bd_mod.split_pairs(pairs, tl.val_ratio, tl.seed)

    from src.preprocessing.build_dataset import TilingStats
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
            overwrite          = tl.overwrite,
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
        "Tiling complete — pairs ✓ %d  ✗ %d  | tiles written %d  dropped %d",
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

    log.info("[data]")
    log.info("  raw_root        : %s", cfg.data.raw_root.resolve())
    log.info("  pansharpened_hr : %s", cfg.data.pansharpened_hr.resolve())
    log.info("  pansharpened_lr : %s", cfg.data.pansharpened_lr.resolve())
    log.info("  processed       : %s", cfg.data.processed.resolve())

    ps = cfg.pansharpening
    log.info("[pansharpening]")
    log.info("  method=%s  resample=%s  dtype=%s  compress=%s  gpu=%s  chunk_rows=%d",
             ps.method, ps.resample_algo, ps.output_dtype,
             ps.compress, ps.use_gpu, ps.chunk_rows)

    dg = cfg.degradation
    log.info("[degradation]")
    log.info("  gpu=%s  device=%d  tile_size=%d  overlap=%d  compress=%s  overwrite=%s",
             dg.use_gpu, dg.gpu_device, dg.tile_size,
             dg.tile_overlap, dg.compress, dg.overwrite)
    for i, step in enumerate(dg.pipeline):
        log.info("  pipeline[%d]: %s", i, step)

    tl = cfg.tiling
    log.info("[tiling]")
    log.info(
        "  tile_size=%d  val_ratio=%.2f  seed=%d  compress=%s  "
        "min_valid=%.2f  overwrite=%s",
        tl.tile_size, tl.val_ratio, tl.seed,
        tl.compress, tl.min_valid_fraction, tl.overwrite,
    )


# =============================================================================
# CLI
# =============================================================================

_ALL_STAGES: Sequence[str] = ("pansharpening", "degradation", "tiling")

_STAGE_RUNNERS = {
    "pansharpening": run_pansharpening,
    "degradation":   run_degradation,
    "tiling":        run_tiling,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "preprocessing.py",
        description = (
            "Run the full Pléiades NEO SR preprocessing pipeline "
            "(pansharpening → degradation → tiling)."
        ),
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type    = Path,
        default = Path("preprocessing.yaml"),
        metavar = "PATH",
        help    = "Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--stages",
        nargs   = "+",
        choices = list(_ALL_STAGES),
        default = list(_ALL_STAGES),
        metavar = "STAGE",
        help    = (
            "One or more stages to execute.  "
            f"Valid choices: {', '.join(_ALL_STAGES)}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Parse and validate the config, print resolved settings, then exit.",
    )
    return parser


# =============================================================================
# Entry point
# =============================================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    # ── Load and validate config ───────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, KeyError, ValueError, yaml.YAMLError) as exc:
        # Logger not yet configured; print directly so the error is always visible.
        print(f"ERROR loading config '{args.config}': {exc}", file=sys.stderr)
        sys.exit(1)

    log = configure_logging(cfg.logging)

    log.info("Config     : %s", args.config.resolve())
    log.info("Stages     : %s", ", ".join(args.stages))

    # ── Dry-run mode ──────────────────────────────────────────────────────────
    if args.dry_run:
        print_config(cfg, log)
        log.info("Dry-run complete — no files written.")
        return

    # ── Ensure output directories exist ───────────────────────────────────────
    for directory in (
        cfg.data.pansharpened_hr,
        cfg.data.pansharpened_lr,
        cfg.data.processed,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # ── Run selected stages ────────────────────────────────────────────────────
    pipeline_start = time.perf_counter()

    for stage_name in args.stages:
        stage_start = time.perf_counter()
        log.info("Starting stage: %s", stage_name)
        try:
            _STAGE_RUNNERS[stage_name](cfg, log)
        except Exception as exc:
            log.error(
                "Stage '%s' failed after %s: %s",
                stage_name, _elapsed(stage_start), exc,
                exc_info=True,
            )
            sys.exit(1)
        log.info("Stage '%s' finished in %s.", stage_name, _elapsed(stage_start))

    # ── Final summary ──────────────────────────────────────────────────────────
    separator = "═" * 60
    tqdm.write(f"\n{separator}")
    tqdm.write(f"  Pipeline complete in {_elapsed(pipeline_start)}")
    tqdm.write(f"  Stages run : {', '.join(args.stages)}")
    tqdm.write(separator)


if __name__ == "__main__":
    main()