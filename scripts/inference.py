"""
scripts/inference.py — SwinIR Super-Resolution Inference
=========================================================
Loads a fine-tuned SwinIR checkpoint and runs tiled super-resolution
inference on one or more LR GeoTIFF files.

Usage
-----
    # Single tile:
    python scripts/inference.py \\
        --config configs/inference.yaml \\
        --input  data/processed/val/LR/WO_.../tile_row0000_col0000.TIF

    # All tiles in a directory (glob):
    python scripts/inference.py \\
        --config configs/inference.yaml \\
        --input  "data/processed/val/LR/**/*.TIF" \\
        --output output/val_sr

    # Override checkpoint and output dir on the fly:
    python scripts/inference.py \\
        --config configs/inference.yaml \\
        --input  my_lr_image.tif \\
        --checkpoint runs/swinir_finetune/checkpoints/best.pth \\
        --output output/

    # CPU inference (slow, useful for debugging):
    python scripts/inference.py \\
        --config configs/inference.yaml \\
        --input  my_lr_image.tif \\
        --device cpu
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── Project root resolution ───────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch

from src.inference.predict import load_model, run_inference
from src.train.utils import DotDict, get_logger, load_config, resolve_device


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog            = "inference.py",
        description     = "Run SwinIR super-resolution on LR GeoTIFF file(s).",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type    = Path,
        default = _PROJECT_ROOT / "configs" / "inference.yaml",
        metavar = "PATH",
        help    = "Path to inference.yaml config file.",
    )
    parser.add_argument(
        "--input",
        type    = str,
        default = None,
        metavar = "PATH_OR_GLOB",
        help    = (
            "Input LR GeoTIFF path or glob pattern.  "
            "Overrides io.input_path in the config.  "
            "Example: 'data/processed/val/LR/**/*.TIF'"
        ),
    )
    parser.add_argument(
        "--output",
        type    = Path,
        default = None,
        metavar = "DIR",
        help    = "Output directory.  Overrides io.output_dir in the config.",
    )
    parser.add_argument(
        "--checkpoint",
        type    = Path,
        default = None,
        metavar = "PATH",
        help    = "Checkpoint path.  Overrides model.checkpoint_path in the config.",
    )
    parser.add_argument(
        "--device",
        type    = str,
        default = None,
        metavar = "DEVICE",
        help    = "Device override: 'cuda', 'cuda:1', 'cpu'.",
    )
    parser.add_argument(
        "--set",
        nargs   = "*",
        default = [],
        metavar = "KEY=VALUE",
        help    = "Override any config key using dot notation (e.g. tiling.tile_size=128).",
    )
    return parser


def _apply_overrides(cfg: DotDict, overrides: List[str]) -> None:
    import yaml as _yaml
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid --set argument '{override}'.  Expected KEY=VALUE.")
        key_str, val_str = override.split("=", 1)
        keys  = key_str.strip().split(".")
        value = _yaml.safe_load(val_str.strip())
        node  = cfg
        for k in keys[:-1]:
            node.setdefault(k, DotDict())
            node = node[k]
        node[keys[-1]] = value


def _collect_inputs(pattern_or_path: str) -> List[Path]:
    """Expand a path or glob pattern into a sorted list of existing files."""
    paths = sorted(Path(p) for p in glob.glob(pattern_or_path, recursive=True))
    paths = [p for p in paths if p.is_file()]
    return paths


def _elapsed(start: float) -> str:
    secs = time.perf_counter() - start
    m, s = divmod(int(secs), 60)
    return f"{m:02d}m {s:02d}s"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config(args.config)

    if args.set:
        _apply_overrides(cfg, args.set)

    # CLI flags override config values.
    if args.input:
        cfg["io"] = cfg.get("io", DotDict())
        cfg["io"]["input_path"] = args.input
    if args.output:
        cfg["io"] = cfg.get("io", DotDict())
        cfg["io"]["output_dir"] = str(args.output)
    if args.checkpoint:
        cfg["model"] = cfg.get("model", DotDict())
        cfg["model"]["checkpoint_path"] = str(args.checkpoint)
    if args.device:
        cfg["misc"] = cfg.get("misc", DotDict())
        cfg["misc"]["device"] = args.device

    # ── Logging ───────────────────────────────────────────────────────────────
    output_dir = Path(cfg.io.output_dir)
    log        = get_logger("inference", output_dir)
    log.info("Config     : %s", args.config.resolve())
    log.info("Output dir : %s", output_dir.resolve())

    # ── Device ────────────────────────────────────────────────────────────────
    device = resolve_device(cfg)
    log.info("Device     : %s", device)

    # ── Collect input files ───────────────────────────────────────────────────
    input_str = str(cfg.io.input_path).strip()
    if not input_str:
        log.error("No input file specified.  Use --input or set io.input_path in the config.")
        sys.exit(1)

    input_files = _collect_inputs(input_str)

    # If the string is not a glob, treat it as a single literal path.
    if not input_files:
        single = Path(input_str)
        if single.is_file():
            input_files = [single]
        else:
            log.error(
                "Input not found: '%s'\n"
                "  Pass a file path or a glob pattern (quote it to prevent shell expansion).",
                input_str,
            )
            sys.exit(1)

    log.info("Input file(s): %d", len(input_files))

    # ── Validate config ───────────────────────────────────────────────────────
    ts = int(cfg.tiling.tile_size)
    ws = int(cfg.model.window_size)
    if ts % ws != 0:
        log.error(
            "tiling.tile_size (%d) must be a multiple of model.window_size (%d).", ts, ws
        )
        sys.exit(1)

    # ── Load model once, reuse for all inputs ─────────────────────────────────
    model_load_start = time.perf_counter()
    model = load_model(cfg, device)
    log.info("Model loaded in %.1fs", time.perf_counter() - model_load_start)

    # ── Run inference ─────────────────────────────────────────────────────────
    success = 0
    failed  = 0
    run_start = time.perf_counter()

    for i, input_path in enumerate(input_files, start=1):
        log.info("─── [%d/%d]  %s", i, len(input_files), input_path.name)
        file_start = time.perf_counter()

        # Point cfg at the current file for this iteration.
        cfg["io"]["input_path"] = str(input_path)

        try:
            out_path = run_inference(cfg, model, device)
            log.info(
                "    ✓  %s  (%.1fs)",
                out_path.name, time.perf_counter() - file_start,
            )
            success += 1
        except Exception as exc:
            log.error("    ✗  FAILED — %s: %s", input_path.name, exc, exc_info=True)
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    separator = "─" * 60
    log.info(separator)
    log.info(
        "Done in %s  │  ✓ %d succeeded  │  ✗ %d failed",
        _elapsed(run_start), success, failed,
    )
    log.info("Output: %s", output_dir.resolve())
    log.info(separator)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
