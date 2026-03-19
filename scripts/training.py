"""
scripts/training.py — Swin2-MoSE Fine-tuning Entry Point
=========================================================
Assembles all components from ``src/train/``, resolves configuration from
YAML + optional CLI overrides, then delegates to ``Trainer.fit()``.

Usage
-----
    # Full run with default config:
    python scripts/training.py --config configs/swinir_finetune.yaml

    # Resume from a checkpoint:
    python scripts/training.py --config configs/swinir_finetune.yaml \\
        --resume runs/swinir_finetune/checkpoints/epoch_0050.pth

    # Override individual config values at the command line:
    python scripts/training.py --config configs/swinir_finetune.yaml \\
        --set training.batch_size=16 training.epochs=200 optimizer.lr=1e-4

    # Validate config without running (dry-run):
    python scripts/training.py --config configs/swinir_finetune.yaml --dry-run

Directory layout expected by this script
-----------------------------------------
    <project_root>/
        configs/swinir_finetune.yaml
        data/processed/train/{HR,LR}/
        data/processed/val/{HR,LR}/
        pretrained/model-70.pt            ← Swin2-MoSE release checkpoint
        swin2-mose/                       ← git clone -b official_code
        scripts/training.py               ← this file
        src/train/                        ← library modules
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Project root resolution ───────────────────────────────────────────────────
# Ensures ``src/`` is importable regardless of the working directory.
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.train.dataset import SRTileDataset, worker_init_fn
from src.train.losses  import build_criterion
from src.train.trainer import Trainer
from src.train.utils   import (
    DotDict,
    build_model,
    build_optimizer,
    build_scheduler,
    get_logger,
    get_writer,
    load_checkpoint,
    load_config,
    resolve_device,
    set_seed,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog            = "training.py",
        description     = "Fine-tune Swin2-MoSE on a paired HR/LR GeoTIFF dataset.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type    = Path,
        default = _PROJECT_ROOT / "configs" / "swinir_finetune.yaml",
        metavar = "PATH",
        help    = "Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type    = Path,
        default = None,
        metavar = "CKPT",
        help    = (
            "Path to a training checkpoint to resume from.  "
            "Overrides misc.resume in the config."
        ),
    )
    parser.add_argument(
        "--device",
        type    = str,
        default = None,
        metavar = "DEVICE",
        help    = "Compute device override (e.g. 'cuda:1', 'cpu').  "
                  "Overrides misc.device in the config.",
    )
    parser.add_argument(
        "--set",
        nargs   = "*",
        default = [],
        metavar = "KEY=VALUE",
        help    = (
            "Override any config key using dot notation.  "
            "Examples: --set training.batch_size=16 optimizer.lr=5e-5"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action  = "store_true",
        help    = "Validate config and print resolved settings without running.",
    )
    return parser


def _apply_overrides(cfg: DotDict, overrides: list[str]) -> None:
    """Apply ``KEY=VALUE`` override strings onto the config in-place.

    Nested keys use dot notation: ``training.batch_size=16`` sets
    ``cfg["training"]["batch_size"] = 16``.  Values are parsed as YAML
    scalars so "true", "1e-4", "[6,6,6]" are handled correctly.
    """
    import yaml as _yaml

    for override in overrides:
        if "=" not in override:
            raise ValueError(
                f"Invalid --set argument '{override}'.  Expected format: KEY=VALUE"
            )
        key_str, val_str = override.split("=", 1)
        keys  = key_str.strip().split(".")
        value = _yaml.safe_load(val_str.strip())

        node = cfg
        for k in keys[:-1]:
            if k not in node:
                node[k] = DotDict()
            if not isinstance(node[k], dict):
                node[k] = DotDict()
            node = node[k]
        node[keys[-1]] = value


def _validate_config(cfg: DotDict, log) -> None:
    """Sanity-check the most common misconfiguration sources."""
    ws      = int(cfg.model.window_size)
    lps     = int(cfg.data.lr_patch_size)
    scale   = int(cfg.data.scale)
    upscale = int(cfg.model.upscale)

    if lps % ws != 0:
        raise ValueError(
            f"data.lr_patch_size ({lps}) must be divisible by "
            f"model.window_size ({ws})."
        )
    if scale != upscale:
        log.warning(
            "data.scale (%d) ≠ model.upscale (%d).  "
            "Make sure both match your dataset's degradation factor.",
            scale, upscale,
        )

    for split, hr_key, lr_key in (
        ("train", "train_hr", "train_lr"),
        ("val",   "val_hr",   "val_lr"),
    ):
        hr_root = Path(cfg.data[hr_key])
        lr_root = Path(cfg.data[lr_key])
        for label, p in ((f"{split} HR", hr_root), (f"{split} LR", lr_root)):
            if not p.exists():
                log.warning("Data directory for %s not found: %s", label, p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args   = parser.parse_args(argv)

    # ── Load and patch config ─────────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.set:
        _apply_overrides(cfg, args.set)

    # CLI flags take precedence over config file values.
    if args.resume is not None:
        cfg["misc"] = cfg.get("misc", DotDict())
        cfg["misc"]["resume"] = str(args.resume)
    if args.device is not None:
        cfg["misc"] = cfg.get("misc", DotDict())
        cfg["misc"]["device"] = args.device

    # ── Logging ───────────────────────────────────────────────────────────────
    run_dir = Path(cfg.logging.run_dir)
    log     = get_logger("training", run_dir)
    log.info("Config : %s", args.config.resolve())
    log.info("Run dir: %s", run_dir.resolve())

    _validate_config(cfg, log)

    # ── Dry-run mode ──────────────────────────────────────────────────────────
    if args.dry_run:
        import json
        log.info(
            "Resolved configuration:\n%s",
            json.dumps(dict(cfg), indent=2, default=str),
        )
        log.info("Dry-run complete — no files written.")
        return

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = int(getattr(cfg.misc, "seed", 42))
    set_seed(seed)
    log.info("Random seed: %d", seed)

    # ── Device ────────────────────────────────────────────────────────────────
    device = resolve_device(cfg)
    log.info("Device: %s", device)

    # ── Datasets & DataLoaders ────────────────────────────────────────────────
    aug_cfg  = dict(cfg.training.augmentation) if hasattr(cfg.training, "augmentation") else {}
    data_cfg = cfg.data

    train_dataset = SRTileDataset(
        hr_root          = Path(data_cfg.train_hr),
        lr_root          = Path(data_cfg.train_lr),
        scale            = int(data_cfg.scale),
        lr_patch_size    = int(data_cfg.lr_patch_size),
        augment          = True,
        augmentation_cfg = aug_cfg,
        dtype_max        = float(data_cfg.dtype_max),
    )
    val_dataset = SRTileDataset(
        hr_root          = Path(data_cfg.val_hr),
        lr_root          = Path(data_cfg.val_lr),
        scale            = int(data_cfg.scale),
        lr_patch_size    = int(data_cfg.lr_patch_size),
        augment          = False,
        dtype_max        = float(data_cfg.dtype_max),
    )

    log.info(
        "Dataset sizes — train: %d tiles  |  val: %d tiles",
        len(train_dataset), len(val_dataset),
    )

    n_workers = int(data_cfg.num_workers)
    loader_kwargs = dict(
        num_workers        = n_workers,
        pin_memory         = bool(data_cfg.pin_memory),
        prefetch_factor    = int(getattr(data_cfg, "prefetch_factor", 2))
                             if n_workers > 0 else None,
        worker_init_fn     = worker_init_fn,
        persistent_workers = n_workers > 0,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size = int(cfg.training.batch_size),
        shuffle    = True,
        drop_last  = True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size = 1,          # evaluate one tile at a time (safest for VRAM)
        shuffle    = False,
        drop_last  = False,
        **loader_kwargs,
    )

    log.info(
        "DataLoaders — train: %d batches/epoch  |  val: %d tiles",
        len(train_loader), len(val_loader),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg, device)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = build_criterion(cfg).to(device)

    # ── TensorBoard writer ────────────────────────────────────────────────────
    writer = get_writer(cfg, run_dir)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        cfg          = cfg,
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        scheduler    = scheduler,
        criterion    = criterion,
        device       = device,
        run_dir      = run_dir,
        writer       = writer,
    )

    # ── Resume or start fresh ─────────────────────────────────────────────────
    start_epoch     = 0
    resume_path_str = getattr(getattr(cfg, "misc", None), "resume", None)
    if resume_path_str:
        resume_path = Path(resume_path_str)
        start_epoch = load_checkpoint(
            path      = resume_path,
            model     = model,
            optimizer = optimizer,
            scheduler = scheduler,
            device    = device,
        )
        trainer.best_psnr = torch.load(
            resume_path, map_location="cpu", weights_only=False
        ).get("best_psnr", 0.0)

    # ── Go ────────────────────────────────────────────────────────────────────
    trainer.fit(start_epoch=start_epoch)


if __name__ == "__main__":
    main()