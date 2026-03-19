"""
src/train/utils.py — Build helpers, checkpoint management, and logging setup.
==============================================================================
All component-construction logic lives here so that scripts/training.py
remains a thin orchestrator and individual components can be unit-tested
in isolation.

Public API
----------
  DotDict                              — dict with attribute-style access
  load_config(path)                    → DotDict
  set_seed(seed)
  resolve_device(cfg)                  → torch.device
  build_model(cfg, device)             → nn.Module
  build_optimizer(cfg, model)          → Optimizer
  build_scheduler(cfg, optimizer)      → LRScheduler | None
  save_checkpoint(state, path)
  load_checkpoint(path, model, optimizer, scheduler, device)  → start_epoch
  get_logger(name, log_dir)            → logging.Logger
  get_writer(cfg, log_dir)             → SummaryWriter | None
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

try:
    import yaml
except ImportError as exc:
    raise ImportError("pyyaml is required.  pip install pyyaml") from exc


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

class DotDict(dict):
    """dict subclass that supports recursive attribute-style access.

    Nested dicts are automatically promoted to DotDict on access so that
    ``cfg.training.batch_size`` works identically to
    ``cfg["training"]["batch_size"]``.
    """

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
            return DotDict(value) if isinstance(value, dict) else value
        except KeyError:
            raise AttributeError(
                f"Configuration has no key '{key}'."
            ) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __repr__(self) -> str:
        return f"DotDict({dict.__repr__(self)})"


def load_config(path: Path) -> DotDict:
    """Parse a YAML file and return a DotDict.

    Raises
    ------
    FileNotFoundError : config file does not exist.
    yaml.YAMLError    : YAML syntax error.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return DotDict(raw)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducibility.

    Note: full determinism on GPU also requires setting
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and calling
    ``torch.use_deterministic_algorithms(True)``.  We do not force
    deterministic algorithms here because some cuDNN kernels used by
    Swin-family models do not have deterministic implementations on all GPUs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(cfg: DotDict) -> torch.device:
    """Return the torch.device specified in cfg.misc.device.

    Falls back to CPU with a warning if CUDA is requested but unavailable.
    """
    requested = str(getattr(getattr(cfg, "misc", DotDict()), "device", "cuda"))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logging.getLogger(__name__).warning(
            "CUDA requested but not available — falling back to CPU."
        )
        return torch.device("cpu")
    return torch.device(requested)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg: DotDict, device: torch.device) -> nn.Module:
    """Instantiate Swin2-MoSE and optionally load pretrained weights.

    Swin2-MoSE integration
    -----------------------
    ``cfg.swin2_mose_root`` is temporarily prepended to ``sys.path`` so that
    ``from swin2_mose_model.model import Swin2MoSE`` resolves to the local
    repo checkout.  The path is removed after import to avoid polluting
    ``sys.path`` for the rest of the process.

    Clone and set up the repo::

        git clone -b official_code \\
            https://github.com/IMPLabUniPr/swin2-mose swin2-mose

    Pretrained weights
    ------------------
    If ``cfg.model.pretrained_path`` points to an existing file, weights are
    loaded with ``strict=False`` (MoE-specific buffers absent from older
    checkpoints are silently skipped).

    The optional ``cfg.model.pretrained_key`` extracts a nested state dict
    from the checkpoint dict.  Set it to ``null`` for Swin2-MoSE ``.pt``
    release assets, which store the state dict at the top level.

    MoE-SM architecture parameters
    --------------------------------
    Swin2-MoSE accepts two extra constructor arguments vs. Swin2SR:

    * ``num_experts``  — Total MoE-SM experts per transformer block.
    * ``k``            — Active experts per input example (Top-K gate).

    Both are read from ``cfg.model`` and default to 4 and 2 respectively,
    matching the configuration reported in the paper.

    Frozen layers
    -------------
    Parameters whose names start with any prefix listed in
    ``cfg.training.frozen_prefixes`` have ``requires_grad`` set to False.
    Swin2-MoSE shares the same top-level layer names as Swin2SR / SwinIR:
    ``patch_embed``, ``layers.0`` … ``layers.5``, ``norm``,
    ``conv_after_body``, ``conv_before_upsample``, ``upsample``,
    ``conv_last``.
    """
    _log = logging.getLogger(__name__)

    swin2_mose_root = Path(cfg.swin2_mose_root).resolve()
    if not swin2_mose_root.exists():
        raise FileNotFoundError(
            f"Swin2-MoSE root not found: {swin2_mose_root}\n"
            "Clone the repo with:\n"
            "  git clone -b official_code "
            "https://github.com/IMPLabUniPr/swin2-mose swin2-mose"
        )

    sys.path.insert(0, str(swin2_mose_root))
    try:
        from swin2_mose_model.model import Swin2MoSE  # noqa: PLC0415
    finally:
        sys.path.remove(str(swin2_mose_root))

    m = cfg.model
    model = Swin2MoSE(
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
        # MoE-SM parameters — unique to Swin2-MoSE.
        num_experts     = int(getattr(m, "num_experts", 4)),
        k               = int(getattr(m, "k",           2)),
    )

    # ── Pretrained weights ────────────────────────────────────────────────────
    pretrained_path = getattr(m, "pretrained_path", None)
    if pretrained_path:
        ckpt_path = Path(pretrained_path)
        if not ckpt_path.exists():
            _log.warning(
                "Pretrained checkpoint not found: %s\n"
                "  Download from the Swin2-MoSE releases page:\n"
                "  https://github.com/IMPLabUniPr/swin2-mose/releases",
                ckpt_path,
            )
        else:
            _log.info("Loading pretrained weights from: %s", ckpt_path)
            raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

            # Resolve the state dict from the (possibly nested) checkpoint.
            # Priority: explicit pretrained_key → "model" key → flat dict.
            key = getattr(m, "pretrained_key", None)
            if key and isinstance(raw, dict) and key in raw:
                state_dict = raw[key]
            elif isinstance(raw, dict) and "model" in raw:
                state_dict = raw["model"]
            elif isinstance(raw, dict) and not any(
                k in raw for k in ("model", "params", "params_ema")
            ):
                # Top-level flat state dict (Swin2-MoSE .pt release format).
                state_dict = raw
            else:
                state_dict = raw.get(key, raw) if isinstance(raw, dict) else raw

            # Strip DataParallel prefix if present.
            if any(k.startswith("module.") for k in state_dict):
                state_dict = {
                    k.removeprefix("module."): v for k, v in state_dict.items()
                }

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                _log.warning(
                    "Pretrained checkpoint — missing keys (%d): %s …",
                    len(missing), missing[:3],
                )
            if unexpected:
                _log.warning(
                    "Pretrained checkpoint — unexpected keys (%d): %s …",
                    len(unexpected), unexpected[:3],
                )
            _log.info("Pretrained weights loaded.")

    # ── Static layer freeze ───────────────────────────────────────────────────
    frozen_prefixes = getattr(getattr(cfg, "training", DotDict()), "frozen_prefixes", None)
    if frozen_prefixes:
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad = False
                frozen_count += 1
        _log.info(
            "Froze %d parameter tensor(s) matching prefixes: %s",
            frozen_count, frozen_prefixes,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    _log.info(
        "Model: Swin2-MoSE — trainable %s / total %s params (%.1f%%)",
        f"{trainable:,}", f"{total:,}", 100.0 * trainable / max(total, 1),
    )

    return model.to(device)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(cfg: DotDict, model: nn.Module) -> torch.optim.Optimizer:
    """Instantiate the optimizer from configuration.

    Only parameters with ``requires_grad=True`` are passed to the optimizer
    so frozen layers are not tracked and do not consume gradient memory.
    """
    _log     = logging.getLogger(__name__)
    opt_cfg  = cfg.optimizer
    opt_type = str(opt_cfg.type).lower()

    params = [p for p in model.parameters() if p.requires_grad]

    kwargs: Dict[str, Any] = dict(
        lr           = float(opt_cfg.lr),
        betas        = tuple(float(b) for b in opt_cfg.betas),
        weight_decay = float(getattr(opt_cfg, "weight_decay", 0.0)),
        eps          = float(getattr(opt_cfg, "eps", 1e-8)),
    )

    if opt_type == "adam":
        optimizer = torch.optim.Adam(params, **kwargs)
    elif opt_type == "adamw":
        optimizer = torch.optim.AdamW(params, **kwargs)
    else:
        raise ValueError(
            f"Unknown optimizer type '{opt_type}'.  Choose from: adam, adamw."
        )

    _log.info(
        "Optimizer: %s  lr=%g  betas=%s  wd=%g",
        opt_type, kwargs["lr"], kwargs["betas"], kwargs["weight_decay"],
    )
    return optimizer


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(
    cfg:       DotDict,
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Instantiate the LR scheduler from configuration.

    Returns None if ``scheduler.type`` is "none".
    """
    _log     = logging.getLogger(__name__)
    sch_cfg  = cfg.scheduler
    sch_type = str(getattr(sch_cfg, "type", "none")).lower()

    if sch_type == "none":
        _log.info("Scheduler: none (constant LR)")
        return None

    if sch_type == "multistep":
        milestones = list(sch_cfg.milestones)
        gamma      = float(getattr(sch_cfg, "gamma", 0.5))
        scheduler  = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
        _log.info(
            "Scheduler: MultiStepLR  milestones=%s  γ=%g", milestones, gamma
        )
        return scheduler

    if sch_type == "cosine":
        t_max   = int(cfg.training.epochs)
        eta_min = float(getattr(sch_cfg, "eta_min", 1e-7))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
        _log.info(
            "Scheduler: CosineAnnealingLR  T_max=%d  η_min=%g", t_max, eta_min
        )
        return scheduler

    raise ValueError(
        f"Unknown scheduler type '{sch_type}'.  "
        f"Choose from: none, multistep, cosine."
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path) -> None:
    """Atomically save *state* to *path* using a temp file + rename."""
    path = Path(path)
    tmp  = path.with_suffix(".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(
    path:      Path,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    device:    torch.device,
) -> int:
    """Load a training checkpoint and return the epoch to resume from.

    Restores model weights, optimizer state, and (if present) scheduler
    state.  If the checkpoint was saved from a DataParallel model the
    ``module.`` prefix is stripped automatically.

    Returns
    -------
    start_epoch : int — next epoch index (checkpoint_epoch + 1).
    """
    _log = logging.getLogger(__name__)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    _log.info("Resuming from checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    _log.info(
        "Resumed at epoch %d  (best PSNR: %.2f dB)",
        start_epoch - 1, ckpt.get("best_psnr", 0.0),
    )
    return start_epoch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Configure logging and return the named training logger.

    Two problems fixed vs. the naive approach:

    1. Root-logger level
       By default the root logger sits at WARNING.  Any helper module that
       calls ``logging.getLogger(__name__)`` will have its INFO messages
       silently dropped, even if a handler exists on the named "training"
       logger, because propagation stops at the first ancestor whose level
       filters out the message.  We set the root level to DEBUG and attach a
       single console handler so that every module in the project can emit
       INFO+ messages without needing its own handler.

    2. Duplicate handlers on re-call
       The root handler is only added once (guard on root.handlers), and the
       named logger is returned early if already configured.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.DEBUG)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root.addHandler(console)
    else:
        root.setLevel(logging.DEBUG)

    logger = logging.getLogger(name)
    # Guard against duplicate FileHandlers when get_logger is called twice.
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # propagate=False avoids double-printing via the root console handler.
        logger.propagate = False
        console_named = logging.StreamHandler(sys.stdout)
        console_named.setLevel(logging.INFO)
        console_named.setFormatter(formatter)
        logger.addHandler(console_named)

    return logger


def get_writer(cfg: DotDict, log_dir: Path) -> Optional[Any]:
    """Return a TensorBoard SummaryWriter if logging.tensorboard is true."""
    if not getattr(cfg.logging, "tensorboard", True):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415

        writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
        logging.getLogger(__name__).info(
            "TensorBoard logs → %s", log_dir / "tensorboard"
        )
        return writer
    except ImportError:
        logging.getLogger(__name__).warning(
            "TensorBoard not installed — scalar logging disabled.\n"
            "  Install with:  pip install tensorboard\n"
            "  Then launch:   tensorboard --logdir %s",
            log_dir / "tensorboard",
        )
        return None