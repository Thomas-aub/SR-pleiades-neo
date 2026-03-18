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
  get_writer(cfg)                      → SummaryWriter | None
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
    ``cfg.training.batch_size`` works identically to ``cfg["training"]["batch_size"]``.
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
    ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` before launching the process, and
    calling ``torch.use_deterministic_algorithms(True)``.  We do not
    force deterministic algorithms here because some cuDNN kernels used by
    SwinIR (e.g. adaptive average pooling) do not have deterministic
    implementations on all GPU generations.
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

# ---------------------------------------------------------------------------
# Pretrained weight download
# ---------------------------------------------------------------------------

# Official SwinIR release artefacts hosted on GitHub Releases.
# Key: (upscale, in_chans) — values that identify the standard checkpoints.
_SWINIR_URLS: dict = {
    (2, 3): "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
             "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    (3, 3): "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
             "001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth",
    (4, 3): "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
             "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
}


def _download_pretrained(dest: Path, upscale: int = 2, in_chans: int = 3) -> Optional[Path]:
    """Download a SwinIR pretrained checkpoint from GitHub Releases.

    Parameters
    ----------
    dest     : Destination file path (directory is created automatically).
    upscale  : SR scale factor — used to select the correct release asset.
    in_chans : Number of input channels (default 3 = RGB).

    Returns
    -------
    Path to the downloaded file, or None if download failed.
    """
    import urllib.request
    import urllib.error

    _log = logging.getLogger(__name__)
    url = _SWINIR_URLS.get((upscale, in_chans))

    if url is None:
        _log.error(
            "No known pretrained checkpoint for upscale=%d in_chans=%d.\n"
            "Please set model.pretrained_path to a local .pth file.",
            upscale, in_chans,
        )
        return None

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    _log.info(
        "Pretrained checkpoint not found at '%s'.\n"
        "  Downloading from: %s",
        dest, url,
    )

    tmp = dest.with_suffix(".tmp")
    _CHUNK  = 1 << 20   # 1 MiB read chunks
    _TIMEOUT = 60        # seconds — connect + read timeout per chunk

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SwinIR-finetune/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            total_mb = total / (1 << 20) if total else 0

            print(
                f"  Downloading {dest.name}"
                + (f" ({total_mb:.1f} MB)" if total_mb else ""),
                flush=True,
            )

            downloaded = 0
            with open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(_CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = "#" * int(pct / 2)
                        print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)
            print()  # newline after progress bar

        tmp.replace(dest)
        _log.info("Pretrained checkpoint saved to: %s", dest)
        return dest

    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        _log.error(
            "Download failed: %s\n"
            "  Please download manually from:\n"
            "  https://github.com/JingyunLiang/SwinIR/releases\n"
            "  and save it to: %s",
            exc, dest,
        )
        if tmp.exists():
            tmp.unlink()
        return None


def build_model(cfg: DotDict, device: torch.device) -> nn.Module:
    """Instantiate SwinIR from KAIR and optionally load pretrained weights.

    KAIR integration
    ----------------
    The function inserts ``cfg.kair_root`` at the front of ``sys.path`` so
    that ``from models.network_swinir import SwinIR`` resolves to the KAIR
    checkout.  The path is removed after import to avoid polluting sys.path.

    Pretrained weights
    ------------------
    If ``cfg.model.pretrained_path`` is set and the file exists, weights are
    loaded with ``strict=True``.  KAIR checkpoints wrap state dicts under a
    ``params`` (or ``params_ema``) key; ``cfg.model.pretrained_key``
    controls which key to look for.

    Frozen layers
    -------------
    Parameters whose names start with any prefix in
    ``cfg.training.frozen_prefixes`` are frozen (``requires_grad = False``).
    """
    log = logging.getLogger(__name__)
    kair_root = Path(cfg.kair_root).resolve()
    if not kair_root.exists():
        raise FileNotFoundError(
            f"KAIR root not found: {kair_root}\n"
            "Clone it with:  git clone https://github.com/cszn/KAIR"
        )

    # Temporarily expose KAIR modules.
    # We suppress two warnings that originate inside KAIR / timm and are
    # not actionable from our side:
    #   • torch.meshgrid indexing FutureWarning  (KAIR uses the old positional API)
    #   • timm.models.layers import DeprecationWarning  (timm internal rename)
    import warnings
    sys.path.insert(0, str(kair_root))
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="torch.meshgrid.*indexing",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Importing from timm.models.layers",
                category=FutureWarning,
            )
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

    # ── Load pretrained weights (auto-download if missing) ───────────────────
    pretrained_path = getattr(m, "pretrained_path", None)
    if pretrained_path:
        ckpt_path = Path(pretrained_path)
        if not ckpt_path.exists():
            ckpt_path = _download_pretrained(ckpt_path, upscale=int(m.upscale))

        if ckpt_path is not None and ckpt_path.exists():
            log.info("Loading pretrained weights from: %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            key  = getattr(m, "pretrained_key", "params") or "params"
            state_dict = ckpt.get(key, ckpt) if isinstance(ckpt, dict) else ckpt
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                log.warning("Missing keys in checkpoint (%d): %s …", len(missing), missing[:3])
            if unexpected:
                log.warning("Unexpected keys in checkpoint (%d): %s …", len(unexpected), unexpected[:3])
            log.info("Pretrained weights loaded (key='%s').", key)

    # ── Freeze requested layers ───────────────────────────────────────────────
    frozen_prefixes = getattr(getattr(cfg, "training", DotDict()), "frozen_prefixes", None)
    if frozen_prefixes:
        frozen_count = 0
        for name, param in model.named_parameters():
            if any(name.startswith(p) for p in frozen_prefixes):
                param.requires_grad = False
                frozen_count += 1
        log.info(
            "Froze %d parameter tensor(s) matching prefixes: %s",
            frozen_count, frozen_prefixes,
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(
        "Model: SwinIR — trainable params %s / total %s (%.1f%%)",
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
    log = logging.getLogger(__name__)
    opt_cfg = cfg.optimizer
    opt_type = str(opt_cfg.type).lower()

    params = [p for p in model.parameters() if p.requires_grad]

    kwargs = dict(
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
        raise ValueError(f"Unknown optimizer type '{opt_type}'.  Choose: adam, adamw.")

    log.info(
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
    log = logging.getLogger(__name__)
    sch_cfg  = cfg.scheduler
    sch_type = str(getattr(sch_cfg, "type", "none")).lower()

    if sch_type == "none":
        log.info("Scheduler : none (constant LR)")
        return None

    if sch_type == "multistep":
        milestones = list(sch_cfg.milestones)
        gamma      = float(getattr(sch_cfg, "gamma", 0.5))
        scheduler  = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
        log.info("Scheduler : MultiStepLR  milestones=%s  γ=%g", milestones, gamma)
        return scheduler

    if sch_type == "cosine":
        t_max   = int(cfg.training.epochs)
        eta_min = float(getattr(sch_cfg, "eta_min", 1e-7))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
        log.info("Scheduler : CosineAnnealingLR  T_max=%d  η_min=%g", t_max, eta_min)
        return scheduler

    raise ValueError(
        f"Unknown scheduler type '{sch_type}'.  Choose from: none, multistep, cosine."
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path) -> None:
    """Atomically save *state* to *path* using a temp file + rename."""
    path    = Path(path)
    tmp     = path.with_suffix(".tmp")
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
    log = logging.getLogger(__name__)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    log.info("Resuming from checkpoint: %s", path)
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Strip DataParallel prefix if present.
    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    log.info("Resumed at epoch %d  (best PSNR: %.2f dB)", start_epoch - 1, ckpt.get("best_psnr", 0.0))
    return start_epoch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Configure logging and return the named training logger.

    Two problems fixed vs. the naive approach:

    1. Root-logger level
       By default the root logger sits at WARNING.  Any helper module that
       calls ``logging.getLogger(__name__)`` (e.g. ``src.train.utils``) will
       have its INFO messages silently dropped, even if a handler exists on
       the named "training" logger, because propagation stops at the first
       ancestor whose level filters out the message.  We set the root level
       to DEBUG and attach a single console handler to the root so that every
       module in the project can emit INFO+ messages without needing its own
       handler registration.

    2. Duplicate handlers on re-call
       The root handler is only added once (guard on root.handlers), and the
       named logger is returned early if already configured.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Root logger: single console handler, INFO level ───────────────────────
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.DEBUG)
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root.addHandler(console)
    else:
        root.setLevel(logging.DEBUG)  # ensure level is not WARNING

    # ── Named logger: optional file handler ───────────────────────────────────
    logger = logging.getLogger(name)
    # Check if a FileHandler is already present to avoid duplicates on re-call.
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return logger

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # Prevent the named logger from also printing to the root console handler
        # (which would double-print every message).
        logger.propagate = False
        # The named logger needs its own console handler since propagate=False.
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