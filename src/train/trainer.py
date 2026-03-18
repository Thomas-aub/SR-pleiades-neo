"""
src/train/trainer.py — SwinIR fine-tuning loop.
================================================
``Trainer`` encapsulates a single training run: epoch loop, validation,
checkpoint management, and metric logging.  It is deliberately stateless
with respect to the filesystem — all I/O paths are derived from the run
directory injected at construction time.

Design decisions
----------------
* Mixed-precision (AMP) is handled via ``torch.cuda.amp.GradScaler`` and
  ``torch.autocast``.  The scaler is only activated when ``cfg.training.use_amp``
  is true *and* the device is CUDA.

* SwinIR has a window_size constraint: H and W of the input must be divisible
  by ``window_size``.  ``_pad_to_window`` handles this for validation so
  full tiles of any size can be evaluated without cropping.

* Best-model tracking compares validation PSNR; the corresponding checkpoint
  is saved as ``best.pth`` independently of the periodic save cadence.

* Periodic checkpoint rotation: only the last N checkpoints on disk are kept
  (configurable via ``logging.keep_last_n_checkpoints``).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train.metrics import MetricTracker, psnr, ssim
from src.train.utils import save_checkpoint


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Padding utility
# ---------------------------------------------------------------------------

def _pad_to_window(
    x:           torch.Tensor,
    window_size: int,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad *x* (N, C, H, W) so that H and W are multiples of *window_size*.

    Returns the padded tensor and the (pad_left, pad_right, pad_top, pad_bottom)
    tuple needed to reverse the padding after inference.
    """
    _, _, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    # F.pad order: (left, right, top, bottom)
    padding = (0, pad_w, 0, pad_h)
    x_padded = torch.nn.functional.pad(x, padding, mode="reflect")
    return x_padded, (0, pad_w, 0, pad_h)


def _unpad(
    x:       torch.Tensor,
    padding: tuple[int, int, int, int],
    scale:   int,
) -> torch.Tensor:
    """Remove the padding added by ``_pad_to_window`` from an SR output.

    *padding* is the (left, right, top, bottom) tuple returned by
    ``_pad_to_window`` applied to the LR input; all values are multiplied
    by *scale* to convert from LR to SR pixel coordinates.
    """
    _, _, h, w  = x.shape
    pad_left, pad_right, pad_top, pad_bottom = padding
    h_end = h - pad_bottom * scale
    w_end = w - pad_right  * scale
    return x[:, :, pad_top * scale : h_end, pad_left * scale : w_end]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Manages the full fine-tuning lifecycle for SwinIR.

    Parameters
    ----------
    cfg          : Full DotDict configuration.
    model        : SwinIR model (already on *device*).
    train_loader : DataLoader yielding {"lr", "hr", "name"} dicts.
    val_loader   : DataLoader for validation (batch_size=1 recommended).
    optimizer    : Torch optimizer.
    scheduler    : LR scheduler or None.
    criterion    : Loss module.
    device       : Compute device.
    run_dir      : Root directory for this run's artefacts.
    writer       : TensorBoard SummaryWriter or None.
    """

    def __init__(
        self,
        cfg:          Any,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        optimizer:    torch.optim.Optimizer,
        scheduler:    Optional[torch.optim.lr_scheduler.LRScheduler],
        criterion:    nn.Module,
        device:       torch.device,
        run_dir:      Path,
        writer:       Optional[Any] = None,
    ) -> None:
        self.cfg          = cfg
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.scheduler    = scheduler
        self.criterion    = criterion
        self.device       = device
        self.run_dir      = Path(run_dir)
        self.writer       = writer

        self.ckpt_dir = self.run_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # AMP scaler — only active on CUDA.
        self._use_amp = (
            getattr(cfg.training, "use_amp", False)
            and device.type == "cuda"
        )
        # torch.amp.GradScaler replaces the deprecated torch.cuda.amp.GradScaler.
        # We keep a CPU-safe fallback: GradScaler is a no-op when enabled=False,
        # but it still requires a valid device string when constructed with the
        # new API (PyTorch ≥ 2.3).
        _scaler_device = device.type if device.type == "cuda" else "cpu"
        self._scaler = torch.amp.GradScaler(_scaler_device, enabled=self._use_amp)

        # Hyperparameters from config.
        self._window_size      = int(cfg.model.window_size)
        self._scale            = int(cfg.model.upscale)
        self._grad_clip        = getattr(cfg.training, "grad_clip_norm", None)
        self._log_interval     = int(getattr(cfg.logging, "log_interval_iters",  100))
        self._val_interval     = int(getattr(cfg.logging, "val_interval_epochs",   5))
        self._save_interval    = int(getattr(cfg.logging, "save_interval_epochs", 10))
        self._keep_n           = int(getattr(cfg.logging, "keep_last_n_checkpoints", 3))

        # State maintained across epochs.
        self.start_epoch: int  = 0
        self.best_psnr:   float = 0.0
        self._periodic_ckpts: Deque[Path] = deque()

        log.info(
            "Trainer ready — AMP=%s  window_size=%d  scale=%d  "
            "val_every=%d  save_every=%d epochs",
            self._use_amp, self._window_size, self._scale,
            self._val_interval, self._save_interval,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self, start_epoch: int = 0) -> None:
        """Run training from *start_epoch* to cfg.training.epochs."""
        self.start_epoch    = start_epoch
        total_epochs = int(self.cfg.training.epochs)

        log.info(
            "Starting fine-tuning: epoch %d → %d  (%d epoch(s) remaining)",
            start_epoch, total_epochs, total_epochs - start_epoch,
        )

        for epoch in range(start_epoch, total_epochs):
            epoch_start = time.perf_counter()
            train_metrics = self._train_epoch(epoch, total_epochs)

            # LR scheduler step (MultiStep / Cosine).
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.perf_counter() - epoch_start
            log.info(
                "Epoch [%d/%d]  loss=%.6f  lr=%g  time=%.1fs",
                epoch + 1, total_epochs,
                train_metrics["loss"], current_lr, elapsed,
            )

            if self.writer:
                self.writer.add_scalar("train/loss", train_metrics["loss"], epoch)
                self.writer.add_scalar("train/lr",   current_lr,            epoch)

            # ── Validation ────────────────────────────────────────────────────
            is_best = False
            if (epoch + 1) % self._val_interval == 0 or epoch == total_epochs - 1:
                val_metrics = self._validate(epoch, total_epochs)
                is_best = val_metrics["psnr"] > self.best_psnr
                if is_best:
                    self.best_psnr = val_metrics["psnr"]

                log.info(
                    "  Val [%d/%d]  PSNR=%.4f dB  SSIM=%.4f%s",
                    epoch + 1, total_epochs,
                    val_metrics["psnr"], val_metrics["ssim"],
                    "  ← best" if is_best else "",
                )

                if self.writer:
                    self.writer.add_scalar("val/psnr", val_metrics["psnr"], epoch)
                    self.writer.add_scalar("val/ssim", val_metrics["ssim"], epoch)

                if is_best:
                    self._save(epoch, best_psnr=self.best_psnr, name="best.pth")

            # ── Periodic checkpoint ───────────────────────────────────────────
            if (epoch + 1) % self._save_interval == 0 or epoch == total_epochs - 1:
                self._save_periodic(epoch)

        log.info("Fine-tuning complete.  Best val PSNR: %.4f dB", self.best_psnr)
        if self.writer:
            self.writer.close()

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        epoch:        int,
        total_epochs: int,
    ) -> Dict[str, float]:
        """Run one full training epoch and return aggregated metrics."""
        self.model.train()
        total_loss      = 0.0
        n_iters         = 0

        for i, batch in enumerate(self.train_loader):
            lr = batch["lr"].to(self.device, non_blocking=True)
            hr = batch["hr"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=self._use_amp):
                sr   = self.model(lr)
                loss = self.criterion(sr, hr)

            self._scaler.scale(loss).backward()

            if self._grad_clip is not None:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=float(self._grad_clip)
                )

            self._scaler.step(self.optimizer)
            self._scaler.update()

            total_loss += loss.item()
            n_iters    += 1

            # ── Iteration-level logging ───────────────────────────────────────
            global_iter = epoch * len(self.train_loader) + i
            if (i + 1) % self._log_interval == 0:
                avg_loss = total_loss / n_iters
                lr_now   = self.optimizer.param_groups[0]["lr"]
                log.debug(
                    "  [%d/%d | iter %d]  loss=%.6f  lr=%g",
                    epoch + 1, total_epochs, global_iter + 1, avg_loss, lr_now,
                )
                if self.writer:
                    self.writer.add_scalar("train/iter_loss", avg_loss, global_iter)

        return {"loss": total_loss / max(n_iters, 1)}

    # ------------------------------------------------------------------
    # Validation epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _validate(
        self,
        epoch:        int,
        total_epochs: int,
    ) -> Dict[str, float]:
        """Run full-tile validation and return PSNR / SSIM.

        Each LR tile is padded to the nearest window_size multiple, passed
        through the model in float32, then unpadded before metric computation.

        Why AMP is disabled here
        ------------------------
        During training, patches are 64×64.  During validation, full tiles
        are used (e.g. 512×512 LR → 1024×1024 SR).  SwinIR's shifted-window
        attention computes large intermediate activation tensors; with fp16
        these overflow to Inf/NaN for inputs larger than ~128 px.  We always
        run validation in float32 regardless of ``use_amp``.
        """
        self.model.eval()
        tracker = MetricTracker()
        nan_skipped = 0

        for batch in self.val_loader:
            lr = batch["lr"].to(self.device, non_blocking=True)
            hr = batch["hr"].to(self.device, non_blocking=True)
            n  = lr.shape[0]

            # Pad LR to satisfy window_size constraint.
            lr_padded, padding = _pad_to_window(lr, self._window_size)

            # Always use float32 for validation — see docstring.
            with torch.autocast(device_type=self.device.type, enabled=False):
                sr_padded = self.model(lr_padded.float())

            sr = _unpad(sr_padded, padding, self._scale)
            sr = sr.float().clamp(0.0, 1.0)
            hr = hr.float()

            # Guard: skip any batch where the model still produced NaN
            # (e.g. corrupt tile on disk).
            if not torch.isfinite(sr).all():
                nan_skipped += n
                log.warning(
                    "Skipping %d val sample(s) at epoch %d — SR output contains NaN/Inf.",
                    n, epoch + 1,
                )
                continue

            tracker.update("psnr", psnr(sr, hr).item(), n)
            tracker.update("ssim", ssim(sr, hr).item(), n)

        if nan_skipped:
            log.warning(
                "Validation: %d / %d tile(s) skipped due to NaN/Inf in SR output.",
                nan_skipped, len(self.val_loader.dataset),
            )

        result = tracker.result()
        if not result:
            log.error(
                "Validation produced no finite metrics — all tiles were skipped. "
                "Check your data normalisation and model checkpoint."
            )
            return {"psnr": float("nan"), "ssim": float("nan")}
        return result

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _checkpoint_state(self, epoch: int, best_psnr: float) -> dict:
        """Build the serialisable checkpoint dict."""
        sch_state = self.scheduler.state_dict() if self.scheduler else None
        return {
            "epoch":     epoch,
            "best_psnr": best_psnr,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": sch_state,
            "scaler":    self._scaler.state_dict(),
        }

    def _save(self, epoch: int, best_psnr: float, name: str) -> Path:
        """Save a named checkpoint and return its path."""
        path  = self.ckpt_dir / name
        state = self._checkpoint_state(epoch, best_psnr)
        save_checkpoint(state, path)
        log.info("Checkpoint saved → %s", path)
        return path

    def _save_periodic(self, epoch: int) -> None:
        """Save a periodic checkpoint and rotate old ones off disk."""
        name = f"epoch_{epoch + 1:04d}.pth"
        path = self._save(epoch, self.best_psnr, name)

        self._periodic_ckpts.append(path)
        while len(self._periodic_ckpts) > self._keep_n:
            old = self._periodic_ckpts.popleft()
            if old.exists() and old.name != "best.pth":
                old.unlink()
                log.debug("Removed old checkpoint: %s", old)