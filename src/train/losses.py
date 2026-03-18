"""
src/train/losses.py — Reconstruction loss functions for SR fine-tuning.
========================================================================
Supported primary losses
-------------------------
  charbonnier  — sqrt((x-y)² + ε²)  — smooth, differentiable L1 proxy.
                 Standard for KAIR SR models; robust to outliers while
                 remaining differentiable at zero.
  l1           — mean absolute error.
  mse          — mean squared error (L2).

Optional auxiliary losses
--------------------------
  perceptual   — VGG feature-space loss (requires torchvision).
                 Weighted combination with the primary pixel-space loss.

Factory
-------
  build_criterion(cfg)  →  loss callable  (sr_pred, hr_target) → scalar
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primary losses
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier (pseudo-Huber / smooth L1) loss.

    L(x, y) = mean( sqrt( (x - y)² + ε² ) )

    NaN-safe: pixels where either sr or hr is non-finite (nodata regions
    normalised to NaN in the dataset) are excluded from the loss.  The
    denominator is the count of *valid* pixels so the scale is unchanged
    when all pixels are valid.

    Parameters
    ----------
    eps : float
        Smoothing constant.  Smaller values → closer to L1; larger → closer
        to L2 near zero.  Default 1e-6 follows KAIR SwinIR training scripts.
    reduction : str
        "mean" | "sum" | "none"
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps       = float(eps)
        self.reduction = reduction

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Build a mask of pixels that are finite in *both* sr and hr.
        # Nodata pixels were set to NaN in the dataset; we exclude them so
        # they never contribute NaN to the loss.
        valid = torch.isfinite(sr) & torch.isfinite(hr)

        # Replace NaN/Inf with 0 before arithmetic — masked out below anyway.
        sr = torch.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
        hr = torch.nan_to_num(hr, nan=0.0, posinf=0.0, neginf=0.0)

        diff = sr - hr
        loss = torch.sqrt(diff * diff + self.eps ** 2)

        # Zero out invalid pixels so they don't affect the sum.
        loss = loss * valid.float()

        if self.reduction == "mean":
            # Divide by valid-pixel count, not total count.
            n_valid = valid.float().sum().clamp_min(1.0)
            return loss.sum() / n_valid
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction={self.reduction}"


# ---------------------------------------------------------------------------
# Optional: Perceptual (VGG) loss
# ---------------------------------------------------------------------------

class _VGGFeatureExtractor(nn.Module):
    """Extract intermediate VGG-19 features for perceptual loss.

    Only instantiated when perceptual loss is enabled.  Requires torchvision.

    Parameters
    ----------
    layer : str
        VGG-19 layer name.  Supported values:
          "relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_4"
    """

    _LAYER_MAP = {
        "relu1_2": 4,
        "relu2_2": 9,
        "relu3_4": 18,
        "relu4_4": 27,
        "relu5_4": 36,
    }

    def __init__(self, layer: str = "relu3_4") -> None:
        super().__init__()
        try:
            from torchvision import models
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for perceptual loss.  pip install torchvision"
            ) from exc

        if layer not in self._LAYER_MAP:
            raise ValueError(
                f"Unsupported VGG layer '{layer}'.  "
                f"Choose from {list(self._LAYER_MAP)}."
            )

        depth  = self._LAYER_MAP[layer]
        vgg    = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:depth])

        # Freeze VGG — we only use it as a fixed feature extractor.
        for p in self.features.parameters():
            p.requires_grad = False

        # ImageNet normalisation constants (VGG was trained on ImageNet).
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is in [0, 1]; normalise to ImageNet statistics before VGG.
        x = (x - self.mean) / self.std
        return self.features(x)


class PerceptualLoss(nn.Module):
    """Perceptual loss: L1 distance in VGG feature space.

    Parameters
    ----------
    layer  : VGG layer used to extract features.
    weight : Relative weight applied to the perceptual term.
    primary_loss : The primary pixel-space loss module (added to perceptual).
    """

    def __init__(
        self,
        layer:        str,
        weight:       float,
        primary_loss: nn.Module,
    ) -> None:
        super().__init__()
        self.extractor    = _VGGFeatureExtractor(layer)
        self.weight       = float(weight)
        self.primary_loss = primary_loss
        log.info("PerceptualLoss: VGG layer=%s  weight=%.4f", layer, weight)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        pixel_loss = self.primary_loss(sr, hr)

        # VGG expects 3-channel input in [0, 1].
        # If tensors have 1 channel (rare) replicate to 3.
        sr_3 = sr if sr.shape[1] == 3 else sr.repeat(1, 3, 1, 1)
        hr_3 = hr if hr.shape[1] == 3 else hr.repeat(1, 3, 1, 1)

        with torch.no_grad():
            feat_hr = self.extractor(hr_3)
        feat_sr = self.extractor(sr_3)

        percep_loss = F.l1_loss(feat_sr, feat_hr)
        return pixel_loss + self.weight * percep_loss


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_criterion(cfg) -> nn.Module:
    """Instantiate and return the training loss from configuration.

    Parameters
    ----------
    cfg : DotDict (or any object supporting attribute access) containing
          at minimum ``loss.type`` and optional ``loss.perceptual.*``.

    Returns
    -------
    nn.Module whose forward(sr, hr) returns a scalar loss tensor.
    """
    loss_type = cfg.loss.type.lower()

    if loss_type == "charbonnier":
        eps      = float(getattr(cfg.loss, "eps", 1e-6))
        primary  = CharbonnierLoss(eps=eps)
        log.info("Primary loss  : CharbonnierLoss(eps=%g)", eps)

    elif loss_type == "l1":
        primary  = nn.L1Loss()
        log.info("Primary loss  : L1Loss")

    elif loss_type == "mse":
        primary  = nn.MSELoss()
        log.info("Primary loss  : MSELoss")

    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'.  "
            f"Choose from: charbonnier, l1, mse."
        )

    # ── Optional perceptual loss ──────────────────────────────────────────────
    percep_cfg = getattr(cfg.loss, "perceptual", None)
    if percep_cfg is not None and getattr(percep_cfg, "enabled", False):
        criterion = PerceptualLoss(
            layer        = getattr(percep_cfg, "layer",  "relu3_4"),
            weight       = float(getattr(percep_cfg, "weight", 0.1)),
            primary_loss = primary,
        )
    else:
        criterion = primary

    return criterion
