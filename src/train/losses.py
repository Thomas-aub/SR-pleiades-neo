"""
src/train/losses.py — Reconstruction loss functions for SR fine-tuning.
========================================================================
Implements the NCC + SSIM combination loss proposed in:

    Rossi et al. (2024). "Swin2-MoSE: A New Single Image Super-Resolution
    Model for Remote Sensing." arXiv:2404.18924.

The authors demonstrate that this combination avoids the over-smoothing
artefacts of MSE-based losses, which disproportionately penalise
high-frequency residuals and produce blurry super-resolved images.

Supported primary losses
-------------------------
  ncc_ssim     — α·NCCLoss + β·SSIMLoss (Rossi et al. 2024, recommended).
  ncc          — Normalized Cross-Correlation: 1 − NCC(sr, hr).
  ssim         — Structural Similarity:        1 − SSIM(sr, hr).
  charbonnier  — sqrt((x−y)² + ε²).  Smooth L1; KAIR default for SwinIR.
  l1           — Mean Absolute Error.
  mse          — Mean Squared Error (L2).

Optional auxiliary loss
------------------------
  perceptual   — L1 in VGG-19 feature space (requires torchvision).
                 Composed on top of any primary loss.

Factory
-------
  build_criterion(cfg)  →  nn.Module  (sr, hr) → scalar tensor

NaN safety
----------
All modules handle nodata regions: pixels set to NaN by the dataset loader
are masked before arithmetic so they never contribute NaN gradients.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NCC Loss
# ---------------------------------------------------------------------------

class NCCLoss(nn.Module):
    """Normalized Cross-Correlation loss: ``1 − NCC(sr, hr)``.

    Computes the Pearson correlation between *sr* and *hr* over all spatial
    dimensions and channels of each image, then averages ``1 − NCC`` over
    the batch.  Identical images yield loss = 0; fully anti-correlated images
    yield loss ≈ 2.

    NCC is invariant to affine intensity transformations (brightness / contrast
    shifts), making it less sensitive to the radiometric differences between
    natural-image pre-training data and satellite imagery.

    Parameters
    ----------
    eps : float
        Added to the denominator for numerical stability when either sr or hr
        has near-zero variance (e.g. flat background tiles).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        if sr.shape != hr.shape:
            raise ValueError(
                f"sr and hr must have the same shape, got {sr.shape} vs {hr.shape}."
            )

        # fp16 overflows on squared residuals for typical satellite tile variance.
        # Cast to float32 for the entire computation regardless of AMP context.
        sr = sr.float()
        hr = hr.float()

        n = sr.shape[0]
        sr_flat = sr.reshape(n, -1)   # (N, C*H*W)
        hr_flat = hr.reshape(n, -1)

        # Mask nodata pixels (set to NaN by the dataset loader).
        valid   = torch.isfinite(sr_flat) & torch.isfinite(hr_flat)
        sr_flat = torch.where(valid, sr_flat, torch.zeros_like(sr_flat))
        hr_flat = torch.where(valid, hr_flat, torch.zeros_like(hr_flat))

        n_valid = valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mu_sr   = (sr_flat * valid.float()).sum(dim=1, keepdim=True) / n_valid
        mu_hr   = (hr_flat * valid.float()).sum(dim=1, keepdim=True) / n_valid

        # Zero-mean residuals; invalid pixels already zeroed via torch.where.
        sr_c = (sr_flat - mu_sr) * valid.float()
        hr_c = (hr_flat - mu_hr) * valid.float()

        cov    = (sr_c * hr_c).sum(dim=1)   # (N,)
        var_sr = (sr_c * sr_c).sum(dim=1)   # (N,)
        var_hr = (hr_c * hr_c).sum(dim=1)   # (N,)

        ncc = cov / (torch.sqrt(var_sr * var_hr) + self.eps)   # (N,)  ∈ [−1, 1]
        # Clamp to valid range: fp32 rounding can push values slightly outside
        # [-1, 1] on flat or near-constant tiles, making 1-NCC negative.
        ncc = ncc.clamp(-1.0, 1.0)
        return (1.0 - ncc).mean()

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


# ---------------------------------------------------------------------------
# SSIM Loss
# ---------------------------------------------------------------------------

class SSIMLoss(nn.Module):
    """Structural Similarity loss: ``1 − SSIM(sr, hr)``.

    Thin differentiable wrapper around the pure-PyTorch SSIM implementation
    in ``src.train.metrics``, ensuring the training loss and the validation
    metric use identical arithmetic.

    Parameters
    ----------
    data_range  : Pixel value range (1.0 for [0, 1]-normalised inputs).
    kernel_size : Gaussian window size (default 11, per Wang et al. 2004).
    sigma       : Gaussian σ (default 1.5).
    """

    def __init__(
        self,
        data_range:  float = 1.0,
        kernel_size: int   = 11,
        sigma:       float = 1.5,
    ) -> None:
        super().__init__()
        self.data_range  = float(data_range)
        self.kernel_size = kernel_size
        self.sigma       = float(sigma)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # Deferred import avoids a package-level circular dependency since
        # metrics.py does not import from losses.py.
        from src.train.metrics import ssim as _ssim  # noqa: PLC0415

        # Cast to float32: the Gaussian-windowed convolutions in SSIM accumulate
        # rounding error in fp16 and can produce values outside [0, 1].
        return 1.0 - _ssim(
            sr.float(),
            hr.float(),
            data_range  = self.data_range,
            kernel_size = self.kernel_size,
            sigma       = self.sigma,
        )

    def extra_repr(self) -> str:
        return (
            f"data_range={self.data_range}, "
            f"kernel_size={self.kernel_size}, "
            f"sigma={self.sigma}"
        )


# ---------------------------------------------------------------------------
# NCC + SSIM Combined Loss  (Rossi et al. 2024)
# ---------------------------------------------------------------------------

class NCCSSIMLoss(nn.Module):
    """NCC + SSIM combination loss from Rossi et al. (2024).

    L = ncc_weight · NCCLoss(sr, hr) + ssim_weight · SSIMLoss(sr, hr)

    Both terms are dimensionless and bounded in [0, 2], so their scale is
    comparable without re-normalisation.  The default equal weighting (0.5 / 0.5)
    matches the configuration used in the Swin2-MoSE paper.

    Rationale
    ----------
    * NCCLoss captures global luminance correlation, penalising structural
      shifts while remaining robust to per-image intensity offsets.
    * SSIMLoss captures local contrast and structure via a Gaussian-windowed
      comparison, complementing the global NCC signal.
    * Neither term reduces to a per-pixel L2 average, so high-frequency
      textures (edges, rooftops, vegetation boundaries) are not systematically
      penalised the way they are under MSE / Charbonnier losses.

    Parameters
    ----------
    ncc_weight  : Weight for the NCCLoss term.
    ssim_weight : Weight for the SSIMLoss term.
    ncc_eps     : Numerical stability term forwarded to NCCLoss.
    data_range  : Pixel value range forwarded to SSIMLoss.
    """

    def __init__(
        self,
        ncc_weight:  float = 0.5,
        ssim_weight: float = 0.5,
        ncc_eps:     float = 1e-8,
        data_range:  float = 1.0,
    ) -> None:
        super().__init__()
        self.ncc_weight  = float(ncc_weight)
        self.ssim_weight = float(ssim_weight)
        self.ncc_loss    = NCCLoss(eps=ncc_eps)
        self.ssim_loss   = SSIMLoss(data_range=data_range)

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        return (
            self.ncc_weight  * self.ncc_loss(sr, hr)
            + self.ssim_weight * self.ssim_loss(sr, hr)
        )

    def extra_repr(self) -> str:
        return f"ncc_weight={self.ncc_weight}, ssim_weight={self.ssim_weight}"


# ---------------------------------------------------------------------------
# Charbonnier Loss
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Charbonnier (pseudo-Huber / smooth L1) loss.

    L(x, y) = mean( sqrt( (x − y)² + ε² ) )

    NaN-safe: pixels where either sr or hr is non-finite (nodata regions
    normalised to NaN in the dataset) are excluded from the loss.  The
    denominator is the count of *valid* pixels so the effective scale is
    unchanged when all pixels are valid.

    Parameters
    ----------
    eps : float
        Smoothing constant.  Default 1e-6 follows KAIR SwinIR training scripts.
    reduction : "mean" | "sum" | "none"
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        self.eps       = float(eps)
        self.reduction = reduction

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        valid = torch.isfinite(sr) & torch.isfinite(hr)

        sr = torch.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
        hr = torch.nan_to_num(hr, nan=0.0, posinf=0.0, neginf=0.0)

        diff = sr - hr
        loss = torch.sqrt(diff * diff + self.eps ** 2)
        loss = loss * valid.float()

        if self.reduction == "mean":
            n_valid = valid.float().sum().clamp_min(1.0)
            return loss.sum() / n_valid
        if self.reduction == "sum":
            return loss.sum()
        return loss   # "none"

    def extra_repr(self) -> str:
        return f"eps={self.eps}, reduction={self.reduction}"


# ---------------------------------------------------------------------------
# Perceptual (VGG) Loss
# ---------------------------------------------------------------------------

class _VGGFeatureExtractor(nn.Module):
    """Fixed VGG-19 feature extractor for perceptual loss.

    Parameters
    ----------
    layer : str
        VGG-19 layer name — one of:
        "relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_4".
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

        depth         = self._LAYER_MAP[layer]
        vgg           = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:depth])

        for p in self.features.parameters():
            p.requires_grad = False

        # ImageNet statistics — VGG was trained on these.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features((x - self.mean) / self.std)


class PerceptualLoss(nn.Module):
    """Perceptual loss: primary loss + weighted L1 in VGG feature space.

    Parameters
    ----------
    layer        : VGG-19 layer used to extract features.
    weight       : Relative weight for the perceptual term.
    primary_loss : Primary loss module (pixel-space or structure-space).
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
        primary = self.primary_loss(sr, hr)

        # VGG expects 3-channel [0, 1] input.
        sr_3 = sr if sr.shape[1] == 3 else sr.repeat(1, 3, 1, 1)
        hr_3 = hr if hr.shape[1] == 3 else hr.repeat(1, 3, 1, 1)

        with torch.no_grad():
            feat_hr = self.extractor(hr_3)
        feat_sr = self.extractor(sr_3)

        return primary + self.weight * F.l1_loss(feat_sr, feat_hr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_criterion(cfg) -> nn.Module:
    """Instantiate and return the training loss from configuration.

    Parameters
    ----------
    cfg : DotDict (or any object with attribute access) containing at minimum
          ``loss.type`` and optional sub-keys per loss type.

    Returns
    -------
    nn.Module whose ``forward(sr, hr)`` returns a scalar loss tensor.

    Configuration reference
    -----------------------
    loss:
      type:        "ncc_ssim"   # "ncc_ssim"|"ncc"|"ssim"|"charbonnier"|"l1"|"mse"
      ncc_weight:  0.5          # [ncc_ssim] weight for NCCLoss term
      ssim_weight: 0.5          # [ncc_ssim] weight for SSIMLoss term
      ncc_eps:     1.0e-8       # [ncc_ssim|ncc] denominator stability
      eps:         1.0e-6       # [charbonnier] smoothing constant

      perceptual:
        enabled: false
        weight:  0.1
        layer:   "relu3_4"
    """
    loss_type = cfg.loss.type.lower()

    if loss_type == "ncc_ssim":
        ncc_w   = float(getattr(cfg.loss, "ncc_weight",  0.5))
        ssim_w  = float(getattr(cfg.loss, "ssim_weight", 0.5))
        ncc_eps = float(getattr(cfg.loss, "ncc_eps",     1e-8))
        primary = NCCSSIMLoss(
            ncc_weight  = ncc_w,
            ssim_weight = ssim_w,
            ncc_eps     = ncc_eps,
        )
        log.info(
            "Primary loss: NCCSSIMLoss(ncc_weight=%g, ssim_weight=%g, ncc_eps=%g)",
            ncc_w, ssim_w, ncc_eps,
        )

    elif loss_type == "ncc":
        ncc_eps = float(getattr(cfg.loss, "ncc_eps", 1e-8))
        primary = NCCLoss(eps=ncc_eps)
        log.info("Primary loss: NCCLoss(eps=%g)", ncc_eps)

    elif loss_type == "ssim":
        primary = SSIMLoss()
        log.info("Primary loss: SSIMLoss")

    elif loss_type == "charbonnier":
        eps     = float(getattr(cfg.loss, "eps", 1e-6))
        primary = CharbonnierLoss(eps=eps)
        log.info("Primary loss: CharbonnierLoss(eps=%g)", eps)

    elif loss_type == "l1":
        primary = nn.L1Loss()
        log.info("Primary loss: L1Loss")

    elif loss_type == "mse":
        primary = nn.MSELoss()
        log.info("Primary loss: MSELoss")

    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'.  "
            f"Choose from: ncc_ssim, ncc, ssim, charbonnier, l1, mse."
        )

    percep_cfg = getattr(cfg.loss, "perceptual", None)
    if percep_cfg is not None and getattr(percep_cfg, "enabled", False):
        return PerceptualLoss(
            layer        = str(getattr(percep_cfg, "layer",  "relu3_4")),
            weight       = float(getattr(percep_cfg, "weight", 0.1)),
            primary_loss = primary,
        )

    return primary