"""
src/train/metrics.py — Image quality metrics for SR evaluation.
================================================================
Pure-PyTorch implementations of PSNR and SSIM that operate on batched
float32 tensors in [0, 1].  No external metric library is required.

Public API
----------
  psnr(sr, hr, data_range=1.0)           → scalar tensor (dB)
  ssim(sr, hr, data_range=1.0, ...)      → scalar tensor [0, 1]
  MetricTracker                          → running average accumulator
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def psnr(
    sr:         torch.Tensor,
    hr:         torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Peak Signal-to-Noise Ratio averaged over a batch.

    Parameters
    ----------
    sr : Predicted super-resolved batch  (N, C, H, W), float32, [0, data_range].
    hr : Ground-truth HR batch           (N, C, H, W), float32, [0, data_range].
    data_range : Maximum possible pixel value after normalisation.

    Returns
    -------
    Scalar tensor — mean PSNR in dB over the batch.
    """
    if sr.shape != hr.shape:
        raise ValueError(
            f"sr and hr must have the same shape, got {sr.shape} vs {hr.shape}."
        )

    # Mask pixels that are finite in both sr and hr (excludes nodata NaN).
    valid = torch.isfinite(sr) & torch.isfinite(hr)

    # Per-image MSE over valid pixels only.
    diff2 = (sr - hr).pow(2)
    diff2 = torch.where(valid, diff2, torch.zeros_like(diff2))

    n_valid = valid.float().sum(dim=[1, 2, 3]).clamp_min(1.0)
    mse_per_image = diff2.sum(dim=[1, 2, 3]) / n_valid   # (N,)

    # Clamp MSE from below to avoid log(0).
    mse_per_image = mse_per_image.clamp_min(1e-10)
    psnr_per_image = 10.0 * torch.log10(data_range ** 2 / mse_per_image)

    return psnr_per_image.mean()


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(kernel_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """1-D Gaussian kernel as a (kernel_size,) tensor."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
    coords -= kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(
    kernel_size: int,
    sigma:       float,
    n_channels:  int,
    device:      torch.device,
) -> torch.Tensor:
    """2-D separable Gaussian kernel shaped (n_channels, 1, K, K)."""
    k1d = _gaussian_kernel_1d(kernel_size, sigma, device)
    k2d = k1d.outer(k1d)
    k2d = k2d.unsqueeze(0).unsqueeze(0)        # (1, 1, K, K)
    return k2d.expand(n_channels, 1, kernel_size, kernel_size)


def ssim(
    sr:          torch.Tensor,
    hr:          torch.Tensor,
    data_range:  float = 1.0,
    kernel_size: int   = 11,
    sigma:       float = 1.5,
) -> torch.Tensor:
    """Structural Similarity Index (SSIM) averaged over a batch.

    Implements Wang et al. (2004), identical to the standard definition used
    in most SR benchmarks (MATLAB reference implementation).

    Parameters
    ----------
    sr, hr      : (N, C, H, W) float32 tensors.
    data_range  : Pixel value range (1.0 for [0,1]-normalised inputs).
    kernel_size : Gaussian window size (default 11).
    sigma       : Gaussian σ (default 1.5).

    Returns
    -------
    Scalar tensor — mean SSIM over all images and channels in the batch.
    """
    if sr.shape != hr.shape:
        raise ValueError(
            f"sr and hr must have the same shape, got {sr.shape} vs {hr.shape}."
        )

    n, c, h, w = sr.shape

    if h < kernel_size or w < kernel_size:
        # Image too small for the default kernel; fall back to pixel MSE proxy.
        return torch.tensor(1.0, device=sr.device) - F.mse_loss(sr, hr)

    kernel = _gaussian_kernel_2d(kernel_size, sigma, c, sr.device)
    pad    = kernel_size // 2

    # Depthwise convolution: each channel filtered with its own kernel slice.
    def _conv(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, kernel, padding=pad, groups=c)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_sr  = _conv(sr)
    mu_hr  = _conv(hr)
    mu_sr2 = mu_sr * mu_sr
    mu_hr2 = mu_hr * mu_hr
    mu_srhr = mu_sr * mu_hr

    sigma_sr2  = _conv(sr * sr) - mu_sr2
    sigma_hr2  = _conv(hr * hr) - mu_hr2
    sigma_srhr = _conv(sr * hr) - mu_srhr

    numerator   = (2 * mu_srhr + C1) * (2 * sigma_srhr + C2)
    denominator = (mu_sr2 + mu_hr2 + C1) * (sigma_sr2 + sigma_hr2 + C2)

    ssim_map = numerator / denominator.clamp_min(1e-12)
    return ssim_map.mean()


# ---------------------------------------------------------------------------
# Running average accumulator
# ---------------------------------------------------------------------------

class MetricTracker:
    """Accumulate scalar metrics and compute their running means.

    Usage
    -----
    ::
        tracker = MetricTracker()
        for batch in val_loader:
            sr, hr = ...
            tracker.update("psnr", psnr(sr, hr).item(), n=batch_size)
            tracker.update("ssim", ssim(sr, hr).item(), n=batch_size)
        summary = tracker.result()   # {"psnr": 35.2, "ssim": 0.92}
        tracker.reset()
    """

    def __init__(self) -> None:
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def reset(self) -> None:
        """Clear all accumulated values."""
        self._sums.clear()
        self._counts.clear()

    def update(self, name: str, value: float, n: int = 1) -> None:
        """Add *n* samples with mean value *value* to metric *name*."""
        self._sums[name]   = self._sums.get(name, 0.0)   + value * n
        self._counts[name] = self._counts.get(name, 0)   + n

    def result(self) -> Dict[str, float]:
        """Return the weighted mean for each tracked metric."""
        return {
            k: self._sums[k] / self._counts[k]
            for k in self._sums
            if self._counts[k] > 0
        }

    def __repr__(self) -> str:
        r = self.result()
        parts = [f"{k}={v:.4f}" for k, v in r.items()]
        return f"MetricTracker({', '.join(parts)})"