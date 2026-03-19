"""
compute_norm.py — Compute dataset-level normalisation statistics.

Samples a random subset of HR tiles, computes per-band percentile statistics,
and prints the recommended dtype_max value to use in swinir_finetune.yaml.

Usage:
    python compute_norm.py \
        --hr_root data/processed/train/HR \
        --lr_root data/processed/train/LR \
        --n_samples 200 \
        --percentile 99.9
"""

import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm


def sample_stats(root: Path, n_samples: int, percentile: float):
    files = sorted(f for f in root.rglob("*")
                   if f.suffix in {".tif", ".TIF", ".tiff", ".TIFF"})
    if not files:
        sys.exit(f"No TIF files found under {root}")

    rng = random.Random(42)
    sample = rng.sample(files, min(n_samples, len(files)))

    all_vals = []
    for path in tqdm(sample, desc=f"Sampling {root.name}"):
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                data = data[data != nodata]
            else:
                data = data[data > 0]   # treat 0 as nodata
        if data.size:
            all_vals.append(data.ravel())

    flat = np.concatenate(all_vals)
    return {
        "min":    float(flat.min()),
        "mean":   float(flat.mean()),
        "median": float(np.median(flat)),
        "p99":    float(np.percentile(flat, 99.0)),
        "p99_9":  float(np.percentile(flat, 99.9)),
        "p99_99": float(np.percentile(flat, 99.99)),
        "max":    float(flat.max()),
        "n_px":   int(flat.size),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hr_root",    default="data/processed/train/HR")
    p.add_argument("--lr_root",    default="data/processed/train/LR")
    p.add_argument("--n_samples",  type=int,   default=200)
    p.add_argument("--percentile", type=float, default=99.9)
    args = p.parse_args()

    print("\n=== HR tiles ===")
    hr = sample_stats(Path(args.hr_root), args.n_samples, args.percentile)
    for k, v in hr.items():
        print(f"  {k:8s}: {v:.2f}" if k != "n_px" else f"  {k:8s}: {v:,}")

    print("\n=== LR tiles ===")
    lr = sample_stats(Path(args.lr_root), args.n_samples, args.percentile)
    for k, v in lr.items():
        print(f"  {k:8s}: {v:.2f}" if k != "n_px" else f"  {k:8s}: {v:,}")

    # Recommended: use the HR 99.9th percentile so ~0.1% of pixels saturate
    # at 1.0, keeping the effective range in [0, ~1] for both HR and LR.
    recommended = hr["p99_9"]
    print(f"\n{'='*50}")
    print(f"  Recommended dtype_max : {recommended:.0f}")
    print(f"  (HR 99.9th percentile)")
    print(f"  Current  dtype_max    : 65535")
    print(f"  Current input range   : [0, {hr['max']/65535:.4f}]")
    print(f"  New input range       : [0, {hr['max']/recommended:.4f}]")
    print(f"{'='*50}\n")
    print(f"  In configs/swinir_finetune.yaml, set:")
    print(f"    data:")
    print(f"      dtype_max: {recommended:.0f}")
    print(f"\n  In configs/inference.yaml, set the same value:")
    print(f"    io:")
    print(f"      dtype_max: {recommended:.0f}")
    print()


if __name__ == "__main__":
    main()