# SR-Pléiades NEO — Satellite Super-Resolution Pipeline

End-to-end pipeline for training and running a **SwinIR super-resolution model** on **Pléiades NEO** satellite imagery. Raw PAN + multispectral acquisitions are pansharpened, degraded to synthetic LR pairs, tiled into a dataset, and used to fine-tune a pretrained SwinIR-M ×2 model.

```
Raw PAN + MS  →  Pansharpening  →  HR GeoTIFF
                                       │
                              Sensor degradation
                                       │
                                   LR GeoTIFF
                                       │
                                Tiling + split
                                       │
                              train / val dataset
                                       │
                              SwinIR fine-tuning
                                       │
                                 best.pth checkpoint
                                       │
                               Inference on LR tile
                                       │
                                SR GeoTIFF output
```

---

## Table of Contents

- [SR-Pléiades NEO — Satellite Super-Resolution Pipeline](#sr-pléiades-neo--satellite-super-resolution-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Project structure](#project-structure)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Data layout](#data-layout)
  - [Stage 1 — Preprocessing](#stage-1--preprocessing)
    - [Pansharpening](#pansharpening)
    - [Sensor degradation](#sensor-degradation)
    - [Dataset tiling](#dataset-tiling)
    - [Running the full pipeline](#running-the-full-pipeline)
  - [Stage 2 — Training](#stage-2--training)
    - [Training configuration](#training-configuration)
    - [Running training](#running-training)
    - [Resuming a run](#resuming-a-run)
    - [Monitoring](#monitoring)
  - [Stage 3 — Inference](#stage-3--inference)
    - [Inference configuration](#inference-configuration)
    - [Running inference](#running-inference)
  - [Configuration reference](#configuration-reference)
    - [`configs/swinir_finetune.yaml`](#configsswinir_finetuneyaml)
    - [`configs/inference.yaml`](#configsinferenceyaml)
  - [Troubleshooting](#troubleshooting)
    - [`loss=nan` during training](#lossnan-during-training)
    - [`PSNR=nan` during validation](#psnrnan-during-validation)
    - [Black SR output image](#black-sr-output-image)
    - [GPU out-of-memory during training](#gpu-out-of-memory-during-training)
    - [GPU out-of-memory during inference](#gpu-out-of-memory-during-inference)
    - [KAIR import warning: `torch.meshgrid`](#kair-import-warning-torchmeshgrid)
    - [`rasterio.errors.RasterBlockError: BLOCKXSIZE must be a multiple of 16`](#rasterioerrorsrasterblockerror-blockxsize-must-be-a-multiple-of-16)
  - [Acknowledgements](#acknowledgements)

---

## Project structure

```
SR-pleiades-neo/
│
├── configs/
│   ├── swinir_finetune.yaml      # Training hyperparameters
│   ├── preprocessing.yaml        # Full preprocessing pipeline config
│   └── inference.yaml            # Inference settings
│
├── scripts/
│   ├── training.py               # Training entry point
│   ├── inference.py              # Inference entry point
│   └── preprocessing.py         # Full preprocessing entry point (optional)
│
├── src/
│   ├── train/
│   │   ├── dataset.py            # Paired GeoTIFF tile dataset
│   │   ├── losses.py             # Charbonnier / L1 / MSE + optional VGG
│   │   ├── metrics.py            # PSNR, SSIM, MetricTracker
│   │   ├── trainer.py            # Training + validation loop
│   │   └── utils.py              # Model builder, optimizer, scheduler, checkpoints
│   └── inference/
│       └── predict.py            # Tiled SR inference engine
│
├── pansharpening.py              # Stage 1a — Brovey / simple-mean pansharpening
├── degrade_pipeline.py           # Stage 1b — Sensor degradation (MTF, downsample, noise)
├── build_dataset.py              # Stage 1c — Tiling + train/val split
│
├── data/
│   ├── raw/                      # Input PAN + MS-FS GeoTIFFs (read-only)
│   ├── pansharpened/
│   │   ├── HR/                   # Pansharpened full-res images
│   │   └── LR/                   # Degraded (synthetic LR) images
│   └── processed/
│       ├── train/
│       │   ├── HR/               # HR tiles for training
│       │   └── LR/               # LR tiles for training
│       └── val/
│           ├── HR/               # HR tiles for validation
│           └── LR/               # LR tiles for validation
│
├── pretrained/                   # Auto-downloaded SwinIR pretrained weights
├── runs/                         # Training run artefacts (checkpoints, logs, TensorBoard)
├── output/                       # Inference SR outputs
└── KAIR/                         # Git clone of https://github.com/cszn/KAIR
```

---

## Requirements

| Dependency | Version tested | Notes |
|---|---|---|
| Python | ≥ 3.12 | |
| PyTorch | ≥ 2.3 | CUDA 12.x recommended |
| rasterio | ≥ 1.3 | Requires GDAL |
| numpy | ≥ 1.26 | |
| tqdm | any | |
| pyyaml | any | |
| timm | any | Pulled in by KAIR |
| tensorboard | any | Optional — for training curves |
| torchvision | any | Optional — for perceptual loss |
| cupy-cuda12x | any | Optional — GPU acceleration for preprocessing |

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/your-org/SR-pleiades-neo.git
cd SR-pleiades-neo

# 2. Clone KAIR (SwinIR model definition)
git clone https://github.com/cszn/KAIR

# 3. Create and activate a conda environment
conda create -n sr_env python=3.12
conda activate sr_env

# 4. Install PyTorch (adjust cuda version to match your driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 5. Install remaining dependencies
pip install rasterio numpy tqdm pyyaml timm tensorboard

# 6. (Optional) GPU-accelerated preprocessing
pip install cupy-cuda12x
```

> **Pretrained weights** are downloaded automatically on the first training run.  
> They are saved to `pretrained/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth`.

---

## Data layout

Place your raw Pléiades NEO acquisitions under `data/raw/`. Each acquisition lives in its own `WO_*` subdirectory containing the multispectral (`_RGB_`) and panchromatic (`_P_`) GeoTIFFs:

```
data/raw/
└── WO_000373512_10_2_.../
    ├── IMG_..._RGB_R1C1.TIF      # Multispectral (MS-FS), ~0.3 m × 4 bands
    └── IMG_..._P_R1C1.TIF        # Panchromatic (PAN), ~0.3 m × 1 band
```

The preprocessing pipeline matches RGB and PAN tiles automatically using shared filename tokens (`STD_*`, `ORT_*`, `R?C?`).

---

## Stage 1 — Preprocessing

The three preprocessing stages can be run individually or all at once via `preprocessing.py`.

### Pansharpening

Fuses each MS + PAN pair into a high-resolution RGB image using the Brovey method (or simple mean — configurable). Output lives in `data/pansharpened/HR/`.

```bash
python pansharpening.py
```

Key parameters in `configs/preprocessing.yaml`:

```yaml
pansharpening:
  method:        "brovey"     # "brovey" | "simple_mean"
  resample_algo: "bilinear"   # upsampling used when resampling MS to PAN resolution
  output_dtype:  "uint16"
  use_gpu:       true
  chunk_rows:    2048          # reduce if GPU OOM
```

### Sensor degradation

Degrades the HR pansharpened images to create synthetic LR training targets. The pipeline applies MTF blur → spatial downsampling → inter-band misregistration → Gaussian noise. Output lives in `data/pansharpened/LR/`.

```bash
python degrade_pipeline.py
```

Key parameters:

```yaml
degradation:
  pipeline:
    - op: mtf_blur
      mtf_nyquist_x: 0.15    # lower = stronger blur
      mtf_nyquist_y: 0.15
    - op: downsample
      scale: 2               # integer scale factor — must match model.upscale
      resampling: "average"
    - op: spectral_misalign
      global_shift_px: [0.3, 0.2]
      per_band_sigma_px: 0.15
      seed: 42
    - op: add_noise
      sigma: 2.0
      seed:  42
```

### Dataset tiling

Tiles the HR and LR full-resolution images into fixed-size patches and splits them into train / val sets. The split is performed at the **acquisition group level** (`WO_*` directory) — all tiles from the same acquisition always land in the same split.

```bash
python build_dataset.py
```

Key parameters:

```yaml
tiling:
  tile_size:          512    # HR tile side (LR tile = 512 // scale = 256)
  val_ratio:          0.2    # fraction of acquisition groups → val
  seed:               42
  min_valid_fraction: 0.1    # discard tiles with < 10 % non-zero pixels
```

### Running the full pipeline

All three stages can be run in sequence with a single command:

```bash
python scripts/preprocessing.py --config configs/preprocessing.yaml

# Run only specific stages:
python scripts/preprocessing.py --config configs/preprocessing.yaml \
    --stages degradation tiling

# Validate config without processing any files:
python scripts/preprocessing.py --config configs/preprocessing.yaml --dry-run
```

---

## Stage 2 — Training

### Training configuration

All hyperparameters live in `configs/swinir_finetune.yaml`. The most important sections:

```yaml
model:
  pretrained_path: "pretrained/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
  pretrained_key:  "params"   # key in the KAIR checkpoint dict
  upscale:         2          # must match degradation.pipeline downsample.scale
  in_chans:        3          # RGB = 3
  img_size:        64         # must equal data.lr_patch_size
  window_size:     8

data:
  lr_patch_size: 64           # must be a multiple of model.window_size
  dtype_max:     65535        # 65535 for uint16 data, 255 for uint8

training:
  epochs:     100
  batch_size: 32
  use_amp:    true            # mixed-precision training (fp16 forward, fp32 grads)

optimizer:
  type: "adam"
  lr:   2.0e-4

scheduler:
  type:       "multistep"
  milestones: [40, 60, 80, 90]
  gamma:      0.5
```

**Important constraint:** `data.lr_patch_size` must be divisible by `model.window_size` (default 8). With `window_size=8`, valid patch sizes are 32, 64, 96, 128, …

### Running training

```bash
python scripts/training.py --config configs/swinir_finetune.yaml
```

Pretrained weights are downloaded automatically on the first run (~56 MB from GitHub Releases).

Override any config key at the command line without editing the YAML:

```bash
# Smaller batch for limited VRAM:
python scripts/training.py --config configs/swinir_finetune.yaml \
    --set training.batch_size=8 optimizer.lr=5e-5

# Freeze early transformer layers and fine-tune only the SR head:
python scripts/training.py --config configs/swinir_finetune.yaml \
    --set "training.frozen_prefixes=[patch_embed,layers.0,layers.1,layers.2]"
```

Checkpoints are saved to `runs/swinir_finetune/checkpoints/`:

| File | Saved when |
|---|---|
| `best.pth` | Validation PSNR improves |
| `epoch_NNNN.pth` | Every `save_interval_epochs` epochs |

Only the last `keep_last_n_checkpoints` periodic checkpoints are kept on disk. `best.pth` is never rotated.

### Resuming a run

```bash
python scripts/training.py --config configs/swinir_finetune.yaml \
    --resume runs/swinir_finetune/checkpoints/epoch_0050.pth
```

This restores model weights, optimizer state, scheduler state, and the best PSNR tracker. Training continues from epoch 51.

### Monitoring

```bash
pip install tensorboard
tensorboard --logdir runs/swinir_finetune/tensorboard
```

Logged scalars:

| Tag | Description |
|---|---|
| `train/loss` | Charbonnier loss per epoch |
| `train/lr` | Learning rate per epoch |
| `train/iter_loss` | Loss every `log_interval_iters` iterations |
| `val/psnr` | Validation PSNR (dB) |
| `val/ssim` | Validation SSIM |

---

## Stage 3 — Inference

### Inference configuration

```yaml
# configs/inference.yaml

model:
  checkpoint_path: "runs/swinir_finetune/checkpoints/best.pth"
  checkpoint_key:  "model"    # "model" for fine-tuned; "params" for raw pretrained

io:
  output_dir:   "output"
  dtype_max:    65535         # must match training dtype_max
  output_dtype: "same"        # "same" | "uint16" | "uint8"
  compress:     "deflate"

tiling:
  tile_size: 256              # LR pixels per tile — reduce if GPU OOM
  overlap:   16               # overlap for seamless blending
```

> **`checkpoint_key`**: fine-tuned checkpoints produced by `trainer.py` store weights under `"model"`. Original KAIR pretrained checkpoints use `"params"`. The inference script tries both automatically if the configured key is not found.

### Running inference

```bash
# Single LR tile:
python scripts/inference.py \
    --config configs/inference.yaml \
    --input  data/processed/val/LR/WO_.../IMG_..._row0058_col0005.TIF

# All validation tiles (glob — quote to prevent shell expansion):
python scripts/inference.py \
    --config configs/inference.yaml \
    --input  "data/processed/val/LR/**/*.TIF" \
    --output output/val_sr

# Override checkpoint and device on the fly:
python scripts/inference.py \
    --config configs/inference.yaml \
    --input  my_lr_tile.tif \
    --checkpoint runs/swinir_finetune/checkpoints/best.pth \
    --device cuda:1
```

Output files are written to the configured `output_dir` with `_SR.TIF` appended to the original filename. Each output is a fully geo-registered GeoTIFF with:
- Spatial resolution halved in the affine transform (SR pixel spacing = LR spacing / 2)
- `PHOTOMETRIC=RGB` tag set explicitly
- Per-band `STATISTICS_MINIMUM/MAXIMUM/MEAN/STDDEV` tags embedded for correct QGIS rendering

---

## Configuration reference

### `configs/swinir_finetune.yaml`

| Key | Default | Description |
|---|---|---|
| `kair_root` | `"KAIR"` | Path to local KAIR clone |
| `data.train_hr` | `"data/processed/train/HR"` | HR training tiles root |
| `data.train_lr` | `"data/processed/train/LR"` | LR training tiles root |
| `data.val_hr` | `"data/processed/val/HR"` | HR validation tiles root |
| `data.val_lr` | `"data/processed/val/LR"` | LR validation tiles root |
| `data.scale` | `2` | LR→HR spatial scale factor |
| `data.dtype_max` | `65535` | Normalisation divisor (`255` for uint8) |
| `data.lr_patch_size` | `64` | LR crop size (must be multiple of `window_size`) |
| `data.num_workers` | `8` | DataLoader worker processes |
| `model.pretrained_path` | `"pretrained/..."` | Path to pretrained `.pth` (auto-downloaded) |
| `model.pretrained_key` | `"params"` | Key in the pretrained checkpoint dict |
| `model.upscale` | `2` | SR scale factor |
| `model.in_chans` | `3` | Input channels |
| `model.img_size` | `64` | Transformer token size (= `lr_patch_size`) |
| `model.window_size` | `8` | Swin attention window size |
| `model.embed_dim` | `180` | Transformer embedding dimension |
| `model.depths` | `[6,6,6,6,6,6]` | Transformer block depths per stage |
| `model.num_heads` | `[6,6,6,6,6,6]` | Attention heads per stage |
| `training.epochs` | `100` | Total training epochs |
| `training.batch_size` | `32` | Batch size |
| `training.frozen_prefixes` | `null` | Layer name prefixes to freeze |
| `training.grad_clip_norm` | `0.01` | Max gradient L2 norm (`null` to disable) |
| `training.use_amp` | `true` | Mixed-precision training |
| `optimizer.type` | `"adam"` | `"adam"` or `"adamw"` |
| `optimizer.lr` | `2e-4` | Initial learning rate |
| `scheduler.type` | `"multistep"` | `"multistep"`, `"cosine"`, or `"none"` |
| `scheduler.milestones` | `[40,60,80,90]` | Epochs at which LR is multiplied by `gamma` |
| `scheduler.gamma` | `0.5` | LR decay factor |
| `loss.type` | `"charbonnier"` | `"charbonnier"`, `"l1"`, or `"mse"` |
| `loss.perceptual.enabled` | `false` | Enable VGG perceptual loss |
| `logging.run_dir` | `"runs/swinir_finetune"` | Root for checkpoints, TensorBoard |
| `logging.val_interval_epochs` | `5` | Run validation every N epochs |
| `logging.save_interval_epochs` | `10` | Save checkpoint every N epochs |
| `logging.keep_last_n_checkpoints` | `3` | Max periodic checkpoints on disk |
| `misc.seed` | `42` | Global random seed |
| `misc.device` | `"cuda"` | Compute device |
| `misc.resume` | `null` | Checkpoint path to resume from |

### `configs/inference.yaml`

| Key | Default | Description |
|---|---|---|
| `model.checkpoint_path` | `"runs/.../best.pth"` | Fine-tuned checkpoint to load |
| `model.checkpoint_key` | `"model"` | State dict key (`"model"` or `"params"`) |
| `io.input_path` | `""` | LR GeoTIFF path or glob (overridden by `--input`) |
| `io.output_dir` | `"output"` | Output directory |
| `io.dtype_max` | `65535` | Must match training `dtype_max` |
| `io.output_dtype` | `"same"` | `"same"`, `"uint16"`, or `"uint8"` |
| `io.compress` | `"deflate"` | Output GeoTIFF compression |
| `tiling.tile_size` | `256` | LR tile size for inference tiling |
| `tiling.overlap` | `16` | Overlap for seam blending |
| `misc.device` | `"cuda"` | Compute device |
| `misc.use_fp16` | `true` | Config flag (currently ignored — see note below) |

> **Note on `misc.use_fp16`:** SwinIR's shifted-window attention overflows fp16 for inputs larger than ~128 px, producing NaN → black output. Inference always runs in float32 regardless of this setting. The flag is kept for future use.

---

## Troubleshooting

### `loss=nan` during training

The Charbonnier loss correctly masks nodata pixels (NaN after normalisation). If loss is still NaN:
- Check `data.dtype_max`: must be `65535` for uint16 tiles, `255` for uint8.
- Verify tiles are not entirely nodata — check `min_valid_fraction` in `build_dataset.py`.
- Lower `optimizer.lr` (try `5e-5`).

### `PSNR=nan` during validation

AMP (fp16) is automatically disabled during validation. If PSNR is still NaN:
- The model may not have learned yet (normal in the first 1–2 epochs with low LR).
- Check that `data.dtype_max` matches the bit depth of your tiles.

### Black SR output image

- Confirm `model.checkpoint_key: "model"` in `inference.yaml` for fine-tuned checkpoints.
- Confirm `io.dtype_max` matches the value used during training.
- Run `gdalinfo output/IMG_..._SR.TIF` and check that min/max statistics are non-zero.

### GPU out-of-memory during training

- Reduce `training.batch_size` (e.g. `8`).
- Reduce `data.lr_patch_size` (e.g. `48`, must remain a multiple of `model.window_size=8`).

### GPU out-of-memory during inference

- Reduce `tiling.tile_size` in `inference.yaml` (e.g. `128`).

### KAIR import warning: `torch.meshgrid`

This warning originates inside KAIR and is suppressed automatically. It does not affect correctness.

### `rasterio.errors.RasterBlockError: BLOCKXSIZE must be a multiple of 16`

This is handled automatically in `build_dataset.py`. If it appears in another context, ensure block sizes are rounded down to the nearest multiple of 16 with a minimum of 16.

---

## Acknowledgements

- **SwinIR** — [Liang et al., 2021](https://arxiv.org/abs/2108.10257)  
  Model definition from [KAIR](https://github.com/cszn/KAIR) by Kai Zhang et al.
- **Pléiades NEO** imagery — © Airbus DS
- Pretrained weights from [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)