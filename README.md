# Single LR tile → output/
python scripts/inference.py \
    --config configs/inference.yaml \
    --input  data/processed/val/LR/WO_000373512_10_2_SAL25349983-2_ACQ_PNEO3_03414708532148/IMG_PNEO3_STD_202309130712484_MS-FS_ORT_PWOI_000373512_10_2_F_1_RGB_R1C1_PANSHARP_row0058_col0005.TIF

# All val LR tiles at once
python scripts/inference.py \
    --config configs/inference.yaml \
    --input  "data/processed/val/LR/**/*.TIF" \
    --output output/val_sr