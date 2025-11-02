#!/bin/bash

ORIGINAL_DATA_FILE_DEV=/mnt/nas_syno/daiki/Datasets/INSCIT/data/dev.json
ORIGINAL_DATA_FILE_TRAIN=/mnt/nas_syno/daiki/Datasets/INSCIT/data/train.json
OUTPUT_DIR=/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT

# dev.json -> 内部 dev.json（同名で出力）
uv run python ./dataset/preprocess_INSCIT.py --original_data_file "$ORIGINAL_DATA_FILE_DEV" --output_dir "$OUTPUT_DIR"
echo "dev.json の内部形式生成が完了しました。"

# train.json -> 内部 train.json（同名で出力）
uv run python ./dataset/preprocess_INSCIT.py --original_data_file "$ORIGINAL_DATA_FILE_TRAIN" --output_dir "$OUTPUT_DIR"
echo "train.json の内部形式生成が完了しました。"
