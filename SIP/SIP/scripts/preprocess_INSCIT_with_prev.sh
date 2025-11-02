#!/bin/bash

set -euo pipefail

# 絶対パスを直接指定（必要に応じて編集してください）
PYTHON_SCRIPT="/home/daiki_shibata/pj/QPP4SIP/SIP/SIP/dataset/preprocess_INSCIT_prev_evidence.py"

INPUT_JSON_DEV="/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/dev.json"
INPUT_CSV_DEV="/home/daiki_shibata/pj/QPP4SIP/QPP/retrieval_data/outputs/dpr_dev_with_prev_evidence.csv"
OUTPUT_DIR="/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT"
OUTPUT_FILENAME_DEV="dev_dpr_with_prev.json"

INPUT_JSON_TRAIN="/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/train.json"
INPUT_CSV_TRAIN="/home/daiki_shibata/pj/QPP4SIP/QPP/retrieval_data/outputs/dpr_train_with_prev_evidence.csv"
OUTPUT_FILENAME_TRAIN="train_dpr_with_prev.json"

echo "[INFO] Processing dev ..."
uv run python "$PYTHON_SCRIPT" \
  --input_json "$INPUT_JSON_DEV" \
  --input_csv "$INPUT_CSV_DEV" \
  --output_dir "$OUTPUT_DIR" \
  --output_filename "$OUTPUT_FILENAME_DEV"

echo "[INFO] Processing train ..."
uv run python "$PYTHON_SCRIPT" \
  --input_json "$INPUT_JSON_TRAIN" \
  --input_csv "$INPUT_CSV_TRAIN" \
  --output_dir "$OUTPUT_DIR" \
  --output_filename "$OUTPUT_FILENAME_TRAIN"

echo "[DONE] All finished."


