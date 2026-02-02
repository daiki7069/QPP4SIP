#!/bin/bash
# 全MoEゲーティング方式を一括実行して比較（script経由で実行）
# 使い方: cd LogReg/MoE && ./script/run_all_moe.sh
#         DATASET=INSCIT RETRIEVAL=bm25 ./script/run_all_moe.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE"

DATASET="${DATASET:-AmbigNQ}"
RETRIEVAL="${RETRIEVAL:-dpr}"

python "$SCRIPT_DIR/run_all_moe.py" \
  --dataset "$DATASET" \
  --retrieval-method "$RETRIEVAL" \
  --no-cv
