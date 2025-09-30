#!/bin/bash

# TSVファイルをPKLファイルに変換するスクリプト
# 使用方法: ./convert_tsv_to_pkl.sh [train_tsv] [dev_tsv] [test_tsv] [output_dir]

# 引数の設定
TRAIN_TSV=${1:-"/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/bm25_train.tsv"}
DEV_TSV=${2:-"/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/bm25_dev.tsv"}
TEST_TSV=${3:-""}
OUTPUT_DIR=${4:-"./dataset/pkl"}

echo "=== TSVファイルをPKLファイルに変換 ==="
echo "訓練データ: $TRAIN_TSV"
echo "開発データ: $DEV_TSV"
echo "テストデータ: $TEST_TSV"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo "================================"

# 出力ディレクトリを作成
mkdir -p $OUTPUT_DIR

# 一括変換を実行
uv run --with pandas python model/tsv_to_pkl_converter.py \
    --batch_convert \
    --tsv_files "$TRAIN_TSV" "$DEV_TSV" $([ -n "$TEST_TSV" ] && echo "$TEST_TSV") \
    --output_dir $OUTPUT_DIR

echo "変換が完了しました。"
echo "生成されたPKLファイル:"
ls -la $OUTPUT_DIR/*.pkl
