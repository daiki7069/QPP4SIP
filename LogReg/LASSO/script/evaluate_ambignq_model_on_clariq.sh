#!/bin/bash
# AmbigNQで学習したモデルをClariQで評価するスクリプト

set -e

cd /home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO

# モデルディレクトリ（AmbigNQで学習したモデルの保存先）
MODEL_DIR="outputs/AmbigNQ/post/clarity_ns50_nqc_smv_wig/models"

# モデルタイプ（評価するモデル）
MODEL_TYPE="l1"

# 特徴量タイプ
FEATURE_TYPES="post"

# 検索手法
RETRIEVAL_METHOD="dpr"

echo "=========================================="
echo "AmbigNQで学習したモデルをClariQで評価"
echo "=========================================="
echo ""
echo "モデルディレクトリ: $MODEL_DIR"
echo "モデルタイプ: $MODEL_TYPE"
echo "特徴量タイプ: $FEATURE_TYPES"
echo "検索手法: $RETRIEVAL_METHOD"
echo ""

# 評価を実行
python evaluate_on_clariq.py \
    --model-dir "$MODEL_DIR" \
    --model-type "$MODEL_TYPE" \
    --feature-types $FEATURE_TYPES \
    --retrieval-method "$RETRIEVAL_METHOD"

echo ""
echo "=========================================="
echo "評価完了"
echo "=========================================="

