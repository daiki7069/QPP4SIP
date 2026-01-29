#!/bin/bash
# AmbigNQで学習したモデルを保存するスクリプト

set -e

cd /home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO

echo "=========================================="
echo "AmbigNQでモデルを学習・保存"
echo "=========================================="
echo ""

# モデルタイプを指定（l1, l2, elasticnet, noneなど）
# 複数のモデルを保存する場合は、このスクリプトを複数回実行するか、ループで実行
MODEL_TYPES=("l1" "l2" "elasticnet" "none")

echo "特徴量タイプ: post"
echo "検索手法: dpr"
echo ""

# 各モデルタイプを学習して保存
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo "=========================================="
    echo "モデルタイプ: $MODEL_TYPE"
    echo "=========================================="
    
    # モデルを学習して保存（--no-cvで通常学習モード）
    python main.py \
        --dataset AmbigNQ \
        --model-type "$MODEL_TYPE" \
        --feature-types post \
        --retrieval-method dpr \
        --use-minmax-normalization \
        --no-cv \
        --save-model
    
    echo ""
done

echo "=========================================="
echo "全てのモデル保存完了"
echo "=========================================="
echo ""
echo "保存先: outputs/AmbigNQ/post/clarity_ns50_nqc_smv_wig/models/"

