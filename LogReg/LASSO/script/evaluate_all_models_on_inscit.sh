#!/bin/bash
# AmbigNQで学習した全てのモデルタイプをINSCITで評価するスクリプト

set -e

cd /home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO

# 特徴量タイプ
FEATURE_TYPES="post"

# 検索手法
RETRIEVAL_METHOD="dpr"

# モデルディレクトリ（特徴量タイプに応じて変更）
if [ "$FEATURE_TYPES" == "post" ]; then
    MODEL_DIR="outputs/AmbigNQ/post/clarity_ns50_nqc_smv_wig/models"
elif [ "$FEATURE_TYPES" == "pre post" ]; then
    MODEL_DIR="outputs/AmbigNQ/pre_post/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig/models"
elif [ "$FEATURE_TYPES" == "pre post bert" ]; then
    MODEL_DIR="outputs/AmbigNQ/pre_post_bert/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_bert/models"
else
    echo "エラー: 不明な特徴量タイプ: $FEATURE_TYPES"
    exit 1
fi

echo "=========================================="
echo "AmbigNQで学習した全てのモデルをINSCITで評価"
echo "=========================================="
echo ""
echo "特徴量タイプ: $FEATURE_TYPES"
echo "モデルディレクトリ: $MODEL_DIR"
echo "検索手法: $RETRIEVAL_METHOD"
echo ""

# 評価するモデルタイプ
MODEL_TYPES=("l1" "l2" "elasticnet" "none")

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo "=========================================="
    echo "モデルタイプ: $MODEL_TYPE"
    echo "=========================================="
    
    # モデルファイルが存在するか確認
    MODEL_FILE="$MODEL_DIR/${MODEL_TYPE}_model.pkl"
    if [ ! -f "$MODEL_FILE" ]; then
        echo "  ⚠️  モデルファイルが見つかりません: $MODEL_FILE"
        echo "  スキップします。"
        echo ""
        continue
    fi
    
    # 評価を実行
    python evaluate_on_inscit.py \
        --model-dir "$MODEL_DIR" \
        --model-type "$MODEL_TYPE" \
        --feature-types $FEATURE_TYPES \
        --retrieval-method "$RETRIEVAL_METHOD"
    
    echo ""
done

echo "=========================================="
echo "全ての評価完了"
echo "=========================================="
echo ""
echo "結果は以下に保存されました:"
echo "  outputs/INSCIT/transfer_from_AmbigNQ/${FEATURE_TYPES// /_}/"

