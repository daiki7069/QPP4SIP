#!/bin/bash
# AmbigNQで学習した全てのモデルをINSCITで評価するスクリプト

set -e

cd /home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO

# 評価するモデルタイプ（全て）
MODEL_TYPES=("l1" "l2" "elasticnet" "none")

# 検索手法
RETRIEVAL_METHOD="dpr"

echo "=========================================="
echo "AmbigNQで学習した全てのモデルをINSCITで評価"
echo "=========================================="
echo ""
echo "評価するモデルタイプ: ${MODEL_TYPES[*]}"
echo "検索手法: $RETRIEVAL_METHOD"
echo ""

# 各特徴量タイプとモデルタイプの組み合わせを評価
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo "=========================================="
    echo "モデルタイプ: $MODEL_TYPE"
    echo "=========================================="
    echo ""
    
    # 1. postのみ
    echo "=== 1. postのみ ==="
    MODEL_FILE="outputs/AmbigNQ/post/clarity_ns50_nqc_smv_wig/models/${MODEL_TYPE}_model.pkl"
    if [ -f "$MODEL_FILE" ]; then
        python evaluate_on_inscit.py \
            --model-dir outputs/AmbigNQ/post/clarity_ns50_nqc_smv_wig/models \
            --model-type "$MODEL_TYPE" \
            --feature-types post \
            --retrieval-method "$RETRIEVAL_METHOD"
    else
        echo "  ⚠️  モデルファイルが見つかりません: $MODEL_FILE"
        echo "  スキップします。"
    fi
    echo ""
    
    # 2. pre_post
    echo "=== 2. pre_post ==="
    MODEL_FILE="outputs/AmbigNQ/pre_post/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig/models/${MODEL_TYPE}_model.pkl"
    if [ -f "$MODEL_FILE" ]; then
        python evaluate_on_inscit.py \
            --model-dir outputs/AmbigNQ/pre_post/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig/models \
            --model-type "$MODEL_TYPE" \
            --feature-types pre post \
            --retrieval-method "$RETRIEVAL_METHOD"
    else
        echo "  ⚠️  モデルファイルが見つかりません: $MODEL_FILE"
        echo "  スキップします。"
    fi
    echo ""
    
    # 3. pre_post_bert
    echo "=== 3. pre_post_bert ==="
    MODEL_FILE="outputs/AmbigNQ/pre_post_bert/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_bert/models/${MODEL_TYPE}_model.pkl"
    if [ -f "$MODEL_FILE" ]; then
        python evaluate_on_inscit.py \
            --model-dir outputs/AmbigNQ/pre_post_bert/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_bert/models \
            --model-type "$MODEL_TYPE" \
            --feature-types pre post bert \
            --retrieval-method "$RETRIEVAL_METHOD"
    else
        echo "  ⚠️  モデルファイルが見つかりません: $MODEL_FILE"
        echo "  スキップします。"
    fi
    echo ""
done

echo "=========================================="
echo "全ての評価完了"
echo "=========================================="
echo ""
echo "結果は以下に保存されました:"
echo "  outputs/INSCIT/transfer_from_AmbigNQ/post/{MODEL_TYPE}_results/"
echo "  outputs/INSCIT/transfer_from_AmbigNQ/pre_post/{MODEL_TYPE}_results/"
echo "  outputs/INSCIT/transfer_from_AmbigNQ/pre_post_bert/{MODEL_TYPE}_results/"

