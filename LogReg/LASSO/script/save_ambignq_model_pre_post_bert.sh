#!/bin/bash
# AmbigNQで学習したモデル（pre_post_bert）を保存するスクリプト

set -e

cd /home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO

echo "=========================================="
echo "AmbigNQでモデルを学習・保存（pre_post_bert）"
echo "=========================================="
echo ""

# モデルタイプを指定（l1, l2, elasticnet, noneなど）
MODEL_TYPE="l1"

echo "モデルタイプ: $MODEL_TYPE"
echo "特徴量タイプ: pre post bert"
echo "検索手法: dpr"
echo ""

# モデルを学習して保存（--no-cvで通常学習モード）
python main.py \
    --dataset AmbigNQ \
    --model-type "$MODEL_TYPE" \
    --feature-types pre post bert \
    --retrieval-method dpr \
    --use-minmax-normalization \
    --no-cv \
    --save-model

echo ""
echo "=========================================="
echo "モデル保存完了"
echo "=========================================="
echo ""
echo "保存先: outputs/AmbigNQ/pre_post_bert/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_bert/models/"

