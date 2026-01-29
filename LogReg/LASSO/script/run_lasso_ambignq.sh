#!/bin/bash
# AmbigNQのLASSO回帰の実行スクリプト
# デフォルトでCVを使用（trainとdevの分布が異なるため）

# 使用する特徴量タイプを指定（post, pre, bert, roberta, transfer）
# 例: FEATURE_TYPES="post pre" または FEATURE_TYPES="post bert"
# 指定しない場合はconfig.pyの設定を使用
FEATURE_TYPES="post pre"

# L1-CVを使用するかどうか（デフォルト: false）
USE_L1_CV="${USE_L1_CV:-false}"

# L1-CVのfold数（デフォルト: 5）
L1_CV_FOLDS="${L1_CV_FOLDS:-5}"

# L1-CVの評価指標（デフォルト: roc_auc）
L1_C_SCORING="${L1_C_SCORING:-roc_auc}"

# L2-CVを使用するかどうか（デフォルト: false）
USE_L2_CV="${USE_L2_CV:-false}"

# L2-CVのfold数（デフォルト: 5）
L2_CV_FOLDS="${L2_CV_FOLDS:-5}"

# L2-CVの評価指標（デフォルト: roc_auc）
L2_C_SCORING="${L2_C_SCORING:-roc_auc}"

# ElasticNet-CVを使用するかどうか（デフォルト: false）
USE_ELASTICNET_CV="${USE_ELASTICNET_CV:-false}"

# ElasticNet-CVのfold数（デフォルト: 5）
ELASTICNET_CV_FOLDS="${ELASTICNET_CV_FOLDS:-5}"

# ElasticNet-CVの評価指標（デフォルト: roc_auc）
ELASTICNET_C_SCORING="${ELASTICNET_C_SCORING:-roc_auc}"

cd "$(dirname "$0")/.." || exit

# 特徴量タイプが指定されている場合は引数に追加
FEATURE_TYPES_ARGS=()
if [ -n "$FEATURE_TYPES" ]; then
    FEATURE_TYPES_ARGS=(--feature-types $FEATURE_TYPES)
fi

# L1-CVオプションを追加
L1_CV_ARGS=()
if [ "$USE_L1_CV" = "true" ]; then
    L1_CV_ARGS=(
        --use-l1-cv
        --l1-cv-folds "$L1_CV_FOLDS"
        --l1-c-scoring "$L1_C_SCORING"
    )
fi

# L2-CVオプションを追加
L2_CV_ARGS=()
if [ "$USE_L2_CV" = "true" ]; then
    L2_CV_ARGS=(
        --use-l2-cv
        --l2-cv-folds "$L2_CV_FOLDS"
        --l2-c-scoring "$L2_C_SCORING"
    )
fi

# ElasticNet-CVオプションを追加
ELASTICNET_CV_ARGS=()
if [ "$USE_ELASTICNET_CV" = "true" ]; then
    ELASTICNET_CV_ARGS=(
        --use-elasticnet-cv
        --elasticnet-cv-folds "$ELASTICNET_CV_FOLDS"
        --elasticnet-c-scoring "$ELASTICNET_C_SCORING"
    )
fi

python main.py \
    --dataset AmbigNQ \
    --use-minmax-normalization \
    --delong-test \
    "${FEATURE_TYPES_ARGS[@]}" \
    "${L1_CV_ARGS[@]}" \
    "${L2_CV_ARGS[@]}" \
    "${ELASTICNET_CV_ARGS[@]}"
    # --bootstrap-test \
    # --bootstrap-samples-path outputs/INSCIT/bootstrap_samples.pkl \
