#!/bin/bash
# AmbigNQのLASSO回帰の実行スクリプト
# デフォルトでCVを使用（trainとdevの分布が異なるため）

# 使用する特徴量タイプを指定（post, pre, bert, roberta, transfer）
# 例: FEATURE_TYPES="post pre" または FEATURE_TYPES="post bert"
# 指定しない場合はconfig.pyの設定を使用
FEATURE_TYPES="post pre"

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
    "${ELASTICNET_CV_ARGS[@]}"
    # --bootstrap-test \
    # --bootstrap-samples-path outputs/INSCIT/bootstrap_samples.pkl \
