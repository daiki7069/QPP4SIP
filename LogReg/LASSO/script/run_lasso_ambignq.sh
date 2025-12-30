#!/bin/bash
# AmbigNQのLASSO回帰の実行スクリプト
# デフォルトでCVを使用（trainとdevの分布が異なるため）

# 使用する特徴量タイプを指定（post, pre, bert, roberta）
# 例: FEATURE_TYPES="post pre" または FEATURE_TYPES="post bert"
# 指定しない場合はconfig.pyの設定を使用
FEATURE_TYPES="post pre"

cd "$(dirname "$0")/.." || exit

# 特徴量タイプが指定されている場合は引数に追加
FEATURE_TYPES_ARGS=()
if [ -n "$FEATURE_TYPES" ]; then
    FEATURE_TYPES_ARGS=(--feature-types $FEATURE_TYPES)
fi

python main.py \
    --dataset AmbigNQ \
    --use-minmax-normalization \
    --delong-test \
    "${FEATURE_TYPES_ARGS[@]}"
    # --bootstrap-test \
    # --bootstrap-samples-path outputs/INSCIT/bootstrap_samples.pkl \
