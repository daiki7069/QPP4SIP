#!/bin/bash
# AmbigNQのLASSO回帰の実行スクリプト
# デフォルトでCVを使用（trainとdevの分布が異なるため）

cd "$(dirname "$0")/.." || exit

python main.py \
    --dataset AmbigNQ \
    --use-minmax-normalization \
    --delong-test \
    # --bootstrap-test \
    # --bootstrap-samples-path outputs/INSCIT/bootstrap_samples.pkl

