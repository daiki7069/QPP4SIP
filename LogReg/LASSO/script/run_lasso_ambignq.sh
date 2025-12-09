#!/bin/bash
# LASSO回帰の実行スクリプト（AmbigNQ）
# デフォルトでCVを使用（trainとdevの分布が異なるため）

cd "$(dirname "$0")/.." || exit

python main.py \
    --dataset AmbigNQ \
    --delong-test

