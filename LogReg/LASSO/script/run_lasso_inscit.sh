#!/bin/bash
# LASSO回帰の実行スクリプト（INSCIT）
# デフォルトでCVを使用（trainとdevの分布が異なるため）

cd "$(dirname "$0")/.." || exit

python main.py \
    --dataset INSCIT \
    --retrieval-method dpr \
    --delong-test \
    --bootstrap \
    --bootstrap-iterations 1000

