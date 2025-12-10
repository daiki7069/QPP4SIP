#!/bin/bash
# ブートストラップ評価の実行スクリプト（単体指標vs回帰モデル）

cd "$(dirname "$0")/.." || exit

# 使用例:
# 1. まず、ブートストラップサンプルを生成（サンプル数は自動取得）
python generate_bootstrap_samples.py \
    --dataset INSCIT \
    --n-iterations 1000

# 2. main.pyを実行して予測結果を生成（--delong-testで予測結果が保存される）
# python main.py --dataset INSCIT --delong-test

# 3. ブートストラップ評価を実行（main.pyで--bootstrap-testを指定するか、bootstrap_evaluation.pyを使用）
# python main.py --dataset INSCIT --delong-test --bootstrap-test --bootstrap-samples-path outputs/INSCIT/bootstrap_samples.pkl

