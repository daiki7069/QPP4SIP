#!/bin/bash
# ブートストラップ評価の実行スクリプト（単体指標vs回帰モデル）

cd "$(dirname "$0")/.." || exit

# 使用例:
# 1. まず、ブートストラップサンプルを生成
# python generate_bootstrap_samples.py \
#     --dataset INSCIT \
#     --n-samples 500 \
#     --n-iterations 1000 \
#     --output-path outputs/INSCIT/bootstrap_samples.pkl
#
# 2. main.pyを実行して予測結果を生成（--delong-testで予測結果が保存される）
# python main.py --dataset INSCIT --retrieval-method dpr --delong-test
#
# 3. ブートストラップ評価を実行
# python bootstrap_evaluation.py \
#     --dataset INSCIT \
#     --retrieval-method dpr \
#     --bootstrap-samples-path outputs/INSCIT/bootstrap_samples.pkl \
#     --predictions-path outputs/INSCIT/pre_post/.../predictions_for_delong.csv

echo "使用方法:"
echo "  1. ブートストラップサンプルを生成:"
echo "     python generate_bootstrap_samples.py \\"
echo "         --dataset INSCIT \\"
echo "         --n-samples 500 \\"
echo "         --n-iterations 1000 \\"
echo "         --output-path outputs/INSCIT/bootstrap_samples.pkl"
echo ""
echo "  2. main.pyを実行して予測結果を生成:"
echo "     python main.py --dataset INSCIT --retrieval-method dpr --delong-test"
echo ""
echo "  3. ブートストラップ評価を実行:"
echo "     python bootstrap_evaluation.py \\"
echo "         --dataset INSCIT \\"
echo "         --retrieval-method dpr \\"
echo "         --bootstrap-samples-path outputs/INSCIT/bootstrap_samples.pkl \\"
echo "         --predictions-path outputs/INSCIT/pre_post/.../predictions_for_delong.csv"

