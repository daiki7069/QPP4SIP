#!/bin/bash
# 回帰モデル同士の比較検定スクリプト

cd "$(dirname "$0")/.." || exit

# 使用例:
# python compare_models.py \
#     --dataset INSCIT \
#     --model-a-path outputs/INSCIT/pre_post/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig/predictions_for_delong.csv \
#     --model-b-path outputs/INSCIT/pre/ictf_idf_maxidf_scq_scs/predictions_for_delong.csv \
#     --model-a-name "Pre+Post" \
#     --model-b-name "Pre only"

echo "使用方法:"
echo "  python compare_models.py \\"
echo "      --dataset INSCIT \\"
echo "      --model-a-path <モデルAの予測結果ファイル> \\"
echo "      --model-b-path <モデルBの予測結果ファイル> \\"
echo "      --model-a-name \"モデルAの名前\" \\"
echo "      --model-b-name \"モデルBの名前\""

