#!/bin/bash
# スコア（QPP、BERT、統合）とQAペア数の相関を分析するスクリプト

set -e

# プロジェクトのルートディレクトリ
PROJECT_ROOT="/home/daiki_shibata/pj/QPP4SIP"
cd "$PROJECT_ROOT"

# Pythonスクリプトのパス
SCRIPT_PATH="LogReg/LASSO/analyze_score_qa_pairs_correlation.py"

# デフォルトの引数
SPLIT="dev"
THRESHOLD_PERCENTILE=75.0

# 引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --threshold_percentile)
            THRESHOLD_PERCENTILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--split dev|train] [--threshold_percentile 75.0]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "スコアとQAペア数の相関分析"
echo "=========================================="
echo "Split: $SPLIT"
echo "Threshold Percentile: $THRESHOLD_PERCENTILE"
echo ""

# Pythonスクリプトを実行
python "$SCRIPT_PATH" \
    --split "$SPLIT" \
    --threshold_percentile "$THRESHOLD_PERCENTILE"

echo ""
echo "=========================================="
echo "分析完了！"
echo "結果は以下に保存されています:"
echo "  LogReg/LASSO/outputs/AmbigNQ/score_qa_pairs_analysis/"
echo "=========================================="

