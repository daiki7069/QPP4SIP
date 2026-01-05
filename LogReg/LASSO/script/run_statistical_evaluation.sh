#!/bin/bash
# 統計的検定（ANOVA/Tukey HSD）用の10-fold CV実験スクリプト
# 各FoldのAUCを記録し、R解析用のCSVを出力する

cd "$(dirname "$0")/.." || exit

# データセット名
DATASET="${DATASET:-AmbigNQ}"

# 出力ディレクトリ
OUTPUT_DIR="outputs/${DATASET}/statistical_evaluation"

# ログディレクトリを作成
LOG_DIR="outputs/${DATASET}/statistical_evaluation_logs"
mkdir -p "$LOG_DIR"

echo "=== 統計的検定用の10-fold CV実験 ==="
echo "データセット: $DATASET"
echo "出力ディレクトリ: $OUTPUT_DIR"
echo ""

# DPRの実行
echo "=== DPRの実行 ==="
python run_statistical_evaluation.py \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --cv-folds 10 \
    --retrieval-method dpr \
    > "$LOG_DIR/run_statistical_evaluation_dpr.log" 2>&1

# DPRの実行結果を確認
if [ $? -eq 0 ]; then
    echo ""
    echo "=== DPR実行完了 ==="
    echo "結果CSV: ${OUTPUT_DIR}/eval_results_dpr.csv"
    echo "ログファイル: ${LOG_DIR}/run_statistical_evaluation_dpr.log"
else
    echo ""
    echo "=== DPR実行でエラーが発生しました ==="
    echo "ログファイルを確認してください: ${LOG_DIR}/run_statistical_evaluation_dpr.log"
    exit 1
fi

# BM25の実行（post-retrievalは検索手法に依存するため）
echo ""
echo "=== BM25の実行 ==="
python run_statistical_evaluation.py \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --cv-folds 10 \
    --retrieval-method bm25 \
    > "$LOG_DIR/run_statistical_evaluation_bm25.log" 2>&1

# BM25の実行結果を確認
if [ $? -eq 0 ]; then
    echo ""
    echo "=== BM25実行完了 ==="
    echo "結果CSV: ${OUTPUT_DIR}/eval_results_bm25.csv"
    echo "ログファイル: ${LOG_DIR}/run_statistical_evaluation_bm25.log"
else
    echo ""
    echo "=== BM25実行でエラーが発生しました ==="
    echo "ログファイルを確認してください: ${LOG_DIR}/run_statistical_evaluation_bm25.log"
    exit 1
fi

echo ""
echo "=== 全ての実行完了 ==="
echo "DPR結果CSV: ${OUTPUT_DIR}/eval_results_dpr.csv"
echo "BM25結果CSV: ${OUTPUT_DIR}/eval_results_bm25.csv"
echo ""
echo "CSVファイルの最初の10行（DPR）:"
head -10 "${OUTPUT_DIR}/eval_results_dpr.csv" 2>/dev/null || echo "CSVファイルが見つかりません"
echo ""
echo "CSVファイルの最初の10行（BM25）:"
head -10 "${OUTPUT_DIR}/eval_results_bm25.csv" 2>/dev/null || echo "CSVファイルが見つかりません"

