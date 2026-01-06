#!/bin/bash
# 複数の特徴量タイプの組み合わせを並列実行するスクリプト

# ElasticNet-CVを使用するかどうか（デフォルト: false）
# 例: USE_ELASTICNET_CV=true ./run_all_feature_combinations.sh
USE_ELASTICNET_CV="${USE_ELASTICNET_CV:-false}"

# ElasticNet-CVのfold数（デフォルト: 5）
ELASTICNET_CV_FOLDS="${ELASTICNET_CV_FOLDS:-5}"

# ElasticNet-CVの評価指標（デフォルト: roc_auc）
ELASTICNET_C_SCORING="${ELASTICNET_C_SCORING:-roc_auc}"

cd "$(dirname "$0")/.." || exit

# 実行する特徴量タイプの組み合わせ（DPR）
FEATURE_COMBINATIONS=(
    "pre"
    "post"
    "pre post"
    "pre post bert"
    "pre bert"
    "post bert"
    "pre post roberta"
    "post roberta"
)

# BM25で実行する特徴量タイプの組み合わせ（postを含むもの）
BM25_FEATURE_COMBINATIONS=(
    "post"
    "pre post"
    "pre post bert"
    "post bert"
    "pre post roberta"
    "post roberta"
    "pre post transfer"
)

# 並列実行数（CPUコア数に応じて調整）
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# ログディレクトリを作成
LOG_DIR="outputs/AmbigNQ/parallel_logs"
mkdir -p "$LOG_DIR"

echo "=== 特徴量タイプの組み合わせを並列実行 ==="
echo "並列実行数: $MAX_PARALLEL"
echo "実行する組み合わせ（DPR）:"
for i in "${!FEATURE_COMBINATIONS[@]}"; do
    echo "  $((i+1)). ${FEATURE_COMBINATIONS[$i]}"
done
echo "実行する組み合わせ（BM25）:"
for i in "${!BM25_FEATURE_COMBINATIONS[@]}"; do
    echo "  $((i+1)). ${BM25_FEATURE_COMBINATIONS[$i]} (BM25)"
done
echo ""

# 各組み合わせを並列実行（DPR）
for combo in "${FEATURE_COMBINATIONS[@]}"; do
    # ログファイル名を生成（スペースをアンダースコアに置換）
    log_name=$(echo "$combo" | tr ' ' '_')
    log_file="$LOG_DIR/${log_name}.log"
    
    echo "実行開始: $combo (DPR, ログ: $log_file)"
    
    # バックグラウンドで実行
    (
        FEATURE_TYPES="$combo" python main.py \
            --dataset AmbigNQ \
            --use-minmax-normalization \
            --no-cv \
            --delong-test \
            --feature-types $combo \
            ${USE_ELASTICNET_CV:+--use-elasticnet-cv} \
            ${ELASTICNET_CV_FOLDS:+--elasticnet-cv-folds "$ELASTICNET_CV_FOLDS"} \
            ${ELASTICNET_C_SCORING:+--elasticnet-c-scoring "$ELASTICNET_C_SCORING"} \
            > "$log_file" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[完了] $combo (DPR)" | tee -a "$LOG_DIR/summary.log"
        else
            echo "[エラー] $combo (DPR)" | tee -a "$LOG_DIR/summary.log"
        fi
    ) &
    
    # 並列実行数を制限
    while [ $(jobs -r | wc -l) -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done
done

# 各組み合わせを並列実行（BM25）
for combo in "${BM25_FEATURE_COMBINATIONS[@]}"; do
    # ログファイル名を生成（スペースをアンダースコアに置換、bm25を追加）
    log_name=$(echo "$combo" | tr ' ' '_')
    log_file="$LOG_DIR/${log_name}_bm25.log"
    
    echo "実行開始: $combo (BM25, ログ: $log_file)"
    
    # バックグラウンドで実行
    (
        FEATURE_TYPES="$combo" python main.py \
            --dataset AmbigNQ \
            --retrieval-method bm25 \
            --use-minmax-normalization \
            --no-cv \
            --delong-test \
            --feature-types $combo \
            ${USE_ELASTICNET_CV:+--use-elasticnet-cv} \
            ${ELASTICNET_CV_FOLDS:+--elasticnet-cv-folds "$ELASTICNET_CV_FOLDS"} \
            ${ELASTICNET_C_SCORING:+--elasticnet-c-scoring "$ELASTICNET_C_SCORING"} \
            > "$log_file" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[完了] $combo (BM25)" | tee -a "$LOG_DIR/summary.log"
        else
            echo "[エラー] $combo (BM25)" | tee -a "$LOG_DIR/summary.log"
        fi
    ) &
    
    # 並列実行数を制限
    while [ $(jobs -r | wc -l) -ge "$MAX_PARALLEL" ]; do
        sleep 1
    done
done

# 全てのジョブの完了を待つ
echo ""
echo "全てのジョブを開始しました。完了を待っています..."
wait

echo ""
echo "=== 実行完了 ==="
echo "ログファイル: $LOG_DIR/"
if [ -f "$LOG_DIR/summary.log" ]; then
    echo ""
    echo "実行結果サマリー:"
    cat "$LOG_DIR/summary.log"
fi

