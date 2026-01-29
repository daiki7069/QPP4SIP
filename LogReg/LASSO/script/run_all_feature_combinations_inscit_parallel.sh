#!/bin/bash
# 複数の特徴量タイプの組み合わせを並列実行するスクリプト（INSCIT用、GNU parallel使用版）

# L1-CVを使用するかどうか（デフォルト: false）
USE_L1_CV="${USE_L1_CV:-false}"

# L1-CVのfold数（デフォルト: 5）
L1_CV_FOLDS="${L1_CV_FOLDS:-5}"

# L1-CVの評価指標（デフォルト: roc_auc）
L1_C_SCORING="${L1_C_SCORING:-roc_auc}"

# L2-CVを使用するかどうか（デフォルト: false）
USE_L2_CV="${USE_L2_CV:-false}"

# L2-CVのfold数（デフォルト: 5）
L2_CV_FOLDS="${L2_CV_FOLDS:-5}"

# L2-CVの評価指標（デフォルト: roc_auc）
L2_C_SCORING="${L2_C_SCORING:-roc_auc}"

# ElasticNet-CVを使用するかどうか（デフォルト: false）
USE_ELASTICNET_CV="${USE_ELASTICNET_CV:-false}"

# ElasticNet-CVのfold数（デフォルト: 5）
ELASTICNET_CV_FOLDS="${ELASTICNET_CV_FOLDS:-5}"

# ElasticNet-CVの評価指標（デフォルト: roc_auc）
ELASTICNET_C_SCORING="${ELASTICNET_C_SCORING:-roc_auc}"

cd "$(dirname "$0")/.." || exit

# 実行する特徴量タイプの組み合わせ
FEATURE_COMBINATIONS=(
    "pre"
    "post"
    "pre post"
    "pre post bert"
    "pre bert"
    "post bert"
    "pre post roberta"
)

# 並列実行数（CPUコア数に応じて調整）
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# ログディレクトリを作成
LOG_DIR="outputs/INSCIT/parallel_logs"
mkdir -p "$LOG_DIR"

echo "=== 特徴量タイプの組み合わせを並列実行（INSCIT、GNU parallel使用） ==="
echo "並列実行数: $MAX_PARALLEL"
echo "実行する組み合わせ:"
for i in "${!FEATURE_COMBINATIONS[@]}"; do
    echo "  $((i+1)). ${FEATURE_COMBINATIONS[$i]}"
done
echo ""

# GNU parallelが利用可能かチェック
if command -v parallel &> /dev/null; then
    echo "GNU parallelを使用します"
    # 一時ファイルに組み合わせを書き込む
    TMPFILE=$(mktemp)
    for combo in "${FEATURE_COMBINATIONS[@]}"; do
        echo "$combo" >> "$TMPFILE"
    done
    
    # parallelで実行
    cat "$TMPFILE" | parallel -j "$MAX_PARALLEL" --tag --line-buffer \
        bash -c 'combo="$1"; log_name=$(echo "$combo" | tr " " "_"); log_file="'"$LOG_DIR"'/${log_name}.log"; \
        echo "[開始] $combo" | tee -a "'"$LOG_DIR"'/summary.log; \
        python main.py \
            --dataset INSCIT \
            --use-minmax-normalization \
            --delong-test \
            --feature-types $combo \
            ${USE_L1_CV:+--use-l1-cv} \
            ${L1_CV_FOLDS:+--l1-cv-folds "$L1_CV_FOLDS"} \
            ${L1_C_SCORING:+--l1-c-scoring "$L1_C_SCORING"} \
            ${USE_L2_CV:+--use-l2-cv} \
            ${L2_CV_FOLDS:+--l2-cv-folds "$L2_CV_FOLDS"} \
            ${L2_C_SCORING:+--l2-c-scoring "$L2_C_SCORING"} \
            ${USE_ELASTICNET_CV:+--use-elasticnet-cv} \
            ${ELASTICNET_CV_FOLDS:+--elasticnet-cv-folds "$ELASTICNET_CV_FOLDS"} \
            ${ELASTICNET_C_SCORING:+--elasticnet-c-scoring "$ELASTICNET_C_SCORING"} \
            > "$log_file" 2>&1; \
        if [ $? -eq 0 ]; then \
            echo "[完了] $combo" | tee -a "'"$LOG_DIR"'/summary.log; \
        else \
            echo "[エラー] $combo" | tee -a "'"$LOG_DIR"'/summary.log; \
        fi' _ {}
    
    rm "$TMPFILE"
else
    echo "GNU parallelが見つかりません。通常の並列実行を使用します。"
    # 通常の並列実行スクリプトを呼び出す
    "$(dirname "$0")/run_all_feature_combinations_inscit.sh"
    exit $?
fi

echo ""
echo "=== 実行完了 ==="
echo "ログファイル: $LOG_DIR/"
if [ -f "$LOG_DIR/summary.log" ]; then
    echo ""
    echo "実行結果サマリー:"
    cat "$LOG_DIR/summary.log"
fi

