#!/bin/bash
# 複数の特徴量タイプの組み合わせを並列実行するスクリプト（INSCIT用、GNU parallel使用版）

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

