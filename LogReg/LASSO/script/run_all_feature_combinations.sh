#!/bin/bash
# 複数の特徴量タイプの組み合わせを並列実行するスクリプト

cd "$(dirname "$0")/.." || exit

# 実行する特徴量タイプの組み合わせ
FEATURE_COMBINATIONS=(
    # "pre"
    # "post"
    # "pre post"
    "pre post bert"
    # "pre bert"
    # "post bert"
    "pre post roberta"
)

# 並列実行数（CPUコア数に応じて調整）
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# ログディレクトリを作成
LOG_DIR="outputs/AmbigNQ/parallel_logs"
mkdir -p "$LOG_DIR"

echo "=== 特徴量タイプの組み合わせを並列実行 ==="
echo "並列実行数: $MAX_PARALLEL"
echo "実行する組み合わせ:"
for i in "${!FEATURE_COMBINATIONS[@]}"; do
    echo "  $((i+1)). ${FEATURE_COMBINATIONS[$i]}"
done
echo ""

# 各組み合わせを並列実行
for combo in "${FEATURE_COMBINATIONS[@]}"; do
    # ログファイル名を生成（スペースをアンダースコアに置換）
    log_name=$(echo "$combo" | tr ' ' '_')
    log_file="$LOG_DIR/${log_name}.log"
    
    echo "実行開始: $combo (ログ: $log_file)"
    
    # バックグラウンドで実行
    (
        FEATURE_TYPES="$combo" python main.py \
            --dataset AmbigNQ \
            --use-minmax-normalization \
            --no-cv \
            --delong-test \
            --feature-types $combo \
            > "$log_file" 2>&1
        
        if [ $? -eq 0 ]; then
            echo "[完了] $combo" | tee -a "$LOG_DIR/summary.log"
        else
            echo "[エラー] $combo" | tee -a "$LOG_DIR/summary.log"
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

