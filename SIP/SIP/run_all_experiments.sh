#!/bin/bash
"""
複数のQPP4SIP実験を一括実行するシェルスクリプト
"""
set -e

# 色付きの出力用
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ログ関数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 実験設定
FEATURE_SETS=("ndcg_only" "precision_only" "recall_only" "at3" "all")
PATTERNS=("feature_fusion" "auxiliary_head" "policy_gating")
EPOCHS=20
BATCH_SIZE=1
LEARNING_RATE=1e-4

# 結果保存用ディレクトリ
RESULTS_DIR="experiments/batch_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# 実験結果を記録するファイル
EXPERIMENT_LOG="$RESULTS_DIR/experiment_log.txt"
SUMMARY_FILE="$RESULTS_DIR/experiment_summary.txt"

# ログファイルを初期化
echo "QPP4SIP バッチ実験ログ - $(date)" > "$EXPERIMENT_LOG"
echo "========================================" >> "$EXPERIMENT_LOG"

# 成功・失敗カウンター
SUCCESS_COUNT=0
FAILED_COUNT=0
TOTAL_COUNT=0

# 実験実行関数
run_experiment() {
    local feature_set=$1
    local pattern=$2
    local model="qpp4sip"
    
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    log_info "実験開始: $model + $pattern + $feature_set"
    echo "実験開始: $model + $pattern + $feature_set - $(date)" >> "$EXPERIMENT_LOG"
    
    # 実験を実行
    if python run.py \
        --mode train \
        --model "$model" \
        --qpp4sip_pattern "$pattern" \
        --input_path "./dataset/train_resolved_retrieved.pkl" \
        --epoch_num "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE"; then
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        log_success "実験完了: $model + $pattern + $feature_set"
        echo "実験完了: $model + $pattern + $feature_set - $(date)" >> "$EXPERIMENT_LOG"
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        log_error "実験失敗: $model + $pattern + $feature_set"
        echo "実験失敗: $model + $pattern + $feature_set - $(date)" >> "$EXPERIMENT_LOG"
    fi
    
    echo "---" >> "$EXPERIMENT_LOG"
}

# MUSICモデルとの比較実験も実行
run_music_experiment() {
    local model="music"
    
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    
    log_info "MUSICモデル実験開始"
    echo "MUSICモデル実験開始 - $(date)" >> "$EXPERIMENT_LOG"
    
    if python run.py \
        --mode train \
        --model "$model" \
        --input_path "./dataset/train_resolved_retrieved.pkl" \
        --epoch_num "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --learning_rate "$LEARNING_RATE"; then
        
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        log_success "MUSICモデル実験完了"
        echo "MUSICモデル実験完了 - $(date)" >> "$EXPERIMENT_LOG"
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
        log_error "MUSICモデル実験失敗"
        echo "MUSICモデル実験失敗 - $(date)" >> "$EXPERIMENT_LOG"
    fi
    
    echo "---" >> "$EXPERIMENT_LOG"
}

# メイン実行
main() {
    log_info "QPP4SIP バッチ実験開始"
    log_info "結果保存先: $RESULTS_DIR"
    log_info "特徴量セット: ${FEATURE_SETS[*]}"
    log_info "パターン: ${PATTERNS[*]}"
    log_info "エポック数: $EPOCHS"
    
    # MUSICモデル（ベースライン）を実行
    run_music_experiment
    
    # QPP4SIPモデルの各組み合わせを実行
    for feature_set in "${FEATURE_SETS[@]}"; do
        for pattern in "${PATTERNS[@]}"; do
            run_experiment "$feature_set" "$pattern"
        done
    done
    
    # 結果サマリーを生成
    log_info "実験結果サマリーを生成中..."
    
    cat > "$SUMMARY_FILE" << EOF
QPP4SIP バッチ実験結果サマリー
================================
実行日時: $(date)
結果保存先: $RESULTS_DIR

実験設定:
- 特徴量セット: ${FEATURE_SETS[*]}
- パターン: ${PATTERNS[*]}
- エポック数: $EPOCHS
- バッチサイズ: $BATCH_SIZE
- 学習率: $LEARNING_RATE

結果:
- 総実験数: $TOTAL_COUNT
- 成功: $SUCCESS_COUNT
- 失敗: $FAILED_COUNT
- 成功率: $(( SUCCESS_COUNT * 100 / TOTAL_COUNT ))%

実験ディレクトリ一覧:
EOF
    
    # 実験ディレクトリの一覧を追加
    for dir in "$RESULTS_DIR"/*; do
        if [ -d "$dir" ]; then
            echo "- $(basename "$dir")" >> "$SUMMARY_FILE"
        fi
    done
    
    log_success "バッチ実験完了"
    log_info "結果サマリー: $SUMMARY_FILE"
    log_info "詳細ログ: $EXPERIMENT_LOG"
    
    # 結果を表示
    echo ""
    echo "========================================"
    cat "$SUMMARY_FILE"
    echo "========================================"
}

# スクリプト実行
main "$@"

