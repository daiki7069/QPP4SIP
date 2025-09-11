#!/bin/bash

# QPP4SIP GPU実験実行スクリプト
# 3つのパターンを並列でGPU実行

set -e  # エラー時に停止

# 基本設定
BASE_DIR="/home/daiki_shibata/pj/QPP4SIP/SIP/SIP"
DATA_DIR="${BASE_DIR}/dataset"
OUTPUT_BASE="${BASE_DIR}/output"
CHECKPOINT_BASE="${BASE_DIR}/checkpoints"
LOG_BASE="${BASE_DIR}/logs"

# データファイル
TRAIN_DATA="${DATA_DIR}/train_resolved_retrieved.pkl"
DEV_DATA="${DATA_DIR}/dev_resolved_retrieved.pkl"
TEST_DATA="${DATA_DIR}/test_resolved_retrieved.pkl"

# 実験設定
EPOCHS=20
BATCH_SIZE=1
LEARNING_RATE=1e-5
HIDDEN_SIZE=768

# パターンリストとGPU割り当て
declare -A PATTERN_GPU=(
    ["feature_fusion"]="0"
    ["auxiliary_head"]="1"
    ["policy_gating"]="2"
)

echo "=== QPP4SIP GPU実験開始 ==="
echo "開始時刻: $(date)"
echo "利用可能GPU: 0, 1, 2, 3, 4, 5"
echo "パターン別GPU割り当て:"
for pattern in "${!PATTERN_GPU[@]}"; do
    echo "  ${pattern}: GPU ${PATTERN_GPU[$pattern]}"
done
echo ""

# 各パターンを並列で実行
for pattern in "${!PATTERN_GPU[@]}"; do
    gpu_id="${PATTERN_GPU[$pattern]}"
    
    echo "=========================================="
    echo "パターン: ${pattern} をGPU ${gpu_id}で開始"
    echo "開始時刻: $(date)"
    echo "=========================================="
    
    # パターン専用のディレクトリ
    OUTPUT_DIR="${OUTPUT_BASE}/qpp4sip_${pattern}"
    CHECKPOINT_DIR="${CHECKPOINT_BASE}/qpp4sip_${pattern}"
    LOG_DIR="${LOG_BASE}/qpp4sip_${pattern}"
    
    # ディレクトリを作成
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${CHECKPOINT_DIR}"
    mkdir -p "${LOG_DIR}"
    
    # GPU指定で学習を実行
    echo "--- 学習開始: ${pattern} (GPU ${gpu_id}) ---"
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup conda run -n sip uv run python "${BASE_DIR}/model/run.py" \
        --mode train \
        --model qpp4sip \
        --qpp4sip_pattern "${pattern}" \
        --input_path "${TRAIN_DATA}" \
        --output_path "${OUTPUT_DIR}" \
        --saved_model_path "${CHECKPOINT_DIR}" \
        --log_path "${LOG_DIR}/train" \
        --epoch_num "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --learning_rate "${LEARNING_RATE}" \
        --hidden_size "${HIDDEN_SIZE}" \
        --dropout 0.1 \
        --BiLSTM_layer 1 \
        --max_utterance_len 50 \
        --max_context_len 512 \
        --random_seed 42 \
        > "${LOG_DIR}/train_${pattern}.log" 2>&1 &
    
    TRAIN_PID=$!
    echo "学習プロセス開始 (PID: ${TRAIN_PID}, GPU: ${gpu_id})"
    
    # 学習完了を待つ
    wait ${TRAIN_PID}
    TRAIN_EXIT_CODE=$?
    
    if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
        echo "✓ 学習完了: ${pattern}"
    else
        echo "✗ 学習失敗: ${pattern} (終了コード: ${TRAIN_EXIT_CODE})"
        continue
    fi
    
    # 開発データでの推論
    echo "--- 開発データ推論開始: ${pattern} (GPU ${gpu_id}) ---"
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup conda run -n sip uv run python "${BASE_DIR}/model/run.py" \
        --mode inference \
        --model qpp4sip \
        --qpp4sip_pattern "${pattern}" \
        --input_path "${DEV_DATA}" \
        --output_path "${OUTPUT_DIR}" \
        --saved_model_path "${CHECKPOINT_DIR}" \
        --log_path "${LOG_DIR}/inference" \
        --epoch_num "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --hidden_size "${HIDDEN_SIZE}" \
        --dropout 0.1 \
        --BiLSTM_layer 1 \
        --max_utterance_len 50 \
        --max_context_len 512 \
        --random_seed 42 \
        > "${LOG_DIR}/inference_dev_${pattern}.log" 2>&1 &
    
    DEV_INF_PID=$!
    echo "開発データ推論プロセス開始 (PID: ${DEV_INF_PID}, GPU: ${gpu_id})"
    
    # テストデータでの推論
    echo "--- テストデータ推論開始: ${pattern} (GPU ${gpu_id}) ---"
    CUDA_VISIBLE_DEVICES=${gpu_id} nohup conda run -n sip uv run python "${BASE_DIR}/model/run.py" \
        --mode inference \
        --model qpp4sip \
        --qpp4sip_pattern "${pattern}" \
        --input_path "${TEST_DATA}" \
        --output_path "${OUTPUT_DIR}" \
        --saved_model_path "${CHECKPOINT_DIR}" \
        --log_path "${LOG_DIR}/inference" \
        --epoch_num "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --hidden_size "${HIDDEN_SIZE}" \
        --dropout 0.1 \
        --BiLSTM_layer 1 \
        --max_utterance_len 50 \
        --max_context_len 512 \
        --random_seed 42 \
        > "${LOG_DIR}/inference_test_${pattern}.log" 2>&1 &
    
    TEST_INF_PID=$!
    echo "テストデータ推論プロセス開始 (PID: ${TEST_INF_PID}, GPU: ${gpu_id})"
    
    # 推論完了を待つ
    wait ${DEV_INF_PID}
    DEV_INF_EXIT_CODE=$?
    wait ${TEST_INF_PID}
    TEST_INF_EXIT_CODE=$?
    
    if [ ${DEV_INF_EXIT_CODE} -eq 0 ]; then
        echo "✓ 開発データ推論完了: ${pattern}"
    else
        echo "✗ 開発データ推論失敗: ${pattern} (終了コード: ${DEV_INF_EXIT_CODE})"
    fi
    
    if [ ${TEST_INF_EXIT_CODE} -eq 0 ]; then
        echo "✓ テストデータ推論完了: ${pattern}"
    else
        echo "✗ テストデータ推論失敗: ${pattern} (終了コード: ${TEST_INF_EXIT_CODE})"
    fi
    
    echo "パターン ${pattern} の実験完了時刻: $(date)"
    echo ""
done

echo "=========================================="
echo "全実験完了時刻: $(date)"
echo "=========================================="

# 結果サマリーを表示
echo ""
echo "=== 実験結果サマリー ==="
for pattern in "${!PATTERN_GPU[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/qpp4sip_${pattern}"
    LOG_DIR="${LOG_BASE}/qpp4sip_${pattern}"
    
    echo ""
    echo "パターン: ${pattern} (GPU ${PATTERN_GPU[$pattern]})"
    echo "  出力ディレクトリ: ${OUTPUT_DIR}"
    echo "  ログディレクトリ: ${LOG_DIR}"
    
    if [ -d "${OUTPUT_DIR}" ]; then
        echo "  生成されたファイル:"
        ls -la "${OUTPUT_DIR}" | head -10
    fi
done

echo ""
echo "GPU実験完了！"
