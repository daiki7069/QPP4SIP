#!/bin/bash

# QPP4SIP GPU実験監視スクリプト

BASE_DIR="/home/daiki_shibata/pj/QPP4SIP/SIP/SIP"
LOG_BASE="${BASE_DIR}/logs"
OUTPUT_BASE="${BASE_DIR}/output"
CHECKPOINT_BASE="${BASE_DIR}/checkpoints"

echo "=== QPP4SIP GPU実験監視 ==="
echo "監視時刻: $(date)"
echo ""

# GPU使用状況の確認
echo "--- GPU使用状況 ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | while IFS=',' read -r gpu_id name util mem_used mem_total temp power; do
    echo "GPU ${gpu_id}: ${name}"
    echo "  使用率: ${util}%"
    echo "  メモリ: ${mem_used}MB / ${mem_total}MB"
    echo "  温度: ${temp}°C"
    echo "  電力: ${power}W"
    echo ""
done

# メインプロセスの確認
echo "--- メインプロセス状況 ---"
if pgrep -f "run_qpp4sip_gpu_experiments.sh" > /dev/null; then
    echo "✓ メインGPU実験プロセス: 実行中"
    MAIN_PID=$(pgrep -f "run_qpp4sip_gpu_experiments.sh")
    echo "  プロセスID: ${MAIN_PID}"
else
    echo "✗ メインGPU実験プロセス: 停止"
fi

# 学習プロセスの確認
echo ""
echo "--- 学習プロセス状況 ---"
PYTHON_PROCS=$(pgrep -f "run.py.*train.*qpp4sip")
if [ -n "$PYTHON_PROCS" ]; then
    echo "✓ 学習プロセス: 実行中"
    for pid in $PYTHON_PROCS; do
        pattern=$(ps -p $pid -o args= | grep -o "qpp4sip_pattern [a-z_]*" | cut -d' ' -f2)
        gpu_id=$(ps -p $pid -o args= | grep -o "CUDA_VISIBLE_DEVICES=[0-9]*" | cut -d'=' -f2)
        echo "  プロセスID: ${pid} (パターン: ${pattern}, GPU: ${gpu_id})"
    done
else
    echo "✗ 学習プロセス: 停止"
fi

# 各パターンの進捗確認
echo ""
echo "--- 各パターンの進捗 ---"
PATTERNS=("feature_fusion" "auxiliary_head" "policy_gating")
declare -A PATTERN_GPU=(
    ["feature_fusion"]="0"
    ["auxiliary_head"]="1"
    ["policy_gating"]="2"
)

for pattern in "${PATTERNS[@]}"; do
    gpu_id="${PATTERN_GPU[$pattern]}"
    echo ""
    echo "パターン: ${pattern} (GPU ${gpu_id})"
    
    # チェックポイントの確認
    checkpoint_dir="${CHECKPOINT_BASE}/qpp4sip_${pattern}"
    if [ -d "$checkpoint_dir" ]; then
        checkpoint_count=$(find "$checkpoint_dir" -name "*.pkl" | wc -l)
        echo "  チェックポイント数: ${checkpoint_count}/20"
        
        if [ $checkpoint_count -gt 0 ]; then
            latest_checkpoint=$(find "$checkpoint_dir" -name "*.pkl" | sort -V | tail -1)
            echo "  最新チェックポイント: $(basename $latest_checkpoint)"
        fi
    else
        echo "  チェックポイント: 未作成"
    fi
    
    # 出力ファイルの確認
    output_dir="${OUTPUT_BASE}/qpp4sip_${pattern}"
    if [ -d "$output_dir" ]; then
        dev_files=$(find "$output_dir" -name "dev.*.txt" | wc -l)
        test_files=$(find "$output_dir" -name "test.*.txt" | wc -l)
        echo "  推論結果: dev=${dev_files}, test=${test_files}"
    else
        echo "  推論結果: 未作成"
    fi
    
    # ログファイルの確認
    log_file="${LOG_BASE}/qpp4sip_${pattern}/train_${pattern}.log"
    if [ -f "$log_file" ]; then
        log_size=$(wc -l < "$log_file")
        echo "  学習ログ: ${log_size} 行"
        if [ $log_size -gt 0 ]; then
            echo "  最新のログ:"
            tail -2 "$log_file" | sed 's/^/    /'
        fi
    else
        echo "  学習ログ: 未作成"
    fi
done

# システムリソースの確認
echo ""
echo "--- システムリソース ---"
echo "CPU使用率:"
top -bn1 | grep "Cpu(s)" | sed 's/^/  /'
echo "メモリ使用率:"
free -h | grep "Mem:" | sed 's/^/  /'
echo "ディスク使用率:"
df -h /home/daiki_shibata/pj/QPP4SIP | tail -1 | sed 's/^/  /'

echo ""
echo "監視完了時刻: $(date)"
