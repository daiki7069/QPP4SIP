#!/bin/bash
# INSCITとAmbigNQのdev/trainを、top_k=10,20,50,100で並行実行するスクリプト
# 各top_kの処理が完了してから次のtop_kに進むため、途中で止めても問題ない
# エラーが発生しても次の処理を続行する
# GPU 0-3のみを使用し、2-4プロセスを並行実行

# プロジェクトのルートディレクトリ
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# データセットとスプリットの組み合わせ
DATASETS=("INSCIT" "AmbigNQ")
SPLITS=("dev" "train")

# top_kの値（順次実行）
TOP_K_VALUES=(10 20 50 100)

# デフォルト設定
MODEL_NAME="bert-base-uncased"
THRESHOLD=0.5
BATCH_SIZE=64

# 並行実行数（2-4プロセス、GPUの余裕に応じて調整可能）
# GPU 0-3の4枚を使用し、各プロセスに2GPUずつ割り当てる場合はMAX_PARALLEL=2
# 各プロセスに1GPUずつ割り当てる場合はMAX_PARALLEL=4
MAX_PARALLEL=8

# 使用可能なGPU（0-3のみ、GPU 4,5は使用しない）
AVAILABLE_GPUS=(0 1 2 3)

# GPU割り当て戦略:
# - MAX_PARALLEL=2の場合: 各プロセスに2GPUずつ (0,1 と 2,3)
# - MAX_PARALLEL=4の場合: 各プロセスに1GPUずつ (0, 1, 2, 3)
# - MAX_PARALLEL=8の場合: 各プロセスに1GPUずつ (0, 1, 2, 3を循環使用)
# GPUは0-3の4枚のみ使用可能なので、実際の並行実行数は最大4プロセス
if [ $MAX_PARALLEL -eq 2 ]; then
    # 2プロセスで、各プロセスに2GPUずつ割り当て（DataParallel使用）
    GPU_ASSIGNMENTS=("0,1" "2,3")
elif [ $MAX_PARALLEL -eq 4 ] || [ $MAX_PARALLEL -ge 4 ]; then
    # 4プロセス以上で、各プロセスに1GPUずつ割り当て（DataParallel不使用）
    # GPUは4枚しかないので、循環的に割り当て
    GPU_ASSIGNMENTS=("0" "1" "2" "3")
    # 実際の並行実行数は最大4プロセスに制限
    if [ $MAX_PARALLEL -gt 4 ]; then
        echo "警告: MAX_PARALLEL=$MAX_PARALLEL ですが、GPUは4枚しかないため、実際の並行実行数は最大4プロセスに制限されます。"
        MAX_PARALLEL=4
    fi
else
    # その他の場合は1GPUずつ割り当て
    GPU_ASSIGNMENTS=("0" "1" "2" "3")
fi

# ログディレクトリ
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 実行関数（GPU割り当て付き）
run_experiment() {
    local DATASET=$1
    local SPLIT=$2
    local TOP_K=$3
    local PROCESS_ID=$4
    local GPU_ASSIGNMENT=$5
    
    echo "[Process $PROCESS_ID, GPUs: $GPU_ASSIGNMENT] Starting: $DATASET / $SPLIT / top_k=$TOP_K"
    
    # ログファイル名
    LOG_FILE="$LOG_DIR/${DATASET}_${SPLIT}_topk${TOP_K}_proc${PROCESS_ID}_${TIMESTAMP}.log"
    
    # CUDA_VISIBLE_DEVICESでGPUを指定
    export CUDA_VISIBLE_DEVICES="$GPU_ASSIGNMENT"
    
    # GPUが複数ある場合はDataParallelを使用、1つの場合は不使用
    if [ $(echo "$GPU_ASSIGNMENT" | tr ',' '\n' | wc -l) -gt 1 ]; then
        MULTI_GPU_FLAG=""
    else
        MULTI_GPU_FLAG="--no_multi_gpu"
    fi
    
    # 実行（エラーが発生しても続行）
    if python main.py \
        --mode graph \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --top_k "$TOP_K" \
        --threshold "$THRESHOLD" \
        --batch_size "$BATCH_SIZE" \
        --model_name "$MODEL_NAME" \
        $MULTI_GPU_FLAG \
        2>&1 | tee "$LOG_FILE"; then
        echo "[Process $PROCESS_ID, GPUs: $GPU_ASSIGNMENT] ✓ Successfully completed: $DATASET / $SPLIT / top_k=$TOP_K"
        return 0
    else
        echo "[Process $PROCESS_ID, GPUs: $GPU_ASSIGNMENT] ✗ Failed: $DATASET / $SPLIT / top_k=$TOP_K (check $LOG_FILE)"
        return 1
    fi
}

# 各top_k値ごとに実行
for TOP_K in "${TOP_K_VALUES[@]}"; do
    echo "=========================================="
    echo "Starting experiments with top_k=$TOP_K"
    echo "=========================================="
    
    # 実行タスクのリストを作成
    TASKS=()
    for DATASET in "${DATASETS[@]}"; do
        for SPLIT in "${SPLITS[@]}"; do
            TASKS+=("$DATASET|$SPLIT|$TOP_K")
        done
    done
    
    # 並行実行
    PROCESS_ID=0
    PIDS=()
    GPU_ASSIGNMENT_INDEX=0
    
    for TASK in "${TASKS[@]}"; do
        IFS='|' read -r DATASET SPLIT TOP_K_TASK <<< "$TASK"
        
        # GPU割り当て（循環的に使用）
        GPU_ASSIGNMENT=${GPU_ASSIGNMENTS[$GPU_ASSIGNMENT_INDEX]}
        GPU_ASSIGNMENT_INDEX=$(( (GPU_ASSIGNMENT_INDEX + 1) % ${#GPU_ASSIGNMENTS[@]} ))
        
        # バックグラウンドで実行
        run_experiment "$DATASET" "$SPLIT" "$TOP_K_TASK" "$PROCESS_ID" "$GPU_ASSIGNMENT" &
        PID=$!
        PIDS+=($PID)
        
        echo "[Process $PROCESS_ID, GPUs: $GPU_ASSIGNMENT] Launched PID $PID: $DATASET / $SPLIT / top_k=$TOP_K_TASK"
        
        PROCESS_ID=$((PROCESS_ID + 1))
        
        # 並行実行数が上限に達したら、1つ完了するまで待機
        while [ ${#PIDS[@]} -ge $MAX_PARALLEL ]; do
            for i in "${!PIDS[@]}"; do
                if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                    # プロセスが終了した
                    wait "${PIDS[$i]}"
                    unset PIDS[$i]
                    # 配列を再構築
                    PIDS=("${PIDS[@]}")
                    break
                fi
            done
            sleep 1
        done
    done
    
    # 残りのプロセスが完了するまで待機
    echo "Waiting for all processes to complete..."
    for PID in "${PIDS[@]}"; do
        wait "$PID"
    done
    
    echo "=========================================="
    echo "Completed all experiments with top_k=$TOP_K"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
echo "=========================================="

