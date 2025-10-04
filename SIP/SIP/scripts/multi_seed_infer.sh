#!/bin/bash

EPOCH_NUM=20
PATTERN="feature_fusion"
QPP_FEATURE_INDICES="0 1 2 3 4 5 6 7 8"
INPUT_PATH=./dataset
QPP_FEATURE_STR=$(echo $QPP_FEATURE_INDICES | tr -d ' ')
LOG_PATH="./logs/qpp4sip_${PATTERN}_${QPP_FEATURE_STR}"
CHECKPOINT_PATH="/mnt/nas_syno/daiki/Models/SIP/qpp4sip_${PATTERN}_${QPP_FEATURE_STR}/v2.0/checkpoints"
OUTPUT_PATH="/mnt/nas_syno/daiki/Models/SIP/qpp4sip_${PATTERN}_${QPP_FEATURE_STR}/v2.0/outputs"

# 複数シード推論用のパラメータ
NUM_RUNS=5

# 必要なディレクトリを作成
echo "Creating necessary directories..."
mkdir -p $LOG_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $OUTPUT_PATH
echo "Directories created successfully."

# 複数シード推論を実行
echo "Starting multi-seed inference..."
uv run python run.py --mode multi_seed_inference --model qpp4sip --qpp4sip_pattern $PATTERN --qpp_feature_indices $QPP_FEATURE_INDICES --input_path $INPUT_PATH/bm25_dev.pkl --saved_model_path $CHECKPOINT_PATH --output_path $OUTPUT_PATH --epoch_num $EPOCH_NUM --num_runs $NUM_RUNS > $LOG_PATH/multi_seed_inference.log 2>&1
if [ $? -eq 0 ]; then
    echo "Multi-seed inference completed successfully."
    echo "Individual results saved to: $OUTPUT_PATH/dev_*_*.txt"
    
    # 結果を集計
    echo "Aggregating results..."
    uv run python scripts/aggregate_multi_seed.py --output_dir $OUTPUT_PATH --num_runs $NUM_RUNS --target_epoch 1 > $LOG_PATH/aggregation.log 2>&1
    if [ $? -eq 0 ]; then
        echo "Aggregation completed successfully."
        echo "Aggregated results saved to: $OUTPUT_PATH/multi_seed_aggregated_epoch_*.txt"
    else
        echo "Aggregation failed. Check $LOG_PATH/aggregation.log for details."
        exit 1
    fi
else
    echo "Multi-seed inference failed. Check $LOG_PATH/multi_seed_inference.log for details."
    exit 1
fi
