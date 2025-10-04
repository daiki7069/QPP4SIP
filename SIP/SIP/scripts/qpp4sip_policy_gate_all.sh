#!/bin/bash

EPOCH_NUM=20
INPUT_PATH=./dataset
LOG_PATH=./logs/qpp4sip_policy_gating_012345678
CHECKPOINT_PATH=./checkpoints/qpp4sip_policy_gating_012345678
OUTPUT_PATH=./output/qpp4sip_policy_gating_012345678

# 必要なディレクトリを作成
echo "Creating necessary directories..."
mkdir -p $LOG_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $OUTPUT_PATH
echo "Directories created successfully."

# 学習
echo "Starting training..."
uv run python run.py --mode train --model qpp4sip --qpp4sip_pattern policy_gating --input_path $INPUT_PATH/bm25_train.pkl --epoch_num $EPOCH_NUM > $LOG_PATH/train.log 2>&1
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed. Check $LOG_PATH/train.log for details."
    exit 1
fi

# 推論
echo "Starting inference..."
uv run python run.py --mode inference --model qpp4sip --qpp4sip_pattern policy_gating --input_path $INPUT_PATH/bm25_dev.pkl --saved_model_path $CHECKPOINT_PATH --epoch_num $EPOCH_NUM > $LOG_PATH/inference.log 2>&1
if [ $? -eq 0 ]; then
    echo "Inference completed successfully."
else
    echo "Inference failed. Check $LOG_PATH/inference.log for details."
    exit 1
fi

# 評価
echo "Starting evaluation..."
uv run python run.py --mode evaluation --model qpp4sip --qpp4sip_pattern policy_gating --input_path $INPUT_PATH/bm25_dev.pkl --output_path $OUTPUT_PATH/result --epoch_num $EPOCH_NUM > $LOG_PATH/evaluation.log 2>&1
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
    echo "All processes completed successfully!"
else
    echo "Evaluation failed. Check $LOG_PATH/evaluation.log for details."
    exit 1
fi
