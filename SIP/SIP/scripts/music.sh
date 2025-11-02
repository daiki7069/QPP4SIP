#!/bin/bash

EPOCH_NUM=20
MODEL_NAME=music
QPP_FEATURE_NAME=f1@5
INPUT_DIR=/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT
INPUT_DEV=dpr_dev.json
INPUT_TRAIN=dpr_train.json
LOG_PATH=./logs/${MODEL_NAME}
CHECKPOINT_PATH=./checkpoints/${MODEL_NAME}
OUTPUT_PATH=./output/${MODEL_NAME}

# 必要なディレクトリを作成
echo "Creating necessary directories..."
mkdir -p $LOG_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $OUTPUT_PATH
echo "Directories created successfully."

# 学習
echo "Starting training..."
uv run python run.py --mode train --model music --qpp_feature_name $QPP_FEATURE_NAME --input_path $INPUT_DIR/$INPUT_TRAIN --epoch_num $EPOCH_NUM > $LOG_PATH/train.log 2>&1
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed. Check $LOG_PATH/train.log for details."
    exit 1
fi

# 推論
echo "Starting inference..."
uv run python run.py --mode inference --model music --qpp_feature_name $QPP_FEATURE_NAME --input_path $INPUT_DIR/$INPUT_DEV --saved_model_path $CHECKPOINT_PATH --epoch_num $EPOCH_NUM > $LOG_PATH/inference.log 2>&1
if [ $? -eq 0 ]; then
    echo "Inference completed successfully."
else
    echo "Inference failed. Check $LOG_PATH/inference.log for details."
    exit 1
fi

# 評価
echo "Starting evaluation..."
uv run python run.py --mode evaluation --model music --qpp_feature_name $QPP_FEATURE_NAME --input_path $INPUT_DIR/$INPUT_DEV --output_path $OUTPUT_PATH/result --epoch_num $EPOCH_NUM > $LOG_PATH/evaluation.log 2>&1
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully."
    echo "All processes completed successfully!"
else
    echo "Evaluation failed. Check $LOG_PATH/evaluation.log for details."
    exit 1
fi