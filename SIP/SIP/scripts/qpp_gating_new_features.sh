#!/bin/bash

EPOCH_NUM=20
MODEL_NAME=qpp_gating
QPP_FEATURE_NAME=ndcg@5
RANDOM_SEED=42
INPUT_DIR=/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT
# JSONファイルを使用（.pklファイルでも動作します）
INPUT_DEV=dev_dpr_with_prev.json
INPUT_TRAIN=train_dpr_with_prev.json

LOG_PATH=./logs/${MODEL_NAME}_new_features/${QPP_FEATURE_NAME}
CHECKPOINT_PATH=./checkpoints/${MODEL_NAME}_new_features/${QPP_FEATURE_NAME}
OUTPUT_PATH=./output/${MODEL_NAME}_new_features/${QPP_FEATURE_NAME}

# 必要なディレクトリを作成
echo "Creating necessary directories..."
mkdir -p $LOG_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $OUTPUT_PATH
echo "Directories created successfully."

# 学習
echo "Starting training with new features..."
uv run python run.py \
  --mode train \
  --model qpp_gating \
  --qpp_feature_name $QPP_FEATURE_NAME \
  --input_path $INPUT_DIR/$INPUT_TRAIN \
  --saved_model_path $CHECKPOINT_PATH \
  --log_path $LOG_PATH \
  --epoch_num $EPOCH_NUM \
  > $LOG_PATH/train.log 2>&1

TRAIN_EXIT_CODE=$?
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully."
else
    echo "Training failed. Check $LOG_PATH/train.log for details."
    exit 1
fi

# 推論
echo "Starting inference with new features..."
uv run python run.py \
  --mode inference \
  --model qpp_gating \
  --qpp_feature_name $QPP_FEATURE_NAME \
  --input_path $INPUT_DIR/$INPUT_DEV \
  --saved_model_path $CHECKPOINT_PATH \
  --output_path $OUTPUT_PATH \
  --log_path $LOG_PATH \
  --epoch_num $EPOCH_NUM \
  --random_seed $RANDOM_SEED \
  > $LOG_PATH/inference.log 2>&1

INFERENCE_EXIT_CODE=$?
if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo "Inference completed successfully."
else
    echo "Inference failed. Check $LOG_PATH/inference.log for details."
    exit 1
fi

# 評価
echo "Starting evaluation with new features..."
uv run python run.py \
  --mode evaluation \
  --model qpp_gating \
  --qpp_feature_name $QPP_FEATURE_NAME \
  --input_path $INPUT_DIR/$INPUT_DEV \
  --output_path $OUTPUT_PATH \
  --log_path $LOG_PATH \
  --epoch_num $EPOCH_NUM \
  --random_seed $RANDOM_SEED \
  > $LOG_PATH/evaluation.log 2>&1

EVAL_EXIT_CODE=$?
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "Evaluation completed successfully."
    echo "All processes completed successfully!"
else
    echo "Evaluation failed. Check $LOG_PATH/evaluation.log for details."
    exit 1
fi

