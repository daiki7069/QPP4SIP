#!/bin/bash

# .envファイルを読み込む（存在する場合）
ENV_FILE="/home/daiki_shibata/pj/QPP4SIP/SIP/SIP/.env"
if [ -f "$ENV_FILE" ]; then
    set -a  # 自動的にexport
    source "$ENV_FILE"
    set +a
fi

EPOCH_NUM=20
MODEL_NAME=qpp_gating
QPP_FEATURE_NAME=lci
RANDOM_SEED=42
INPUT_DIR=/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT
# JSONファイルを使用（.pklファイルでも動作します）
INPUT_DEV=dev_lci.json
INPUT_TRAIN=train_lci.json

LOG_PATH=./logs/${MODEL_NAME}_new_features/${QPP_FEATURE_NAME}
CHECKPOINT_PATH=./checkpoints/${MODEL_NAME}_new_features/${QPP_FEATURE_NAME}
OUTPUT_PATH=./output/${MODEL_NAME}_new_features/${QPP_FEATURE_NAME}

# Slack通知関数
send_slack_notification() {
    local message="$1"
    local webhook_url="${SLACK_WEBHOOK_URL:-}"
    
    if [ -z "$webhook_url" ]; then
        return 0  # Webhook URLが設定されていない場合は何もしない
    fi
    
    local payload=$(cat <<EOF
{
    "text": "${message}"
}
EOF
)
    curl -s -X POST -H 'Content-type: application/json' --data "$payload" "$webhook_url" > /dev/null 2>&1 || true
}

# 必要なディレクトリを作成
echo "Creating necessary directories..."
mkdir -p $LOG_PATH
mkdir -p $CHECKPOINT_PATH
mkdir -p $OUTPUT_PATH
echo "Directories created successfully."

send_slack_notification "🚀 実験開始: ${MODEL_NAME} (${QPP_FEATURE_NAME})"

# 学習
echo "Starting training with new features..."
send_slack_notification "📚 学習開始: ${MODEL_NAME} (${QPP_FEATURE_NAME})"
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
    send_slack_notification "✅ 学習完了: ${MODEL_NAME} (${QPP_FEATURE_NAME})"
else
    echo "Training failed. Check $LOG_PATH/train.log for details."
    send_slack_notification "❌ 学習失敗: ${MODEL_NAME} (${QPP_FEATURE_NAME}) - $LOG_PATH/train.log を確認"
    exit 1
fi

# 推論
echo "Starting inference with new features..."
send_slack_notification "🔮 推論開始: ${MODEL_NAME} (${QPP_FEATURE_NAME})"
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
    send_slack_notification "✅ 推論完了: ${MODEL_NAME} (${QPP_FEATURE_NAME})"
else
    echo "Inference failed. Check $LOG_PATH/inference.log for details."
    send_slack_notification "❌ 推論失敗: ${MODEL_NAME} (${QPP_FEATURE_NAME}) - $LOG_PATH/inference.log を確認"
    exit 1
fi

# 評価
echo "Starting evaluation with new features..."
send_slack_notification "📊 評価開始: ${MODEL_NAME} (${QPP_FEATURE_NAME})"
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
    send_slack_notification "🎉 全工程完了: ${MODEL_NAME} (${QPP_FEATURE_NAME})"
else
    echo "Evaluation failed. Check $LOG_PATH/evaluation.log for details."
    send_slack_notification "❌ 評価失敗: ${MODEL_NAME} (${QPP_FEATURE_NAME}) - $LOG_PATH/evaluation.log を確認"
    exit 1
fi

