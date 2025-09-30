#!/bin/bash

# TSVファイルを使用したSIP学習スクリプト
# 使用方法: ./run_tsv_training.sh [train_tsv] [dev_tsv] [model_type] [pattern]

# 引数の設定
TRAIN_TSV=${1:-"/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/bm25_train.tsv"}
DEV_TSV=${2:-"/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/bm25_dev.tsv"}
MODEL_TYPE=${3:-"qpp4sip"}
PATTERN=${4:-"feature_fusion"}

echo "=== TSVファイルを使用したSIP学習 ==="
echo "訓練データ: $TRAIN_TSV"
echo "開発データ: $DEV_TSV"
echo "モデルタイプ: $MODEL_TYPE"
echo "パターン: $PATTERN"
echo "================================"

# 学習の実行
uv run python model/run.py \
    --mode train \
    --task SIP \
    --name QPP4SIP \
    --dataset QPP4SIP \
    --model $MODEL_TYPE \
    --qpp4sip_pattern $PATTERN \
    --input_path $TRAIN_TSV \
    --output_path ./output/qpp4sip_${PATTERN} \
    --saved_model_path ./checkpoints/qpp4sip_${PATTERN} \
    --log_path ./logs/qpp4sip_${PATTERN} \
    --hidden_size 768 \
    --dropout 0.1 \
    --BiLSTM_layers 1 \
    --epoch_num 20 \
    --batch_size 1 \
    --learning_rate 2e-5 \
    --lr_crf 1e-3 \
    --accumulation_steps 1 \
    --clip 1.0 \
    --max_utterance_len 128 \
    --max_context_len 384 \
    --random_seed 42 \
    --class_imbalance_ratio 6.5 \
    --focal_gamma 2.0

echo "学習が完了しました。"
