#!/bin/bash

# TSVファイルを使用したSIP推論スクリプト
# 使用方法: ./run_tsv_inference.sh [test_tsv] [model_type] [pattern]

# 引数の設定
TEST_TSV=${1:-"/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/bm25_dev.tsv"}
MODEL_TYPE=${2:-"qpp4sip"}
PATTERN=${3:-"feature_fusion"}

echo "=== TSVファイルを使用したSIP推論 ==="
echo "テストデータ: $TEST_TSV"
echo "モデルタイプ: $MODEL_TYPE"
echo "パターン: $PATTERN"
echo "================================"

# 推論の実行
uv runpython model/run.py \
    --mode inference \
    --task SIP \
    --name QPP4SIP \
    --dataset QPP4SIP \
    --model $MODEL_TYPE \
    --qpp4sip_pattern $PATTERN \
    --input_path $TEST_TSV \
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

echo "推論が完了しました。"
