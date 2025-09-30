#!/bin/bash

# TSVファイルをPKLに変換してから学習・推論を実行する統合スクリプト
# 使用方法: ./run_with_tsv_conversion.sh [mode] [train_tsv] [dev_tsv] [test_tsv] [model_type] [pattern]

# 引数の設定
MODE=${1:-"train"}
TRAIN_TSV=${2:-"/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/bm25_train.tsv"}
DEV_TSV=${3:-"/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/bm25_dev.tsv"}
TEST_TSV=${4:-""}
MODEL_TYPE=${5:-"qpp4sip"}
PATTERN=${6:-"feature_fusion"}

# 出力ディレクトリの設定
PKL_DIR="./dataset/pkl"
OUTPUT_DIR="./output/qpp4sip_${PATTERN}"
CHECKPOINT_DIR="./checkpoints/qpp4sip_${PATTERN}"
LOG_DIR="./logs/qpp4sip_${PATTERN}"

echo "=== TSV変換 + SIP処理統合スクリプト ==="
echo "モード: $MODE"
echo "訓練データ: $TRAIN_TSV"
echo "開発データ: $DEV_TSV"
echo "テストデータ: $TEST_TSV"
echo "モデルタイプ: $MODEL_TYPE"
echo "パターン: $PATTERN"
echo "PKL出力ディレクトリ: $PKL_DIR"
echo "================================"

# 1. TSVファイルをPKLに変換
echo "Step 1: TSVファイルをPKLに変換中..."
./scripts/convert_tsv_to_pkl.sh "$TRAIN_TSV" "$DEV_TSV" "$TEST_TSV" "$PKL_DIR"

# 変換されたPKLファイルのパスを設定
TRAIN_PKL="$PKL_DIR/$(basename "$TRAIN_TSV" .tsv).pkl"
DEV_PKL="$PKL_DIR/$(basename "$DEV_TSV" .tsv).pkl"
TEST_PKL=""
if [ -n "$TEST_TSV" ]; then
    TEST_PKL="$PKL_DIR/$(basename "$TEST_TSV" .tsv).pkl"
fi

echo "変換されたPKLファイル:"
echo "  訓練: $TRAIN_PKL"
echo "  開発: $DEV_PKL"
if [ -n "$TEST_PKL" ]; then
    echo "  テスト: $TEST_PKL"
fi

# 2. 学習または推論を実行
if [ "$MODE" = "train" ]; then
    echo "Step 2: 学習を実行中..."
    uv run python model/run.py \
        --mode train \
        --task SIP \
        --name QPP4SIP \
        --dataset QPP4SIP \
        --model $MODEL_TYPE \
        --qpp4sip_pattern $PATTERN \
        --input_path $TRAIN_PKL \
        --output_path $OUTPUT_DIR \
        --saved_model_path $CHECKPOINT_DIR \
        --log_path $LOG_DIR \
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

elif [ "$MODE" = "inference" ]; then
    # 推論用のデータを選択（テストデータがあればそれを使用、なければ開発データを使用）
    INFERENCE_PKL=$DEV_PKL
    if [ -n "$TEST_PKL" ] && [ -f "$TEST_PKL" ]; then
        INFERENCE_PKL=$TEST_PKL
    fi
    
    echo "Step 2: 推論を実行中..."
    echo "推論データ: $INFERENCE_PKL"
    
    uv run python model/run.py \
        --mode inference \
        --task SIP \
        --name QPP4SIP \
        --dataset QPP4SIP \
        --model $MODEL_TYPE \
        --qpp4sip_pattern $PATTERN \
        --input_path $INFERENCE_PKL \
        --output_path $OUTPUT_DIR \
        --saved_model_path $CHECKPOINT_DIR \
        --log_path $LOG_DIR \
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

else
    echo "エラー: サポートされていないモード: $MODE"
    echo "使用可能なモード: train, inference"
    exit 1
fi

echo "処理が完了しました。"
echo "PKLファイルは $PKL_DIR に保存されています。"
