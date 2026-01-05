#!/bin/bash
# ClariQデータセットでBERTとRoBERTaモデルを学習するスクリプト

set -e

cd /home/daiki_shibata/pj/QPP4SIP/SIP/FT-PLM

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "ClariQデータセットでBERT/RoBERTaモデルを学習"
echo "=========================================="
echo "ログファイル: $LOG_DIR/train_clariq_bert.log および train_clariq_roberta.log"
echo ""

# BERTモデルの学習（K-fold交差検証）
echo "=== BERTモデルの学習（K-fold=5） ==="
echo "ログ: $LOG_DIR/train_clariq_bert.log"
nohup python main.py --mode train \
    --dataset ClariQ \
    --model_name bert-base-uncased \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --k_fold 5 \
    --use_class_weights \
    --seed 42 \
    --output_dir ./output \
    > "$LOG_DIR/train_clariq_bert.log" 2>&1 &

BERT_PID=$!
echo "BERT学習プロセスID: $BERT_PID"
echo ""

# RoBERTaモデルの学習（K-fold交差検証、Early Stopping）
echo "=== RoBERTaモデルの学習（K-fold=5, Early Stopping） ==="
echo "ログ: $LOG_DIR/train_clariq_roberta.log"
nohup python main.py --mode train \
    --dataset ClariQ \
    --model_name roberta-base \
    --num_epochs 20 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --k_fold 5 \
    --early_stopping \
    --use_class_weights \
    --seed 42 \
    --output_dir ./output \
    > "$LOG_DIR/train_clariq_roberta.log" 2>&1 &

ROBERTA_PID=$!
echo "RoBERTa学習プロセスID: $ROBERTA_PID"
echo ""

echo "=========================================="
echo "学習を開始しました（バックグラウンド実行）"
echo "=========================================="
echo "BERTプロセスID: $BERT_PID"
echo "RoBERTaプロセスID: $ROBERTA_PID"
echo ""
echo "進捗確認:"
echo "  tail -f $LOG_DIR/train_clariq_bert.log"
echo "  tail -f $LOG_DIR/train_clariq_roberta.log"
echo ""
echo "プロセス確認:"
echo "  ps aux | grep $BERT_PID"
echo "  ps aux | grep $ROBERTA_PID"

