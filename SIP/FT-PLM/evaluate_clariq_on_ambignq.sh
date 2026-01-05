#!/bin/bash
# ClariQで学習したモデルを使ってAmbigNQを予測するスクリプト

set -e

cd /home/daiki_shibata/pj/QPP4SIP/SIP/FT-PLM

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "ClariQで学習したモデルでAmbigNQを予測"
echo "=========================================="
echo ""

# BERTモデルでAmbigNQを予測
echo "=== BERTモデル（ClariQ学習）でAmbigNQを予測 ==="
echo "ログ: $LOG_DIR/evaluate_clariq_bert_on_ambignq.log"
nohup python main.py --mode transfer_evaluate \
    --dataset AmbigNQ \
    --train_dataset ClariQ \
    --experiment_name ClariQ_bert-base_lr2e-05_bs16_kfold5 \
    --batch_size 16 \
    --output_dir ./output \
    > "$LOG_DIR/evaluate_clariq_bert_on_ambignq.log" 2>&1 &

BERT_PID=$!
echo "BERT予測プロセスID: $BERT_PID"
echo ""

# RoBERTaモデルでAmbigNQを予測
echo "=== RoBERTaモデル（ClariQ学習）でAmbigNQを予測 ==="
echo "ログ: $LOG_DIR/evaluate_clariq_roberta_on_ambignq.log"
nohup python main.py --mode transfer_evaluate \
    --dataset AmbigNQ \
    --train_dataset ClariQ \
    --experiment_name ClariQ_roberta-base_lr2e-05_bs16_earlystop_kfold5 \
    --batch_size 16 \
    --output_dir ./output \
    > "$LOG_DIR/evaluate_clariq_roberta_on_ambignq.log" 2>&1 &

ROBERTA_PID=$!
echo "RoBERTa予測プロセスID: $ROBERTA_PID"
echo ""

echo "=========================================="
echo "予測を開始しました（バックグラウンド実行）"
echo "=========================================="
echo "BERTプロセスID: $BERT_PID"
echo "RoBERTaプロセスID: $ROBERTA_PID"
echo ""
echo "進捗確認:"
echo "  tail -f $LOG_DIR/evaluate_clariq_bert_on_ambignq.log"
echo "  tail -f $LOG_DIR/evaluate_clariq_roberta_on_ambignq.log"
echo ""
echo "プロセス確認:"
echo "  ps aux | grep $BERT_PID"
echo "  ps aux | grep $ROBERTA_PID"
echo ""
echo "予測結果は以下に保存されます:"
echo "  ./output/AmbigNQ/AmbigNQ_transfer_from_ClariQ/"

