#!/bin/bash

# QPP4SIP実験評価スクリプト
# 3つのパターンに対してevaluationを実行

set -e  # エラー時に停止

# 基本設定
BASE_DIR="/home/daiki_shibata/pj/QPP4SIP/SIP/SIP"
OUTPUT_BASE="${BASE_DIR}/output"
LOG_BASE="${BASE_DIR}/logs"

# データファイル
DEV_DATA="${BASE_DIR}/dataset/dev_resolved_retrieved.pkl"
TEST_DATA="${BASE_DIR}/dataset/test_resolved_retrieved.pkl"

# 実験設定
EPOCHS=20

# パターンリスト
PATTERNS=("feature_fusion" "auxiliary_head" "policy_gating")

echo "=== QPP4SIP実験評価開始 ==="
echo "開始時刻: $(date)"
echo "評価パターン: ${PATTERNS[@]}"
echo "エポック数: ${EPOCHS}"
echo ""

# 各パターンに対して評価を実行
for pattern in "${PATTERNS[@]}"; do
    echo "=========================================="
    echo "パターン: ${pattern} の評価を開始"
    echo "開始時刻: $(date)"
    echo "=========================================="
    
    # パターン専用のディレクトリ
    OUTPUT_DIR="${OUTPUT_BASE}/qpp4sip_${pattern}"
    LOG_DIR="${LOG_BASE}/qpp4sip_${pattern}"
    
    # ログディレクトリを作成
    mkdir -p "${LOG_DIR}"
    
    # 1. 開発データでの評価 (Development Evaluation)
    echo "--- 開発データ評価開始: ${pattern} ---"
    nohup conda run -n sip uv run python "${BASE_DIR}/evaluation.py" \
        --prediction_path "${OUTPUT_DIR}" \
        --label_path "${DEV_DATA}" \
        --task SIP \
        --dataset_type dev \
        --epoch_num "${EPOCHS}" \
        --dataset QPP4SIP \
        > "${LOG_DIR}/evaluation_dev_${pattern}.log" 2>&1 &
    
    DEV_EVAL_PID=$!
    echo "開発データ評価プロセス開始 (PID: ${DEV_EVAL_PID})"
    
    # 2. テストデータでの評価 (Test Evaluation)
    echo "--- テストデータ評価開始: ${pattern} ---"
    nohup conda run -n sip uv run python "${BASE_DIR}/evaluation.py" \
        --prediction_path "${OUTPUT_DIR}" \
        --label_path "${TEST_DATA}" \
        --task SIP \
        --dataset_type test \
        --epoch_num "${EPOCHS}" \
        --dataset QPP4SIP \
        > "${LOG_DIR}/evaluation_test_${pattern}.log" 2>&1 &
    
    TEST_EVAL_PID=$!
    echo "テストデータ評価プロセス開始 (PID: ${TEST_EVAL_PID})"
    
    # 評価完了を待つ
    wait ${DEV_EVAL_PID}
    DEV_EVAL_EXIT_CODE=$?
    wait ${TEST_EVAL_PID}
    TEST_EVAL_EXIT_CODE=$?
    
    if [ ${DEV_EVAL_EXIT_CODE} -eq 0 ]; then
        echo "✓ 開発データ評価完了: ${pattern}"
    else
        echo "✗ 開発データ評価失敗: ${pattern} (終了コード: ${DEV_EVAL_EXIT_CODE})"
    fi
    
    if [ ${TEST_EVAL_EXIT_CODE} -eq 0 ]; then
        echo "✓ テストデータ評価完了: ${pattern}"
    else
        echo "✗ テストデータ評価失敗: ${pattern} (終了コード: ${TEST_EVAL_EXIT_CODE})"
    fi
    
    echo "パターン ${pattern} の評価完了時刻: $(date)"
    echo ""
done

echo "=========================================="
echo "全評価完了時刻: $(date)"
echo "=========================================="

# 結果サマリーを表示
echo ""
echo "=== 評価結果サマリー ==="
for pattern in "${PATTERNS[@]}"; do
    OUTPUT_DIR="${OUTPUT_BASE}/qpp4sip_${pattern}"
    LOG_DIR="${LOG_BASE}/qpp4sip_${pattern}"
    
    echo ""
    echo "パターン: ${pattern}"
    echo "  出力ディレクトリ: ${OUTPUT_DIR}"
    echo "  ログディレクトリ: ${LOG_DIR}"
    
    # 評価結果ファイルを確認
    if [ -f "${OUTPUT_DIR}/result.dev.txt" ]; then
        echo "  開発データ評価結果:"
        echo "    $(wc -l < ${OUTPUT_DIR}/result.dev.txt) エポックの結果"
        echo "    最新の結果:"
        tail -3 "${OUTPUT_DIR}/result.dev.txt" | sed 's/^/      /'
    else
        echo "  ✗ 開発データ評価結果ファイルが見つかりません"
    fi
    
    if [ -f "${OUTPUT_DIR}/result.test.txt" ]; then
        echo "  テストデータ評価結果:"
        echo "    $(wc -l < ${OUTPUT_DIR}/result.test.txt) エポックの結果"
        echo "    最新の結果:"
        tail -3 "${OUTPUT_DIR}/result.test.txt" | sed 's/^/      /'
    else
        echo "  ✗ テストデータ評価結果ファイルが見つかりません"
    fi
done

echo ""
echo "評価完了！"
