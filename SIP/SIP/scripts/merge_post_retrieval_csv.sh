#!/bin/bash

set -euo pipefail

# 絶対パスを直接指定
PYTHON_SCRIPT="/home/daiki_shibata/pj/QPP4SIP/SIP/SIP/dataset/merge_csv_to_json.py"
BASE_JSON_DEV="/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/dev.json"
BASE_JSON_TRAIN="/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT/train.json"
CSV_DIR="/home/daiki_shibata/pj/QPP4SIP/QPP/post_retrieval/outputs"
OUTPUT_DIR="/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT"

# 処理するCSVファイルのリスト（dev用）
CSV_FILES_DEV=(
    "dev_lci.csv"
    "dev_entropy.csv"
    "dev_similarity.csv"
    "dev_nqc.csv"
    "dev_unique_titles.csv"
)

# 処理するCSVファイルのリスト（train用、存在する場合）
CSV_FILES_TRAIN=(
    "train_lci.csv"
    "train_entropy.csv"
    "train_similarity.csv"
    "train_nqc.csv"
    "train_unique_titles.csv"
)

echo "=== Post-retrieval CSV to JSON 変換スクリプト ==="
echo ""

# devデータの処理
if [ -f "$BASE_JSON_DEV" ]; then
    echo "[INFO] Processing dev data..."
    for csv_file in "${CSV_FILES_DEV[@]}"; do
        csv_path="$CSV_DIR/$csv_file"
        if [ -f "$csv_path" ]; then
            # CSVファイル名から出力JSONファイル名を生成
            # dev_lci.csv -> dev_lci.json
            output_name="${csv_file%.csv}.json"
            output_json="$OUTPUT_DIR/$output_name"
            
            echo "  Processing: $csv_file -> $output_name"
            uv run python "$PYTHON_SCRIPT" \
                --base_json "$BASE_JSON_DEV" \
                --csv_file "$csv_path" \
                --output_json "$output_json"
            echo ""
        else
            echo "  [SKIP] CSVファイルが見つかりません: $csv_path"
        fi
    done
else
    echo "[WARN] ベースJSONファイルが見つかりません: $BASE_JSON_DEV"
fi

# trainデータの処理
if [ -f "$BASE_JSON_TRAIN" ]; then
    echo "[INFO] Processing train data..."
    for csv_file in "${CSV_FILES_TRAIN[@]}"; do
        csv_path="$CSV_DIR/$csv_file"
        if [ -f "$csv_path" ]; then
            # CSVファイル名から出力JSONファイル名を生成
            # train_lci.csv -> train_lci.json
            output_name="${csv_file%.csv}.json"
            output_json="$OUTPUT_DIR/$output_name"
            
            echo "  Processing: $csv_file -> $output_name"
            uv run python "$PYTHON_SCRIPT" \
                --base_json "$BASE_JSON_TRAIN" \
                --csv_file "$csv_path" \
                --output_json "$output_json"
            echo ""
        else
            echo "  [SKIP] CSVファイルが見つかりません: $csv_path"
        fi
    done
else
    echo "[WARN] ベースJSONファイルが見つかりません: $BASE_JSON_TRAIN"
fi

echo "[DONE] All finished."

