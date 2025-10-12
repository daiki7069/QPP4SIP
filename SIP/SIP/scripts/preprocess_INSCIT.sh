#!/bin/bash

INPUT_FILE_DEV=/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/retrieval_outputs_own/results/dpr_dev.json
INPUT_FILE_TRAIN=/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/retrieval_outputs_own/results/dpr_train.json
ORIGINAL_DATA_FILE_DEV=/mnt/nas_syno/daiki/Datasets/INSCIT/data/dev.json
ORIGINAL_DATA_FILE_TRAIN=/mnt/nas_syno/daiki/Datasets/INSCIT/data/train.json
OUTPUT_DIR=../dataset/INSCIT

# dpr_dev.jsonの処理
uv run python dataset/preprocess_INSCIT.py --input_file $INPUT_FILE_DEV --original_data_file $ORIGINAL_DATA_FILE_DEV --output_dir $OUTPUT_DIR --save_json

echo "dpr_dev.jsonの処理が完了しました。"

# dpr_train.jsonの処理
uv run python dataset/preprocess_INSCIT.py --input_file $INPUT_FILE_TRAIN --original_data_file $ORIGINAL_DATA_FILE_TRAIN --output_dir $OUTPUT_DIR --save_json

echo "dpr_train.jsonの処理が完了しました。"
