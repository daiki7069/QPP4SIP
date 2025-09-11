import json
import pickle
import argparse
import os
from pathlib import Path


def convert_json_to_pkl(input_file, output_file):
    """JSONファイルをPKL形式に変換する"""
    print(f"Converting {input_file} to {output_file}")
    
    # JSONファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # PKL形式で保存
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Successfully converted {input_file} to {output_file}")


def main(args):
    if args.input_files:
        # 複数ファイルを処理
        for input_file in args.input_files:
            if os.path.exists(input_file):
                # 出力ファイル名を生成（拡張子を.pklに変更）
                input_path = Path(input_file)
                output_file = input_path.parent / f"{input_path.stem}.pkl"
                convert_json_to_pkl(input_file, str(output_file))
            else:
                print(f"Warning: File {input_file} does not exist")
    else:
        # 単一ファイルを処理（従来の動作）
        convert_json_to_pkl(args.input_file, args.output_file)


if __name__ == "__main__":
    """
    Usage examples:
    python preprocess.py --input_files file1.json file2.json file3.json
    python preprocess.py --input_file input.json --output_file output.pkl
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='input.json')
    parser.add_argument('--output_file', type=str, default='output.pkl')
    parser.add_argument('--input_files', nargs='+', help='List of input JSON files to convert')
    args = parser.parse_args()

    main(args)