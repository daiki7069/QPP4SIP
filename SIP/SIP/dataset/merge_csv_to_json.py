#!/usr/bin/env python3
"""
CSVファイルをベースJSONにマージして新しいJSONファイルを生成するスクリプト

使い方:
    python merge_csv_to_json.py \
        --base_json dataset/INSCIT/dev.json \
        --csv_file outputs/dev_lci.csv \
        --output_json dataset/INSCIT/dev_lci.json
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Tuple, Optional


def load_base_json(json_path: str) -> List[List[Dict[str, Any]]]:
    """
    ベースJSONファイルを読み込む
    
    Args:
        json_path: JSONファイルのパス
        
    Returns:
        会話のリスト（各会話はターンのリスト）
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"ベースJSONファイルが見つかりません: {json_path}")
    
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    # 期待構造: [[{turn...}, ...], ...] ただし単一会話のフラットな可能性もある
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return [data]
    return data


def load_csv_data(csv_path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    CSVファイルを読み込んで(conv_id, turn_id)をキーとする辞書に変換
    
    Args:
        csv_path: CSVファイルのパス
        
    Returns:
        (conv_id, turn_id)をキーとし、CSVの列（conv_id, turn_id以外）を値とする辞書
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
    
    key_to_data: Dict[Tuple[str, int], Dict[str, Any]] = {}
    
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # 必須列の確認
        if "conv_id" not in reader.fieldnames:
            raise ValueError("CSVファイルに'conv_id'列がありません")
        if "turn_id" not in reader.fieldnames:
            raise ValueError("CSVファイルに'turn_id'列がありません")
        
        # conv_id, turn_id以外の列を取得
        data_columns = [col for col in reader.fieldnames if col not in ("conv_id", "turn_id")]
        
        if not data_columns:
            raise ValueError("CSVファイルにconv_id, turn_id以外の列がありません")
        
        print(f"CSV列: {', '.join(data_columns)}")
        
        for row_num, row in enumerate(reader, start=2):  # ヘッダー行を除いて2から開始
            try:
                conv_id = row["conv_id"].strip()
                turn_id = int(row["turn_id"])
            except (KeyError, ValueError) as e:
                print(f"警告: 行{row_num}でconv_idまたはturn_idの読み込みに失敗しました: {e}")
                continue
            
            # データ列の値を適切な型に変換
            data_dict: Dict[str, Any] = {}
            for col in data_columns:
                val = row[col].strip()
                
                # 空文字列の処理
                if val == "":
                    data_dict[col] = None
                    continue
                
                # 数値型の判定と変換
                try:
                    # 整数として解釈可能か試す
                    if "." not in val and "e" not in val.lower():
                        data_dict[col] = int(val)
                    else:
                        data_dict[col] = float(val)
                except ValueError:
                    # 数値でない場合は文字列として扱う
                    data_dict[col] = val
            
            key_to_data[(conv_id, turn_id)] = data_dict
    
    print(f"CSVから{len(key_to_data)}件のデータを読み込みました")
    return key_to_data


def merge_csv_to_json(
    conversations: List[List[Dict[str, Any]]],
    csv_data: Dict[Tuple[str, int], Dict[str, Any]],
    default_value: Optional[Any] = None
) -> Tuple[List[List[Dict[str, Any]]], int, int]:
    """
    ベースJSONとCSVデータをマージ
    
    Args:
        conversations: ベースJSONの会話データ
        csv_data: CSVから読み込んだデータ（(conv_id, turn_id)をキーとする辞書）
        default_value: マッチしない場合のデフォルト値（Noneの場合はスキップ）
        
    Returns:
        (マージ後の会話データ, マッチしたターン数, マッチしなかったターン数)
    """
    matched_count = 0
    unmatched_count = 0
    
    for conv in conversations:
        for turn in conv:
            conv_id = str(turn.get("conv_id", "")).strip()
            try:
                turn_id = int(turn.get("turn_id", 0))
            except (ValueError, TypeError):
                print(f"警告: turn_idが無効です (conv_id={conv_id}, turn_id={turn.get('turn_id')})")
                unmatched_count += 1
                continue
            
            key = (conv_id, turn_id)
            csv_row = csv_data.get(key)
            
            if csv_row is not None:
                # CSVデータをターンに追加
                for k, v in csv_row.items():
                    turn[k] = v
                matched_count += 1
            else:
                # マッチしない場合
                if default_value is not None:
                    # デフォルト値を設定（CSVの列名が必要なので、最初の行から取得）
                    if csv_data:
                        first_row = next(iter(csv_data.values()))
                        for k in first_row.keys():
                            turn[k] = default_value
                unmatched_count += 1
    
    return conversations, matched_count, unmatched_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSVファイルをベースJSONにマージして新しいJSONファイルを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python merge_csv_to_json.py \\
    --base_json dataset/INSCIT/dev.json \\
    --csv_file outputs/dev_lci.csv \\
    --output_json dataset/INSCIT/dev_lci.json

  python merge_csv_to_json.py \\
    --base_json dataset/INSCIT/dev.json \\
    --csv_file outputs/dev_entropy.csv \\
    --output_json dataset/INSCIT/dev_entropy.json \\
    --default_value 0.0
        """
    )
    
    parser.add_argument(
        "--base_json",
        type=str,
        required=True,
        help="ベースJSONファイルのパス（例: dataset/INSCIT/dev.json）"
    )
    parser.add_argument(
        "--csv_file",
        type=str,
        required=True,
        help="入力CSVファイルのパス（例: outputs/dev_lci.csv）"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="出力JSONファイルのパス（例: dataset/INSCIT/dev_lci.json）"
    )
    parser.add_argument(
        "--default_value",
        type=str,
        default=None,
        help="マッチしない場合のデフォルト値（オプション、例: 0.0, None）"
    )
    
    args = parser.parse_args()
    
    # デフォルト値の型変換
    default_value: Optional[Any] = None
    if args.default_value is not None:
        if args.default_value.lower() == "none":
            default_value = None
        else:
            # 数値として解釈を試す
            try:
                if "." in args.default_value or "e" in args.default_value.lower():
                    default_value = float(args.default_value)
                else:
                    default_value = int(args.default_value)
            except ValueError:
                default_value = args.default_value
    
    try:
        # ベースJSONの読み込み
        print(f"ベースJSONを読み込み中: {args.base_json}")
        conversations = load_base_json(args.base_json)
        num_conversations = len(conversations)
        num_turns = sum(len(c) for c in conversations)
        print(f"  会話数: {num_conversations}, ターン数: {num_turns}")
        
        # CSVの読み込み
        print(f"CSVファイルを読み込み中: {args.csv_file}")
        csv_data = load_csv_data(args.csv_file)
        
        # マージ処理
        print("マージ処理を実行中...")
        merged_conversations, matched_count, unmatched_count = merge_csv_to_json(
            conversations,
            csv_data,
            default_value=default_value
        )
        
        # 出力ディレクトリの作成
        output_dir = os.path.dirname(args.output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"出力ディレクトリを作成しました: {output_dir}")
        
        # JSONファイルの出力
        print(f"JSONファイルを出力中: {args.output_json}")
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(merged_conversations, f, ensure_ascii=False, indent=2)
        
        # 統計情報の表示
        print("\n=== 処理結果 ===")
        print(f"ベースJSON: {args.base_json}")
        print(f"  - 会話数: {num_conversations}")
        print(f"  - ターン数: {num_turns}")
        print(f"CSVファイル: {args.csv_file}")
        print(f"  - CSV行数: {len(csv_data)}")
        print(f"マッチング結果:")
        print(f"  - マッチしたターン: {matched_count}")
        print(f"  - マッチしなかったターン: {unmatched_count}")
        if unmatched_count > 0 and default_value is None:
            print(f"  - 警告: {unmatched_count}件のターンにCSVデータがありません")
        print(f"出力ファイル: {args.output_json}")
        print("処理が完了しました。")
        
    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"予期しないエラーが発生しました: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

