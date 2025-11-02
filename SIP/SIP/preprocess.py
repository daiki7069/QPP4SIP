#!/usr/bin/env python3
"""
データ前処理のエントリポイント
PKLファイルの生成、データセットの準備を行う
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

# パスを追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.load_pkl import convert_dialogue_to_conversations
from model.dataset import Dataset
from config.dataset_config import Config
from sklearn.preprocessing import MultiLabelBinarizer


def prepare_dataset(pkl_path, dataset_type="train", max_utterance_len=128, max_context_len=384):
    """
    データセットを準備（データの読み込みと検証）
    
    Args:
        pkl_path (str): PKLファイルのパス
        dataset_type (str): データセットタイプ
        max_utterance_len (int): 最大発話長
        max_context_len (int): 最大コンテキスト長
        
    Returns:
        tuple: (conversations, dataset_info)
    """
    print(f"=== データセット準備 ===")
    print(f"PKLファイル: {pkl_path}")
    print(f"データセットタイプ: {dataset_type}")
    
    # PKLファイルを読み込み
    with open(pkl_path, 'rb') as f:
        dialogue_data = pickle.load(f)
    
    # 会話データに変換
    conversations = convert_dialogue_to_conversations(dialogue_data)
    
    # データセット情報を収集
    dataset_info = {
        "dataset_type": dataset_type,
        "total_conversations": len(conversations),
        "total_turns": sum(len(conv['turns']) for conv in conversations),
        "max_utterance_len": max_utterance_len,
        "max_context_len": max_context_len,
        "pkl_file": pkl_path
    }
    
    print(f"会話数: {dataset_info['total_conversations']}")
    print(f"総ターン数: {dataset_info['total_turns']}")
    
    return conversations, dataset_info


def validate_dataset(conversations, dataset_type="train"):
    """
    データセットの妥当性を検証
    
    Args:
        conversations (list): 会話データのリスト
        dataset_type (str): データセットタイプ
        
    Returns:
        bool: 検証結果
    """
    print(f"=== データセット検証 ===")
    
    if not conversations:
        print("エラー: 会話データが空です")
        return False
    
    # 基本的な検証
    total_turns = 0
    for conv in conversations:
        if 'turns' not in conv:
            print("エラー: 会話にturnsがありません")
            return False
        total_turns += len(conv['turns'])
    
    print(f"検証完了: {len(conversations)}会話, {total_turns}ターン")
    return True


def save_dataset_info(dataset_info, output_dir):
    """
    データセット情報をJSONファイルに保存
    
    Args:
        dataset_info (dict): データセット情報
        output_dir (str): 出力ディレクトリ
    """
    import json
    
    os.makedirs(output_dir, exist_ok=True)
    info_file = os.path.join(output_dir, f"{dataset_info['dataset_type']}_info.json")
    
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"データセット情報を保存しました: {info_file}")


def main():
    parser = argparse.ArgumentParser(description="データ前処理")
    parser.add_argument("--input_pkl", type=str, help="入力PKLファイルのパス")
    parser.add_argument("--output_dir", type=str, default="./dataset", help="出力ディレクトリ")
    parser.add_argument("--dataset_type", type=str, default="train", 
                       choices=["train", "dev", "test"], help="データセットタイプ")
    parser.add_argument("--max_utterance_len", type=int, default=128, help="最大発話長")
    parser.add_argument("--max_context_len", type=int, default=384, help="最大コンテキスト長")
    parser.add_argument("--prepare_dataset", action="store_true", help="データセットを準備")
    parser.add_argument("--validate_dataset", action="store_true", help="データセットを検証")
    parser.add_argument("--all", action="store_true", help="すべての処理を実行")
    
    args = parser.parse_args()
    
    print(f"データセットタイプ: {args.dataset_type}")
    
    if args.prepare_dataset or args.all:
        if not args.input_pkl:
            print("エラー: データセット準備には --input_pkl が必要です")
            return
        
        conversations, dataset_info = prepare_dataset(
            args.input_pkl, 
            args.dataset_type, 
            args.max_utterance_len, 
            args.max_context_len
        )
        
        # データセット情報を保存
        save_dataset_info(dataset_info, args.output_dir)
        
        # 検証も実行
        if not validate_dataset(conversations, args.dataset_type):
            print("エラー: データセット検証に失敗しました")
            return
        
        print("データセット準備完了")
    
    elif args.validate_dataset:
        if not args.input_pkl:
            print("エラー: データセット検証には --input_pkl が必要です")
            return
        
        conversations, _ = prepare_dataset(args.input_pkl, args.dataset_type)
        if not validate_dataset(conversations, args.dataset_type):
            print("エラー: データセット検証に失敗しました")
            return
        
        print("データセット検証完了")


if __name__ == "__main__":
    main()