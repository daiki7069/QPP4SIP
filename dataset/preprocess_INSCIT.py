#!/usr/bin/env python3
"""
INSCITデータセットをdataset/train.json形式に変換するスクリプト
元データ: /mnt/nas_syno/daiki/Datasets/INSCIT/data/train.json (辞書形式)
完成形: dataset/INSCIT/train.json (会話のリストのリスト形式)
"""

import argparse
import json
from typing import List, Dict, Any
from pathlib import Path


def convert_inscit_to_format(inscit_data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    元のINSCITデータ（辞書形式）を会話のリストのリスト形式に変換
    
    元データ形式:
    {
        "conv_id": {
            "seedArticle": {...},
            "turns": [
                {
                    "context": ["query1", "response1", "query2", ...],
                    "prevEvidence": [...],
                    "labels": [
                        {
                            "responseType": "directAnswer",
                            "response": "answer text",
                            "evidence": [
                                {
                                    "passage_id": "Beer bottle:36",
                                    "passage_text": "...",
                                    "passage_titles": [...]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }
    
    完成形:
    [
        [
            {
                "conv_id": "food_level3_dial26",
                "turn_id": 1,
                "query": "How does ultraviolet spoil beer?",
                "answer": ["answer1", "answer2"],
                "response_type": "directAnswer [SEP] directAnswer",
                "dialogue_history": ["query1", "response1", "query2", ...]
            }
        ]
    ]
    
    Args:
        inscit_data: 元のINSCITデータ（辞書形式）
        
    Returns:
        会話のリストのリスト形式のデータ
    """
    conversations: List[List[Dict[str, Any]]] = []
    
    print(f"[INFO] Converting {len(inscit_data)} conversations...")
    
    for conv_id, conv_data in inscit_data.items():
        turns = conv_data.get("turns", [])
        conversation: List[Dict[str, Any]] = []
        
        for turn_idx, turn in enumerate(turns, start=1):
            # contextからqueryを取得（最後の要素が現在のクエリ）
            context = turn.get("context", [])
            if not context:
                continue
            
            query = context[-1]  # 最後の要素が現在のクエリ
            
            # labelsからanswerとresponse_typeを抽出
            labels = turn.get("labels", [])
            answers = []
            response_types = []
            
            for label in labels:
                response = label.get("response", "")
                if response:
                    answers.append(response)
                
                response_type = label.get("responseType", "")
                if response_type:
                    response_types.append(response_type)
            
            # response_typeを結合（複数ある場合は " [SEP] " で結合）
            response_type_joined = " [SEP] ".join(response_types) if response_types else ""
            
            # dialogue_historyはcontextをそのまま使用
            dialogue_history = context.copy()
            
            turn_dict = {
                "conv_id": conv_id,
                "turn_id": turn_idx,
                "query": query,
                "answer": answers,
                "response_type": response_type_joined,
                "dialogue_history": dialogue_history
            }
            
            conversation.append(turn_dict)
        
        if conversation:
            conversations.append(conversation)
    
    print(f"[INFO] Conversion complete:")
    print(f"  - Total conversations: {len(conversations)}")
    print(f"  - Total turns: {sum(len(conv) for conv in conversations)}")
    
    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Convert original INSCIT dataset to dataset/train.json format"
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "train", "test"],
        help="データセットのスプリット（dev, train, またはtest）"
    )
    parser.add_argument(
        "--input_dir",
        default="/mnt/nas_syno/daiki/Datasets/INSCIT/data",
        help="元のINSCITデータセットのディレクトリ"
    )
    parser.add_argument(
        "--output_dir",
        default="./INSCIT",
        help="出力ディレクトリ"
    )
    
    args = parser.parse_args()
    
    # 入力ファイルパス
    input_file = Path(args.input_dir) / f"{args.split}.json"
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データを読み込む
    print(f"[INFO] Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        inscit_data = json.load(f)
    
    print(f"[INFO] Loaded {len(inscit_data)} conversations")
    
    # 形式変換
    converted_data = convert_inscit_to_format(inscit_data)
    
    # 出力ファイル名
    output_file = output_dir / f"{args.split}.json"
    
    # JSONファイルに保存
    print(f"[INFO] Writing output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] Successfully created {output_file}")
    print(f"  - Total conversations: {len(converted_data)}")
    print(f"  - Total turns: {sum(len(conv) for conv in converted_data)}")


if __name__ == "__main__":
    main()

