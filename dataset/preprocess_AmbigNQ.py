#!/usr/bin/env python3
"""
AmbigQAデータセットをINSCIT形式のdev.json/train.jsonに変換するスクリプト
"""

import argparse
import json
import os
from typing import List, Dict, Any
from pathlib import Path


def map_response_type(ambigqa_type: str) -> str:
    """
    AmbigQAのresponse typeをINSCITのresponse_typeにマッピング
    
    Args:
        ambigqa_type: AmbigQAのtype（例: "singleAnswer", "multipleQAs"）
        
    Returns:
        INSCIT形式のresponse_type
    """
    # AmbigQAのtypeをINSCITのresponse_typeにマッピング
    type_mapping = {
        "singleAnswer": "directAnswer",
        "multipleQAs": "clarification",  # multipleQAsはclarificationにマッピング
    }
    return type_mapping.get(ambigqa_type, "directAnswer")


def extract_answers_and_types(annotations: List[Dict[str, Any]]) -> tuple[List[str], List[str]]:
    """
    annotationsから回答とresponse_typeを抽出
    
    Args:
        annotations: AmbigQAのannotationsリスト
        
    Returns:
        (answers, response_types) のタプル
    """
    answers = []
    response_types = []
    
    for ann in annotations:
        # response_typeを取得
        ann_type = ann.get("type", "singleAnswer")
        mapped_type = map_response_type(ann_type)
        response_types.append(mapped_type)
        
        # answerを取得
        if ann_type == "multipleQAs":
            # multipleQAsの場合はqaPairsから回答を抽出
            if "qaPairs" in ann:
                for qa_pair in ann["qaPairs"]:
                    if "answer" in qa_pair:
                        if isinstance(qa_pair["answer"], list):
                            answers.extend(qa_pair["answer"])
                        else:
                            answers.append(qa_pair["answer"])
        elif "answer" in ann:
            # 通常の場合はanswerフィールドから取得
            if isinstance(ann["answer"], list):
                answers.extend(ann["answer"])
            else:
                answers.append(ann["answer"])
    
    # 重複除去（順序は保持）
    seen = set()
    unique_answers = []
    for ans in answers:
        if ans not in seen:
            seen.add(ans)
            unique_answers.append(ans)
    
    return unique_answers, response_types


def create_inscit_format(questions: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    AmbigQAデータをINSCIT形式に変換
    
    INSCIT形式:
    - 最上位は会話のリスト（各会話はターンのリスト）
    - 各ターンは以下のキーを持つ:
      - conv_id: 会話ID
      - turn_id: ターンID（整数、1始まり）
      - query: 質問文
      - answer: 回答のリスト
      - response_type: レスポンスタイプ（複数ある場合は " [SEP] " で結合）
      - dialogue_history: 対話履歴（文字列のリスト）
    
    Args:
        questions: AmbigQAの質問データのリスト
        
    Returns:
        INSCIT形式のデータ（会話のリストのリスト）
    """
    conversations: List[List[Dict[str, Any]]] = []
    
    print(f"[INFO] Converting {len(questions)} questions to INSCIT format...")
    
    for question in questions:
        # 質問IDを取得
        q_id = question.get("id", "")
        
        # 質問文
        q_text = question.get("question", "")
        
        # 回答とresponse_typeを抽出
        annotations = question.get("annotations", [])
        answers, response_types = extract_answers_and_types(annotations)
        
        # response_typeを結合（複数ある場合は " [SEP] " で結合）
        response_type_joined = " [SEP] ".join(response_types) if response_types else ""
        
        # 各質問を1つの会話として扱い、1ターンだけの会話を作成
        turn = {
            "conv_id": str(q_id),
            "turn_id": 1,  # 最初のターン
            "query": q_text,
            "answer": answers,
            "response_type": response_type_joined,
            "dialogue_history": [q_text]  # 質問文のみを含む対話履歴
        }
        
        # 1ターンだけの会話として追加
        conversations.append([turn])
    
    print(f"[INFO] Conversion complete:")
    print(f"  - Total conversations: {len(conversations)}")
    print(f"  - Total turns: {sum(len(conv) for conv in conversations)}")
    
    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Convert AmbigQA dataset to INSCIT format (dev.json/train.json)"
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "train"],
        help="データセットのスプリット（devまたはtrain）"
    )
    parser.add_argument(
        "--ambigqa_dir",
        default="/mnt/nas_syno/daiki/Datasets/AmbigQA",
        help="AmbigQAデータセットのルートディレクトリ"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ"
    )
    
    args = parser.parse_args()
    
    # ファイルパスを構築
    # まず codes/data/ambigqa/ を試し、なければ ambignq_light/ を使用
    ambigqa_dir = Path(args.ambigqa_dir)
    questions_file_1 = ambigqa_dir / "codes" / "data" / "ambigqa" / f"{args.split}_light.json"
    questions_file_2 = ambigqa_dir / "ambignq_light" / f"{args.split}_light.json"
    
    if questions_file_1.exists():
        questions_file = questions_file_1
    elif questions_file_2.exists():
        questions_file = questions_file_2
    else:
        raise FileNotFoundError(
            f"Questions file not found. Tried:\n"
            f"  - {questions_file_1}\n"
            f"  - {questions_file_2}"
        )
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データを読み込む
    print(f"[INFO] Loading questions from {questions_file}...")
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"[INFO] Loaded {len(questions)} questions")
    
    # INSCIT形式に変換
    inscit_data = create_inscit_format(questions)
    
    # 出力ファイル名
    output_file = output_dir / f"{args.split}.json"
    
    # JSONファイルに保存
    print(f"[INFO] Writing output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inscit_data, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] Successfully created {output_file}")
    print(f"  - Total conversations: {len(inscit_data)}")


if __name__ == "__main__":
    main()

