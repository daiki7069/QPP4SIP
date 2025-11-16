#!/usr/bin/env python3
"""
AmbigQAデータセットをINSCIT形式のdpr_dev.json/dpr_train.jsonに変換するスクリプト
"""

import argparse
import json
import gzip
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


def load_passages(psgs_file: str) -> Dict[int, Dict[str, str]]:
    """
    Wikipedia passagesファイルを読み込み、passage ID -> passage情報のマッピングを作成
    
    Args:
        psgs_file: passagesファイルのパス（.tsv.gz形式）
        
    Returns:
        passage ID -> {id, title, text} のマッピング
    """
    passages = {}
    
    print(f"[INFO] Loading passages from {psgs_file}...")
    with gzip.open(psgs_file, 'rt', encoding='utf-8') as f:
        # ヘッダー行をスキップ
        header = f.readline().strip()
        if header != "id\ttext\ttitle":
            print(f"[WARN] Unexpected header: {header}")
        
        for line_num, line in enumerate(f, start=2):  # ヘッダーをスキップしたので2から
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t', 2)  # 最大2回分割（textにタブが含まれる可能性があるため）
            if len(parts) != 3:
                print(f"[WARN] Skipping malformed line {line_num}: {len(parts)} parts")
                continue
            
            passage_id_str, text, title = parts
            try:
                passage_id = int(passage_id_str)
                passages[passage_id] = {
                    "id": passage_id_str,
                    "title": title,
                    "text": text
                }
            except ValueError:
                print(f"[WARN] Skipping line {line_num}: invalid passage ID '{passage_id_str}'")
                continue
    
    print(f"[INFO] Loaded {len(passages)} passages")
    return passages


def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """
    AmbigQAの質問データを読み込む
    
    Args:
        questions_file: dev_light.jsonまたはtrain_light.jsonのパス
        
    Returns:
        質問データのリスト
    """
    print(f"[INFO] Loading questions from {questions_file}...")
    with open(questions_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    
    print(f"[INFO] Loaded {len(questions)} questions")
    return questions


def load_predictions(predictions_file: str) -> List[List[int]]:
    """
    AmbigQAのpredictionデータを読み込む
    
    Args:
        predictions_file: ambigqa_dev_predictions_2020.jsonまたはambigqa_train_predictions_2020.jsonのパス
        
    Returns:
        各質問に対応するpassage IDリストのリスト
    """
    print(f"[INFO] Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    print(f"[INFO] Loaded predictions for {len(predictions)} questions")
    return predictions


def extract_answers(annotations: List[Dict[str, Any]]) -> List[str]:
    """
    annotationsから回答を抽出してフラット化
    
    Args:
        annotations: AmbigQAのannotationsリスト
        
    Returns:
        回答のリスト（重複除去済み）
    """
    answers = []
    for ann in annotations:
        if "answer" in ann:
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
    
    return unique_answers


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
        (回答のリスト, response_typeのリスト)
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
    
    # response_typeの重複除去
    unique_types = list(dict.fromkeys(response_types))  # 順序を保持しながら重複除去
    
    return unique_answers, unique_types


def create_inscit_format(
    questions: List[Dict[str, Any]],
    predictions: List[List[int]],
    passages: Dict[int, Dict[str, str]],
    top_k: int = 100
) -> List[List[Dict[str, Any]]]:
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
      - dialogue_history: 対話履歴（文字列のリスト、AmbigNQの場合は空）
      - ctxs: 検索されたpassageのリスト（オプション、musicモデルでは使用しない）
    
    Args:
        questions: 質問データのリスト
        predictions: 各質問に対応するpassage IDリストのリスト
        passages: passage ID -> passage情報のマッピング
        top_k: 使用するpassageの最大数
        
    Returns:
        INSCIT形式のデータ（会話のリストのリスト）
    """
    if len(questions) != len(predictions):
        raise ValueError(
            f"Questions ({len(questions)}) and predictions ({len(predictions)}) "
            f"must have the same length"
        )
    
    conversations: List[List[Dict[str, Any]]] = []
    missing_passages = 0
    total_passages = 0
    
    print(f"[INFO] Converting {len(questions)} questions to INSCIT format...")
    
    for i, question in enumerate(questions):
        # 質問IDを取得（存在しない場合はインデックスを使用）
        q_id = question.get("id", str(i))
        
        # 質問文
        q_text = question.get("question", "")
        
        # 回答とresponse_typeを抽出
        annotations = question.get("annotations", [])
        answers, response_types = extract_answers_and_types(annotations)
        
        # response_typeを結合（複数ある場合は " [SEP] " で結合）
        response_type_joined = " [SEP] ".join(response_types) if response_types else ""
        
        # 対応するpassage IDリストを取得
        passage_ids = predictions[i][:top_k]  # top_kまで使用
        
        # ctxsを作成
        ctxs = []
        for passage_id in passage_ids:
            total_passages += 1
            if passage_id in passages:
                passage = passages[passage_id]
                ctx = {
                    "id": passage["id"],
                    "title": passage["title"],
                    "text": passage["text"],
                    "score": "0.0",  # スコアは適当な値（必要に応じて変更可能）
                    "has_answer": False  # デフォルトはFalse（必要に応じて計算可能）
                }
                ctxs.append(ctx)
            else:
                missing_passages += 1
                print(f"[WARN] Missing passage ID: {passage_id} for question {q_id}")
        
        # 各質問を1つの会話として扱い、1ターンだけの会話を作成
        turn = {
            "conv_id": str(q_id),
            "turn_id": 1,  # 最初のターン
            "query": q_text,
            "answer": answers,
            "response_type": response_type_joined,
            "dialogue_history": [],  # AmbigNQは各質問が独立しているため空
            "ctxs": ctxs  # 検索されたpassageのリスト（musicモデルでは使用しないが、データ形式として保持）
        }
        
        # 1ターンだけの会話として追加
        conversations.append([turn])
    
    print(f"[INFO] Conversion complete:")
    print(f"  - Total conversations: {len(conversations)}")
    print(f"  - Total turns: {sum(len(conv) for conv in conversations)}")
    print(f"  - Total passages: {total_passages}")
    print(f"  - Missing passages: {missing_passages}")
    
    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Convert AmbigQA dataset to INSCIT format (dpr_dev.json/dpr_train.json)"
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="使用するpassageの最大数（デフォルト: 100）"
    )
    
    args = parser.parse_args()
    
    # ファイルパスを構築
    ambigqa_dir = Path(args.ambigqa_dir)
    questions_file = ambigqa_dir / "ambignq_light" / f"{args.split}_light.json"
    predictions_file = ambigqa_dir / "ambgqa_prediction" / f"ambigqa_{args.split}_predictions_2020.json"
    psgs_file = ambigqa_dir / "codes" / "data" / "wikipedia_split" / "psgs_w100_20200201.tsv.gz"
    
    # ファイルの存在確認
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    if not psgs_file.exists():
        raise FileNotFoundError(f"Passages file not found: {psgs_file}")
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データを読み込む
    questions = load_questions(str(questions_file))
    predictions = load_predictions(str(predictions_file))
    passages = load_passages(str(psgs_file))
    
    # INSCIT形式に変換
    inscit_data = create_inscit_format(questions, predictions, passages, top_k=args.top_k)
    
    # 出力ファイル名
    output_file = output_dir / f"dpr_{args.split}.json"
    
    # JSONファイルに保存
    print(f"[INFO] Writing output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inscit_data, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] Successfully created {output_file}")
    print(f"  - Total entries: {len(inscit_data)}")


if __name__ == "__main__":
    main()

