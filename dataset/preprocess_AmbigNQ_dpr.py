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

try:
    from tqdm import tqdm
except ImportError:
    # tqdmがない場合は通常のenumerateを使用
    def tqdm(iterable, desc=None, total=None, **kwargs):
        if desc:
            print(f"[INFO] {desc}...")
        return iterable


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
        
        # 進捗バー付きで処理（gzipファイルのため総行数は不明）
        line_num = 2
        for line in tqdm(f, desc="Loading passages", unit="lines"):
            line = line.strip()
            if not line:
                line_num += 1
                continue
            
            parts = line.split('\t', 2)  # 最大2回分割（textにタブが含まれる可能性があるため）
            if len(parts) != 3:
                if line_num % 100000 == 0:  # 警告を減らすため、100000行ごとにのみ表示
                    print(f"\n[WARN] Skipping malformed line {line_num}: {len(parts)} parts")
                line_num += 1
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
                if line_num % 100000 == 0:  # 警告を減らすため、100000行ごとにのみ表示
                    print(f"\n[WARN] Skipping line {line_num}: invalid passage ID '{passage_id_str}'")
                line_num += 1
                continue
            
            line_num += 1
    
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


def load_dpr_predictions(predictions_file: str) -> List[Dict[str, Any]]:
    """
    DPRのpredictionデータを読み込む（idとscoreを含む）
    
    Args:
        predictions_file: dev_predictions.jsonまたはtrain_for_inference_predictions.jsonのパス
        
    Returns:
        各質問に対応する{passage_ids: List[int], scores: List[float]}のリスト
    """
    print(f"[INFO] Loading DPR predictions from {predictions_file}...")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    print(f"[INFO] Loaded DPR predictions for {len(predictions)} questions")
    return predictions


def load_nqopen_index_mapping(nqopen_file: str) -> Dict[Any, int]:
    """
    nqopen/dev.jsonから質問ID -> インデックスのマッピングを作成
    
    Args:
        nqopen_file: nqopen/dev.jsonのパス
        
    Returns:
        質問ID -> インデックスのマッピング
    """
    print(f"[INFO] Loading nqopen index mapping from {nqopen_file}...")
    with open(nqopen_file, 'r', encoding='utf-8') as f:
        nqopen_data = json.load(f)
    
    # 質問ID -> インデックスのマッピングを作成
    id_to_index = {}
    for index, item in enumerate(tqdm(nqopen_data, desc="Creating index mapping", unit="items")):
        q_id = item.get('id')
        if q_id is not None:
            id_to_index[q_id] = index
    
    print(f"[INFO] Created index mapping for {len(id_to_index)} questions")
    return id_to_index


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
    dpr_predictions: List[Dict[str, Any]],
    passages: Dict[int, Dict[str, str]],
    nqopen_id_to_index: Optional[Dict[Any, int]] = None,
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
        dpr_predictions: 各質問に対応する{passage_ids: List[int], scores: List[float]}のリスト
        passages: passage ID -> passage情報のマッピング
        top_k: 使用するpassageの最大数
        
    Returns:
        INSCIT形式のデータ（会話のリストのリスト）
    """
    conversations: List[List[Dict[str, Any]]] = []
    missing_passages = 0
    total_passages = 0
    missing_dpr_predictions = 0
    
    print(f"[INFO] Converting {len(questions)} questions to INSCIT format...")
    
    for i, question in enumerate(tqdm(questions, desc="Converting to INSCIT format", unit="questions")):
        # 質問IDを取得（存在しない場合はインデックスを使用）
        q_id = question.get("id", str(i))
        
        # 質問文
        q_text = question.get("question", "")
        
        # 回答とresponse_typeを抽出
        annotations = question.get("annotations", [])
        answers, response_types = extract_answers_and_types(annotations)
        
        # response_typeを結合（複数ある場合は " [SEP] " で結合）
        response_type_joined = " [SEP] ".join(response_types) if response_types else ""
        
        # DPR予測データからpassage IDとscoreを取得
        # nqopenのインデックスマッピングがある場合はそれを使用、なければインデックスで直接アクセス
        if nqopen_id_to_index is not None and q_id in nqopen_id_to_index:
            dpr_index = nqopen_id_to_index[q_id]
            if dpr_index >= len(dpr_predictions):
                print(f"[WARN] DPR index {dpr_index} out of range for question {q_id}")
                missing_dpr_predictions += 1
                continue
            dpr_pred = dpr_predictions[dpr_index]
        elif len(dpr_predictions) > len(questions):
            # インデックスマッピングがない場合、最初のN件を使用（警告を出す）
            if i == 0:
                print(f"[WARN] No nqopen index mapping provided, using first {len(questions)} DPR predictions")
            if i >= len(dpr_predictions):
                print(f"[WARN] DPR index {i} out of range for question {q_id}")
                missing_dpr_predictions += 1
                continue
            dpr_pred = dpr_predictions[i]
        else:
            # 長さが一致している場合
            if i >= len(dpr_predictions):
                print(f"[WARN] DPR index {i} out of range for question {q_id}")
                missing_dpr_predictions += 1
                continue
            dpr_pred = dpr_predictions[i]
        passage_ids = dpr_pred.get("passage_ids", [])[:top_k]
        scores = dpr_pred.get("scores", [])[:top_k]
        
        # passage_idsとscoresの長さが一致していることを確認
        if len(passage_ids) != len(scores):
            print(f"[WARN] Mismatch in passage_ids and scores length for question {q_id}: "
                  f"{len(passage_ids)} vs {len(scores)}")
            min_len = min(len(passage_ids), len(scores))
            passage_ids = passage_ids[:min_len]
            scores = scores[:min_len]
        
        # ctxsを作成
        ctxs = []
        for passage_id, score in zip(passage_ids, scores):
            total_passages += 1
            if passage_id in passages:
                passage = passages[passage_id]
                ctx = {
                    "id": passage["id"],
                    "title": passage["title"],
                    "text": passage["text"],
                    "score": str(score),  # DPRから取得したスコアを使用
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
    if missing_dpr_predictions > 0:
        print(f"  - Missing DPR predictions: {missing_dpr_predictions}")
    
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
    
    # DPR予測ファイルのパス（新しい形式）
    if args.split == "dev":
        dpr_predictions_file = Path("/mnt/nas_syno/daiki/Datasets/AmbigQA/codes/out/dpr/dev_predictions.json")
        nqopen_file = Path("/mnt/nas_syno/daiki/Datasets/AmbigQA/codes/data/nqopen/dev.json")
    elif args.split == "train":
        dpr_predictions_file = Path("/mnt/nas_syno/daiki/Datasets/AmbigQA/codes/out/dpr/train_for_inference_predictions.json")
        nqopen_file = None  # trainの場合はnqopenファイルがない可能性がある
    else:
        raise ValueError(f"Unknown split: {args.split}")
    
    # passagesファイル（従来通り）
    psgs_file = ambigqa_dir / "codes" / "data" / "wikipedia_split" / "psgs_w100.tsv.gz"
    
    # ファイルの存在確認
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")
    if not dpr_predictions_file.exists():
        raise FileNotFoundError(f"DPR predictions file not found: {dpr_predictions_file}")
    if not psgs_file.exists():
        raise FileNotFoundError(f"Passages file not found: {psgs_file}")
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データを読み込む
    questions = load_questions(str(questions_file))
    dpr_predictions = load_dpr_predictions(str(dpr_predictions_file))
    passages = load_passages(str(psgs_file))
    
    # nqopenファイルからインデックスマッピングを読み込む（devの場合）
    nqopen_id_to_index = None
    if nqopen_file is not None and nqopen_file.exists():
        nqopen_id_to_index = load_nqopen_index_mapping(str(nqopen_file))
    elif nqopen_file is not None:
        print(f"[WARN] nqopen file not found: {nqopen_file}, using direct index mapping")
    
    # INSCIT形式に変換
    inscit_data = create_inscit_format(questions, dpr_predictions, passages, nqopen_id_to_index=nqopen_id_to_index, top_k=args.top_k)
    
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

