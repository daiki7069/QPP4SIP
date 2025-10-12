import json
import pickle
import argparse
import os
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple


def calculate_hit_at_k(relevant_docs: List[bool], k: int) -> float:
    """Hit@kを計算する"""
    if not relevant_docs:
        return 0.0
    return 1.0 if any(relevant_docs[:k]) else 0.0


def calculate_precision_at_k(relevant_docs: List[bool], k: int) -> float:
    """Precision@kを計算する"""
    if not relevant_docs or k == 0:
        return 0.0
    relevant_in_k = sum(relevant_docs[:k])
    return relevant_in_k / k


def calculate_recall_at_k(relevant_docs: List[bool], k: int, total_relevant: int) -> float:
    """Recall@kを計算する"""
    if not relevant_docs or total_relevant == 0:
        return 0.0
    relevant_in_k = sum(relevant_docs[:k])
    return relevant_in_k / total_relevant


def calculate_f1_at_k(relevant_docs: List[bool], k: int, total_relevant: int) -> float:
    """F1@kを計算する"""
    precision = calculate_precision_at_k(relevant_docs, k)
    recall = calculate_recall_at_k(relevant_docs, k, total_relevant)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_ndcg_at_k(relevant_docs: List[bool], k: int) -> float:
    """NDCG@kを計算する"""
    if not relevant_docs or k == 0:
        return 0.0
    
    # 理想的な順序（関連文書が上位に来る）
    ideal_relevant = [True] * sum(relevant_docs) + [False] * (len(relevant_docs) - sum(relevant_docs))
    
    def dcg_at_k(relevances: List[bool], k: int) -> float:
        dcg = 0.0
        for i in range(min(k, len(relevances))):
            if relevances[i]:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
        return dcg
    
    dcg = dcg_at_k(relevant_docs, k)
    idcg = dcg_at_k(ideal_relevant, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_mrr(relevant_docs: List[bool]) -> float:
    """MRR（Mean Reciprocal Rank）を計算する"""
    if not relevant_docs:
        return 0.0
    
    for i, is_relevant in enumerate(relevant_docs):
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def calculate_mean_rank(relevant_docs: List[bool]) -> float:
    """Mean Rankを計算する"""
    if not relevant_docs:
        return 0.0
    
    for i, is_relevant in enumerate(relevant_docs):
        if is_relevant:
            return float(i + 1)
    return float(len(relevant_docs) + 1)  # 関連文書が見つからない場合


def calculate_found_ratio(relevant_docs: List[bool]) -> float:
    """Found Ratioを計算する（関連文書が存在するかどうか）"""
    return 1.0 if any(relevant_docs) else 0.0


def calculate_all_metrics(relevant_docs: List[bool], total_relevant: int) -> Dict[str, float]:
    """すべての評価指標を計算する"""
    metrics = {}
    
    # 各k値に対して指標を計算
    k_values = [1, 5, 10, 20, 50]
    
    for k in k_values:
        metrics[f"hit@{k}"] = calculate_hit_at_k(relevant_docs, k)
        metrics[f"precision@{k}"] = calculate_precision_at_k(relevant_docs, k)
        metrics[f"recall@{k}"] = calculate_recall_at_k(relevant_docs, k, total_relevant)
        metrics[f"f1@{k}"] = calculate_f1_at_k(relevant_docs, k, total_relevant)
        metrics[f"ndcg@{k}"] = calculate_ndcg_at_k(relevant_docs, k)
    
    # その他の指標
    metrics["mrr"] = calculate_mrr(relevant_docs)
    metrics["mean_rank"] = calculate_mean_rank(relevant_docs)
    metrics["found_ratio"] = calculate_found_ratio(relevant_docs)
    
    return metrics


def convert_inscit_to_msdialog_format(processed_data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """INSCITデータをMSDialog形式の構造に変換する（会話ごとにグループ化）"""
    # 会話IDでグループ化
    conversations = {}
    for turn in processed_data:
        conv_id = turn['conv_id']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(turn)
    
    # MSDialog形式の構造に変換（会話のリスト → ターンのリスト）
    msdialog_conversations = []
    
    for conv_id, turns in conversations.items():
        # ターンをソート（turn_id順）
        turns.sort(key=lambda x: x['turn_id'])
        
        # 会話として追加（INSCITのフィールドをそのまま保持）
        msdialog_conversations.append(turns)
    
    return msdialog_conversations


def process_inscit_data(input_file: str, original_data_file: str) -> List[Dict[str, Any]]:
    """INSCITデータを処理してターン単位のデータセットに変換する"""
    print(f"Processing {input_file}...")
    
    # DPR結果ファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        dpr_data = json.load(f)
    
    # 元のINSCITデータを読み込む
    with open(original_data_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    processed_data = []
    
    for item in dpr_data:
        # データの整合性チェック
        conv_id = item.get("conv_id", "")
        turn_id = int(item.get("turn_id", 0))
        
        # 元データから対応する会話を取得
        assert conv_id in original_data, f"conv_id {conv_id} not found in original data"
        conversation = original_data[conv_id]
        turns = conversation.get("turns", [])
        
        # ターンIDのチェック
        assert 1 <= turn_id <= len(turns), f"turn_id {turn_id} out of range for conv_id {conv_id} (max: {len(turns)})"
            
        # 対応するターンのデータを取得
        turn_data = turns[turn_id - 1]  # turn_idは1から始まるので-1
        labels = turn_data.get("labels", [])
        
        # labelsの存在チェック
        assert len(labels) > 0, f"No labels found for conv_id {conv_id}, turn_id {turn_id}"
        
        # response_typeを抽出（2つのラベルのresponseTypeを[SEP]で結合）
        response_types = []
        for label in labels:
            response_type = label.get("responseType", "")
            assert response_type, f"Empty responseType in conv_id {conv_id}, turn_id {turn_id}"
            response_types.append(response_type)
        
        response_type_str = " [SEP] ".join(response_types)
        
        # questionから対話履歴を分割
        question_with_history = item["question"]
        question_parts = question_with_history.split(" [SEP] ")
        
        # 最後の部分が現在のクエリ
        current_query = question_parts[-1].strip() if question_parts else ""
        
        # 対話履歴
        dialogue_history = question_with_history if len(question_parts) > 1 else ""
        
        # 関連文書の情報を取得
        ctxs = item.get("ctxs", [])
        relevant_docs = [ctx.get("has_answer", False) for ctx in ctxs]
        total_relevant = sum(relevant_docs)
        
        # 評価指標を計算
        metrics = calculate_all_metrics(relevant_docs, total_relevant)
        
        # ターン単位のデータを作成
        processed_turn_data = {
            "conv_id": conv_id,
            "turn_id": turn_id,
            "query": current_query,
            "question": question_with_history,
            "answer": item.get("answers", []),
            "response_type": response_type_str,
            "dialogue_history": dialogue_history,
            "mrr": metrics["mrr"],
            "found_ratio": metrics["found_ratio"],
            "mean_rank": metrics["mean_rank"],
            "hit@1": metrics["hit@1"],
            "hit@5": metrics["hit@5"],
            "hit@10": metrics["hit@10"],
            "hit@20": metrics["hit@20"],
            "hit@50": metrics["hit@50"],
            "precision@1": metrics["precision@1"],
            "precision@5": metrics["precision@5"],
            "precision@10": metrics["precision@10"],
            "precision@20": metrics["precision@20"],
            "precision@50": metrics["precision@50"],
            "recall@1": metrics["recall@1"],
            "recall@5": metrics["recall@5"],
            "recall@10": metrics["recall@10"],
            "recall@20": metrics["recall@20"],
            "recall@50": metrics["recall@50"],
            "f1@1": metrics["f1@1"],
            "f1@5": metrics["f1@5"],
            "f1@10": metrics["f1@10"],
            "f1@20": metrics["f1@20"],
            "f1@50": metrics["f1@50"],
            "ndcg@1": metrics["ndcg@1"],
            "ndcg@5": metrics["ndcg@5"],
            "ndcg@10": metrics["ndcg@10"],
            "ndcg@20": metrics["ndcg@20"],
            "ndcg@50": metrics["ndcg@50"]
        }
        
        processed_data.append(processed_turn_data)
    
    print(f"Processed {len(processed_data)} turns from {input_file}")
    return processed_data


def save_data(data: List[Dict[str, Any]], output_path: str, save_json: bool = False):
    """データをtorch形式で保存する（オプションでJSON形式も保存）"""
    # torch形式で保存
    torch.save(data, output_path)
    print(f"Saved torch file: {output_path}")
    
    # オプションでJSON形式も保存（人間が読める形式）
    if save_json:
        output_json = output_path.replace('.pkl', '.json')
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON file: {output_json}")


def main():
    parser = argparse.ArgumentParser(description='INSCITデータセットの前処理')
    parser.add_argument('--input_file', type=str, required=True,
                       help='DPR結果JSONファイルのパス')
    parser.add_argument('--original_data_file', type=str, required=True,
                       help='元のINSCITデータJSONファイルのパス')
    parser.add_argument('--output_dir', type=str, default='./',
                       help='出力ディレクトリのパス')
    parser.add_argument('--save_json', action='store_true',
                       help='JSON形式でも保存する（オプション）')
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
        
    if not os.path.exists(args.original_data_file):
        print(f"Error: Original data file {args.original_data_file} does not exist")
        return
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 入力ファイル名から出力ファイル名を生成
    input_path = Path(args.input_file)
    base_name = input_path.stem  # 拡張子を除いたファイル名
    
    # ファイル名にdevまたはtrainが含まれているかチェック
    if 'dev' in base_name:
        split_name = 'dev'
    elif 'train' in base_name:
        split_name = 'train'
    else:
        print("Warning: File name does not contain 'dev' or 'train'")
        split_name = 'unknown'
    
    # 出力ファイル名を生成
    output_path = os.path.join(args.output_dir, f"{base_name}.pkl")
    
    # データを処理
    processed_data = process_inscit_data(args.input_file, args.original_data_file)
    
    # MSDialog形式に変換（各ターンを独立した会話として扱う）
    msdialog_data = convert_inscit_to_msdialog_format(processed_data)
    
    # データを保存
    save_data(msdialog_data, output_path, args.save_json)
    
    print(f"Processing completed successfully!")
    print(f"Output file:")
    print(f"  Torch: {output_path}")
    if args.save_json:
        output_json = output_path.replace('.pkl', '.json')
        print(f"  JSON: {output_json}")


if __name__ == "__main__":
    main()
