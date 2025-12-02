#!/usr/bin/env python3
"""
DPR結果にhas_answerラベルを追加するスクリプト
元データ: /mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/retrieval_outputs_own/results/dpr_train.json
完成形: dataset/INSCIT/dpr_train.json (has_answerラベルが正しく設定されている)
"""

import argparse
import json
from typing import List, Dict, Any, Set
from pathlib import Path


def extract_evidence_passage_ids(inscit_data: Dict[str, Any]) -> Dict[tuple, Set[str]]:
    """
    INSCITデータからevidenceのpassage_idを抽出
    
    Args:
        inscit_data: 元のINSCITデータ（辞書形式）
        
    Returns:
        {(conv_id, turn_id): {passage_id, ...}} の辞書
    """
    evidence_dict = {}
    
    for conv_id, conv_data in inscit_data.items():
        turns = conv_data.get("turns", [])
        
        for turn_idx, turn in enumerate(turns, start=1):
            labels = turn.get("labels", [])
            passage_ids = set()
            
            # labels[0].evidenceからpassage_idを抽出
            if labels:
                first_label = labels[0]
                evidence = first_label.get("evidence", [])
                for ev in evidence:
                    passage_id = ev.get("passage_id", "")
                    if passage_id:
                        passage_ids.add(passage_id)
            
            key = (str(conv_id), str(turn_idx))
            evidence_dict[key] = passage_ids
    
    return evidence_dict


def extract_evidence_passage_ids_from_converted(converted_data: List[List[Dict[str, Any]]]) -> Dict[tuple, Set[str]]:
    """
    変換済みINSCITデータからevidenceのpassage_idを抽出
    （変換済みデータにはevidence情報がないため、空のセットを返す）
    
    実際には、元のINSCITデータから抽出する必要があるため、
    この関数は使用されませんが、インターフェースの一貫性のために残します。
    """
    evidence_dict = {}
    
    for conversation in converted_data:
        for turn in conversation:
            conv_id = str(turn.get("conv_id", ""))
            turn_id = str(turn.get("turn_id", ""))
            key = (conv_id, turn_id)
            evidence_dict[key] = set()  # 変換済みデータにはevidence情報がない
    
    return evidence_dict


def add_has_answer_labels(
    dpr_data: List[Dict[str, Any]],
    evidence_dict: Dict[tuple, Set[str]]
) -> List[Dict[str, Any]]:
    """
    DPR結果にhas_answerラベルを追加
    
    Args:
        dpr_data: DPR結果のリスト
        evidence_dict: {(conv_id, turn_id): {passage_id, ...}} の辞書
        
    Returns:
        has_answerラベルが追加されたDPR結果
    """
    updated_data = []
    total_docs = 0
    relevant_docs = 0
    
    print(f"[INFO] Adding has_answer labels to {len(dpr_data)} DPR results...")
    
    for item in dpr_data:
        conv_id = str(item.get("conv_id", ""))
        turn_id = str(item.get("turn_id", ""))
        key = (conv_id, turn_id)
        
        # 該当するevidence passage_idsを取得
        evidence_passage_ids = evidence_dict.get(key, set())
        
        # ctxsの各文書に対してhas_answerを設定
        ctxs = item.get("ctxs", [])
        for ctx in ctxs:
            total_docs += 1
            passage_id = ctx.get("id", "")
            
            # passage_idがevidenceに含まれていればtrue
            has_answer = passage_id in evidence_passage_ids
            ctx["has_answer"] = has_answer
            
            if has_answer:
                relevant_docs += 1
        
        updated_data.append(item)
    
    print(f"[INFO] Label assignment complete:")
    print(f"  - Total documents: {total_docs}")
    print(f"  - Relevant documents: {relevant_docs}")
    print(f"  - Relevance rate: {relevant_docs/total_docs*100:.2f}%")
    
    return updated_data


def main():
    parser = argparse.ArgumentParser(
        description="Add has_answer labels to DPR results based on INSCIT evidence"
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "train", "test"],
        help="データセットのスプリット（dev, train, またはtest）"
    )
    parser.add_argument(
        "--dpr_input_dir",
        default="/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/retrieval_outputs_own/results",
        help="DPR結果の入力ディレクトリ"
    )
    parser.add_argument(
        "--inscit_input_dir",
        default="/mnt/nas_syno/daiki/Datasets/INSCIT/data",
        help="元のINSCITデータセットのディレクトリ（evidence抽出用）"
    )
    parser.add_argument(
        "--output_dir",
        default="./INSCIT",
        help="出力ディレクトリ"
    )
    
    args = parser.parse_args()
    
    # 入力ファイルパス
    dpr_input_file = Path(args.dpr_input_dir) / f"dpr_{args.split}.json"
    inscit_input_file = Path(args.inscit_input_dir) / f"{args.split}.json"
    
    if not dpr_input_file.exists():
        raise FileNotFoundError(f"DPR input file not found: {dpr_input_file}")
    if not inscit_input_file.exists():
        raise FileNotFoundError(f"INSCIT input file not found: {inscit_input_file}")
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # DPRデータを読み込む
    print(f"[INFO] Loading DPR data from {dpr_input_file}...")
    with open(dpr_input_file, 'r', encoding='utf-8') as f:
        dpr_data = json.load(f)
    
    print(f"[INFO] Loaded {len(dpr_data)} DPR results")
    
    # INSCITデータを読み込んでevidence passage_idsを抽出
    print(f"[INFO] Loading INSCIT data from {inscit_input_file}...")
    with open(inscit_input_file, 'r', encoding='utf-8') as f:
        inscit_data = json.load(f)
    
    print(f"[INFO] Extracting evidence passage IDs...")
    evidence_dict = extract_evidence_passage_ids(inscit_data)
    print(f"[INFO] Extracted evidence for {len(evidence_dict)} (conv_id, turn_id) pairs")
    
    # has_answerラベルを追加
    updated_dpr_data = add_has_answer_labels(dpr_data, evidence_dict)
    
    # 出力ファイル名
    output_file = output_dir / f"dpr_{args.split}.json"
    
    # JSONファイルに保存
    print(f"[INFO] Writing output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(updated_dpr_data, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] Successfully created {output_file}")
    print(f"  - Total DPR results: {len(updated_dpr_data)}")


if __name__ == "__main__":
    main()

