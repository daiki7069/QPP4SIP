#!/usr/bin/env python3
"""
ClariQデータセットをINSCIT形式のdpr_dev.json/dpr_train.jsonに変換するスクリプト
参考: /home/daiki_shibata/pj/SIP/dataset/preprocess_ClariQ.py
出力フォーマット: /home/daiki_shibata/pj/QPP4SIP/dataset/AmbigNQ/dpr_dev.json
"""

import argparse
import json
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


def load_clariq_data(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    ClariQデータを読み込む
    
    Args:
        file_path: ClariQのTSVファイルのパス
        
    Returns:
        {topic_id: {initial_request, clarification_need, question}} の辞書
    """
    ClariQ = {}
    
    print(f"[INFO] Loading ClariQ data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as r:
        for line in r:
            columns = line.strip().split('\t')
            if columns[0] == "conversation_id" or columns[0] == "topic_id":
                continue
            
            conversation_id = columns[0]
            
            # 既に存在する場合はスキップ（1つのconversation_idにつき1つのエントリのみ）
            if conversation_id in ClariQ:
                continue
            
            # TSVファイルの構造:
            # conversation_id, example_id, user_utterance, user_action, user_I_label,
            # system_utterance, system_action, system_I_label
            user_utterance = columns[2] if len(columns) > 2 else ""
            user_I_label = columns[4] if len(columns) > 4 else ""
            system_utterance = columns[5] if len(columns) > 5 else ""
            system_I_label = columns[7] if len(columns) > 7 else ""
            
            # system_I_labelが"Initiative"の場合はclarification、そうでない場合はdirectAnswer
            # ただし、system_utteranceが空の場合はdirectAnswerと判断
            if system_I_label == "Initiative" and system_utterance:
                clarification_need = "2"  # clarification (2,3,4のいずれかとして扱う)
            else:
                clarification_need = "1"  # directAnswer
            
            ClariQ[conversation_id] = {
                "initial_request": user_utterance,
                "clarification_need": clarification_need,
                "question": system_utterance
            }
    
    print(f"[INFO] Loaded {len(ClariQ)} topics")
    return ClariQ


def load_dpr_predictions(predictions_file: Optional[str]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    DPRのpredictionデータを読み込む（オプション）
    
    Args:
        predictions_file: DPR予測ファイルのパス（JSON形式）
        
    Returns:
        {topic_id: [{id, title, text, score, has_answer}, ...]} の辞書、またはNone
    """
    if predictions_file is None:
        return None
    
    print(f"[INFO] Loading DPR predictions from {predictions_file}...")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        dpr_data = json.load(f)
    
    # DPRデータをtopic_idでインデックス化
    dpr_dict = {}
    for item in dpr_data:
        # DPRデータの形式に応じてtopic_idを取得
        # 形式が不明な場合は、インデックスや他のキーを使用する必要がある
        topic_id = item.get("topic_id") or item.get("conv_id") or item.get("id")
        if topic_id:
            ctxs = item.get("ctxs", [])
            dpr_dict[str(topic_id)] = ctxs
    
    print(f"[INFO] Loaded DPR predictions for {len(dpr_dict)} topics")
    return dpr_dict


def create_inscit_format(
    clariq_data: Dict[str, Dict[str, Any]],
    dpr_predictions: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> List[List[Dict[str, Any]]]:
    """
    ClariQデータをINSCIT形式に変換
    
    INSCIT形式:
    - 最上位は会話のリスト（各会話はターンのリスト）
    - 各ターンは以下のキーを持つ:
      - conv_id: 会話ID（topic_id）
      - turn_id: ターンID（整数、1始まり）
      - query: 質問文
      - answer: 回答のリスト（ClariQには回答がないため空リスト）
      - response_type: レスポンスタイプ
      - dialogue_history: 対話履歴（文字列のリスト）
      - ctxs: 検索されたpassageのリスト（DPR結果がある場合）
    
    Args:
        clariq_data: ClariQデータの辞書
        dpr_predictions: DPR予測データ（オプション）
        
    Returns:
        INSCIT形式のデータ（会話のリストのリスト）
    """
    conversations: List[List[Dict[str, Any]]] = []
    I_num = 0
    N_num = 0
    
    print(f"[INFO] Converting {len(clariq_data)} topics to INSCIT format...")
    
    for topic_id, conv_data in tqdm(clariq_data.items(), desc="Converting to INSCIT format", unit="topics"):
        initial_request = conv_data["initial_request"]
        clarification_need = int(conv_data["clarification_need"])
        question = conv_data["question"]
        
        # clarification_needに基づいてresponse_typeを決定
        # 2,3,4: Initiative (clarification question)
        # 1: Non-initiative (direct answer)
        if clarification_need in [2, 3, 4]:
            response_type = "clarification"
            I_num += 1
        elif clarification_need == 1:
            response_type = "directAnswer"
            N_num += 1
        else:
            print(f"[WARN] Unknown clarification_need value: {clarification_need} for topic {topic_id}")
            response_type = "directAnswer"
        
        # DPR予測がある場合は取得
        ctxs = []
        if dpr_predictions is not None and topic_id in dpr_predictions:
            ctxs = dpr_predictions[topic_id]
        
        # FT-PLM用: ターン1のみを作成（ユーザーのクエリとシステムの応答タイプ）
        # INSCIT/AmbigNQと同じ形式で、各ターンはユーザーのクエリとシステムの応答タイプを持つ
        turn = {
            "conv_id": str(topic_id),
            "turn_id": 1,
            "query": initial_request,  # ユーザーの質問
            "answer": [],  # ClariQには回答がないため空リスト
            "response_type": response_type,  # システムの応答タイプ（clarification or directAnswer）
            "dialogue_history": [],  # 最初のターンなので対話履歴は空
            "ctxs": ctxs  # DPR結果がある場合は追加
        }
        
        # 1ターンの会話として追加（FT-PLMは各ターンを独立して扱うため）
        conversations.append([turn])
    
    print(f"[INFO] Conversion complete:")
    print(f"  - Total conversations: {len(conversations)}")
    print(f"  - Total turns: {sum(len(conv) for conv in conversations)}")
    print(f"  - Initiative (clarification): {I_num}")
    print(f"  - Non-initiative (direct answer): {N_num}")
    
    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Convert ClariQ dataset to INSCIT format (dpr_dev.json/dpr_train.json)"
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="ClariQのTSVファイルのパス"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "train", "test"],
        help="データセットのスプリット（dev, train, またはtest）"
    )
    parser.add_argument(
        "--dpr_predictions",
        type=str,
        default=None,
        help="DPR予測ファイルのパス（オプション、JSON形式）"
    )
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    input_file = Path(args.input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # DPR予測ファイルの存在確認（指定されている場合）
    dpr_file = None
    if args.dpr_predictions:
        dpr_file_path = Path(args.dpr_predictions)
        if not dpr_file_path.exists():
            print(f"[WARN] DPR predictions file not found: {dpr_file_path}, continuing without DPR data")
        else:
            dpr_file = str(dpr_file_path)
    
    # 出力ディレクトリを作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ClariQデータを読み込む
    clariq_data = load_clariq_data(str(input_file))
    
    # DPR予測を読み込む（オプション）
    dpr_predictions = None
    if dpr_file:
        dpr_predictions = load_dpr_predictions(dpr_file)
    
    # INSCIT形式に変換
    inscit_data = create_inscit_format(clariq_data, dpr_predictions)
    
    # 出力ファイル名
    output_file = output_dir / f"dpr_{args.split}.json"
    
    # JSONファイルに保存
    print(f"[INFO] Writing output to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(inscit_data, f, ensure_ascii=False, indent=2)
    
    print(f"[DONE] Successfully created {output_file}")
    print(f"  - Total conversations: {len(inscit_data)}")


if __name__ == "__main__":
    main()

