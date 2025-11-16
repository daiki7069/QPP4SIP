"""
データ読み込み関連のモジュール
"""
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_json_data(json_path: Path) -> List[List[Dict[str, Any]]]:
    """JSONファイルを読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_labels(json_data: List[List[Dict[str, Any]]]) -> Dict[Tuple[str, int], int]:
    """
    response_typeからラベルを抽出
    [SEP]で区切られている場合は前方を採用
    clarificationを正例（1）、それ以外を負例（0）とする
    """
    labels = {}
    for conversation in json_data:
        for turn in conversation:
            # conv_idとturn_idを文字列/整数に統一
            conv_id = str(turn['conv_id'])
            turn_id = int(turn['turn_id'])
            key = (conv_id, turn_id)
            
            response_type = turn.get('response_type', '')
            # [SEP]で分割した場合は前方を採用
            if ' [SEP] ' in response_type:
                response_type = response_type.split(' [SEP] ')[0]
            
            # clarificationを正例（1）、それ以外を負例（0）
            label = 1 if response_type.strip() == 'clarification' else 0
            labels[key] = label
    
    return labels


def extract_base_scores(json_data: List[List[Dict[str, Any]]]) -> Dict[Tuple[str, int], float]:
    """ベースモデルのlogit_clarificationを抽出"""
    scores = {}
    for conversation in json_data:
        for turn in conversation:
            # conv_idとturn_idを文字列/整数に統一
            conv_id = str(turn['conv_id'])
            turn_id = int(turn['turn_id'])
            key = (conv_id, turn_id)
            
            logit = turn.get('logit_clarification')
            if logit is not None:
                scores[key] = float(logit)
    
    return scores


def load_qpp_scores(split: str, qpp_output_dir: Path) -> Dict[str, Dict[Tuple[str, int], float]]:
    """
    QPPスコアを読み込む
    戻り値: {metric_name: {(conv_id, turn_id): score}}
    """
    qpp_scores = {}
    
    # QPPスコアの定義
    qpp_configs = {
        # 'nqc': ('nqc.csv', 'nqc'),
        'similarity': ('similarity.csv', 'mean_similarity'),
        # 'wig': ('wig.csv', 'wig'),
        # 'entropy': ('entropy.csv', 'entropy'),
        # 'lci': ('lci.csv', 'lci'),
        # 'unique_titles': ('unique_titles.csv', 'num_unique_titles'),
    }
    
    for metric_name, (filename, column_name) in qpp_configs.items():
        csv_path = qpp_output_dir / f"{split}_{filename}"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {metric_name}")
            continue
        
        # conv_idを文字列として読み込む（科学記数法を避けるため）
        df = pd.read_csv(csv_path, dtype={'conv_id': str})
        scores = {}
        for _, row in df.iterrows():
            # conv_idは既に文字列として読み込まれている
            conv_id = str(row['conv_id'])
            turn_id = int(row['turn_id'])
            key = (conv_id, turn_id)
            
            value = row[column_name]
            if pd.notna(value):
                scores[key] = float(value)
        
        qpp_scores[metric_name] = scores
        print(f"Loaded {len(scores)} {metric_name} scores for {split}")
    
    return qpp_scores

