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


def find_common_nsp_top_k(nsp_output_dir: Path, splits: List[str] = None) -> int:
    """
    複数のスプリットで共通して存在するNSPのtop_k値の最大値を返す
    
    Args:
        nsp_output_dir: next_sentence_predictionの出力ディレクトリ
        splits: 確認するスプリットのリスト（Noneの場合は'train'と'dev'を確認）
    
    Returns:
        共通して存在する最大のtop_k値、存在しない場合はNone
    """
    if splits is None:
        splits = ['train', 'dev']
    
    nsp_top_k_values = [10, 20, 50, 100]
    common_top_k = []
    
    for top_k in nsp_top_k_values:
        # 全てのスプリットで存在するか確認
        all_exist = True
        for split in splits:
            csv_path = nsp_output_dir / f"{split}_nsp_graph_topk{top_k}.csv"
            if not csv_path.exists():
                all_exist = False
                break
        
        if all_exist:
            common_top_k.append(top_k)
    
    return max(common_top_k) if common_top_k else None


def load_base_scores(split: str, dataset: str, base_dir: Path, base_experiment_names: List[str] = None) -> Dict[str, Dict[Tuple[str, int], float]]:
    """
    ベーススコア（logit_clarification）を読み込む
    戻り値: {feature_name: {(conv_id, turn_id): score}}
    
    Args:
        split: データセットのスプリット（'train' または 'dev'）
        dataset: データセット名
        base_dir: ベースディレクトリ（SIP/FT-PLM/outputの親ディレクトリ）
        base_experiment_names: 使用するSIP実験名のリスト（Noneの場合は関数内のデフォルト設定を使用）
    
    Returns:
        ベーススコアの辞書 {prefix_logit_clarification: {key: score}}
    """
    base_scores = {}
    
    # ベース実験名の定義（コメントアウトで選択）
    # 使用したい実験名をコメントアウト解除して有効化
    default_base_configs = [
        # f"{dataset}_bert-base_lr2e-05_bs16_kfold5",
        # f"{dataset}_roberta-base_lr2e-05_bs16_earlystop_kfold5",  # earlystopあり
        # f"{dataset}_roberta-base_lr2e-05_bs16_kfold5",  # earlystopなし
        # 他の実験名も追加可能
    ]
    
    # base_experiment_namesが指定されていない場合は、デフォルト設定を使用
    if base_experiment_names is None:
        base_experiment_names = default_base_configs
    
    if len(base_experiment_names) == 0:
        return base_scores
    
    def extract_prefix(experiment_name: str) -> str:
        """実験名からプレフィックスを抽出"""
        parts = experiment_name.split('_')
        if len(parts) > 1:
            model_part = parts[1]
            if model_part.startswith('bert-'):
                return 'bert_'
            elif model_part.startswith('roberta-'):
                return 'roberta_'
            else:
                return f"{model_part.split('-')[0]}_"
        return ""
    
    # 存在する実験名のみをフィルタリング
    available_experiments = []
    for exp_name in base_experiment_names:
        exp_output_dir = base_dir / "SIP" / "FT-PLM" / "output" / dataset / exp_name
        pred_json_path = exp_output_dir / f"{split}_with_predictions.json"
        if pred_json_path.exists():
            available_experiments.append(exp_name)
        else:
            print(f"Warning: {pred_json_path} not found, skipping {exp_name}")
            print(f"  → 実験ディレクトリが存在しないか、ファイル名が異なります")
    
    if len(available_experiments) == 0:
        print(f"Warning: No available base experiments found. Returning empty base_scores.")
        return base_scores
    
    for exp_name in available_experiments:
        prefix = extract_prefix(exp_name)
        exp_output_dir = base_dir / "SIP" / "FT-PLM" / "output" / dataset / exp_name
        pred_json_path = exp_output_dir / f"{split}_with_predictions.json"
        
        json_data = load_json_data(pred_json_path)
        scores = {}
        for conversation in json_data:
            for turn in conversation:
                conv_id = str(turn['conv_id'])
                turn_id = int(turn['turn_id'])
                key = (conv_id, turn_id)
                
                logit = turn.get('logit_clarification')
                if logit is not None:
                    scores[key] = float(logit)
        
        feature_name = f"{prefix}logit_clarification" if prefix else "logit_clarification"
        base_scores[feature_name] = scores
        print(f"Loaded {len(scores)} {feature_name} scores for {split} (experiment: {exp_name})")
    
    return base_scores


def load_qpp_scores(split: str, qpp_output_dir: Path, nsp_output_dir: Path = None, nsp_top_k: int = None) -> Dict[str, Dict[Tuple[str, int], float]]:
    """
    QPPスコアを読み込む（post_retrievalとnspの両方）
    戻り値: {metric_name: {(conv_id, turn_id): score}}
    
    Args:
        split: データセットのスプリット（'train' または 'dev'）
        qpp_output_dir: post_retrievalの出力ディレクトリ
        nsp_output_dir: next_sentence_predictionの出力ディレクトリ（オプション）
        nsp_top_k: 使用するNSPのtop_k値（Noneの場合は自動検出）
    """
    qpp_scores = {}
    
    # Post-retrieval QPPスコアの定義
    # 使用したいメトリクスをコメントアウト解除して有効化
    post_qpp_configs = {
        # 'nqc': ('nqsc.csv', 'nqc'),
        'similarity': ('similarity.csv', 'mean_similarity'),
        # 'clarity': ('clarity.csv', 'clarity'),
        # 'wig': ('wig.csv', 'wig'),
        'smv': ('smv.csv', 'smv'),
        # 'nsv': ('nsv.csv', 'nsv'),
        # 'entropy': ('entropy.csv', 'entropy'),
        # 'lci': ('lci.csv', 'lci'),
        # 'unique_titles': ('unique_titles.csv', 'num_unique_titles'),
        # 'acc': ('coherency.csv', 'acc'),
        # 'wacc': ('coherency.csv', 'wacc'),
    }
    
    # Post-retrievalスコアを読み込む
    for metric_name, (filename, column_name) in post_qpp_configs.items():
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
    
    # NSPスコアを読み込む（オプション）
    if nsp_output_dir is not None:
        # top_k値の決定
        if nsp_top_k is None:
            # 自動検出：このスプリットで存在する最大のtop_k値を使用
            nsp_top_k_values = [10, 20, 50, 100]
            available_top_k = []
            
            for top_k in nsp_top_k_values:
                csv_path = nsp_output_dir / f"{split}_nsp_graph_topk{top_k}.csv"
                if csv_path.exists():
                    available_top_k.append(top_k)
            
            if available_top_k:
                top_k = max(available_top_k)
            else:
                print(f"Warning: No NSP score files found in {nsp_output_dir} for {split}")
                return qpp_scores
        else:
            top_k = nsp_top_k
        
        csv_path = nsp_output_dir / f"{split}_nsp_graph_topk{top_k}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping NSP scores for {split}")
            return qpp_scores
        
        # conv_idを文字列として読み込む
        df = pd.read_csv(csv_path, dtype={'conv_id': str})
        
        # NSPメトリクスを読み込む
        nsp_metrics = {
            # 'node_connectivity': f'nsp_node_connectivity_topk{top_k}',
            # 'average_node_connectivity': f'nsp_avg_node_connectivity_topk{top_k}',
            # 'density': f'nsp_density_topk{top_k}',
        }
        
        for column_name, metric_name in nsp_metrics.items():
            if column_name not in df.columns:
                print(f"Warning: {column_name} not found in {csv_path}, skipping")
                continue
            
            scores = {}
            for _, row in df.iterrows():
                conv_id = str(row['conv_id'])
                turn_id = int(row['turn_id'])
                key = (conv_id, turn_id)
                
                value = row[column_name]
                if pd.notna(value):
                    scores[key] = float(value)
            
            qpp_scores[metric_name] = scores
            print(f"Loaded {len(scores)} {metric_name} scores for {split} (top_k={top_k})")
    
    return qpp_scores

