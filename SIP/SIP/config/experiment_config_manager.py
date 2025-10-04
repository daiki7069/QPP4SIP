"""
実験設定管理関連のユーティリティ関数
"""
import os
import json
import torch
from datetime import datetime


def save_experiment_config(config, save_dir, experiment_name=None):
    """
    実験設定をJSONファイルとして保存
    
    Args:
        config (dict): 実験設定の辞書
        save_dir (str): 保存先ディレクトリ
        experiment_name (str): 実験名（指定しない場合はタイムスタンプを使用）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    config_path = os.path.join(save_dir, f"{experiment_name}_config.json")
    
    # torch.Tensorをリストに変換
    config_serializable = {}
    for key, value in config.items():
        if isinstance(value, torch.Tensor):
            config_serializable[key] = value.tolist()
        else:
            config_serializable[key] = value
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"実験設定を保存しました: {config_path}")


def load_experiment_config(config_path):
    """
    実験設定をJSONファイルから読み込み
    
    Args:
        config_path (str): 設定ファイルのパス
        
    Returns:
        dict: 実験設定の辞書
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def create_experiment_config_dict(args, experiment_name=None):
    """
    実験設定辞書を作成
    
    Args:
        args: コマンドライン引数
        experiment_name (str): 実験名
        
    Returns:
        dict: 実験設定の辞書
    """
    config = {
        'experiment_name': experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'timestamp': datetime.now().isoformat(),
        'model_type': getattr(args, 'model_type', 'unknown'),
        'dataset': getattr(args, 'dataset', 'unknown'),
        'qpp_pattern': getattr(args, 'qpp_pattern', None),
        'qpp_feature_indices': getattr(args, 'qpp_feature_indices', None),
        'learning_rate': getattr(args, 'learning_rate', None),
        'batch_size': getattr(args, 'batch_size', None),
        'epochs': getattr(args, 'epochs', None),
        'hidden_size': getattr(args, 'hidden_size', None),
        'dropout': getattr(args, 'dropout', None),
    }
    
    return config
