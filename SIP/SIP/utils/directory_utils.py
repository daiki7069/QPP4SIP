"""
実験ディレクトリ管理関連のユーティリティ関数
"""
import os
from datetime import datetime


def create_experiment_directory(base_dir, experiment_name=None):
    """
    実験用ディレクトリを作成
    
    Args:
        base_dir (str): ベースディレクトリ
        experiment_name (str): 実験名（指定しない場合はタイムスタンプを使用）
        
    Returns:
        str: 作成された実験ディレクトリのパス
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    experiment_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # サブディレクトリも作成
    subdirs = ['logs', 'models', 'results', 'configs']
    for subdir in subdirs:
        os.makedirs(os.path.join(experiment_dir, subdir), exist_ok=True)
    
    return experiment_dir


def get_experiment_timestamp():
    """実験用のタイムスタンプを取得"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
