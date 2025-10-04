"""
実験結果サマリー関連のユーティリティ関数
"""
import os
import json
import pandas as pd
from datetime import datetime


def generate_experiment_summary(experiment_dir, results_dict):
    """
    実験結果のサマリーを生成
    
    Args:
        experiment_dir (str): 実験ディレクトリ
        results_dict (dict): 実験結果の辞書
        
    Returns:
        str: サマリーファイルのパス
    """
    summary_dir = os.path.join(experiment_dir, 'results')
    os.makedirs(summary_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(summary_dir, f"summary_{timestamp}.json")
    
    # 結果をJSONで保存
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    return summary_file


def create_experiment_comparison_table(experiment_dirs, output_file=None):
    """
    複数の実験結果を比較するテーブルを作成
    
    Args:
        experiment_dirs (list): 実験ディレクトリのリスト
        output_file (str, optional): 出力ファイルのパス
        
    Returns:
        pd.DataFrame: 比較テーブル
    """
    results = []
    
    for exp_dir in experiment_dirs:
        # 各実験ディレクトリから結果を読み込み
        results_dir = os.path.join(exp_dir, 'results')
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.startswith('summary_') and file.endswith('.json'):
                    file_path = os.path.join(results_dir, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                        result['experiment_dir'] = exp_dir
                        results.append(result)
    
    if not results:
        return pd.DataFrame()
    
    # DataFrameに変換
    df = pd.DataFrame(results)
    
    # 出力ファイルが指定されている場合は保存
    if output_file:
        df.to_csv(output_file, index=False, encoding='utf-8')
    
    return df


def print_experiment_summary(results_dict):
    """
    実験結果のサマリーをコンソールに出力
    
    Args:
        results_dict (dict): 実験結果の辞書
    """
    print("\n" + "=" * 60)
    print("実験結果サマリー")
    print("=" * 60)
    
    for key, value in results_dict.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for sub_key, sub_value in value.items():
                print(f"  {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 60)
