"""
Neural QPP指標の可視化
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os


def visualize_all_metrics(
    csv_path: str,
    metric_configs: Dict[str, Dict[str, str]],
    merge_config: Dict,
    output_dir: str
):
    """
    全Neural QPP指標を1つの図にまとめて可視化する
    
    Args:
        csv_path: csvのパス
        metric_configs: メトリクス名とCSVパス、カラム名の辞書
                       例: {'bertqpp_bi': {'csv_path': '...', 'column': 'bertqpp_bi'}, ...}
        merge_config: マージ設定（left_on, right_on, how）
        output_dir: 出力ディレクトリ
    """
    # dev.csvを読み込む
    df = pd.read_csv(csv_path)
    
    # 全メトリクスのデータをマージ
    merged_data = {}
    for metric_name, config in metric_configs.items():
        metric_df = pd.read_csv(config['csv_path'])
        merged_df = df.merge(
            metric_df,
            left_on=merge_config['left_on'],
            right_on=merge_config['right_on'],
            how=merge_config['how']
        )
        merged_data[metric_name] = merged_df
    
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 全指標を1つの図にまとめる（ボックスプロット）
    num_metrics = len(metric_configs)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))
    if num_metrics == 1:
        axes = [axes]
    
    response_type_col = 'response_type'
    
    for idx, (metric_name, merged_df) in enumerate(merged_data.items()):
        ax = axes[idx]
        
        # カラム名を取得
        column_name = metric_configs[metric_name]['column']
        
        # response_typeでグループ化
        direct_answer_values = merged_df[merged_df[response_type_col].str.contains('directAnswer', na=False)][column_name].dropna()
        clarification_values = merged_df[merged_df[response_type_col].str.contains('clarification', na=False)][column_name].dropna()
        
        # ボックスプロット
        box_data = [direct_answer_values, clarification_values]
        bp = ax.boxplot(box_data, labels=['Direct Answer', 'Clarification'], patch_artist=True)
        
        # 色を設定
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_title(f'{metric_name}', fontsize=14, fontweight='bold')
        ax.set_ylabel('QPP Score', fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, 'neural_qpp_metrics_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可視化結果を保存しました: {output_path}")
    plt.close()

