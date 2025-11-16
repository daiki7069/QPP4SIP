"""
Post-retrieval QPP指標の可視化
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List


def visualize_all_metrics(
    csv_path: str,
    metric_configs: Dict[str, Dict[str, str]],
    merge_config: Dict,
    output_dir: str
):
    """
    全QPP指標を1つの図にまとめて可視化する
    
    Args:
        csv_path: csvのパス
        metric_configs: メトリクス名とCSVパス、カラム名の辞書
                       例: {'entropy': {'csv_path': '...', 'column': 'entropy'}, ...}
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
    
    for idx, (metric_name, config) in enumerate(metric_configs.items()):
        merged_df = merged_data[metric_name]
        column_name = config['column']
        
        # NaN値を除外
        merged_df = merged_df.dropna(subset=[column_name])
        
        # ボックスプロットを作成
        # [SEP]で分割されている場合があるので、最初の部分を取得して正規化
        merged_df['response_type_clean'] = merged_df[response_type_col].str.split(' [SEP] ', regex=False).str[0].str.strip()
        response_types = merged_df['response_type_clean'].unique()
        data_to_plot = [merged_df[merged_df['response_type_clean'] == rt][column_name].values 
                       for rt in response_types]
        
        bp = axes[idx].boxplot(data_to_plot, labels=response_types, patch_artist=True)
        
        # ボックスに色を付ける
        colors = plt.cm.Set3(np.linspace(0, 1, len(response_types)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        axes[idx].set_title(f'{metric_name.upper()}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Response Type', fontsize=10)
        axes[idx].set_ylabel(metric_name.upper(), fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    output_path = f"{output_dir}/all_metrics_by_response_type.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"全指標の可視化を保存しました: {output_path}")
    
    plt.show()
    
    # 統計情報を表示
    print("\n=== 統計情報 ===")
    for metric_name, config in metric_configs.items():
        merged_df = merged_data[metric_name]
        column_name = config['column']
        merged_df = merged_df.dropna(subset=[column_name])
        
        print(f"\n{metric_name.upper()}:")
        print(f"  結合後のデータ数: {len(merged_df)}")
        print(f"  {column_name}の統計:")
        print(merged_df[column_name].describe())
