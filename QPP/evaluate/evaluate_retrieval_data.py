"""
Retrieval data QPP指標の可視化
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict


def visualize_all_metrics(
    csv_path: str,
    metric_configs: Dict[str, Dict[str, str]],
    merge_config: Dict,
    output_dir: str
):
    """
    全Retrieval指標を1つの図にまとめて可視化する
    
    Args:
        csv_path: csvのパス
        metric_configs: メトリクス名とCSVパス、カラム名の辞書
                       例: {'found_ratio': {'csv_path': '...', 'column': 'found_ratio'}, ...}
        merge_config: マージ設定（left_on, right_on, how）
        output_dir: 出力ディレクトリ
    """
    # dev.csvを読み込む
    df = pd.read_csv(csv_path)
    
    # 全メトリクスのデータをマージ
    merged_data = {}
    for metric_name, config in metric_configs.items():
        metric_df = pd.read_csv(config['csv_path'])
        
        # num_evidence_docsより左のカラムを無視（可視化に不要なカラムを除外）
        if 'num_evidence_docs' in metric_df.columns:
            num_evidence_idx = metric_df.columns.get_loc('num_evidence_docs')
            # num_evidence_docsより左のカラムを除外
            columns_to_use = metric_df.columns[num_evidence_idx:].tolist()
            # マージに必要なカラム（right_on）も含める
            for col in merge_config['right_on']:
                if col in metric_df.columns and col not in columns_to_use:
                    columns_to_use.insert(0, col)
            metric_df = metric_df[columns_to_use]
        
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
    # 複数行に分けて表示（1行に5つまで）
    cols = min(5, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    response_type_col = 'response_type'
    
    for idx, (metric_name, config) in enumerate(metric_configs.items()):
        merged_df = merged_data[metric_name]
        column_name = config['column']
        
        # NaN値を除外
        merged_df = merged_df.dropna(subset=[column_name])
        
        # ボックスプロットを作成
        response_types = merged_df[response_type_col].unique()
        data_to_plot = [merged_df[merged_df[response_type_col] == rt][column_name].values 
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
    
    # 余ったサブプロットを非表示
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    # 保存
    output_path = f"{output_dir}/all_retrieval_metrics_by_response_type.png"
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

