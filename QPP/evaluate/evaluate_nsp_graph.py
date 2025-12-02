"""
NSP Graph QPP指標の可視化
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
    output_dir: str,
    top_k_values: List[int] = None
):
    """
    全NSP Graph QPP指標を1つの図にまとめて可視化する（全top_k値を含む）
    
    Args:
        csv_path: csvのパス
        metric_configs: メトリクス名とCSVパス、カラム名の辞書
                       例: {'nsp_nc_topk10': {'csv_path': '...', 'column': 'node_connectivity', 'top_k': 10}, ...}
        merge_config: マージ設定（left_on, right_on, how）
        output_dir: 出力ディレクトリ
        top_k_values: top_k値のリスト（オプション）
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
    
    # メトリクスタイプごとにグループ化（NC, ANC, density）
    metric_types = {
        'NC': 'node_connectivity',
        'ANC': 'average_node_connectivity',
        'Density': 'density'
    }
    
    # 3つのメトリクスタイプごとにサブプロットを作成
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    response_type_col = 'response_type'
    
    for idx, (metric_type, column_name) in enumerate(metric_types.items()):
        ax = axes[idx]
        
        # このメトリクスタイプに該当する全てのtop_k値のデータを収集
        all_data = []
        labels = []
        
        # response_typeの種類を先に取得
        response_types_set = set()
        for metric_name, merged_df in merged_data.items():
            config = metric_configs[metric_name]
            if config['column'] == column_name:
                df_clean = merged_df.dropna(subset=[column_name])
                df_clean['response_type_clean'] = df_clean[response_type_col].str.split(' [SEP] ', regex=False).str[0].str.strip()
                response_types_set.update(df_clean['response_type_clean'].unique())
        
        response_types_list = sorted(list(response_types_set))
        
        # top_k値ごと、response_typeごとにデータを整理
        for top_k in (top_k_values or []):
            for metric_name, merged_df in merged_data.items():
                config = metric_configs[metric_name]
                if config['column'] == column_name and config.get('top_k') == top_k:
                    # NaN値を除外
                    df_clean = merged_df.dropna(subset=[column_name])
                    df_clean['response_type_clean'] = df_clean[response_type_col].str.split(' [SEP] ', regex=False).str[0].str.strip()
                    
                    # 各response_typeごとにデータを追加
                    for rt in response_types_list:
                        values = df_clean[df_clean['response_type_clean'] == rt][column_name].values
                        if len(values) > 0:
                            all_data.append(values)
                            labels.append(f'topk{top_k}\n{rt}')
                    break
        
        if all_data:
            # ボックスプロットを作成
            bp = ax.boxplot(all_data, labels=labels, patch_artist=True)
            
            # ボックスに色を付ける（top_kごとに異なる色）
            if top_k_values:
                colors = plt.cm.Set3(np.linspace(0, 1, len(top_k_values)))
                num_response_types = len(response_types_list)
                for i, patch in enumerate(bp['boxes']):
                    # top_k値に応じて色を割り当て
                    top_k_idx = i // num_response_types if num_response_types > 0 else 0
                    patch.set_facecolor(colors[top_k_idx % len(colors)])
                    patch.set_alpha(0.7)
            
            ax.set_title(f'{metric_type}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Top-k / Response Type', fontsize=10)
            ax.set_ylabel(metric_type, fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, f'nsp_graph_metrics_all_topk_by_response_type.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"全指標の可視化を保存しました: {output_path}")
    
    plt.close()
    
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

