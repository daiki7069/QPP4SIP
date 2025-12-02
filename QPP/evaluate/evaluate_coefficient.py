"""
各指標同士のピアソン相関係数をヒートマップで可視化
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict
import os
import json


def plot_correlation_heatmaps(
    csv_path: str,
    metric_configs: Dict[str, Dict[str, str]],
    merge_config: Dict,
    output_dir: str,
    output_filename: str = 'correlation_heatmaps.png'
):
    """
    各指標同士のピアソン相関係数をヒートマップで可視化する
    
    Args:
        csv_path: csvのパス
        metric_configs: メトリクス名とCSVパス、カラム名の辞書
                       例: {'nqc': {'csv_path': '...', 'column': 'nqc'}, ...}
        merge_config: マージ設定（left_on, right_on, how）
        output_dir: 出力ディレクトリ
    """
    # ベースCSVを読み込む
    df = pd.read_csv(csv_path)
    
    # 全メトリクスのデータをマージして1つのDataFrameにまとめる
    # まずベースデータを準備
    merged_df = df[merge_config['left_on'] + ['response_type']].copy()
    
    # 各メトリクスのデータを読み込んでマージ
    for metric_name, config in metric_configs.items():
        # JSONファイルの場合は特別な処理
        if config.get('is_json', False):
            # JSONファイルからデータを読み込む
            with open(config['csv_path'], 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # ネストされたリストをフラット化
            flattened_data = []
            for conversation in json_data:
                for turn in conversation:
                    flattened_data.append(turn)
            
            # DataFrameに変換
            metric_df = pd.DataFrame(flattened_data)
            column_name = config['column']
            
            # マージキーを準備（conv_id -> dialogue_id）
            metric_subset = metric_df[['conv_id', 'turn_id', column_name]].copy()
            metric_subset = metric_subset.rename(columns={'conv_id': 'dialogue_id'})
            metric_subset = metric_subset.rename(columns={column_name: metric_name})
        else:
            # CSVファイルの場合
            metric_df = pd.read_csv(config['csv_path'])
            column_name = config['column']
            
            # 必要なカラムのみを選択（マージキーとメトリクス値）
            # マージキーをleft_onに合わせるために、right_onをleft_onにマッピング
            metric_subset = metric_df[merge_config['right_on'] + [column_name]].copy()
            
            # right_onカラムをleft_onカラム名に変更（マージキーの統一）
            rename_dict = dict(zip(merge_config['right_on'], merge_config['left_on']))
            metric_subset = metric_subset.rename(columns=rename_dict)
            
            # メトリクス値のカラム名をメトリクス名に変更
            metric_subset = metric_subset.rename(columns={column_name: metric_name})
        
        # マージ（left_onとright_onが同じカラム名になる）
        merged_df = merged_df.merge(
            metric_subset,
            on=merge_config['left_on'],
            how=merge_config['how']
        )
    
    # response_typeでラベルを分ける
    response_type_col = 'response_type'
    if response_type_col not in merged_df.columns:
        raise ValueError(f"'{response_type_col}'カラムが見つかりません")
    
    # [SEP]で分割されている場合があるので、最初の部分を取得して正規化
    merged_df['response_type_clean'] = merged_df[response_type_col].str.split(' [SEP] ', regex=False).str[0].str.strip()
    
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 指標のスコアカラムのみを抽出
    metric_columns = list(metric_configs.keys())
    
    # NaN値を除外
    merged_df_clean = merged_df[metric_columns + ['response_type_clean']].dropna()
    
    # 相関係数を計算するためのデータを準備
    # 1. 全体の相関係数
    correlation_all = merged_df_clean[metric_columns].corr(method='pearson')
    
    # 2. ラベル別の相関係数
    response_types = sorted(merged_df_clean['response_type_clean'].unique())
    correlations_by_label = {}
    
    for rt in response_types:
        df_label = merged_df_clean[merged_df_clean['response_type_clean'] == rt]
        if len(df_label) > 1:  # 相関係数を計算するには最低2行必要
            correlations_by_label[rt] = df_label[metric_columns].corr(method='pearson')
    
    # ヒートマップを作成（3つ：全体、ラベル1、ラベル2）
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # カラーマップの範囲を統一（-1から1）
    vmin, vmax = -1, 1
    
    # 1. 全体の相関係数ヒートマップ
    sns.heatmap(
        correlation_all,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'shrink': 0.8},
        ax=axes[0]
    )
    axes[0].set_title('All Labels', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Metrics', fontsize=12)
    axes[0].set_ylabel('Metrics', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # 2-3. ラベル別の相関係数ヒートマップ
    label_idx = 1
    for rt in response_types[:2]:  # 最大2つのラベルを表示
        if rt in correlations_by_label:
            sns.heatmap(
                correlations_by_label[rt],
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'shrink': 0.8},
                ax=axes[label_idx]
            )
            axes[label_idx].set_title(f'Label: {rt}', fontsize=14, fontweight='bold')
            axes[label_idx].set_xlabel('Metrics', fontsize=12)
            axes[label_idx].set_ylabel('Metrics', fontsize=12)
            axes[label_idx].tick_params(axis='x', rotation=45)
            axes[label_idx].tick_params(axis='y', rotation=0)
            label_idx += 1
    
    # 3つ目のラベルがない場合は非表示
    if label_idx < 3:
        axes[2].set_visible(False)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"相関係数ヒートマップを保存しました: {output_path}")
    
    plt.close()
    
    # 統計情報を表示
    print("\n=== 相関係数の統計情報 ===")
    print(f"全体のデータ数: {len(merged_df_clean)}")
    print("\n全体の相関係数:")
    print(correlation_all)
    
    for rt in response_types[:2]:
        if rt in correlations_by_label:
            print(f"\n{rt}の相関係数 (データ数: {len(merged_df_clean[merged_df_clean['response_type_clean'] == rt])}):")
            print(correlations_by_label[rt])


def evaluate_coefficients(split: str = 'train', dataset: str = 'INSCIT', mode: str = 'post_retrieval', include_ftplm: bool = True):
    """
    各指標同士のピアソン相関係数をヒートマップで可視化する
    
    Args:
        split: データセットの種類 (train または dev)
        dataset: データセット名 (INSCIT または AmbigNQ)
        mode: 評価モード (post_retrieval, retrieval_data, neural_qpp, nsp_graph)
        include_ftplm: FT-PLM指標を含めるかどうか（デフォルト: True）
    """
    # ベースパスの設定
    base_dir = '/home/daiki_shibata/pj/QPP4SIP'
    
    # 入力ファイルの設定
    csv_path = os.path.join(base_dir, 'QPP', 'raw_data', 'outputs', dataset, f'{split}.csv')
    output_dir = os.path.join(base_dir, 'QPP', 'evaluate', 'outputs', dataset)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # マージ設定
    merge_config = {
        'left_on': ['dialogue_id', 'turn_id'],
        'right_on': ['conv_id', 'turn_id'],
        'how': 'inner'
    }
    
    # モードに応じてメトリクス設定を取得
    if mode == 'post_retrieval':
        post_retrieval_outputs_dir = os.path.join(base_dir, 'QPP', 'post_retrieval', 'outputs', dataset)
        metric_configs = {
            'nqc': {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_nqc.csv'),
                'column': 'nqc'
            },
            'smv': {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_smv.csv'),
                'column': 'smv'
            },
            'nsv': {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_nsv.csv'),
                'column': 'nsv'
            },
            'similarity': {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_similarity.csv'),
                'column': 'mean_similarity'
            },
            'wig': {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_wig.csv'),
                'column': 'wig'
            },
            'acc': {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_coherency.csv'),
                'column': 'acc'
            },
            'clarity': {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{split}_clarity.csv'),
                'column': 'clarity'
            },
        }
    elif mode == 'retrieval_data':
        retrieval_csv_path = os.path.join(base_dir, 'QPP', 'retrieval_data', 'outputs', dataset, f'dpr_{split}_only_evidence.csv')
        metric_configs = {
            'num_evidence_docs': {
                'csv_path': retrieval_csv_path,
                'column': 'num_evidence_docs'
            },
            'mrr': {
                'csv_path': retrieval_csv_path,
                'column': 'mrr'
            },
            'ndcg@1': {
                'csv_path': retrieval_csv_path,
                'column': 'ndcg@1'
            },
            'ndcg@5': {
                'csv_path': retrieval_csv_path,
                'column': 'ndcg@5'
            },
            'ndcg@10': {
                'csv_path': retrieval_csv_path,
                'column': 'ndcg@10'
            },
            'ndcg@20': {
                'csv_path': retrieval_csv_path,
                'column': 'ndcg@20'
            },
            'ndcg@50': {
                'csv_path': retrieval_csv_path,
                'column': 'ndcg@50'
            },
            'ndcg@100': {
                'csv_path': retrieval_csv_path,
                'column': 'ndcg@100'
            }
        }
        # 不足している評価指標があれば追加（CSVに存在する場合のみ）
        retrieval_df = pd.read_csv(retrieval_csv_path)
        additional_metrics = {
            'map': 'map',
            'precision@1': 'precision@1',
            'precision@5': 'precision@5',
            'precision@10': 'precision@10',
            'precision@20': 'precision@20',
            'precision@50': 'precision@50',
            'precision@100': 'precision@100',
            'recall@1': 'recall@1',
            'recall@5': 'recall@5',
            'recall@10': 'recall@10',
            'recall@20': 'recall@20',
            'recall@50': 'recall@50',
            'recall@100': 'recall@100',
        }
        for metric_name, column_name in additional_metrics.items():
            if column_name in retrieval_df.columns:
                metric_configs[metric_name] = {
                    'csv_path': retrieval_csv_path,
                    'column': column_name
                }
    elif mode == 'neural_qpp':
        neural_qpp_outputs_dir = os.path.join(base_dir, 'QPP', 'neural_qpp', 'outputs', dataset)
        metric_configs = {}
        for model_type in ['bi', 'cross']:
            for metric in ['map@20', 'map']:
                model_dir = os.path.join(neural_qpp_outputs_dir, f'{model_type}_{metric}')
                csv_file = os.path.join(model_dir, f'{split}_bertqpp_{model_type}_{metric}.csv')
                if os.path.exists(csv_file):
                    metric_name = f'bertqpp_{model_type}_{metric}'
                    metric_configs[metric_name] = {
                        'csv_path': csv_file,
                        'column': f'bertqpp_{model_type}'
                    }
    elif mode == 'nsp_graph':
        nsp_graph_outputs_dir = os.path.join(base_dir, 'QPP', 'next_sentence_prediction', 'outputs', dataset)
        metric_configs = {}
        TOP_K_VALUES = []
        if os.path.exists(nsp_graph_outputs_dir):
            for filename in os.listdir(nsp_graph_outputs_dir):
                if filename.startswith(f'{split}_nsp_graph_topk') and filename.endswith('.csv'):
                    try:
                        top_k_str = filename.replace(f'{split}_nsp_graph_topk', '').replace('.csv', '')
                        top_k = int(top_k_str)
                        TOP_K_VALUES.append(top_k)
                    except ValueError:
                        continue
        TOP_K_VALUES = sorted(TOP_K_VALUES)
        for top_k in TOP_K_VALUES:
            csv_file = os.path.join(nsp_graph_outputs_dir, f'{split}_nsp_graph_topk{top_k}.csv')
            metric_name_nc = f'nsp_nc_topk{top_k}'
            metric_configs[metric_name_nc] = {
                'csv_path': csv_file,
                'column': 'node_connectivity',
                'top_k': top_k
            }
            metric_name_anc = f'nsp_anc_topk{top_k}'
            metric_configs[metric_name_anc] = {
                'csv_path': csv_file,
                'column': 'average_node_connectivity',
                'top_k': top_k
            }
            metric_name_density = f'nsp_density_topk{top_k}'
            metric_configs[metric_name_density] = {
                'csv_path': csv_file,
                'column': 'density',
                'top_k': top_k
            }
    else:
        raise ValueError(f"未知のモード: {mode}")
    
    if not metric_configs:
        print(f"警告: {mode}モードのメトリクスが見つかりません。")
        return
    
    # FT-PLM指標も追加
    if include_ftplm:
        ftplm_outputs_dir = os.path.join(base_dir, 'SIP', 'FT-PLM', 'output', dataset)
        if os.path.exists(ftplm_outputs_dir):
            # 利用可能なFT-PLMモデルを検出
            for exp_dir in os.listdir(ftplm_outputs_dir):
                exp_path = os.path.join(ftplm_outputs_dir, exp_dir)
                if not os.path.isdir(exp_path):
                    continue
                
                # JSONファイルを探す
                json_filename = f'{split}_with_predictions.json'
                json_path = os.path.join(exp_path, json_filename)
                
                if os.path.exists(json_path):
                    # モデル名を抽出（例: INSCIT_bert-base_lr2e-05_bs16_kfold5 -> ftplm_bert-base）
                    model_name = exp_dir
                    if 'bert-base' in model_name.lower():
                        metric_name = 'ftplm_bert-base'
                    elif 'roberta-base' in model_name.lower():
                        metric_name = 'ftplm_roberta-base'
                    elif 'bert' in model_name.lower():
                        metric_name = 'ftplm_bert'
                    elif 'roberta' in model_name.lower():
                        metric_name = 'ftplm_roberta'
                    else:
                        # デフォルトでモデル名を使用
                        metric_name = f'ftplm_{exp_dir}'
                    
                    metric_configs[metric_name] = {
                        'csv_path': json_path,
                        'column': 'prob_clarification',
                        'is_json': True
                    }
                    print(f"FT-PLM指標を追加: {metric_name} ({json_path})")
    
    # 相関係数ヒートマップの可視化
    print(f"=== {mode.upper()} 指標の相関係数ヒートマップ ===")
    output_filename = f'correlation_heatmaps_{mode}_{split}.png'
    plot_correlation_heatmaps(
        csv_path=csv_path,
        metric_configs=metric_configs,
        merge_config=merge_config,
        output_dir=output_dir,
        output_filename=output_filename
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="各指標同士のピアソン相関係数をヒートマップで可視化")
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=['INSCIT', 'AmbigNQ'],
        help='データセット名（INSCIT または AmbigNQ）'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['post_retrieval', 'retrieval_data', 'neural_qpp', 'nsp_graph'],
        default='post_retrieval',
        help='評価モード'
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=['train', 'dev'],
        default='train',
        help='データセットの種類 (train または dev, デフォルト: train)'
    )
    parser.add_argument(
        '--include_ftplm',
        action='store_true',
        default=True,
        help='FT-PLM指標を含めるかどうか（デフォルト: True）'
    )
    parser.add_argument(
        '--no_ftplm',
        dest='include_ftplm',
        action='store_false',
        help='FT-PLM指標を含めない'
    )
    
    args = parser.parse_args()
    
    evaluate_coefficients(split=args.split, dataset=args.dataset, mode=args.mode, include_ftplm=args.include_ftplm)

