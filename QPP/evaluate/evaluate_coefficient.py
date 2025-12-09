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
    output_filename: str = 'correlation_heatmaps.png',
    use_cv: bool = False,
    train_csv_path: str = None,
    dev_csv_path: str = None
):
    """
    各指標同士のピアソン相関係数をヒートマップで可視化する
    
    Args:
        csv_path: csvのパス（use_cv=Trueの場合はNone）
        metric_configs: メトリクス名とCSVパス、カラム名の辞書
                       例: {'nqc': {'csv_path': '...', 'column': 'nqc'}, ...}
                       use_cv=Trueの場合は {'nqc': {'csv_paths': [...], 'column': 'nqc'}, ...}
        merge_config: マージ設定（left_on, right_on, how）
        output_dir: 出力ディレクトリ
        use_cv: trainとdevを統合するかどうか
        train_csv_path: train CSVのパス（use_cv=Trueの場合）
        dev_csv_path: dev CSVのパス（use_cv=Trueの場合）
    """
    # ベースCSVを読み込む（use_cvの場合は統合）
    if use_cv:
        train_df = pd.read_csv(train_csv_path)
        dev_df = pd.read_csv(dev_csv_path)
        df = pd.concat([train_df, dev_df], axis=0, ignore_index=True)
        print(f"trainとdevを統合: train={len(train_df)}サンプル, dev={len(dev_df)}サンプル, 合計={len(df)}サンプル")
    else:
        df = pd.read_csv(csv_path)
    
    # 全メトリクスのデータをマージして1つのDataFrameにまとめる
    # まずベースデータを準備
    merged_df = df[merge_config['left_on'] + ['response_type']].copy()
    
    # 型を統一（マージキーの型を統一）
    merged_df['dialogue_id'] = merged_df['dialogue_id'].astype(str)
    merged_df['turn_id'] = merged_df['turn_id'].astype(int)
    
    # 各メトリクスのデータを読み込んでマージ
    for metric_name, config in metric_configs.items():
        # JSONファイルの場合は特別な処理（最初にチェック）
        if config.get('is_json', False):
            # use_cvの場合は複数のJSONファイルから統合
            if use_cv and 'csv_paths' in config:
                # 複数のJSONファイルを読み込んで統合
                all_flattened_data = []
                for json_path_item in config['csv_paths']:
                    if os.path.exists(json_path_item):
                        with open(json_path_item, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        
                        # ネストされたリストをフラット化
                        for conversation in json_data:
                            for turn in conversation:
                                all_flattened_data.append(turn)
                
                if not all_flattened_data:
                    print(f"警告: {metric_name}のJSONファイルが見つかりません。スキップします。")
                    continue
                
                # DataFrameに変換
                metric_df = pd.DataFrame(all_flattened_data)
                column_name = config['column']
                
                # マージキーを準備（conv_id -> dialogue_id）
                metric_subset = metric_df[['conv_id', 'turn_id', column_name]].copy()
                metric_subset = metric_subset.rename(columns={'conv_id': 'dialogue_id'})
                metric_subset = metric_subset.rename(columns={column_name: metric_name})
                
                # 型を統一（turn_idをint型に変換、dialogue_idを文字列型に変換）
                metric_subset['turn_id'] = metric_subset['turn_id'].astype(int)
                metric_subset['dialogue_id'] = metric_subset['dialogue_id'].astype(str)
            else:
                # 単一のJSONファイルからデータを読み込む
                json_path = config.get('csv_path') or (config.get('csv_paths', [None])[0] if 'csv_paths' in config else None)
                if json_path and os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
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
                    
                    # 型を統一（turn_idをint型に変換、dialogue_idを文字列型に変換）
                    metric_subset['turn_id'] = metric_subset['turn_id'].astype(int)
                    metric_subset['dialogue_id'] = metric_subset['dialogue_id'].astype(str)
                else:
                    print(f"警告: {metric_name}のJSONファイルが見つかりません。スキップします。")
                    continue
        # CSVファイルの場合
        else:
            # use_cvの場合は複数のCSVパスから統合
            if use_cv and 'csv_paths' in config:
                # 複数のCSVファイルを読み込んで統合
                metric_dfs = []
                for csv_path_item in config['csv_paths']:
                    if os.path.exists(csv_path_item):
                        metric_df_item = pd.read_csv(csv_path_item)
                        metric_dfs.append(metric_df_item)
                
                if not metric_dfs:
                    print(f"警告: {metric_name}のCSVファイルが見つかりません。スキップします。")
                    continue
                
                # 統合
                metric_df = pd.concat(metric_dfs, axis=0, ignore_index=True)
                column_name = config['column']
                
                # 必要なカラムのみを選択（マージキーとメトリクス値）
                metric_subset = metric_df[merge_config['right_on'] + [column_name]].copy()
                
                # right_onカラムをleft_onカラム名に変更（マージキーの統一）
                rename_dict = dict(zip(merge_config['right_on'], merge_config['left_on']))
                metric_subset = metric_subset.rename(columns=rename_dict)
                
                # メトリクス値のカラム名をメトリクス名に変更
                metric_subset = metric_subset.rename(columns={column_name: metric_name})
                
                # 型を統一（turn_idをint型に変換、dialogue_idを文字列型に変換）
                if 'turn_id' in metric_subset.columns:
                    metric_subset['turn_id'] = metric_subset['turn_id'].astype(int)
                if 'dialogue_id' in metric_subset.columns:
                    metric_subset['dialogue_id'] = metric_subset['dialogue_id'].astype(str)
            else:
                # 単一のCSVファイルの場合
                csv_path_item = config.get('csv_path') or (config.get('csv_paths', [None])[0] if 'csv_paths' in config else None)
                if csv_path_item and os.path.exists(csv_path_item):
                    metric_df = pd.read_csv(csv_path_item)
                    column_name = config['column']
                    
                    # 必要なカラムのみを選択（マージキーとメトリクス値）
                    # マージキーをleft_onに合わせるために、right_onをleft_onにマッピング
                    metric_subset = metric_df[merge_config['right_on'] + [column_name]].copy()
                    
                    # right_onカラムをleft_onカラム名に変更（マージキーの統一）
                    rename_dict = dict(zip(merge_config['right_on'], merge_config['left_on']))
                    metric_subset = metric_subset.rename(columns=rename_dict)
                    
                    # メトリクス値のカラム名をメトリクス名に変更
                    metric_subset = metric_subset.rename(columns={column_name: metric_name})
                    
                    # 型を統一（turn_idをint型に変換、dialogue_idを文字列型に変換）
                    if 'turn_id' in metric_subset.columns:
                        metric_subset['turn_id'] = metric_subset['turn_id'].astype(int)
                    if 'dialogue_id' in metric_subset.columns:
                        metric_subset['dialogue_id'] = metric_subset['dialogue_id'].astype(str)
                else:
                    print(f"警告: {metric_name}のCSVファイルが見つかりません。スキップします。")
                    continue
        
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
    # 1. 全体の相関係数（ピアソン）
    correlation_all_pearson = merged_df_clean[metric_columns].corr(method='pearson')
    
    # 2. 全体の相関係数（スピアマン）
    correlation_all_spearman = merged_df_clean[metric_columns].corr(method='spearman')
    
    # 3. 全体の相関係数（ケンドール）
    correlation_all_kendall = merged_df_clean[metric_columns].corr(method='kendall')
    
    # 4. ラベル別の相関係数（ピアソン、スピアマン、ケンドール）
    response_types = sorted(merged_df_clean['response_type_clean'].unique())
    correlations_by_label_pearson = {}
    correlations_by_label_spearman = {}
    correlations_by_label_kendall = {}
    
    for rt in response_types:
        df_label = merged_df_clean[merged_df_clean['response_type_clean'] == rt]
        if len(df_label) > 1:  # 相関係数を計算するには最低2行必要
            correlations_by_label_pearson[rt] = df_label[metric_columns].corr(method='pearson')
            correlations_by_label_spearman[rt] = df_label[metric_columns].corr(method='spearman')
            correlations_by_label_kendall[rt] = df_label[metric_columns].corr(method='kendall')
    
    # ヒートマップを作成（ピアソン、スピアマン、ケンドールの3行、各3列：全体、ラベル1、ラベル2）
    # 図のサイズを大きくする（メトリクス数に応じて調整）
    num_metrics = len(metric_columns)
    fig_width = max(24, num_metrics * 1.5)  # メトリクス数に応じて幅を調整
    fig_height = max(24, num_metrics * 2.4)  # メトリクス数に応じて高さを調整（3行なので3倍）
    fig, axes = plt.subplots(3, 3, figsize=(fig_width, fig_height))
    
    # カラーマップの範囲を統一（-1から1）
    vmin, vmax = -1, 1
    
    # フォントサイズを小さく設定
    annot_fontsize = max(6, min(10, 100 // num_metrics))  # メトリクス数に応じて調整
    label_fontsize = max(7, min(10, 120 // num_metrics))
    title_fontsize = max(8, min(12, 140 // num_metrics))
    
    # ピアソン相関係数のヒートマップ（1行目）
    # 1. 全体の相関係数ヒートマップ（ピアソン）
    sns.heatmap(
        correlation_all_pearson,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'shrink': 0.8},
        ax=axes[0, 0],
        annot_kws={'size': annot_fontsize}  # 注釈のフォントサイズ
    )
    axes[0, 0].set_title('All Labels (Pearson)', fontsize=title_fontsize, fontweight='bold')
    axes[0, 0].set_xlabel('Metrics', fontsize=label_fontsize)
    axes[0, 0].set_ylabel('Metrics', fontsize=label_fontsize)
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
    axes[0, 0].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
    
    # 2-3. ラベル別の相関係数ヒートマップ（ピアソン）
    label_idx = 1
    for rt in response_types[:2]:  # 最大2つのラベルを表示
        if rt in correlations_by_label_pearson:
            sns.heatmap(
                correlations_by_label_pearson[rt],
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'shrink': 0.8},
                ax=axes[0, label_idx],
                annot_kws={'size': annot_fontsize}  # 注釈のフォントサイズ
            )
            axes[0, label_idx].set_title(f'Label: {rt} (Pearson)', fontsize=title_fontsize, fontweight='bold')
            axes[0, label_idx].set_xlabel('Metrics', fontsize=label_fontsize)
            axes[0, label_idx].set_ylabel('Metrics', fontsize=label_fontsize)
            axes[0, label_idx].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
            axes[0, label_idx].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
            label_idx += 1
    
    # 3つ目のラベルがない場合は非表示（ピアソン）
    if label_idx < 3:
        axes[0, 2].set_visible(False)
    
    # スピアマン相関係数のヒートマップ（2行目）
    # 1. 全体の相関係数ヒートマップ（スピアマン）
    sns.heatmap(
        correlation_all_spearman,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'shrink': 0.8},
        ax=axes[1, 0],
        annot_kws={'size': annot_fontsize}  # 注釈のフォントサイズ
    )
    axes[1, 0].set_title('All Labels (Spearman)', fontsize=title_fontsize, fontweight='bold')
    axes[1, 0].set_xlabel('Metrics', fontsize=label_fontsize)
    axes[1, 0].set_ylabel('Metrics', fontsize=label_fontsize)
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
    axes[1, 0].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
    
    # 2-3. ラベル別の相関係数ヒートマップ（スピアマン）
    label_idx = 1
    for rt in response_types[:2]:  # 最大2つのラベルを表示
        if rt in correlations_by_label_spearman:
            sns.heatmap(
                correlations_by_label_spearman[rt],
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'shrink': 0.8},
                ax=axes[1, label_idx],
                annot_kws={'size': annot_fontsize}  # 注釈のフォントサイズ
            )
            axes[1, label_idx].set_title(f'Label: {rt} (Spearman)', fontsize=title_fontsize, fontweight='bold')
            axes[1, label_idx].set_xlabel('Metrics', fontsize=label_fontsize)
            axes[1, label_idx].set_ylabel('Metrics', fontsize=label_fontsize)
            axes[1, label_idx].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
            axes[1, label_idx].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
            label_idx += 1
    
    # 3つ目のラベルがない場合は非表示（スピアマン）
    if label_idx < 3:
        axes[1, 2].set_visible(False)
    
    # ケンドール相関係数のヒートマップ（3行目）
    # 1. 全体の相関係数ヒートマップ（ケンドール）
    sns.heatmap(
        correlation_all_kendall,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'shrink': 0.8},
        ax=axes[2, 0],
        annot_kws={'size': annot_fontsize}  # 注釈のフォントサイズ
    )
    axes[2, 0].set_title('All Labels (Kendall)', fontsize=title_fontsize, fontweight='bold')
    axes[2, 0].set_xlabel('Metrics', fontsize=label_fontsize)
    axes[2, 0].set_ylabel('Metrics', fontsize=label_fontsize)
    axes[2, 0].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
    axes[2, 0].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
    
    # 2-3. ラベル別の相関係数ヒートマップ（ケンドール）
    label_idx = 1
    for rt in response_types[:2]:  # 最大2つのラベルを表示
        if rt in correlations_by_label_kendall:
            sns.heatmap(
                correlations_by_label_kendall[rt],
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'shrink': 0.8},
                ax=axes[2, label_idx],
                annot_kws={'size': annot_fontsize}  # 注釈のフォントサイズ
            )
            axes[2, label_idx].set_title(f'Label: {rt} (Kendall)', fontsize=title_fontsize, fontweight='bold')
            axes[2, label_idx].set_xlabel('Metrics', fontsize=label_fontsize)
            axes[2, label_idx].set_ylabel('Metrics', fontsize=label_fontsize)
            axes[2, label_idx].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
            axes[2, label_idx].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
            label_idx += 1
    
    # 3つ目のラベルがない場合は非表示（ケンドール）
    if label_idx < 3:
        axes[2, 2].set_visible(False)
    
    plt.tight_layout()
    
    # 保存
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"相関係数ヒートマップを保存しました: {output_path}")
    
    plt.close()
    
    # 統計情報を表示
    print("\n=== 相関係数の統計情報 ===")
    print(f"全体のデータ数: {len(merged_df_clean)}")
    print("\n全体の相関係数（ピアソン）:")
    print(correlation_all_pearson)
    print("\n全体の相関係数（スピアマン）:")
    print(correlation_all_spearman)
    print("\n全体の相関係数（ケンドール）:")
    print(correlation_all_kendall)
    
    for rt in response_types[:2]:
        if rt in correlations_by_label_pearson:
            print(f"\n{rt}の相関係数（ピアソン） (データ数: {len(merged_df_clean[merged_df_clean['response_type_clean'] == rt])}):")
            print(correlations_by_label_pearson[rt])
        if rt in correlations_by_label_spearman:
            print(f"\n{rt}の相関係数（スピアマン） (データ数: {len(merged_df_clean[merged_df_clean['response_type_clean'] == rt])}):")
            print(correlations_by_label_spearman[rt])
        if rt in correlations_by_label_kendall:
            print(f"\n{rt}の相関係数（ケンドール） (データ数: {len(merged_df_clean[merged_df_clean['response_type_clean'] == rt])}):")
            print(correlations_by_label_kendall[rt])


def evaluate_coefficients(split: str = 'train', dataset: str = 'INSCIT', mode: str = 'post_retrieval', include_ftplm: bool = True, use_cv: bool = False):
    """
    各指標同士のピアソン相関係数をヒートマップで可視化する
    
    Args:
        split: データセットの種類 (train または dev、use_cv=Trueの場合は無視される)
        dataset: データセット名 (INSCIT または AmbigNQ)
        mode: 評価モード (post_retrieval, retrieval_data, neural_qpp, nsp_graph)
        include_ftplm: FT-PLM指標を含めるかどうか（デフォルト: True）
        use_cv: trainとdevを統合して相関係数を計算するかどうか（デフォルト: False）
    """
    # ベースパスの設定
    base_dir = '/home/daiki_shibata/pj/QPP4SIP'
    
    # 入力ファイルの設定
    if use_cv:
        # trainとdevを統合
        train_csv_path = os.path.join(base_dir, 'QPP', 'raw_data', 'outputs', dataset, 'train.csv')
        dev_csv_path = os.path.join(base_dir, 'QPP', 'raw_data', 'outputs', dataset, 'dev.csv')
        csv_path = None  # 後で統合データを使用
    else:
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
    
    # 使用するsplitを決定（use_cvの場合は両方）
    splits_to_use = ['train', 'dev'] if use_cv else [split]
    
    # モードに応じてメトリクス設定を取得
    if mode == 'post_retrieval':
        post_retrieval_outputs_dir = os.path.join(base_dir, 'QPP', 'post_retrieval', 'outputs', dataset)
        metric_configs = {}
        
        # 各splitのメトリクスを収集
        for current_split in splits_to_use:
            metric_configs[f'nqc_{current_split}'] = {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{current_split}_nqc.csv'),
                'column': 'nqc',
                'split': current_split
            }
            metric_configs[f'smv_{current_split}'] = {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{current_split}_smv.csv'),
                'column': 'smv',
                'split': current_split
            }
            metric_configs[f'nsv_{current_split}'] = {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{current_split}_nsv.csv'),
                'column': 'nsv',
                'split': current_split
            }
            metric_configs[f'n_sigma_50_{current_split}'] = {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{current_split}_n_sigma_50.csv'),
                'column': 'n_sigma_50',
                'split': current_split
            }
            metric_configs[f'similarity_{current_split}'] = {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{current_split}_similarity.csv'),
                'column': 'mean_similarity',
                'split': current_split
            }
            metric_configs[f'wig_{current_split}'] = {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{current_split}_wig.csv'),
                'column': 'wig',
                'split': current_split
            }
            metric_configs[f'acc_{current_split}'] = {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{current_split}_coherency.csv'),
                'column': 'acc',
                'split': current_split
            }
            metric_configs[f'clarity_{current_split}'] = {
                'csv_path': os.path.join(post_retrieval_outputs_dir, f'{current_split}_clarity.csv'),
                'column': 'clarity',
                'split': current_split
            }
        
        # use_cvの場合は、メトリクス名からsplitサフィックスを削除して統合
        if use_cv:
            # 統合されたメトリクス設定を作成
            unified_metric_configs = {}
            base_metrics = ['nqc', 'smv', 'nsv', 'n_sigma_50', 'similarity', 'wig', 'acc', 'clarity']
            for base_metric in base_metrics:
                unified_metric_configs[base_metric] = {
                    'csv_paths': [config['csv_path'] for name, config in metric_configs.items() if name.startswith(f'{base_metric}_')],
                    'column': metric_configs[f'{base_metric}_train']['column'],
                    'split': 'combined'
                }
            metric_configs = unified_metric_configs
        else:
            # splitサフィックスを削除
            simplified_metric_configs = {}
            for name, config in metric_configs.items():
                base_name = name.replace(f'_{split}', '')
                simplified_metric_configs[base_name] = config
            metric_configs = simplified_metric_configs
    elif mode == 'retrieval_data':
        metric_configs = {}
        
        # 各splitのメトリクスを収集
        for current_split in splits_to_use:
            retrieval_csv_path = os.path.join(base_dir, 'QPP', 'retrieval_data', 'outputs', dataset, f'dpr_{current_split}_only_evidence.csv')
            if os.path.exists(retrieval_csv_path):
                retrieval_df = pd.read_csv(retrieval_csv_path)
                
                base_metrics = {
                    'num_evidence_docs': 'num_evidence_docs',
                    'mrr': 'mrr',
                    'ndcg@1': 'ndcg@1',
                    'ndcg@5': 'ndcg@5',
                    'ndcg@10': 'ndcg@10',
                    'ndcg@20': 'ndcg@20',
                    'ndcg@50': 'ndcg@50',
                    'ndcg@100': 'ndcg@100',
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
                
                for metric_name, column_name in base_metrics.items():
                    if column_name in retrieval_df.columns:
                        metric_configs[f'{metric_name}_{current_split}'] = {
                            'csv_path': retrieval_csv_path,
                            'column': column_name,
                            'split': current_split
                        }
        
        # use_cvの場合は、メトリクス名からsplitサフィックスを削除して統合
        if use_cv:
            unified_metric_configs = {}
            base_metrics = set([name.rsplit('_', 1)[0] for name in metric_configs.keys()])
            for base_metric in base_metrics:
                csv_paths = [config['csv_path'] for name, config in metric_configs.items() if name.startswith(f'{base_metric}_')]
                if csv_paths:
                    unified_metric_configs[base_metric] = {
                        'csv_paths': csv_paths,
                        'column': metric_configs[f'{base_metric}_train']['column'],
                        'split': 'combined'
                    }
            metric_configs = unified_metric_configs
        else:
            # splitサフィックスを削除
            simplified_metric_configs = {}
            for name, config in metric_configs.items():
                base_name = name.replace(f'_{split}', '')
                simplified_metric_configs[base_name] = config
            metric_configs = simplified_metric_configs
    elif mode == 'neural_qpp':
        neural_qpp_outputs_dir = os.path.join(base_dir, 'QPP', 'neural_qpp', 'outputs', dataset)
        metric_configs = {}
        
        for current_split in splits_to_use:
            for model_type in ['bi', 'cross']:
                for metric in ['map@20', 'map']:
                    model_dir = os.path.join(neural_qpp_outputs_dir, f'{model_type}_{metric}')
                    csv_file = os.path.join(model_dir, f'{current_split}_bertqpp_{model_type}_{metric}.csv')
                    if os.path.exists(csv_file):
                        metric_name = f'bertqpp_{model_type}_{metric}_{current_split}'
                        metric_configs[metric_name] = {
                            'csv_path': csv_file,
                            'column': f'bertqpp_{model_type}',
                            'split': current_split
                        }
        
        # use_cvの場合は統合
        if use_cv:
            unified_metric_configs = {}
            base_metrics = set([name.rsplit('_', 1)[0] for name in metric_configs.keys()])
            for base_metric in base_metrics:
                csv_paths = [config['csv_path'] for name, config in metric_configs.items() if name.startswith(f'{base_metric}_')]
                if csv_paths:
                    unified_metric_configs[base_metric] = {
                        'csv_paths': csv_paths,
                        'column': metric_configs[f'{base_metric}_train']['column'],
                        'split': 'combined'
                    }
            metric_configs = unified_metric_configs
        else:
            # splitサフィックスを削除
            simplified_metric_configs = {}
            for name, config in metric_configs.items():
                base_name = name.replace(f'_{split}', '')
                simplified_metric_configs[base_name] = config
            metric_configs = simplified_metric_configs
    elif mode == 'nsp_graph':
        nsp_graph_outputs_dir = os.path.join(base_dir, 'QPP', 'next_sentence_prediction', 'outputs', dataset)
        metric_configs = {}
        
        # 各splitのメトリクスを収集
        for current_split in splits_to_use:
            TOP_K_VALUES = []
            if os.path.exists(nsp_graph_outputs_dir):
                for filename in os.listdir(nsp_graph_outputs_dir):
                    if filename.startswith(f'{current_split}_nsp_graph_topk') and filename.endswith('.csv'):
                        try:
                            top_k_str = filename.replace(f'{current_split}_nsp_graph_topk', '').replace('.csv', '')
                            top_k = int(top_k_str)
                            TOP_K_VALUES.append(top_k)
                        except ValueError:
                            continue
            TOP_K_VALUES = sorted(TOP_K_VALUES)
            for top_k in TOP_K_VALUES:
                csv_file = os.path.join(nsp_graph_outputs_dir, f'{current_split}_nsp_graph_topk{top_k}.csv')
                if os.path.exists(csv_file):
                    metric_name_nc = f'nsp_nc_topk{top_k}_{current_split}'
                    metric_configs[metric_name_nc] = {
                        'csv_path': csv_file,
                        'column': 'node_connectivity',
                        'top_k': top_k,
                        'split': current_split
                    }
                    metric_name_anc = f'nsp_anc_topk{top_k}_{current_split}'
                    metric_configs[metric_name_anc] = {
                        'csv_path': csv_file,
                        'column': 'average_node_connectivity',
                        'top_k': top_k,
                        'split': current_split
                    }
                    metric_name_density = f'nsp_density_topk{top_k}_{current_split}'
                    metric_configs[metric_name_density] = {
                        'csv_path': csv_file,
                        'column': 'density',
                        'top_k': top_k,
                        'split': current_split
                    }
        
        # use_cvの場合は統合
        if use_cv:
            unified_metric_configs = {}
            # top_kごとに統合
            top_k_values = set([config['top_k'] for config in metric_configs.values()])
            for top_k in top_k_values:
                for metric_type in ['nc', 'anc', 'density']:
                    base_metric = f'nsp_{metric_type}_topk{top_k}'
                    csv_paths = [config['csv_path'] for name, config in metric_configs.items() if name.startswith(f'{base_metric}_')]
                    if csv_paths:
                        column_map = {'nc': 'node_connectivity', 'anc': 'average_node_connectivity', 'density': 'density'}
                        unified_metric_configs[base_metric] = {
                            'csv_paths': csv_paths,
                            'column': column_map[metric_type],
                            'top_k': top_k,
                            'split': 'combined'
                        }
            metric_configs = unified_metric_configs
        else:
            # splitサフィックスを削除
            simplified_metric_configs = {}
            for name, config in metric_configs.items():
                base_name = name.replace(f'_{split}', '')
                simplified_metric_configs[base_name] = config
            metric_configs = simplified_metric_configs
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
                
                # モデル名を抽出（例: INSCIT_bert-base_lr2e-05_bs16_kfold5 -> ftplm_bert-base）
                model_name = exp_dir
                if 'bert-base' in model_name.lower():
                    base_metric_name = 'ftplm_bert-base'
                elif 'roberta-base' in model_name.lower():
                    base_metric_name = 'ftplm_roberta-base'
                elif 'bert' in model_name.lower():
                    base_metric_name = 'ftplm_bert'
                elif 'roberta' in model_name.lower():
                    base_metric_name = 'ftplm_roberta'
                else:
                    # デフォルトでモデル名を使用
                    base_metric_name = f'ftplm_{exp_dir}'
                
                if use_cv:
                    # trainとdevのJSONファイルを探す
                    json_paths = []
                    for current_split in ['train', 'dev']:
                        json_filename = f'{current_split}_with_predictions.json'
                        json_path = os.path.join(exp_path, json_filename)
                        if os.path.exists(json_path):
                            json_paths.append(json_path)
                    
                    if json_paths:
                        metric_configs[base_metric_name] = {
                            'csv_paths': json_paths,
                            'column': 'prob_clarification',
                            'is_json': True,
                            'split': 'combined'
                        }
                        print(f"FT-PLM指標を追加: {base_metric_name} ({len(json_paths)}ファイル統合)")
                else:
                    # JSONファイルを探す
                    json_filename = f'{split}_with_predictions.json'
                    json_path = os.path.join(exp_path, json_filename)
                    
                    if os.path.exists(json_path):
                        metric_configs[base_metric_name] = {
                            'csv_path': json_path,
                            'column': 'prob_clarification',
                            'is_json': True
                        }
                        print(f"FT-PLM指標を追加: {base_metric_name} ({json_path})")
    
    # 相関係数ヒートマップの可視化
    print(f"=== {mode.upper()} 指標の相関係数ヒートマップ ===")
    if use_cv:
        print("（trainとdevを統合して計算）")
        output_filename = f'correlation_heatmaps_{mode}_cv.png'
    else:
        output_filename = f'correlation_heatmaps_{mode}_{split}.png'
    
    plot_correlation_heatmaps(
        csv_path=csv_path,
        metric_configs=metric_configs,
        merge_config=merge_config,
        output_dir=output_dir,
        output_filename=output_filename,
        use_cv=use_cv,
        train_csv_path=train_csv_path if use_cv else None,
        dev_csv_path=dev_csv_path if use_cv else None
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
        help='データセットの種類 (train または dev, デフォルト: train、--use-cvが指定されている場合は無視される)'
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
    parser.add_argument(
        '--use-cv',
        action='store_true',
        help='trainとdevを統合して相関係数を計算する（デフォルト: False）'
    )
    
    args = parser.parse_args()
    
    evaluate_coefficients(split=args.split, dataset=args.dataset, mode=args.mode, include_ftplm=args.include_ftplm, use_cv=args.use_cv)

