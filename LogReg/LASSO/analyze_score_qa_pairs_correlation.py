#!/usr/bin/env python3
"""
スコア（QPP、BERT、RoBERTa、統合）とQAペア数の相関を分析し、可視化するスクリプト
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 日本語フォントの設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_qa_pairs_count(data_path: Path) -> Dict[Tuple[str, int], int]:
    """
    dev.jsonからqa_pairs_countを読み込む
    
    Returns:
        {(conv_id, turn_id): qa_pairs_count}の辞書
    """
    print(f"[INFO] Loading QA pairs count from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    qa_pairs_dict = {}
    for conversation in data:
        for turn in conversation:
            conv_id = str(turn.get('conv_id', ''))
            turn_id = int(turn.get('turn_id', 1))
            qa_pairs_count = int(turn.get('qa_pairs_count', 0))
            key = (conv_id, turn_id)
            qa_pairs_dict[key] = qa_pairs_count
    
    print(f"[INFO] Loaded QA pairs count for {len(qa_pairs_dict)} entries")
    return qa_pairs_dict


def load_bert_scores(predictions_path: Path) -> Dict[Tuple[str, int], float]:
    """
    BERTの予測スコア（logit_clarification）を読み込む
    
    Returns:
        {(conv_id, turn_id): logit_clarification}の辞書
    """
    print(f"[INFO] Loading BERT scores from {predictions_path}...")
    with open(predictions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scores_dict = {}
    for conversation in data:
        for turn in conversation:
            conv_id = str(turn.get('conv_id', ''))
            turn_id = int(turn.get('turn_id', 1))
            logit = turn.get('logit_clarification')
            if logit is not None:
                key = (conv_id, turn_id)
                scores_dict[key] = float(logit)
    
    print(f"[INFO] Loaded BERT scores for {len(scores_dict)} entries")
    return scores_dict


def load_roberta_scores(predictions_path: Path) -> Dict[Tuple[str, int], float]:
    """
    RoBERTaの予測スコア（logit_clarification）を読み込む
    
    Returns:
        {(conv_id, turn_id): logit_clarification}の辞書
    """
    print(f"[INFO] Loading RoBERTa scores from {predictions_path}...")
    with open(predictions_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    scores_dict = {}
    for conversation in data:
        for turn in conversation:
            conv_id = str(turn.get('conv_id', ''))
            turn_id = int(turn.get('turn_id', 1))
            logit = turn.get('logit_clarification')
            if logit is not None:
                key = (conv_id, turn_id)
                scores_dict[key] = float(logit)
    
    print(f"[INFO] Loaded RoBERTa scores for {len(scores_dict)} entries")
    return scores_dict


def load_qpp_scores(qpp_dir: Path, split: str = 'dev') -> Dict[str, Dict[Tuple[str, int], float]]:
    """
    QPPスコアを読み込む
    
    Returns:
        {metric_name: {(conv_id, turn_id): score}}の辞書
    """
    print(f"[INFO] Loading QPP scores from {qpp_dir}...")
    qpp_scores = {}
    
    # 主要なQPP指標
    qpp_files = {
        'clarity': f'{split}_clarity.json',
        'wig': f'{split}_wig.json',
        'nqc': f'{split}_nqc.json',
        'smv': f'{split}_smv.json',
        'n_sigma_50': f'{split}_n_sigma_50.json',
        'avgidf': f'{split}_avgidf.json',
        'maxidf': f'{split}_maxidf.json',
    }
    
    for metric_name, filename in qpp_files.items():
        file_path = qpp_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scores_dict = {}
            for conversation in data:
                for turn in conversation:
                    conv_id = str(turn.get('conv_id', ''))
                    turn_id = int(turn.get('turn_id', 1))
                    # フィールド名は指標名（clarity, wig, nqcなど）
                    score = turn.get(metric_name)
                    if score is None:
                        # フォールバック: scoreフィールドを試す
                        score = turn.get('score')
                    if score is not None:
                        key = (conv_id, turn_id)
                        scores_dict[key] = float(score)
            
            if scores_dict:
                qpp_scores[metric_name] = scores_dict
                print(f"[INFO] Loaded {metric_name} scores for {len(scores_dict)} entries")
        else:
            print(f"[WARN] QPP file not found: {file_path}")
    
    return qpp_scores


def load_integrated_score(
    predictions_path: Path,
    data_path: Path,
    split: str = 'dev'
) -> Dict[Tuple[str, int], float]:
    """
    統合スコア（LogRegの予測確率）を読み込む
    データの順序に基づいてマッピングする
    
    Returns:
        {(conv_id, turn_id): probability}の辞書
    """
    print(f"[INFO] Loading integrated scores from {predictions_path}...")
    if not predictions_path.exists():
        print(f"[WARN] Integrated score file not found: {predictions_path}")
        return {}
    
    # データセットの順序を取得
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data_keys = []
    for conversation in data:
        for turn in conversation:
            conv_id = str(turn.get('conv_id', ''))
            turn_id = int(turn.get('turn_id', 1))
            data_keys.append((conv_id, turn_id))
    
    # CSVファイルを読み込む
    df = pd.read_csv(predictions_path)
    
    scores_dict = {}
    if 'y_pred_proba' in df.columns:
        # データの順序に基づいてマッピング
        for i, (key, prob) in enumerate(zip(data_keys, df['y_pred_proba'])):
            if pd.notna(prob):
                scores_dict[key] = float(prob)
    else:
        print(f"[WARN] y_pred_proba column not found in {predictions_path}")
    
    print(f"[INFO] Loaded integrated scores for {len(scores_dict)} entries")
    return scores_dict


def create_qa_pairs_category(qa_pairs_count: int) -> str:
    """
    QAペア数をカテゴリに分類
    先行研究に合わせて、0 QAペアを1 QAペアとして扱う
    """
    if qa_pairs_count == 0:
        return "1 QA pair"  # 0 QAペアを1 QAペアとして扱う
    elif qa_pairs_count == 1:
        return "1 QA pair"
    elif qa_pairs_count == 2:
        return "2 QA pairs"
    elif qa_pairs_count == 3:
        return "3 QA pairs"
    elif qa_pairs_count >= 4:
        return "4+ QA pairs"
    else:
        return "1 QA pair"


def calculate_high_ambiguity_percentage(
    scores: pd.Series,
    qa_pairs_category: pd.Series,
    threshold_percentile: float = 75.0
) -> pd.DataFrame:
    """
    各QAペア数カテゴリで、高曖昧性と予測されたクエリの割合を計算
    
    Args:
        scores: スコアのSeries
        qa_pairs_category: QAペア数カテゴリのSeries
        threshold_percentile: 高曖昧性の閾値（パーセンタイル）
    
    Returns:
        各カテゴリとメソッドごとの割合を含むDataFrame
    """
    # 高曖昧性の閾値を計算
    threshold = np.percentile(scores.dropna(), threshold_percentile)
    
    # 高曖昧性と判定されたクエリ
    high_ambiguity = scores >= threshold
    
    # 各カテゴリごとに集計
    results = []
    for category in qa_pairs_category.unique():
        if pd.isna(category):
            continue
        category_mask = qa_pairs_category == category
        category_total = category_mask.sum()
        if category_total > 0:
            category_high_ambiguity = (category_mask & high_ambiguity).sum()
            percentage = (category_high_ambiguity / category_total) * 100
            results.append({
                'qa_pairs_category': category,
                'percentage': percentage,
                'count': category_high_ambiguity,
                'total': category_total
            })
    
    return pd.DataFrame(results)


def create_bar_chart(
    data: pd.DataFrame,
    output_path: Path,
    title: str = "Percentage of queries predicted as ambiguous for each method on different classes of ambiguity in AmbigNQ"
):
    """
    写真のような棒グラフを作成
    """
    # メソッドとカテゴリの順序を定義（先行研究に合わせて）
    # 先行研究: QPP (NQC), BART, Integrated
    # 現在の実装: QPP (NQC), BERT (ClariQ学習、転移学習), Integrated
    method_order = ['QPP', 'BERT', 'Integrated']
    # データに存在するカテゴリのみを使用
    available_categories = sorted(data['qa_pairs_category'].unique())
    # カテゴリの順序を定義（写真に合わせて、存在するもののみ）
    category_order = [cat for cat in ['0 QA pair', '1 QA pair', '2 QA pairs', '3 QA pairs', '4+ QA pairs'] 
                     if cat in available_categories]
    
    # データをピボット
    pivot_data = data.pivot(index='method', columns='qa_pairs_category', values='percentage')
    
    # 順序を適用
    pivot_data = pivot_data.reindex(method_order, fill_value=0)
    pivot_data = pivot_data.reindex(columns=category_order, fill_value=0)
    
    # グラフを作成
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # カテゴリごとの色とパターンを定義（写真に合わせて）
    # ライトブルー（クロスハッチ）、ミディアムブルー（ソリッド）、ダークブルー（ドット）、ダークストブルー（タイトドット）
    # カテゴリのインデックスに基づいて色とパターンを割り当て
    color_map = {
        '0 QA pair': ('#E6F3FF', '///'),  # 非常にライトブルー、クロスハッチ
        '1 QA pair': ('#87CEEB', '///'),  # ライトブルー、クロスハッチ
        '2 QA pairs': ('#4682B4', ''),    # ミディアムブルー、ソリッド
        '3 QA pairs': ('#1E90FF', '...'), # ダークブルー、ドット
        '4+ QA pairs': ('#000080', 'xxx')  # ダークストブルー、タイトドット
    }
    
    x = np.arange(len(method_order))
    n_categories = len(category_order)
    width = 0.8 / n_categories  # バーの幅を調整
    
    # 各カテゴリのバーを描画
    for i, category in enumerate(category_order):
        if category in pivot_data.columns:
            values = pivot_data[category].values
            offset = (i - n_categories / 2 + 0.5) * width
            color, pattern = color_map.get(category, ('#87CEEB', ''))
            bars = ax.bar(x + offset, values, width, label=category, 
                         color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # パターンを適用
            if pattern:
                for bar in bars:
                    bar.set_hatch(pattern)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('% of queries predicted with high ambiguity', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(method_order)
    ax.legend(title='QA Pairs', loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(data['percentage'].max() * 1.2, 35))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved bar chart to {output_path}")
    plt.close()


def create_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    output_path: Path
):
    """
    相関行列のヒートマップを作成
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    ax.set_title('Correlation Matrix: Scores vs QA Pairs Count', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved correlation heatmap to {output_path}")
    plt.close()


def create_scatter_plots(
    df: pd.DataFrame,
    output_dir: Path
):
    """
    各スコアとQAペア数の散布図を作成
    """
    score_columns = ['QPP', 'BERT', 'Integrated']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, score_col in enumerate(score_columns):
        if score_col not in df.columns:
            continue
        
        ax = axes[i]
        
        # 散布図
        scatter = ax.scatter(
            df['qa_pairs_count'],
            df[score_col],
            alpha=0.5,
            s=20,
            c=df['qa_pairs_count'],
            cmap='viridis'
        )
        
        # 回帰直線
        mask = ~(pd.isna(df[score_col]) | pd.isna(df['qa_pairs_count']))
        if mask.sum() > 1:
            x = df.loc[mask, 'qa_pairs_count']
            y = df.loc[mask, score_col]
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend line')
            
            # 相関係数を計算
            corr, p_value = pearsonr(x, y)
            ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_value:.3e}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('QA Pairs Count', fontsize=11)
        ax.set_ylabel(f'{score_col} Score', fontsize=11)
        ax.set_title(f'{score_col} vs QA Pairs Count', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_plots.png', dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved scatter plots to {output_dir / 'scatter_plots.png'}")
    plt.close()


def create_box_plots(
    df: pd.DataFrame,
    output_dir: Path,
    score_columns: List[str]
):
    """
    QAペア数カテゴリごとのスコア分布を箱ひげ図で可視化
    """
    # 0 QA pairを除外
    df_plot = df[df['qa_pairs_category'] != '0 QA pair'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, score_col in enumerate(score_columns):
        if score_col not in df_plot.columns:
            continue
        
        ax = axes[i]
        
        # カテゴリの順序
        category_order = ['2 QA pairs', '3 QA pairs', '4+ QA pairs']
        available_categories = [cat for cat in category_order if cat in df_plot['qa_pairs_category'].unique()]
        
        # データを準備
        plot_data = []
        plot_labels = []
        for category in available_categories:
            mask = (df_plot['qa_pairs_category'] == category) & ~pd.isna(df_plot[score_col])
            values = df_plot.loc[mask, score_col].values
            if len(values) > 0:
                plot_data.append(values)
                plot_labels.append(category)
        
        if plot_data:
            bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
            
            # 色を設定
            colors_box = ['#4682B4', '#1E90FF', '#000080']
            for patch, color in zip(bp['boxes'], colors_box[:len(plot_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_xlabel('QA Pairs Category', fontsize=11)
        ax.set_ylabel(f'{score_col} Score', fontsize=11)
        ax.set_title(f'{score_col} Distribution by QA Pairs Count', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'box_plots.png', dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved box plots to {output_dir / 'box_plots.png'}")
    plt.close()


def create_violin_plots(
    df: pd.DataFrame,
    output_dir: Path,
    score_columns: List[str]
):
    """
    QAペア数カテゴリごとのスコア分布をバイオリンプロットで可視化
    """
    # 0 QA pairを除外
    df_plot = df[df['qa_pairs_category'] != '0 QA pair'].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, score_col in enumerate(score_columns):
        if score_col not in df_plot.columns:
            continue
        
        ax = axes[i]
        
        # カテゴリの順序
        category_order = ['2 QA pairs', '3 QA pairs', '4+ QA pairs']
        available_categories = [cat for cat in category_order if cat in df_plot['qa_pairs_category'].unique()]
        
        # データを準備
        plot_data = []
        plot_labels = []
        for category in available_categories:
            mask = (df_plot['qa_pairs_category'] == category) & ~pd.isna(df_plot[score_col])
            values = df_plot.loc[mask, score_col].values
            if len(values) > 0:
                plot_data.append(values)
                plot_labels.append(category)
        
        if plot_data:
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), showmeans=True, showmedians=True)
            
            # 色を設定
            colors_violin = ['#4682B4', '#1E90FF', '#000080']
            for pc, color in zip(parts['bodies'], colors_violin[:len(plot_data)]):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels)
        ax.set_xlabel('QA Pairs Category', fontsize=11)
        ax.set_ylabel(f'{score_col} Score', fontsize=11)
        ax.set_title(f'{score_col} Distribution by QA Pairs Count (Violin Plot)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'violin_plots.png', dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved violin plots to {output_dir / 'violin_plots.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze correlation between scores and QA pairs count"
    )
    parser.add_argument(
        '--dataset_dir',
        type=Path,
        default=Path('/home/daiki_shibata/pj/QPP4SIP/dataset/AmbigNQ'),
        help='Dataset directory'
    )
    parser.add_argument(
        '--sip_output_dir',
        type=Path,
        default=Path('/home/daiki_shibata/pj/QPP4SIP/SIP/FT-PLM/output/AmbigNQ'),
        help='SIP model output directory'
    )
    parser.add_argument(
        '--logreg_output_dir',
        type=Path,
        default=Path('/home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO/outputs/AmbigNQ'),
        help='LogReg output directory'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('/home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO/outputs/AmbigNQ/score_qa_pairs_analysis'),
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='dev',
        choices=['dev', 'train'],
        help='Dataset split'
    )
    parser.add_argument(
        '--threshold_percentile',
        type=float,
        default=75.0,
        help='Percentile threshold for high ambiguity (default: 75.0)'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # データを読み込む
    data_path = args.dataset_dir / f'{args.split}.json'
    qa_pairs_dict = load_qa_pairs_count(data_path)
    
    # BERTスコアを読み込む（AmbigNQで学習）
    bert_path = args.sip_output_dir / 'AmbigNQ_bert-base_lr2e-05_bs16_kfold5' / f'{args.split}_with_predictions.json'
    bert_scores = load_bert_scores(bert_path)
    
    # BERTスコアを読み込む（ClariQで学習、転移学習）- 先行研究のBARTの代わり
    # 転移学習の結果はAmbigNQ_transfer_from_ClariQディレクトリに保存される
    bert_clariq_path = args.sip_output_dir / 'AmbigNQ_transfer_from_ClariQ' / f'{args.split}_with_predictions.json'
    if not bert_clariq_path.exists():
        # モデル名を含むディレクトリも確認
        bert_clariq_path_alt = args.sip_output_dir / 'AmbigNQ_transfer_from_ClariQ_bert-base' / f'{args.split}_with_predictions.json'
        if bert_clariq_path_alt.exists():
            bert_clariq_path = bert_clariq_path_alt
    bert_clariq_scores = load_bert_scores(bert_clariq_path) if bert_clariq_path.exists() else {}
    if bert_clariq_scores:
        print(f"[INFO] Loaded BERT (ClariQ-trained) scores for {len(bert_clariq_scores)} entries")
    else:
        print(f"[WARN] BERT (ClariQ-trained) scores not found at {bert_clariq_path}")
    
    # RoBERTaスコアを読み込む（ClariQで学習、転移学習）- 使用しないが読み込みは可能
    roberta_clariq_path = args.sip_output_dir / 'AmbigNQ_transfer_from_ClariQ' / f'{args.split}_with_predictions.json'
    if not roberta_clariq_path.exists():
        roberta_clariq_path = args.sip_output_dir / 'AmbigNQ_transfer_from_ClariQ_roberta-base' / f'{args.split}_with_predictions.json'
    roberta_clariq_scores = load_roberta_scores(roberta_clariq_path) if roberta_clariq_path.exists() else {}
    
    # RoBERTaスコアを読み込む（分析には使用しないが、データ確認用）
    roberta_path = args.sip_output_dir / 'AmbigNQ_roberta-base_lr2e-05_bs16_earlystop_kfold5' / f'{args.split}_with_predictions.json'
    roberta_scores = load_roberta_scores(roberta_path)
    # 注意: AmbigNQで学習したRoBERTaは分析には含めない（BERTとRoBERTaを同時に扱う実験設定はないため）
    
    # QPPスコアを読み込む（主要指標の平均を使用）
    qpp_scores_dict = load_qpp_scores(args.dataset_dir, split=args.split)
    
    # 統合スコアを読み込む（LogRegの予測確率）
    # 転移学習したBERTを使ったIntegratedスコアを探す
    # transfer特徴量はbaseとして扱われ、pre_post_bertディレクトリに保存される
    # 最新のファイル（転移学習の結果）を優先的に使用
    integrated_score = {}
    logreg_dirs = list(args.logreg_output_dir.glob('**/pre_post_bert/**/csv/predictions_for_delong.csv'))
    if logreg_dirs:
        # ファイルの更新時刻でソートして、最新のものを使用
        logreg_dirs_with_time = [(p, p.stat().st_mtime) for p in logreg_dirs]
        logreg_dirs_with_time.sort(key=lambda x: x[1], reverse=True)
        latest_file = logreg_dirs_with_time[0][0]
        integrated_score = load_integrated_score(latest_file, data_path, split=args.split)
        print(f"[INFO] Loaded integrated scores from (latest): {latest_file}")
        print(f"[INFO] File modification time: {logreg_dirs_with_time[0][1]}")
    else:
        print(f"[WARN] No LogReg predictions file found, skipping integrated score")
    
    # データを結合
    all_keys = set(qa_pairs_dict.keys())
    all_keys.update(bert_clariq_scores.keys())  # ClariQで学習したBERTのみ使用
    if integrated_score:
        all_keys.update(integrated_score.keys())
    
    data_list = []
    for key in all_keys:
        conv_id, turn_id = key
        row = {
            'conv_id': conv_id,
            'turn_id': turn_id,
            'qa_pairs_count': qa_pairs_dict.get(key, 0),
            'BERT': bert_clariq_scores.get(key, np.nan),  # ClariQで学習したBERT（転移学習）
            'RoBERTa': roberta_scores.get(key, np.nan),  # 使用しない
        }
        
        # QPPスコア（先行研究に合わせてNQCのみを使用）
        # 先行研究: "best of QPP methods (NQC)"
        if 'nqc' in qpp_scores_dict and key in qpp_scores_dict['nqc']:
            # NQCは値が低いほど曖昧性が高い可能性があるため、符号を反転
            # 先行研究のFigure 4では、QPPは1 QA pairから4+ QA pairsまで増加している
            # これは、NQCの符号を反転させる必要があることを示唆
            nqc_value = qpp_scores_dict['nqc'][key]
            row['QPP'] = -nqc_value  # 符号を反転（値が高いほど曖昧性が高い）
        else:
            row['QPP'] = np.nan
        
        # 統合スコア
        if integrated_score:
            row['Integrated'] = integrated_score.get(key, np.nan)
        else:
            row['Integrated'] = np.nan
        
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    # QAペア数カテゴリを追加
    df['qa_pairs_category'] = df['qa_pairs_count'].apply(create_qa_pairs_category)
    
    print(f"\n[INFO] Total entries: {len(df)}")
    print(f"[INFO] QA pairs count distribution:")
    print(df['qa_pairs_category'].value_counts().sort_index())
    
    # 0 QAペアを1 QAペアとして扱うため、フィルタリングは不要
    # ただし、相関分析では1+ QAペアのみを使用（先行研究に合わせて）
    print("\n[INFO] Note: 0 QA pairs are treated as 1 QA pair (as in prior work)")
    df_filtered = df[df['qa_pairs_count'] >= 1].copy()  # 相関分析用
    print(f"[INFO] Entries for correlation analysis (1+ QA pairs): {len(df_filtered)} (from {len(df)} total)")
    
    # 相関分析
    print("\n[INFO] Calculating correlations...")
    # 先行研究に合わせて: QPP (NQC), BART, Integrated
    # 現在の実装: QPP (NQC), BERT (ClariQ学習、転移学習), Integrated
    score_columns = ['QPP', 'BERT', 'Integrated']
    available_columns = [col for col in score_columns if col in df_filtered.columns]
    
    correlation_results = []
    for score_col in available_columns:
        mask = ~(pd.isna(df_filtered[score_col]) | pd.isna(df_filtered['qa_pairs_count']))
        if mask.sum() > 1:
            x = df_filtered.loc[mask, 'qa_pairs_count']
            y = df_filtered.loc[mask, score_col]
            pearson_corr, pearson_p = pearsonr(x, y)
            spearman_corr, spearman_p = spearmanr(x, y)
            
            correlation_results.append({
                'Score': score_col,
                'Pearson_r': pearson_corr,
                'Pearson_p': pearson_p,
                'Spearman_rho': spearman_corr,
                'Spearman_p': spearman_p,
                'N': mask.sum()
            })
    
    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(args.output_dir / 'correlation_results.csv', index=False)
    print("\n[INFO] Correlation Results:")
    print(corr_df.to_string(index=False))
    
    # 相関行列を作成
    corr_matrix_data = df_filtered[['qa_pairs_count'] + available_columns].corr()
    corr_matrix_data.to_csv(args.output_dir / 'correlation_matrix.csv')
    create_correlation_heatmap(corr_matrix_data, args.output_dir / 'correlation_heatmap.png')
    
    # 各メソッドごとに高曖昧性の割合を計算（先行研究に合わせて）
    # 先行研究では、各QAペア数カテゴリで「曖昧性と予測された割合」を計算
    # これは、スコアが閾値以上の場合に「曖昧性が高い」と判定
    # 閾値は全データ（0 QAペア含む）で計算する
    bar_chart_data = []
    for score_col in available_columns:
        # 全データを使用（0 QAペアも含む、閾値計算用）
        mask_all = ~pd.isna(df[score_col])
        if mask_all.sum() > 0:
            # 全データで閾値を計算
            scores_all = df.loc[mask_all, score_col]
            threshold = np.percentile(scores_all.dropna(), args.threshold_percentile)
            
            # 各カテゴリごとに集計（全データを使用）
            results = []
            for category in df.loc[mask_all, 'qa_pairs_category'].unique():
                if pd.isna(category):
                    continue
                category_mask = (df.loc[mask_all, 'qa_pairs_category'] == category)
                category_total = category_mask.sum()
                if category_total > 0:
                    # 高曖昧性と判定されたクエリ
                    category_scores = scores_all[category_mask]
                    category_high_ambiguity = (category_scores >= threshold).sum()
                    percentage = (category_high_ambiguity / category_total) * 100
                    results.append({
                        'qa_pairs_category': category,
                        'percentage': percentage,
                        'count': category_high_ambiguity,
                        'total': category_total
                    })
            
            if results:
                results_df = pd.DataFrame(results)
                results_df['method'] = score_col
                bar_chart_data.append(results_df)
    
    if bar_chart_data:
        bar_chart_df = pd.concat(bar_chart_data, ignore_index=True)
        # 1, 2, 3, 4+ QAペアのみ（先行研究に合わせて）
        bar_chart_df = bar_chart_df[bar_chart_df['qa_pairs_category'].isin(['1 QA pair', '2 QA pairs', '3 QA pairs', '4+ QA pairs'])].copy()
        bar_chart_df.to_csv(args.output_dir / 'high_ambiguity_percentage.csv', index=False)
        create_bar_chart(bar_chart_df, args.output_dir / 'bar_chart_high_ambiguity.png')
    
    # 散布図を作成（全データを使用、0 QAペアは1 QAペアとして表示）
    create_scatter_plots(df, args.output_dir)
    
    # 箱ひげ図を作成（1+ QAペアのみ、先行研究に合わせて）
    create_box_plots(df_filtered, args.output_dir, available_columns)
    
    # バイオリンプロットを作成（1+ QAペアのみ、先行研究に合わせて）
    create_violin_plots(df_filtered, args.output_dir, available_columns)
    
    # 統計サマリーを保存
    summary = {
        'total_entries': len(df),
        'filtered_entries': len(df_filtered),
        'qa_pairs_distribution': df['qa_pairs_category'].value_counts().to_dict(),
        'filtered_qa_pairs_distribution': df_filtered['qa_pairs_category'].value_counts().to_dict(),
        'correlations': corr_df.to_dict('records'),
        'threshold_percentile': args.threshold_percentile,
        'note': 'Analysis follows prior work: QPP uses NQC only, filtered to 1+ QA pairs'
    }
    
    with open(args.output_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n[INFO] Analysis complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()

