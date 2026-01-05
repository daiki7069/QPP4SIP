"""
LogRegが正解し、ベースモデルが外したクエリの可視化スクリプト
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')

# scipy.special.expit (sigmoid関数) を使用するため
try:
    from scipy.special import expit
except ImportError:
    # scipyが利用できない場合はnumpyで実装
    def expit(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# フォントの設定（日本語フォントが利用可能な場合は使用、なければ英語のみ）
import matplotlib
try:
    # 日本語フォントを試す
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
except:
    plt.rcParams['font.family'] = 'DejaVu Sans'

# マイナス記号の文字化けを防ぐ
plt.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_data(results_csv_path: Path, analysis_csv_path: Path) -> tuple:
    """データを読み込む"""
    results_df = pd.read_csv(results_csv_path)
    analysis_df = pd.read_csv(analysis_csv_path)
    return results_df, analysis_df


def plot_feature_distributions(
    results_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """特徴量の分布を比較"""
    # 特徴量カラムを取得
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']
    feature_cols = [col for col in results_df.columns if col not in exclude_cols]
    
    # グラフのサイズを調整（特徴量の数に応じて）
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        
        # データを準備
        target_data = analysis_df[col].dropna()
        all_data = results_df[col].dropna()
        
        # ヒストグラムを描画
        ax.hist(all_data, bins=30, alpha=0.5, label='All', density=True, color='blue')
        ax.hist(target_data, bins=30, alpha=0.7, label=f'LogReg correct, {base_model} wrong', density=True, color='red')
        
        # 平均値を表示
        all_mean = all_data.mean()
        target_mean = target_data.mean()
        ax.axvline(all_mean, color='blue', linestyle='--', linewidth=2, label=f'All mean: {all_mean:.3f}')
        ax.axvline(target_mean, color='red', linestyle='--', linewidth=2, label=f'Target mean: {target_mean:.3f}')
        
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Score', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 余分なサブプロットを非表示
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / f'feature_distributions_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特徴量分布を保存: {output_path}")


def plot_feature_boxplots(
    results_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """特徴量のボックスプロットを比較"""
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']
    feature_cols = [col for col in results_df.columns if col not in exclude_cols]
    
    # データを準備
    plot_data = []
    for col in feature_cols:
        # 全体データ
        for val in results_df[col].dropna():
            plot_data.append({'Feature': col, 'Value': val, 'Group': 'All'})
        # 対象データ
        for val in analysis_df[col].dropna():
            plot_data.append({'Feature': col, 'Value': val, 'Group': f'LogReg correct, {base_model} wrong'})
    
    plot_df = pd.DataFrame(plot_data)
    
    # グラフを作成
    n_features = len(feature_cols)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, col in enumerate(feature_cols):
        ax = axes[idx]
        col_data = plot_df[plot_df['Feature'] == col]
        
        sns.boxplot(data=col_data, x='Group', y='Value', ax=ax)
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Score', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # 余分なサブプロットを非表示
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / f'feature_boxplots_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特徴量ボックスプロットを保存: {output_path}")


def plot_query_length_distribution(
    analysis_df: pd.DataFrame,
    all_query_lengths: List[int],
    output_dir: Path,
    base_model: str
):
    """クエリ長の分布を比較"""
    if 'query_length' not in analysis_df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ヒストグラム
    ax1 = axes[0]
    ax1.hist(all_query_lengths, bins=20, alpha=0.5, label='All', density=True, color='blue')
    ax1.hist(analysis_df['query_length'], bins=20, alpha=0.7, 
             label=f'LogReg correct, {base_model} wrong', density=True, color='red')
    ax1.axvline(np.mean(all_query_lengths), color='blue', linestyle='--', linewidth=2, 
                label=f'All mean: {np.mean(all_query_lengths):.2f}')
    ax1.axvline(analysis_df['query_length'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Target mean: {analysis_df["query_length"].mean():.2f}')
    ax1.set_xlabel('Query Length (words)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Query Length Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ボックスプロット
    ax2 = axes[1]
    plot_data = pd.DataFrame({
        'Group': ['All'] * len(all_query_lengths) + [f'LogReg correct, {base_model} wrong'] * len(analysis_df),
        'Query Length': all_query_lengths + analysis_df['query_length'].tolist()
    })
    sns.boxplot(data=plot_data, x='Group', y='Query Length', ax=ax2)
    ax2.set_title('Query Length Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('')
    ax2.set_ylabel('Query Length (words)', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'query_length_distribution_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"クエリ長分布を保存: {output_path}")


def plot_score_comparison(
    analysis_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """LogRegとベースモデルのスコアを比較"""
    if f'{base_model}_score' not in analysis_df.columns:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 散布図
    ax1 = axes[0]
    ax1.scatter(analysis_df['LogReg_score'], analysis_df[f'{base_model}_score'], 
                alpha=0.6, s=50, c=analysis_df['label'], cmap='coolwarm')
    
    # LogRegは確率値（0-1）、ベースモデルはlogit値なので、y=xの線は描画しない
    # 代わりに、ベースモデルの閾値（0.0）とLogRegの閾値（0.5）を示す線を描画
    ax1.axhline(y=0.0, color='red', linestyle='--', alpha=0.5, label=f'{base_model} threshold (0.0)')
    ax1.axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='LogReg threshold (0.5)')
    
    ax1.set_xlabel('LogReg Score (probability)', fontsize=12)
    ax1.set_ylabel(f'{base_model} Score (logit)', fontsize=12)
    ax1.set_title(f'LogReg vs {base_model} Score Comparison', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # LogRegは0-1の範囲、ベースモデルはlogit値なので範囲を自動調整
    ax1.set_xlim(0, 1)
    # logit値の範囲を自動調整（データに基づいて）
    y_min = analysis_df[f'{base_model}_score'].min() - 0.2
    y_max = analysis_df[f'{base_model}_score'].max() + 0.2
    ax1.set_ylim(y_min, y_max)
    
    # スコア差の分布（注意: LogRegは確率値、ベースモデルはlogit値なので直接比較はできない）
    # 代わりに、両方のスコアの分布を表示
    ax2 = axes[1]
    
    # LogRegスコアの分布（確率値）
    ax2.hist(analysis_df['LogReg_score'], bins=30, alpha=0.5, label='LogReg (probability)', 
             color='blue', density=True)
    # ベースモデルスコアを確率に変換（sigmoid関数を使用）
    base_proba = expit(analysis_df[f'{base_model}_score'])
    ax2.hist(base_proba, bins=30, alpha=0.5, label=f'{base_model} (logit->probability)', 
             color='red', density=True)
    
    ax2.axvline(0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Threshold (0.5)')
    ax2.set_xlabel('Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Score Distribution (converted to probability)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    output_path = output_dir / f'score_comparison_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"スコア比較を保存: {output_path}")


def plot_feature_correlation_heatmap(
    analysis_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """特徴量間の相関行列を可視化"""
    exclude_cols = ['index', 'label', 'LogReg_score', 'LogReg_pred', 
                   'BERT_score', 'BERT_pred', 'RoBERTa_score', 'RoBERTa_pred',
                   'Transfer_score', 'Transfer_pred', 'query', 'answer',
                   'conv_id', 'turn_id', 'retrieval_titles', 'query_length']
    feature_cols = [col for col in analysis_df.columns 
                   if col not in exclude_cols and analysis_df[col].dtype in [np.float64, np.int64]]
    
    if len(feature_cols) < 2:
        return
    
    corr_matrix = analysis_df[feature_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Feature Correlation Matrix (LogReg correct, {base_model} wrong)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / f'feature_correlation_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"特徴量相関行列を保存: {output_path}")


def plot_statistical_summary(
    results_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """統計サマリーを可視化"""
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']
    feature_cols = [col for col in results_df.columns if col not in exclude_cols]
    
    # 統計を計算
    stats_data = []
    for col in feature_cols:
        if col in analysis_df.columns:
            target_mean = analysis_df[col].mean()
            all_mean = results_df[col].mean()
            diff = target_mean - all_mean
            stats_data.append({
                'Feature': col,
                'All_mean': all_mean,
                'Target_mean': target_mean,
                'Difference': diff
            })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values('Difference', key=abs, ascending=False)
    
    # バープロット
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(stats_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, stats_df['All_mean'], width, label='All mean', alpha=0.7)
    bars2 = ax.bar(x + width/2, stats_df['Target_mean'], width, label=f'LogReg correct, {base_model} wrong', alpha=0.7)
    
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Feature Mean Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['Feature'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / f'statistical_summary_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"統計サマリーを保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LogRegが正解し、ベースモデルが外したクエリの可視化"
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        required=True,
        help="results.csvのパス"
    )
    parser.add_argument(
        "--analysis-csv",
        type=str,
        required=True,
        help="分析結果CSV（logreg_correct_*_wrong.csv）のパス"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="ベースモデル名（BERT/RoBERTa/Transfer、Noneの場合は自動検出）"
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="データセットJSON（クエリ長の計算用、オプション）"
    )
    
    args = parser.parse_args()
    
    results_csv_path = Path(args.results_csv)
    analysis_csv_path = Path(args.analysis_csv)
    output_dir = Path(args.output_dir)
    
    if not results_csv_path.exists():
        print(f"エラー: results.csvが見つかりません: {results_csv_path}")
        return
    
    if not analysis_csv_path.exists():
        print(f"エラー: 分析結果CSVが見つかりません: {analysis_csv_path}")
        return
    
    # ベースモデルを自動検出
    base_model = args.base_model
    if base_model is None:
        if 'RoBERTa_score' in pd.read_csv(analysis_csv_path, nrows=1).columns:
            base_model = 'RoBERTa'
        elif 'BERT_score' in pd.read_csv(analysis_csv_path, nrows=1).columns:
            base_model = 'BERT'
        elif 'Transfer_score' in pd.read_csv(analysis_csv_path, nrows=1).columns:
            base_model = 'Transfer'
        else:
            print("エラー: ベースモデルを自動検出できませんでした。--base-modelを指定してください。")
            return
    
    print(f"=== 可視化を開始 ===")
    print(f"ベースモデル: {base_model}")
    print()
    
    # データを読み込む
    results_df, analysis_df = load_data(results_csv_path, analysis_csv_path)
    print(f"results.csv: {len(results_df)} 行")
    print(f"分析結果CSV: {len(analysis_df)} 行")
    print()
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # クエリ長を計算（可能な場合）
    all_query_lengths = []
    if args.dataset_json and Path(args.dataset_json).exists():
        import json
        with open(args.dataset_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for conversation in data:
            for turn in conversation:
                query = turn.get('query', turn.get('question', ''))
                all_query_lengths.append(len(str(query).split()))
        
        if 'query' in analysis_df.columns:
            analysis_df['query_length'] = analysis_df['query'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) else 0
            )
    
    # 可視化を実行
    print("可視化を生成中...")
    plot_feature_distributions(results_df, analysis_df, output_dir, base_model)
    plot_feature_boxplots(results_df, analysis_df, output_dir, base_model)
    
    if len(all_query_lengths) > 0:
        plot_query_length_distribution(analysis_df, all_query_lengths, output_dir, base_model)
    
    plot_score_comparison(analysis_df, output_dir, base_model)
    plot_feature_correlation_heatmap(analysis_df, output_dir, base_model)
    plot_statistical_summary(results_df, analysis_df, output_dir, base_model)
    
    print()
    print("=== 可視化完了 ===")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()

