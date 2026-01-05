"""
RoBERTaが弱い部分の共通特徴を探索するスクリプト
1. RoBERTaが弱いときの共通するQPP特徴量パターン
2. RoBERTaが弱いけどQPP統合が効果的なときの共通特徴
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
warnings.filterwarnings('ignore')

try:
    from scipy.special import expit
except ImportError:
    def expit(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")

# フォント設定
import matplotlib
import matplotlib.font_manager as fm
try:
    jp_font_candidates = [
        'Noto Sans CJK JP', 'Noto Sans Japanese', 'Takao', 'TakaoGothic', 'TakaoPGothic',
        'IPAexGothic', 'IPAPGothic', 'IPAPMincho', 'VL PGothic', 'VL Gothic',
        'Yu Gothic', 'YuGothic', 'Meiryo', 'MS PGothic', 'MS Gothic',
        'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Osaka', 'Arial Unicode MS'
    ]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    found_jp_font = None
    for font_name in jp_font_candidates:
        if font_name in available_fonts:
            found_jp_font = font_name
            break
    if found_jp_font:
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [found_jp_font] + matplotlib.rcParams['font.sans-serif']
        matplotlib.rcParams['axes.unicode_minus'] = False
    else:
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False

sns.set_style("whitegrid")
sns.set_palette("husl")


def identify_weak_cases(
    results_df: pd.DataFrame,
    base_model: str,
    error_rate_threshold: float = 0.4,
    improvement_threshold: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """
    RoBERTaが弱いケースを特定し、QPP統合の効果も分析
    
    Returns:
        dict with keys: 'weak_all', 'weak_improved', 'weak_not_improved'
    """
    results_df = results_df.copy()
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # 予測を計算
    median_threshold = results_df[f'{base_model}_proba'].median()
    results_df[f'{base_model}_pred'] = (results_df[f'{base_model}_proba'] >= median_threshold).astype(int)
    
    if 'LogReg' not in results_df.columns:
        print("LogRegの予測結果が見つかりません。")
        return {}
    
    results_df['LogReg_pred'] = (results_df['LogReg'] >= 0.5).astype(int)
    
    # 誤分類を計算
    results_df['base_error'] = (results_df[f'{base_model}_pred'] != results_df['label']).astype(int)
    results_df['logreg_error'] = (results_df['LogReg_pred'] != results_df['label']).astype(int)
    results_df['base_accuracy'] = (results_df[f'{base_model}_pred'] == results_df['label']).astype(int)
    results_df['logreg_accuracy'] = (results_df['LogReg_pred'] == results_df['label']).astype(int)
    
    # 改善量を計算
    results_df['improvement'] = results_df['logreg_accuracy'] - results_df['base_accuracy']
    
    # RoBERTaが弱いケース（誤分類したケース）
    weak_all = results_df[results_df['base_error'] == 1].copy()
    
    # RoBERTaが弱いけどQPP統合が効果的なケース
    weak_improved = results_df[
        (results_df['base_error'] == 1) & 
        (results_df['improvement'] >= improvement_threshold)
    ].copy()
    
    # RoBERTaが弱くてQPP統合も効果的でないケース
    weak_not_improved = results_df[
        (results_df['base_error'] == 1) & 
        (results_df['improvement'] < improvement_threshold)
    ].copy()
    
    return {
        'weak_all': weak_all,
        'weak_improved': weak_improved,
        'weak_not_improved': weak_not_improved
    }


def analyze_common_features(
    weak_cases: Dict[str, pd.DataFrame],
    all_cases: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    弱いケースの共通特徴を分析
    """
    # QPP特徴量を取得
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer',
                   'BERT_proba', 'RoBERTa_proba', 'Transfer_proba', 'LogReg_pred',
                   'BERT_pred', 'RoBERTa_pred', 'Transfer_pred', 'threshold', 'group',
                   'score_percentile', 'percentile_bin', 'base_error', 'logreg_error',
                   'base_accuracy', 'logreg_accuracy', 'improvement']
    qpp_features = [col for col in all_cases.columns 
                   if col not in exclude_cols and all_cases[col].dtype in [np.float64, np.int64]]
    
    # 統計分析結果を保存
    stats_results = []
    
    for feature in qpp_features:
        if feature not in all_cases.columns:
            continue
        
        all_values = all_cases[feature].dropna()
        weak_all_values = weak_cases['weak_all'][feature].dropna() if len(weak_cases['weak_all']) > 0 else pd.Series()
        weak_improved_values = weak_cases['weak_improved'][feature].dropna() if len(weak_cases['weak_improved']) > 0 else pd.Series()
        weak_not_improved_values = weak_cases['weak_not_improved'][feature].dropna() if len(weak_cases['weak_not_improved']) > 0 else pd.Series()
        
        if len(weak_all_values) == 0:
            continue
        
        # 統計的検定
        stats_data = {
            'feature': feature,
            'all_mean': all_values.mean(),
            'all_std': all_values.std(),
            'weak_all_mean': weak_all_values.mean(),
            'weak_all_std': weak_all_values.std(),
            'weak_improved_mean': weak_improved_values.mean() if len(weak_improved_values) > 0 else np.nan,
            'weak_improved_std': weak_improved_values.std() if len(weak_improved_values) > 0 else np.nan,
            'weak_not_improved_mean': weak_not_improved_values.mean() if len(weak_not_improved_values) > 0 else np.nan,
            'weak_not_improved_std': weak_not_improved_values.std() if len(weak_not_improved_values) > 0 else np.nan,
        }
        
        # 統計的検定
        if len(weak_all_values) > 0 and len(all_values) > 0:
            try:
                _, p_weak_vs_all = stats.mannwhitneyu(weak_all_values, all_values, alternative='two-sided')
                stats_data['p_weak_vs_all'] = p_weak_vs_all
            except:
                stats_data['p_weak_vs_all'] = np.nan
        else:
            stats_data['p_weak_vs_all'] = np.nan
        
        if len(weak_improved_values) > 0 and len(weak_not_improved_values) > 0:
            try:
                _, p_improved_vs_not = stats.mannwhitneyu(weak_improved_values, weak_not_improved_values, alternative='two-sided')
                stats_data['p_improved_vs_not'] = p_improved_vs_not
            except:
                stats_data['p_improved_vs_not'] = np.nan
        else:
            stats_data['p_improved_vs_not'] = np.nan
        
        # 効果量（Cohen's d）
        def cohens_d(group1, group2):
            if len(group1) == 0 or len(group2) == 0:
                return np.nan
            n1, n2 = len(group1), len(group2)
            var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std == 0:
                return np.nan
            return (group1.mean() - group2.mean()) / pooled_std
        
        stats_data['cohens_d_weak_vs_all'] = cohens_d(weak_all_values, all_values)
        stats_data['cohens_d_improved_vs_not'] = cohens_d(weak_improved_values, weak_not_improved_values) if len(weak_improved_values) > 0 and len(weak_not_improved_values) > 0 else np.nan
        
        stats_results.append(stats_data)
    
    stats_df = pd.DataFrame(stats_results)
    
    # CSVに保存
    stats_path = output_dir / f'weakness_common_features_{base_model.lower()}.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"弱いケースの共通特徴分析を保存: {stats_path}")
    
    return stats_df


def plot_feature_comparison(
    weak_cases: Dict[str, pd.DataFrame],
    all_cases: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    弱いケースと全体の特徴量を比較可視化
    """
    # 統計的に有意な特徴を選択
    significant_features = stats_df[
        (stats_df['p_weak_vs_all'] < 0.05) | 
        (stats_df['p_improved_vs_not'] < 0.05)
    ].copy()
    
    if len(significant_features) == 0:
        print("統計的に有意な特徴が見つかりませんでした。")
        return
    
    # 効果量でソート
    significant_features['max_effect'] = significant_features[
        ['cohens_d_weak_vs_all', 'cohens_d_improved_vs_not']
    ].abs().max(axis=1)
    significant_features = significant_features.sort_values('max_effect', ascending=False)
    
    top_features = significant_features.head(15)['feature'].tolist()
    
    # グラフを作成
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # データを準備
        all_values = all_cases[feature].dropna()
        weak_all_values = weak_cases['weak_all'][feature].dropna() if len(weak_cases['weak_all']) > 0 else pd.Series()
        weak_improved_values = weak_cases['weak_improved'][feature].dropna() if len(weak_cases['weak_improved']) > 0 else pd.Series()
        weak_not_improved_values = weak_cases['weak_not_improved'][feature].dropna() if len(weak_cases['weak_not_improved']) > 0 else pd.Series()
        
        # ボックスプロット
        plot_data = []
        for values, group_name in [
            (all_values, 'All'),
            (weak_all_values, f'{base_model} Weak'),
            (weak_improved_values, 'Weak + QPP Improved'),
            (weak_not_improved_values, 'Weak + QPP Not Improved')
        ]:
            if len(values) > 0:
                for val in values:
                    plot_data.append({'Group': group_name, 'Value': val})
        
        plot_df = pd.DataFrame(plot_data)
        
        if len(plot_df) > 0:
            sns.boxplot(data=plot_df, x='Group', y='Value', ax=ax,
                       order=['All', f'{base_model} Weak', 'Weak + QPP Improved', 'Weak + QPP Not Improved'])
            
            # 統計情報を表示
            stats_row = stats_df[stats_df['feature'] == feature].iloc[0]
            title_parts = [feature]
            if stats_row['p_weak_vs_all'] < 0.05:
                title_parts.append(f'p={stats_row["p_weak_vs_all"]:.3f}')
            if stats_row['p_improved_vs_not'] < 0.05:
                title_parts.append(f'p_imp={stats_row["p_improved_vs_not"]:.3f}')
            
            ax.set_title(' '.join(title_parts), fontsize=10, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Value', fontsize=9)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    # 余分なサブプロットを非表示
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / f'weakness_feature_comparison_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"弱いケースの特徴比較を保存: {output_path}")


def cluster_weak_cases(
    weak_cases: Dict[str, pd.DataFrame],
    all_cases: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    n_clusters: int = 3
):
    """
    弱いケースをクラスタリングしてパターンを発見
    """
    # QPP特徴量を取得
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer',
                   'BERT_proba', 'RoBERTa_proba', 'Transfer_proba', 'LogReg_pred',
                   'BERT_pred', 'RoBERTa_pred', 'Transfer_pred', 'threshold', 'group',
                   'score_percentile', 'percentile_bin', 'base_error', 'logreg_error',
                   'base_accuracy', 'logreg_accuracy', 'improvement']
    qpp_features = [col for col in all_cases.columns 
                   if col not in exclude_cols and all_cases[col].dtype in [np.float64, np.int64]]
    
    if len(qpp_features) == 0:
        print("QPP特徴量が見つかりませんでした。")
        return
    
    # データを準備
    weak_all = weak_cases['weak_all'].copy()
    if len(weak_all) == 0:
        print("弱いケースが見つかりませんでした。")
        return
    
    # 欠損値を処理
    weak_data = weak_all[qpp_features].dropna()
    
    if len(weak_data) < n_clusters:
        print(f"クラスタリングに十分なデータがありません（{len(weak_data)} < {n_clusters}）。")
        return
    
    # 標準化
    scaler = StandardScaler()
    weak_data_scaled = scaler.fit_transform(weak_data)
    
    # K-meansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(weak_data_scaled)
    
    weak_data['cluster'] = clusters
    weak_data['improvement'] = weak_all.loc[weak_data.index, 'improvement'].values
    
    # 各クラスタの特徴を分析
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_data = weak_data[weak_data['cluster'] == cluster_id]
        if len(cluster_data) == 0:
            continue
        
        cluster_info = {
            'cluster': cluster_id,
            'n_samples': len(cluster_data),
            'avg_improvement': cluster_data['improvement'].mean() if 'improvement' in cluster_data.columns else np.nan
        }
        
        for feature in qpp_features:
            if feature in cluster_data.columns:
                cluster_info[f'{feature}_mean'] = cluster_data[feature].mean()
                cluster_info[f'{feature}_std'] = cluster_data[feature].std()
        
        cluster_stats.append(cluster_info)
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    # CSVに保存
    cluster_path = output_dir / f'weakness_clusters_{base_model.lower()}.csv'
    cluster_stats_df.to_csv(cluster_path, index=False)
    print(f"弱いケースのクラスタ分析を保存: {cluster_path}")
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PCAで2次元に投影
    pca = PCA(n_components=2, random_state=42)
    weak_data_pca = pca.fit_transform(weak_data_scaled)
    
    # クラスタを可視化
    ax1 = axes[0]
    scatter = ax1.scatter(weak_data_pca[:, 0], weak_data_pca[:, 1], 
                         c=clusters, cmap='viridis', alpha=0.6, s=50)
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=11)
    ax1.set_title(f'Weak Cases Clustering (n={len(weak_data)})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # 改善量で色分け
    ax2 = axes[1]
    if 'improvement' in weak_data.columns:
        scatter2 = ax2.scatter(weak_data_pca[:, 0], weak_data_pca[:, 1], 
                             c=weak_data['improvement'], cmap='RdYlGn', alpha=0.6, s=50)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=11)
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=11)
        ax2.set_title('QPP Improvement by Cluster', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Improvement')
    
    plt.tight_layout()
    cluster_viz_path = output_dir / f'weakness_clusters_visualization_{base_model.lower()}.png'
    plt.savefig(cluster_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"クラスタ可視化を保存: {cluster_viz_path}")
    
    return cluster_stats_df


def analyze_improved_vs_not_improved(
    weak_cases: Dict[str, pd.DataFrame],
    all_cases: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    QPP統合が効果的なケースと効果的でないケースの違いを分析
    """
    weak_improved = weak_cases['weak_improved']
    weak_not_improved = weak_cases['weak_not_improved']
    
    if len(weak_improved) == 0 or len(weak_not_improved) == 0:
        print("QPP統合が効果的なケースまたは効果的でないケースが見つかりませんでした。")
        return
    
    # QPP特徴量を取得
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer',
                   'BERT_proba', 'RoBERTa_proba', 'Transfer_proba', 'LogReg_pred',
                   'BERT_pred', 'RoBERTa_pred', 'Transfer_pred', 'threshold', 'group',
                   'score_percentile', 'percentile_bin', 'base_error', 'logreg_error',
                   'base_accuracy', 'logreg_accuracy', 'improvement']
    qpp_features = [col for col in all_cases.columns 
                   if col not in exclude_cols and all_cases[col].dtype in [np.float64, np.int64]]
    
    # 統計分析
    comparison_results = []
    
    for feature in qpp_features:
        if feature not in all_cases.columns:
            continue
        
        improved_values = weak_improved[feature].dropna()
        not_improved_values = weak_not_improved[feature].dropna()
        
        if len(improved_values) == 0 or len(not_improved_values) == 0:
            continue
        
        # 統計的検定
        try:
            _, pvalue = stats.mannwhitneyu(improved_values, not_improved_values, alternative='two-sided')
        except:
            pvalue = np.nan
        
        # 効果量
        def cohens_d(group1, group2):
            if len(group1) == 0 or len(group2) == 0:
                return np.nan
            n1, n2 = len(group1), len(group2)
            var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std == 0:
                return np.nan
            return (group1.mean() - group2.mean()) / pooled_std
        
        cohens_d_val = cohens_d(improved_values, not_improved_values)
        
        comparison_results.append({
            'feature': feature,
            'improved_mean': improved_values.mean(),
            'improved_std': improved_values.std(),
            'improved_n': len(improved_values),
            'not_improved_mean': not_improved_values.mean(),
            'not_improved_std': not_improved_values.std(),
            'not_improved_n': len(not_improved_values),
            'pvalue': pvalue,
            'cohens_d': cohens_d_val,
            'significant': pvalue < 0.05 if not np.isnan(pvalue) else False
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values('cohens_d', key=abs, ascending=False)
    
    # CSVに保存
    comparison_path = output_dir / f'improved_vs_not_improved_{base_model.lower()}.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"QPP統合効果の比較分析を保存: {comparison_path}")
    
    # 可視化
    significant_features = comparison_df[comparison_df['significant']].head(15)
    
    if len(significant_features) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x_pos = np.arange(len(significant_features))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, significant_features['improved_mean'], width,
                      label='Weak + QPP Improved', alpha=0.7, color='green')
        bars2 = ax.bar(x_pos + width/2, significant_features['not_improved_mean'], width,
                      label='Weak + QPP Not Improved', alpha=0.7, color='red')
        
        # エラーバー
        ax.errorbar(x_pos - width/2, significant_features['improved_mean'],
                   yerr=significant_features['improved_std'], fmt='none', color='black', capsize=3)
        ax.errorbar(x_pos + width/2, significant_features['not_improved_mean'],
                   yerr=significant_features['not_improved_std'], fmt='none', color='black', capsize=3)
        
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Mean Value', fontsize=12)
        ax.set_title('Features Distinguishing QPP-Improved vs Not-Improved Cases', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(significant_features['feature'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        comparison_viz_path = output_dir / f'improved_vs_not_improved_comparison_{base_model.lower()}.png'
        plt.savefig(comparison_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"QPP統合効果の比較可視化を保存: {comparison_viz_path}")
    
    return comparison_df


def main():
    parser = argparse.ArgumentParser(
        description="RoBERTaが弱い部分の共通特徴を探索"
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        required=True,
        help="results.csvのパス"
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
        "--error-threshold",
        type=float,
        default=0.0,
        help="弱いケースの誤分類率閾値（デフォルト: 0.0 = 誤分類した全て）"
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.01,
        help="QPP統合が効果的とみなす改善量の閾値（デフォルト: 0.01）"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="クラスタリングのクラスタ数（デフォルト: 3）"
    )
    
    args = parser.parse_args()
    
    results_csv_path = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    
    if not results_csv_path.exists():
        print(f"エラー: results.csvが見つかりません: {results_csv_path}")
        return
    
    # ベースモデルを自動検出
    base_model = args.base_model
    if base_model is None:
        results_df = pd.read_csv(results_csv_path, nrows=1)
        if 'RoBERTa' in results_df.columns:
            base_model = 'RoBERTa'
        elif 'BERT' in results_df.columns:
            base_model = 'BERT'
        elif 'Transfer' in results_df.columns:
            base_model = 'Transfer'
        else:
            print("エラー: ベースモデルを自動検出できませんでした。--base-modelを指定してください。")
            return
    
    print(f"=== RoBERTaが弱い部分の共通特徴を探索 ===")
    print(f"ベースモデル: {base_model}")
    print()
    
    # データを読み込む
    results_df = pd.read_csv(results_csv_path)
    print(f"results.csvを読み込み: {len(results_df)} 行")
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 弱いケースを特定
    print("弱いケースを特定中...")
    weak_cases = identify_weak_cases(
        results_df, 
        base_model, 
        error_rate_threshold=args.error_threshold,
        improvement_threshold=args.improvement_threshold
    )
    
    print(f"弱いケース（全体）: {len(weak_cases['weak_all'])} 件")
    print(f"弱い + QPP統合が効果的: {len(weak_cases['weak_improved'])} 件")
    print(f"弱い + QPP統合が効果的でない: {len(weak_cases['weak_not_improved'])} 件")
    print()
    
    # 共通特徴を分析
    print("共通特徴を分析中...")
    stats_df = analyze_common_features(weak_cases, results_df, output_dir, base_model)
    
    # 可視化
    print("可視化を生成中...")
    plot_feature_comparison(weak_cases, results_df, stats_df, output_dir, base_model)
    
    # クラスタリング
    print("クラスタリングを実行中...")
    cluster_stats_df = cluster_weak_cases(weak_cases, results_df, output_dir, base_model, args.n_clusters)
    
    # QPP統合が効果的なケースと効果的でないケースの比較
    print("QPP統合効果の比較分析中...")
    comparison_df = analyze_improved_vs_not_improved(weak_cases, results_df, output_dir, base_model)
    
    print()
    print("=== 分析完了 ===")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()

