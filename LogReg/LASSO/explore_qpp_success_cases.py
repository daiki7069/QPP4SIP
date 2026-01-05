"""
RoBERTaが誤分類したがQPP統合で正解したケースを探索的に分析
閾値依存を考慮し、複数の閾値で分析を行う
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, roc_auc_score
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
        USE_JAPANESE = True
    else:
        matplotlib.rcParams['font.family'] = 'DejaVu Sans'
        matplotlib.rcParams['axes.unicode_minus'] = False
        USE_JAPANESE = False
except Exception as e:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'
    matplotlib.rcParams['axes.unicode_minus'] = False
    USE_JAPANESE = False

sns.set_style("whitegrid")
sns.set_palette("husl")


def identify_success_cases_threshold_free(
    results_df: pd.DataFrame,
    base_model: str,
    n_thresholds: int = 20,
    use_median_threshold: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    閾値に依存しない方法で、RoBERTaが誤分類したがQPP統合で正解したケースを特定
    
    Args:
        use_median_threshold: Trueの場合、中央値閾値を使用。Falseの場合、複数閾値で分析
    
    Returns:
        dict with keys: 'success_cases', 'all_roberta_errors', 'statistics'
    """
    results_df = results_df.copy()
    
    # RoBERTaのlogit値を確率に変換
    roberta_logit = results_df[base_model].values
    roberta_proba = expit(roberta_logit)
    
    # LogRegのlogit値を確率に変換
    if 'LogReg' not in results_df.columns:
        print("LogRegの予測結果が見つかりません。")
        return {}
    
    logreg_logit = results_df['LogReg'].values
    logreg_proba = expit(logreg_logit)
    
    labels = results_df['label'].values
    
    if use_median_threshold:
        # 中央値閾値を使用（より厳密な分析）
        roberta_thresh = np.median(roberta_proba)
        logreg_thresh = 0.5
        
        roberta_pred = (roberta_proba >= roberta_thresh).astype(int)
        logreg_pred = (logreg_proba >= logreg_thresh).astype(int)
        
        # RoBERTaが誤分類、LogRegが正解
        roberta_error = (roberta_pred != labels)
        logreg_correct = (logreg_pred == labels)
        success_mask = roberta_error & logreg_correct
        
        success_indices = np.where(success_mask)[0]
        roberta_error_indices = np.where(roberta_error)[0]
        
        success_cases = results_df.iloc[success_indices].copy() if len(success_indices) > 0 else pd.DataFrame()
        all_roberta_errors = results_df.iloc[roberta_error_indices].copy() if len(roberta_error_indices) > 0 else pd.DataFrame()
        
        statistics = {
            'n_success_cases': len(success_indices),
            'n_roberta_errors': len(roberta_error_indices),
            'success_rate': len(success_indices) / len(roberta_error_indices) if len(roberta_error_indices) > 0 else 0,
            'roberta_threshold': roberta_thresh,
            'logreg_threshold': logreg_thresh,
            'method': 'median_threshold'
        }
    else:
        # 複数の閾値で分析（より広範囲な探索）
        roberta_thresholds = np.percentile(roberta_proba, np.linspace(10, 90, n_thresholds))
        logreg_thresholds = np.linspace(0.1, 0.9, n_thresholds)
        
        success_indices_set = set()
        
        # 各閾値の組み合わせで成功ケースを収集
        for roberta_thresh in roberta_thresholds:
            for logreg_thresh in logreg_thresholds:
                roberta_pred = (roberta_proba >= roberta_thresh).astype(int)
                logreg_pred = (logreg_proba >= logreg_thresh).astype(int)
                
                # RoBERTaが誤分類、LogRegが正解
                roberta_error = (roberta_pred != labels)
                logreg_correct = (logreg_pred == labels)
                success_mask = roberta_error & logreg_correct
                
                success_indices = np.where(success_mask)[0]
                success_indices_set.update(success_indices)
        
        # 成功ケースを取得
        success_indices_list = list(success_indices_set)
        success_cases = results_df.iloc[success_indices_list].copy() if len(success_indices_list) > 0 else pd.DataFrame()
        
        # RoBERTaが誤分類した全ケース（中央値閾値で）
        roberta_thresh = np.median(roberta_proba)
        roberta_pred = (roberta_proba >= roberta_thresh).astype(int)
        roberta_error = (roberta_pred != labels)
        roberta_error_indices = np.where(roberta_error)[0]
        all_roberta_errors = results_df.iloc[roberta_error_indices].copy() if len(roberta_error_indices) > 0 else pd.DataFrame()
        
        statistics = {
            'n_success_cases': len(success_indices_list),
            'n_roberta_errors': len(roberta_error_indices),
            'success_rate': len(success_indices_list) / len(roberta_error_indices) if len(roberta_error_indices) > 0 else 0,
            'n_thresholds_tested': n_thresholds * n_thresholds,
            'method': 'multiple_thresholds'
        }
    
    return {
        'success_cases': success_cases,
        'all_roberta_errors': all_roberta_errors,
        'statistics': statistics
    }


def analyze_success_case_features(
    success_cases: pd.DataFrame,
    all_roberta_errors: pd.DataFrame,
    all_cases: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    成功ケースの特徴を分析
    """
    if len(success_cases) == 0:
        print("成功ケースが見つかりませんでした。")
        return
    
    # QPP特徴量を取得
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']
    qpp_features = [col for col in all_cases.columns 
                   if col not in exclude_cols and all_cases[col].dtype in [np.float64, np.int64]]
    
    # 統計分析結果
    stats_results = []
    
    for feature in qpp_features:
        if feature not in all_cases.columns:
            continue
        
        success_values = success_cases[feature].dropna()
        error_values = all_roberta_errors[feature].dropna()
        all_values = all_cases[feature].dropna()
        
        if len(success_values) == 0:
            continue
        
        # 統計的検定
        stats_data = {
            'feature': feature,
            'success_mean': success_values.mean(),
            'success_std': success_values.std(),
            'success_median': success_values.median(),
            'error_mean': error_values.mean() if len(error_values) > 0 else np.nan,
            'error_std': error_values.std() if len(error_values) > 0 else np.nan,
            'error_median': error_values.median() if len(error_values) > 0 else np.nan,
            'all_mean': all_values.mean(),
            'all_std': all_values.std(),
        }
        
        # Mann-Whitney U test
        if len(success_values) > 0 and len(error_values) > 0:
            try:
                stat, p = mannwhitneyu(success_values, error_values, alternative='two-sided')
                stats_data['p_success_vs_error'] = p
                stats_data['statistic'] = stat
            except:
                stats_data['p_success_vs_error'] = np.nan
                stats_data['statistic'] = np.nan
        else:
            stats_data['p_success_vs_error'] = np.nan
            stats_data['statistic'] = np.nan
        
        # Cohen's d
        if len(success_values) > 0 and len(error_values) > 0:
            pooled_std = np.sqrt((success_values.std()**2 + error_values.std()**2) / 2)
            if pooled_std > 0:
                cohens_d = (success_values.mean() - error_values.mean()) / pooled_std
                stats_data['cohens_d'] = cohens_d
            else:
                stats_data['cohens_d'] = np.nan
        else:
            stats_data['cohens_d'] = np.nan
        
        stats_results.append(stats_data)
    
    stats_df = pd.DataFrame(stats_results)
    
    # 統計的に有意な特徴を抽出（緩い条件も含む）
    significant_features_strict = stats_df[
        (stats_df['p_success_vs_error'] < 0.05) & 
        (stats_df['cohens_d'].abs() > 0.2)
    ].sort_values('cohens_d', key=abs, ascending=False)
    
    significant_features_loose = stats_df[
        (stats_df['p_success_vs_error'] < 0.1) & 
        (stats_df['cohens_d'].abs() > 0.1)
    ].sort_values('cohens_d', key=abs, ascending=False)
    
    significant_features = significant_features_strict if len(significant_features_strict) > 0 else significant_features_loose
    
    # 結果を保存
    output_path = output_dir / f'qpp_success_case_features_{base_model.lower()}.csv'
    stats_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    # 統計的に有意な特徴を表示
    if len(significant_features_strict) > 0:
        print(f"\n=== 統計的に有意な特徴 (p < 0.05, |Cohen's d| > 0.2) ===")
        print(f"Found {len(significant_features_strict)} significant features\n")
        display_features = significant_features_strict
    else:
        print(f"\n=== 統計的に有意な特徴 (緩い条件: p < 0.1, |Cohen's d| > 0.1) ===")
        print(f"Found {len(significant_features_loose)} significant features\n")
        display_features = significant_features_loose
    
    print(f"Found {len(significant_features)} significant features\n")
    for _, row in display_features.head(20).iterrows():
        diff = row['success_mean'] - row['error_mean']
        pct_diff = (diff / row['error_mean'] * 100) if row['error_mean'] != 0 else 0
        print(f"{row['feature']:15s}: Success={row['success_mean']:8.4f}, Error={row['error_mean']:8.4f}, "
              f"Diff={diff:+.4f} ({pct_diff:+.1f}%), p={row['p_success_vs_error']:.4f}, d={row['cohens_d']:.3f}")
    
    return stats_df, significant_features


def plot_success_case_analysis(
    results_df: pd.DataFrame,
    success_cases: pd.DataFrame,
    all_roberta_errors: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    成功ケースの可視化
    """
    if len(success_cases) == 0:
        print("成功ケースが見つかりませんでした。可視化をスキップします。")
        return
    
    # RoBERTaとLogRegのlogit値と確率値を計算
    roberta_logit = results_df[base_model].values
    roberta_proba = expit(roberta_logit)
    logreg_logit = results_df['LogReg'].values
    logreg_proba = expit(logreg_logit)
    labels = results_df['label'].values
    
    success_indices = success_cases.index
    error_indices = all_roberta_errors.index
    
    # 1. RoBERTa logit値の分布比較
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1-1. RoBERTa logit値の分布
    ax = axes[0, 0]
    ax.hist(roberta_logit[error_indices], bins=50, alpha=0.5, 
           label='All RoBERTa Errors', color='coral', density=True)
    ax.hist(roberta_logit[success_indices], bins=50, alpha=0.7,
           label='QPP Success Cases', color='green', density=True)
    ax.set_xlabel(f'{base_model} Logit', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    if USE_JAPANESE:
        ax.set_title(f'{base_model} Logit Distribution: Success vs All Errors', fontsize=13)
    else:
        ax.set_title(f'{base_model} Logit Distribution: Success vs All Errors', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 平均値を表示
    ax.axvline(roberta_logit[error_indices].mean(), color='coral', 
              linestyle='--', linewidth=2, label=f'Error Mean: {roberta_logit[error_indices].mean():.3f}')
    ax.axvline(roberta_logit[success_indices].mean(), color='green',
              linestyle='--', linewidth=2, label=f'Success Mean: {roberta_logit[success_indices].mean():.3f}')
    
    # 1-2. RoBERTa確率値の分布
    ax = axes[0, 1]
    ax.hist(roberta_proba[error_indices], bins=50, alpha=0.5,
           label='All RoBERTa Errors', color='coral', density=True)
    ax.hist(roberta_proba[success_indices], bins=50, alpha=0.7,
           label='QPP Success Cases', color='green', density=True)
    ax.set_xlabel(f'{base_model} Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    if USE_JAPANESE:
        ax.set_title(f'{base_model} Probability Distribution: Success vs All Errors', fontsize=13)
    else:
        ax.set_title(f'{base_model} Probability Distribution: Success vs All Errors', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='black', linestyle=':', linewidth=1)
    
    # 1-3. LogReg logit値の分布
    ax = axes[0, 2]
    ax.hist(logreg_logit[error_indices], bins=50, alpha=0.5,
           label='All RoBERTa Errors', color='coral', density=True)
    ax.hist(logreg_logit[success_indices], bins=50, alpha=0.7,
           label='QPP Success Cases', color='green', density=True)
    ax.set_xlabel('LogReg Logit', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    if USE_JAPANESE:
        ax.set_title('LogReg Logit Distribution: Success vs All Errors', fontsize=13)
    else:
        ax.set_title('LogReg Logit Distribution: Success vs All Errors', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. RoBERTa logit vs LogReg logit（散布図）
    ax = axes[1, 0]
    ax.scatter(roberta_logit[error_indices], logreg_logit[error_indices],
              alpha=0.3, s=10, color='coral', label='All RoBERTa Errors')
    ax.scatter(roberta_logit[success_indices], logreg_logit[success_indices],
              alpha=0.7, s=30, color='green', label='QPP Success Cases', edgecolors='black', linewidths=0.5)
    ax.set_xlabel(f'{base_model} Logit', fontsize=12)
    ax.set_ylabel('LogReg Logit', fontsize=12)
    if USE_JAPANESE:
        ax.set_title('RoBERTa Logit vs LogReg Logit', fontsize=13)
    else:
        ax.set_title('RoBERTa Logit vs LogReg Logit', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linestyle=':', linewidth=1)
    ax.axvline(0, color='black', linestyle=':', linewidth=1)
    
    # 3. RoBERTa確率 vs LogReg確率（散布図）
    ax = axes[1, 1]
    ax.scatter(roberta_proba[error_indices], logreg_proba[error_indices],
              alpha=0.3, s=10, color='coral', label='All RoBERTa Errors')
    ax.scatter(roberta_proba[success_indices], logreg_proba[success_indices],
              alpha=0.7, s=30, color='green', label='QPP Success Cases', edgecolors='black', linewidths=0.5)
    ax.set_xlabel(f'{base_model} Probability', fontsize=12)
    ax.set_ylabel('LogReg Probability', fontsize=12)
    if USE_JAPANESE:
        ax.set_title('RoBERTa Probability vs LogReg Probability', fontsize=13)
    else:
        ax.set_title('RoBERTa Probability vs LogReg Probability', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1)
    ax.axvline(0.5, color='black', linestyle=':', linewidth=1)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    # 4. LogReg logit値の変化（RoBERTa logitとの差）
    ax = axes[1, 2]
    logit_diff_error = logreg_logit[error_indices] - roberta_logit[error_indices]
    logit_diff_success = logreg_logit[success_indices] - roberta_logit[success_indices]
    
    ax.hist(logit_diff_error, bins=50, alpha=0.5,
           label='All RoBERTa Errors', color='coral', density=True)
    ax.hist(logit_diff_success, bins=50, alpha=0.7,
           label='QPP Success Cases', color='green', density=True)
    ax.set_xlabel('LogReg Logit - RoBERTa Logit', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    if USE_JAPANESE:
        ax.set_title('Logit Difference: Success vs All Errors', fontsize=13)
    else:
        ax.set_title('Logit Difference: Success vs All Errors', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linestyle=':', linewidth=1)
    
    plt.tight_layout()
    output_path = output_dir / f'qpp_success_case_analysis_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_qpp_feature_patterns(
    success_cases: pd.DataFrame,
    all_roberta_errors: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    top_n: int = 10
):
    """
    成功ケースのQPP特徴量パターンを可視化
    """
    if len(success_cases) == 0:
        return
    
    # QPP特徴量を取得
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']
    qpp_features = [col for col in success_cases.columns 
                   if col not in exclude_cols and success_cases[col].dtype in [np.float64, np.int64]]
    
    if len(qpp_features) == 0:
        return
    
    # 統計的に有意な特徴を特定
    stats_results = []
    for feature in qpp_features:
        if feature not in success_cases.columns or feature not in all_roberta_errors.columns:
            continue
        
        success_values = success_cases[feature].dropna()
        error_values = all_roberta_errors[feature].dropna()
        
        if len(success_values) == 0 or len(error_values) == 0:
            continue
        
        try:
            stat, p = mannwhitneyu(success_values, error_values, alternative='two-sided')
            pooled_std = np.sqrt((success_values.std()**2 + error_values.std()**2) / 2)
            cohens_d = (success_values.mean() - error_values.mean()) / pooled_std if pooled_std > 0 else np.nan
            
            stats_results.append({
                'feature': feature,
                'pvalue': p,
                'cohens_d': cohens_d,
                'success_mean': success_values.mean(),
                'error_mean': error_values.mean()
            })
        except:
            continue
    
    if len(stats_results) == 0:
        return
    
    stats_df = pd.DataFrame(stats_results)
    significant_features_strict = stats_df[
        (stats_df['pvalue'] < 0.05) & 
        (stats_df['cohens_d'].abs() > 0.2)
    ].sort_values('cohens_d', key=abs, ascending=False)
    
    significant_features_loose = stats_df[
        (stats_df['pvalue'] < 0.1) & 
        (stats_df['cohens_d'].abs() > 0.1)
    ].sort_values('cohens_d', key=abs, ascending=False)
    
    significant_features = significant_features_strict if len(significant_features_strict) > 0 else significant_features_loose
    
    if len(significant_features) == 0:
        print("統計的に有意な特徴が見つかりませんでした。")
        return
    
    # トップNの特徴を可視化
    top_features = significant_features.head(top_n)['feature'].tolist()
    
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        success_values = success_cases[feature].dropna()
        error_values = all_roberta_errors[feature].dropna()
        
        ax.hist(error_values, bins=30, alpha=0.5, label='All RoBERTa Errors', 
               color='coral', density=True)
        ax.hist(success_values, bins=30, alpha=0.7, label='QPP Success Cases',
               color='green', density=True)
        
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        
        row = significant_features[significant_features['feature'] == feature].iloc[0]
        ax.set_title(f'{feature}\np={row["pvalue"]:.4f}, d={row["cohens_d"]:.3f}', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 平均値を表示
        ax.axvline(error_values.mean(), color='coral', linestyle='--', linewidth=1.5)
        ax.axvline(success_values.mean(), color='green', linestyle='--', linewidth=1.5)
    
    # 余ったサブプロットを非表示
    for idx in range(len(top_features), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = output_dir / f'qpp_success_feature_patterns_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='RoBERTaが誤分類したがQPP統合で正解したケースを探索的に分析')
    parser.add_argument('--results-csv', type=str, required=True,
                       help='Path to results.csv file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--base-model', type=str, default='RoBERTa',
                       help='Base model name (default: RoBERTa)')
    parser.add_argument('--n-thresholds', type=int, default=20,
                       help='Number of thresholds to test (default: 20)')
    parser.add_argument('--use-median-threshold', action='store_true', default=True,
                       help='Use median threshold for more strict analysis (default: True)')
    parser.add_argument('--use-multiple-thresholds', action='store_true', default=False,
                       help='Use multiple thresholds for broader exploration (default: False)')
    
    args = parser.parse_args()
    
    results_csv_path = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {results_csv_path}")
    results_df = pd.read_csv(results_csv_path)
    print(f"Loaded {len(results_df)} samples")
    
    print("\n=== 閾値に依存しない成功ケースの特定 ===")
    use_median = not args.use_multiple_thresholds  # デフォルトは中央値閾値
    success_data = identify_success_cases_threshold_free(
        results_df, args.base_model, args.n_thresholds, use_median_threshold=use_median
    )
    
    if len(success_data) == 0 or len(success_data['success_cases']) == 0:
        print("成功ケースが見つかりませんでした。")
        return
    
    success_cases = success_data['success_cases']
    all_roberta_errors = success_data['all_roberta_errors']
    statistics = success_data['statistics']
    
    print(f"\n成功ケース数: {statistics['n_success_cases']}")
    print(f"RoBERTa誤分類数: {statistics['n_roberta_errors']}")
    print(f"成功率: {statistics['success_rate']:.4f} ({statistics['success_rate']*100:.2f}%)")
    if 'roberta_threshold' in statistics:
        print(f"使用した閾値: RoBERTa={statistics['roberta_threshold']:.4f}, LogReg={statistics['logreg_threshold']:.4f}")
    if 'n_thresholds_tested' in statistics:
        print(f"テストした閾値の組み合わせ数: {statistics['n_thresholds_tested']}")
    print(f"分析方法: {statistics.get('method', 'unknown')}")
    
    print("\n=== 成功ケースの特徴分析 ===")
    stats_df, significant_features = analyze_success_case_features(
        success_cases, all_roberta_errors, results_df, output_dir, args.base_model
    )
    
    print("\n=== 成功ケースの可視化 ===")
    plot_success_case_analysis(
        results_df, success_cases, all_roberta_errors, output_dir, args.base_model
    )
    
    print("\n=== QPP特徴量パターンの可視化 ===")
    plot_qpp_feature_patterns(
        success_cases, all_roberta_errors, output_dir, args.base_model, top_n=15
    )
    
    print("\n=== 分析完了 ===")


if __name__ == '__main__':
    main()

