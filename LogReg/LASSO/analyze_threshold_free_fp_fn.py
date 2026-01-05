"""
閾値に依存しないFP/FN分析スクリプト
1. ROC曲線上の各点でのFP/FN分析
2. パーセンタイルベースの分析
3. FP固有の特徴と全体傾向を分離する可視化
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
from sklearn.metrics import roc_curve, precision_recall_curve
warnings.filterwarnings('ignore')

# scipy.special.expit (sigmoid関数) を使用するため
try:
    from scipy.special import expit
except ImportError:
    def expit(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")

# フォントの設定（日本語フォントが利用可能な場合は使用、なければ英語のみ）
import matplotlib
import matplotlib.font_manager as fm
try:
    # 利用可能な日本語フォントを検索
    jp_font_candidates = [
        'Noto Sans CJK JP', 'Noto Sans Japanese', 'Takao', 'TakaoGothic', 'TakaoPGothic',
        'IPAexGothic', 'IPAPGothic', 'IPAPMincho', 'VL PGothic', 'VL Gothic',
        'Yu Gothic', 'YuGothic', 'Meiryo', 'MS PGothic', 'MS Gothic',
        'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Osaka', 'Arial Unicode MS'
    ]
    
    # システムにインストールされているフォントを確認
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


def classify_by_roc_points(
    results_df: pd.DataFrame,
    base_model: str,
    n_points: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    ROC曲線上の各点でFP/FNを分類
    閾値に依存しない分析のため、ROC曲線上の複数の点で分析を行う
    """
    # ベースモデルのlogit値を確率に変換
    results_df = results_df.copy()
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # ROC曲線を計算
    fpr, tpr, thresholds = roc_curve(results_df['label'], results_df[f'{base_model}_proba'])
    
    # 等間隔でn_points個の閾値を選択
    threshold_indices = np.linspace(0, len(thresholds) - 1, n_points, dtype=int)
    selected_thresholds = thresholds[threshold_indices]
    
    # 各閾値でのFP/FNを分類
    classifications = []
    
    for thresh in selected_thresholds:
        # 予測を計算
        pred = (results_df[f'{base_model}_proba'] >= thresh).astype(int)
        
        # TP, FP, TN, FNを分類
        tp_mask = (pred == 1) & (results_df['label'] == 1)
        fp_mask = (pred == 1) & (results_df['label'] == 0)
        tn_mask = (pred == 0) & (results_df['label'] == 0)
        fn_mask = (pred == 0) & (results_df['label'] == 1)
        
        # 各グループに閾値情報を追加
        for mask, group_name in [(tp_mask, 'TP'), (fp_mask, 'FP'), (tn_mask, 'TN'), (fn_mask, 'FN')]:
            if mask.sum() > 0:
                group_df = results_df[mask].copy()
                group_df['threshold'] = thresh
                group_df['group'] = group_name
                classifications.append(group_df)
    
    # 全ての分類を結合
    if classifications:
        all_classifications = pd.concat(classifications, ignore_index=True)
    else:
        all_classifications = pd.DataFrame()
    
    return all_classifications


def classify_by_percentiles(
    results_df: pd.DataFrame,
    base_model: str,
    n_percentiles: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    パーセンタイルベースでFP/FNを分類
    予測スコアの順位に基づいて分析を行う
    """
    results_df = results_df.copy()
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # 予測スコアのパーセンタイルを計算
    results_df['score_percentile'] = results_df[f'{base_model}_proba'].rank(pct=True) * 100
    
    # パーセンタイル区間で分類
    percentile_bins = np.linspace(0, 100, n_percentiles + 1)
    results_df['percentile_bin'] = pd.cut(results_df['score_percentile'], bins=percentile_bins, 
                                         labels=[f'{int(percentile_bins[i])}-{int(percentile_bins[i+1])}' 
                                                for i in range(len(percentile_bins)-1)])
    
    # 各パーセンタイル区間でTP/FP/TN/FNを分類
    classifications = []
    
    for bin_label in results_df['percentile_bin'].cat.categories:
        bin_mask = results_df['percentile_bin'] == bin_label
        bin_df = results_df[bin_mask]
        
        if len(bin_df) == 0:
            continue
        
        # この区間での予測（上位パーセンタイルは正例と予測）
        percentile_upper = float(bin_label.split('-')[1])
        # 50パーセンタイル以上を正例と予測（調整可能）
        pred = (percentile_upper >= 50).astype(int)
        
        # TP, FP, TN, FNを分類
        tp_mask = (bin_df['label'] == 1)
        fp_mask = (bin_df['label'] == 0) & (pred == 1)
        tn_mask = (bin_df['label'] == 0) & (pred == 0)
        fn_mask = (bin_df['label'] == 1) & (pred == 0)
        
        for mask, group_name in [(tp_mask, 'TP'), (fp_mask, 'FP'), (tn_mask, 'TN'), (fn_mask, 'FN')]:
            if mask.sum() > 0:
                group_df = bin_df[mask].copy()
                group_df['percentile_bin'] = bin_label
                group_df['group'] = group_name
                classifications.append(group_df)
    
    if classifications:
        all_classifications = pd.concat(classifications, ignore_index=True)
    else:
        all_classifications = pd.DataFrame()
    
    return all_classifications


def analyze_fp_specific_features(
    results_df: pd.DataFrame,
    target_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    feature_cols: Optional[List[str]] = None,
    group_name: str = "FP"
) -> pd.DataFrame:
    """
    FP/FN固有の特徴と全体傾向を分離する分析
    FP/FN vs TP/TN/全体を比較し、統計的検定を行う
    
    Args:
        target_df: FPまたはFNのデータフレーム
        group_name: "FP" または "FN"
    """
    results_df = results_df.copy()
    
    # 特徴量カラムを取得
    if feature_cols is None:
        exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer',
                       'BERT_proba', 'RoBERTa_proba', 'Transfer_proba', 'LogReg_pred',
                       'BERT_pred', 'RoBERTa_pred', 'Transfer_pred', 'threshold', 'group',
                       'score_percentile', 'percentile_bin']
        feature_cols = [col for col in results_df.columns 
                      if col not in exclude_cols and results_df[col].dtype in [np.float64, np.int64]]
    
    # 全体データからTP/TN/FNを分類（簡易版：予測スコアの中央値で分類）
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    median_threshold = results_df[f'{base_model}_proba'].median()
    pred = (results_df[f'{base_model}_proba'] >= median_threshold).astype(int)
    
    tp_df = results_df[(pred == 1) & (results_df['label'] == 1)].copy()
    tn_df = results_df[(pred == 0) & (results_df['label'] == 0)].copy()
    fn_df = results_df[(pred == 0) & (results_df['label'] == 1)].copy()
    
    # 統計分析結果を保存
    stats_results = []
    prefix = group_name.lower()
    
    for feature in feature_cols:
        if feature not in results_df.columns:
            continue
        
        # 各グループの統計
        target_values = target_df[feature].dropna()
        tp_values = tp_df[feature].dropna() if feature in tp_df.columns else pd.Series()
        tn_values = tn_df[feature].dropna() if feature in tn_df.columns else pd.Series()
        fn_values = fn_df[feature].dropna() if feature in fn_df.columns else pd.Series()
        all_values = results_df[feature].dropna()
        
        if len(target_values) == 0:
            continue
        
        # 平均値と標準偏差
        target_mean = target_values.mean()
        target_std = target_values.std()
        all_mean = all_values.mean()
        all_std = all_values.std()
        
        # ターゲット vs 全体の統計的検定（Mann-Whitney U test）
        if len(all_values) > 0:
            try:
                stat_target_all, pvalue_target_all = stats.mannwhitneyu(target_values, all_values, alternative='two-sided')
            except:
                stat_target_all, pvalue_target_all = np.nan, np.nan
        else:
            stat_target_all, pvalue_target_all = np.nan, np.nan
        
        # ターゲット vs TPの統計的検定
        if len(tp_values) > 0:
            try:
                stat_target_tp, pvalue_target_tp = stats.mannwhitneyu(target_values, tp_values, alternative='two-sided')
            except:
                stat_target_tp, pvalue_target_tp = np.nan, np.nan
        else:
            stat_target_tp, pvalue_target_tp = np.nan, np.nan
        
        # ターゲット vs TNの統計的検定
        if len(tn_values) > 0:
            try:
                stat_target_tn, pvalue_target_tn = stats.mannwhitneyu(target_values, tn_values, alternative='two-sided')
            except:
                stat_target_tn, pvalue_target_tn = np.nan, np.nan
        else:
            stat_target_tn, pvalue_target_tn = np.nan, np.nan
        
        # ターゲット vs FNの統計的検定
        if len(fn_values) > 0:
            try:
                stat_target_fn, pvalue_target_fn = stats.mannwhitneyu(target_values, fn_values, alternative='two-sided')
            except:
                stat_target_fn, pvalue_target_fn = np.nan, np.nan
        else:
            stat_target_fn, pvalue_target_fn = np.nan, np.nan
        
        # 効果量（Cohen's d）を計算
        def cohens_d(group1, group2):
            if len(group1) == 0 or len(group2) == 0:
                return np.nan
            n1, n2 = len(group1), len(group2)
            var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            if pooled_std == 0:
                return np.nan
            return (group1.mean() - group2.mean()) / pooled_std
        
        cohens_d_target_all = cohens_d(target_values, all_values)
        cohens_d_target_tp = cohens_d(target_values, tp_values) if len(tp_values) > 0 else np.nan
        cohens_d_target_tn = cohens_d(target_values, tn_values) if len(tn_values) > 0 else np.nan
        cohens_d_target_fn = cohens_d(target_values, fn_values) if len(fn_values) > 0 else np.nan
        
        stats_results.append({
            'feature': feature,
            f'{prefix}_mean': target_mean,
            f'{prefix}_std': target_std,
            f'{prefix}_n': len(target_values),
            'all_mean': all_mean,
            'all_std': all_std,
            'all_n': len(all_values),
            'tp_mean': tp_values.mean() if len(tp_values) > 0 else np.nan,
            'tn_mean': tn_values.mean() if len(tn_values) > 0 else np.nan,
            'fn_mean': fn_values.mean() if len(fn_values) > 0 else np.nan,
            f'{prefix}_vs_all_pvalue': pvalue_target_all,
            f'{prefix}_vs_tp_pvalue': pvalue_target_tp,
            f'{prefix}_vs_tn_pvalue': pvalue_target_tn,
            f'{prefix}_vs_fn_pvalue': pvalue_target_fn,
            f'{prefix}_vs_all_cohens_d': cohens_d_target_all,
            f'{prefix}_vs_tp_cohens_d': cohens_d_target_tp,
            f'{prefix}_vs_tn_cohens_d': cohens_d_target_tn,
            f'{prefix}_vs_fn_cohens_d': cohens_d_target_fn,
            f'{prefix}_specific': pvalue_target_all < 0.05 and abs(cohens_d_target_all) > 0.2,  # グループ固有の特徴かどうか
            f'{prefix}_vs_others_significant': (pvalue_target_tp < 0.05 or pvalue_target_tn < 0.05 or pvalue_target_fn < 0.05)
        })
    
    stats_df = pd.DataFrame(stats_results)
    
    # CSVに保存
    stats_path = output_dir / f'{group_name.lower()}_specific_features_analysis_{base_model.lower()}.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"{group_name}固有特徴分析を保存: {stats_path}")
    
    return stats_df


def plot_fp_vs_all_comparison(
    results_df: pd.DataFrame,
    target_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    group_name: str = "FP",
    top_n: int = 15
):
    """
    FP/FN vs 全体/TP/TN/FNの比較を可視化
    FP/FN固有の特徴と全体傾向を分離して表示
    """
    prefix = group_name.lower()
    
    # 統計的に有意な特徴を選択
    pvalue_cols = [f'{prefix}_vs_all_pvalue', f'{prefix}_vs_tp_pvalue', 
                   f'{prefix}_vs_tn_pvalue', f'{prefix}_vs_fn_pvalue']
    significant_features = stats_df[
        stats_df[pvalue_cols].min(axis=1) < 0.05
    ].copy()
    
    # 効果量の絶対値でソート
    cohens_d_cols = [f'{prefix}_vs_all_cohens_d', f'{prefix}_vs_tp_cohens_d', 
                     f'{prefix}_vs_tn_cohens_d', f'{prefix}_vs_fn_cohens_d']
    significant_features['max_effect_size'] = significant_features[cohens_d_cols].abs().max(axis=1)
    significant_features = significant_features.sort_values('max_effect_size', ascending=False)
    
    # 上位N個の特徴を選択
    top_features = significant_features.head(top_n)['feature'].tolist()
    
    if len(top_features) == 0:
        print(f"統計的に有意な特徴が見つかりませんでした。")
        return
    
    # 全体データからTP/TN/FNを分類
    results_df = results_df.copy()
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    median_threshold = results_df[f'{base_model}_proba'].median()
    pred = (results_df[f'{base_model}_proba'] >= median_threshold).astype(int)
    
    tp_df = results_df[(pred == 1) & (results_df['label'] == 1)].copy()
    tn_df = results_df[(pred == 0) & (results_df['label'] == 0)].copy()
    fn_df = results_df[(pred == 0) & (results_df['label'] == 1)].copy()
    
    # グラフを作成
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # データを準備
        target_values = target_df[feature].dropna()
        tp_values = tp_df[feature].dropna() if feature in tp_df.columns else pd.Series()
        tn_values = tn_df[feature].dropna() if feature in tn_df.columns else pd.Series()
        fn_values = fn_df[feature].dropna() if feature in fn_df.columns else pd.Series()
        all_values = results_df[feature].dropna()
        
        # ボックスプロット
        plot_data = []
        for values, group_label in [(target_values, group_name), (tp_values, 'TP'), 
                                   (tn_values, 'TN'), (fn_values, 'FN'), (all_values, 'All')]:
            if len(values) > 0:
                for val in values:
                    plot_data.append({'Group': group_label, 'Value': val})
        
        plot_df = pd.DataFrame(plot_data)
        
        if len(plot_df) > 0:
            sns.boxplot(data=plot_df, x='Group', y='Value', ax=ax, 
                       order=[group_name, 'TP', 'TN', 'FN', 'All'])
            
            # 統計情報を表示
            stats_row = stats_df[stats_df['feature'] == feature].iloc[0]
            title_parts = [feature]
            if stats_row[f'{prefix}_specific']:
                title_parts.append(f'({group_name}固有)')
            pvalue = stats_row[f'{prefix}_vs_all_pvalue']
            if pvalue < 0.05:
                title_parts.append(f'p={pvalue:.3f}')
            
            ax.set_title(' '.join(title_parts), fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('Value', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
    
    # 余分なサブプロットを非表示
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / f'{group_name.lower()}_vs_all_comparison_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{group_name} vs 全体比較を保存: {output_path}")


def plot_fp_specific_heatmap(
    stats_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    group_name: str = "FP"
):
    """
    FP/FN固有の特徴をヒートマップで可視化
    p値と効果量を組み合わせて表示
    """
    prefix = group_name.lower()
    pvalue_cols = [f'{prefix}_vs_all_pvalue', f'{prefix}_vs_tp_pvalue', 
                   f'{prefix}_vs_tn_pvalue', f'{prefix}_vs_fn_pvalue']
    
    # 統計的に有意な特徴のみを選択
    significant_features = stats_df[
        stats_df[pvalue_cols].min(axis=1) < 0.05
    ].copy()
    
    if len(significant_features) == 0:
        print("統計的に有意な特徴が見つかりませんでした。")
        return
    
    # ヒートマップ用のデータを準備
    heatmap_data = []
    for _, row in significant_features.iterrows():
        feature = row['feature']
        for comparison in ['all', 'tp', 'tn', 'fn']:
            pvalue_col = f'{prefix}_vs_{comparison}_pvalue'
            cohens_d_col = f'{prefix}_vs_{comparison}_cohens_d'
            
            if pd.notna(row[pvalue_col]) and pd.notna(row[cohens_d_col]):
                # -log10(p値) * 効果量の符号で重み付け
                weight = -np.log10(row[pvalue_col] + 1e-10) * np.sign(row[cohens_d_col]) * abs(row[cohens_d_col])
                heatmap_data.append({
                    'Feature': feature,
                    'Comparison': comparison.upper(),
                    'Weight': weight,
                    'P-value': row[pvalue_col],
                    "Cohen's d": row[cohens_d_col]
                })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    if len(heatmap_df) == 0:
        return
    
    # ピボットテーブルを作成
    pivot_weight = heatmap_df.pivot(index='Feature', columns='Comparison', values='Weight')
    pivot_pvalue = heatmap_df.pivot(index='Feature', columns='Comparison', values='P-value')
    
    # ヒートマップを描画
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(pivot_weight) * 0.5)))
    
    # 重み付けヒートマップ
    ax1 = axes[0]
    sns.heatmap(pivot_weight, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                square=False, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax1)
    ax1.set_title(f'{group_name} vs Others: Weighted Significance\n(-log10(p) * sign(Cohen\'s d) * |Cohen\'s d|)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Comparison Group', fontsize=11)
    ax1.set_ylabel('Feature', fontsize=11)
    
    # p値のヒートマップ
    ax2 = axes[1]
    # p値を-log10変換（視覚的に分かりやすく）
    pivot_pvalue_log = -np.log10(pivot_pvalue + 1e-10)
    sns.heatmap(pivot_pvalue_log, annot=True, fmt='.2f', cmap='YlOrRd',
                square=False, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "-log10(p-value)"}, ax=ax2)
    ax2.set_title(f'{group_name} vs Others: Statistical Significance\n(-log10(p-value))', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Comparison Group', fontsize=11)
    ax2.set_ylabel('Feature', fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / f'{group_name.lower()}_specific_heatmap_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"{group_name}固有特徴ヒートマップを保存: {output_path}")


def plot_roberta_weakness_and_qpp_improvement(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    qpp_features: List[str] = None
):
    """
    RoBERTaの弱い部分（誤分類率が高いQPP値域）を特定し、
    QPP統合による改善を可視化
    """
    results_df = results_df.copy()
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # 予測を計算
    median_threshold = results_df[f'{base_model}_proba'].median()
    results_df[f'{base_model}_pred'] = (results_df[f'{base_model}_proba'] >= median_threshold).astype(int)
    
    # LogRegの予測も計算
    if 'LogReg' not in results_df.columns:
        print("LogRegの予測結果が見つかりません。QPP統合による改善分析をスキップします。")
        return
    
    results_df['LogReg_pred'] = (results_df['LogReg'] >= 0.5).astype(int)
    
    # 誤分類を計算
    results_df['base_error'] = (results_df[f'{base_model}_pred'] != results_df['label']).astype(int)
    results_df['logreg_error'] = (results_df['LogReg_pred'] != results_df['label']).astype(int)
    results_df['base_accuracy'] = (results_df[f'{base_model}_pred'] == results_df['label']).astype(int)
    results_df['logreg_accuracy'] = (results_df['LogReg_pred'] == results_df['label']).astype(int)
    
    # 全体の性能を計算（基準線として使用）
    overall_base_error = results_df['base_error'].mean()
    overall_logreg_error = results_df['logreg_error'].mean()
    overall_base_accuracy = results_df['base_accuracy'].mean()
    overall_logreg_accuracy = results_df['logreg_accuracy'].mean()
    
    # QPP特徴量を取得
    if qpp_features is None:
        exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer',
                       'BERT_proba', 'RoBERTa_proba', 'Transfer_proba', 'LogReg_pred',
                       'BERT_pred', 'RoBERTa_pred', 'Transfer_pred', 'threshold', 'group',
                       'score_percentile', 'percentile_bin', 'base_error', 'logreg_error',
                       'base_accuracy', 'logreg_accuracy']
        qpp_features = [col for col in results_df.columns 
                       if col not in exclude_cols and results_df[col].dtype in [np.float64, np.int64]]
    
    # 各QPP特徴量について分析
    n_features = len(qpp_features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(qpp_features):
        if feature not in results_df.columns:
            continue
        
        ax = axes[idx]
        
        # 特徴量の値を5つの区間に分割
        feature_values = results_df[feature].dropna()
        if len(feature_values) == 0:
            continue
        
        # パーセンタイルで区間を定義
        bins = np.percentile(feature_values, [0, 20, 40, 60, 80, 100])
        labels = []
        for i in range(len(bins)-1):
            label = f'{bins[i]:.2f}-{bins[i+1]:.2f}'
            if label not in labels:
                labels.append(label)
            else:
                labels.append(f'{bins[i]:.4f}-{bins[i+1]:.4f}')
        
        results_df['feature_bin'] = pd.cut(results_df[feature], bins=bins, 
                                          labels=labels, 
                                          include_lowest=True,
                                          duplicates='drop')
        
        # 各区間での性能を計算
        bin_stats = []
        for bin_label in results_df['feature_bin'].cat.categories:
            bin_mask = results_df['feature_bin'] == bin_label
            bin_data = results_df[bin_mask]
            
            if len(bin_data) == 0:
                continue
            
            base_error_rate = bin_data['base_error'].mean()
            logreg_error_rate = bin_data['logreg_error'].mean()
            base_accuracy = bin_data['base_accuracy'].mean()
            logreg_accuracy = bin_data['logreg_accuracy'].mean()
            improvement = logreg_accuracy - base_accuracy  # 精度の改善
            
            bin_stats.append({
                'bin': bin_label,
                'base_error_rate': base_error_rate,
                'logreg_error_rate': logreg_error_rate,
                'base_accuracy': base_accuracy,
                'logreg_accuracy': logreg_accuracy,
                'improvement': improvement,
                'n_samples': len(bin_data)
            })
        
        bin_stats_df = pd.DataFrame(bin_stats)
        
        if len(bin_stats_df) == 0:
            continue
        
        # プロット（RoBERTaの誤分類率とQPP統合による改善）
        x_pos = np.arange(len(bin_stats_df))
        width = 0.35
        
        # 左Y軸: 誤分類率
        ax2 = ax.twinx()
        
        # RoBERTaの誤分類率（各値域での性能）
        bars1 = ax.bar(x_pos - width/2, bin_stats_df['base_error_rate'], width, 
                      label=f'{base_model} Error Rate (by {feature} range)', alpha=0.7, color='red')
        
        # LogRegの誤分類率（各値域での性能）
        bars2 = ax.bar(x_pos + width/2, bin_stats_df['logreg_error_rate'], width, 
                      label=f'LogReg Error Rate (by {feature} range)', alpha=0.7, color='blue')
        
        # 全体の基準線を表示
        ax.axhline(y=overall_base_error, color='red', linestyle='--', alpha=0.5, linewidth=1,
                  label=f'{base_model} Overall ({overall_base_error:.3f})')
        ax.axhline(y=overall_logreg_error, color='blue', linestyle='--', alpha=0.5, linewidth=1,
                  label=f'LogReg Overall ({overall_logreg_error:.3f})')
        
        # 右Y軸: 改善量（各値域での改善）
        line = ax2.plot(x_pos, bin_stats_df['improvement'], 'o-', 
                       label='Accuracy Improvement (by range)', color='green', 
                       linewidth=2, markersize=8)
        
        # 全体の改善量を基準線として表示
        overall_improvement = overall_logreg_accuracy - overall_base_accuracy
        ax2.axhline(y=overall_improvement, color='green', linestyle='--', alpha=0.5, linewidth=1,
                   label=f'Overall Improvement ({overall_improvement:+.3f})')
        
        # 改善量をテキストで表示
        for i, (pos, imp) in enumerate(zip(x_pos, bin_stats_df['improvement'])):
            if abs(imp) > 0.01:  # 改善が1%以上の場合のみ表示
                ax2.text(pos, imp, f'{imp:+.3f}', ha='center', va='bottom' if imp > 0 else 'top',
                        fontsize=8, fontweight='bold', color='green')
        
        ax.set_xlabel(f'{feature} Value Range', fontsize=10)
        ax.set_ylabel('Error Rate', fontsize=10, color='black')
        ax2.set_ylabel('Accuracy Improvement', fontsize=10, color='green')
        ax.set_title(f'{feature}: {base_model} Weakness & QPP Improvement\n(Red bars: {base_model} errors by range, Blue bars: LogReg errors by range)', 
                    fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_stats_df['bin'], rotation=45, ha='right', fontsize=8)
        
        # 凡例を結合
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)
        
        ax.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='green')
        ax.grid(True, alpha=0.3, axis='y')
        
        # RoBERTaの弱い部分（誤分類率が高い）を強調
        max_error_idx = bin_stats_df['base_error_rate'].idxmax()
        if max_error_idx is not None:
            ax.axvline(x=max_error_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(max_error_idx, bin_stats_df.loc[max_error_idx, 'base_error_rate'] * 1.1,
                   f'Weakest\n({bin_stats_df.loc[max_error_idx, "base_error_rate"]:.2%})',
                   ha='center', fontsize=7, color='red', fontweight='bold')
    
    # 余分なサブプロットを非表示
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / f'roberta_weakness_qpp_improvement_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RoBERTaの弱い部分とQPP改善を保存: {output_path}")


def plot_qpp_threshold_analysis(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    qpp_features: List[str] = None
):
    """
    QPP特徴量の値域ごとにBERTの誤分類率を可視化
    「QPPを使うべき場面」を明確にする
    """
    results_df = results_df.copy()
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # 予測を計算（中央値で閾値設定）
    median_threshold = results_df[f'{base_model}_proba'].median()
    results_df[f'{base_model}_pred'] = (results_df[f'{base_model}_proba'] >= median_threshold).astype(int)
    
    # FPとFNを計算
    results_df['is_fp'] = ((results_df[f'{base_model}_pred'] == 1) & (results_df['label'] == 0)).astype(int)
    results_df['is_fn'] = ((results_df[f'{base_model}_pred'] == 0) & (results_df['label'] == 1)).astype(int)
    results_df['is_correct'] = (results_df[f'{base_model}_pred'] == results_df['label']).astype(int)
    
    # QPP特徴量を取得
    if qpp_features is None:
        exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer',
                       'BERT_proba', 'RoBERTa_proba', 'Transfer_proba', 'LogReg_pred',
                       'BERT_pred', 'RoBERTa_pred', 'Transfer_pred', 'threshold', 'group',
                       'score_percentile', 'percentile_bin', 'is_fp', 'is_fn', 'is_correct']
        qpp_features = [col for col in results_df.columns 
                       if col not in exclude_cols and results_df[col].dtype in [np.float64, np.int64]]
    
    # 各QPP特徴量について分析
    n_features = len(qpp_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(qpp_features):
        if feature not in results_df.columns:
            continue
        
        ax = axes[idx]
        
        # 特徴量の値を5つの区間に分割
        feature_values = results_df[feature].dropna()
        if len(feature_values) == 0:
            continue
        
        # パーセンタイルで区間を定義
        bins = np.percentile(feature_values, [0, 20, 40, 60, 80, 100])
        # 重複を避けるため、ラベルを生成
        labels = []
        for i in range(len(bins)-1):
            label = f'{bins[i]:.2f}-{bins[i+1]:.2f}'
            # 重複チェック
            if label not in labels:
                labels.append(label)
            else:
                # 重複している場合は、より詳細なラベルを使用
                labels.append(f'{bins[i]:.4f}-{bins[i+1]:.4f}')
        
        results_df['feature_bin'] = pd.cut(results_df[feature], bins=bins, 
                                          labels=labels, 
                                          include_lowest=True,
                                          duplicates='drop')
        
        # 各区間での誤分類率を計算
        bin_stats = []
        for bin_label in results_df['feature_bin'].cat.categories:
            bin_mask = results_df['feature_bin'] == bin_label
            bin_data = results_df[bin_mask]
            
            if len(bin_data) == 0:
                continue
            
            fp_rate = bin_data['is_fp'].mean()
            fn_rate = bin_data['is_fn'].mean()
            error_rate = (bin_data['is_fp'].sum() + bin_data['is_fn'].sum()) / len(bin_data)
            correct_rate = bin_data['is_correct'].mean()
            
            bin_stats.append({
                'bin': bin_label,
                'fp_rate': fp_rate,
                'fn_rate': fn_rate,
                'error_rate': error_rate,
                'correct_rate': correct_rate,
                'n_samples': len(bin_data)
            })
        
        bin_stats_df = pd.DataFrame(bin_stats)
        
        if len(bin_stats_df) == 0:
            continue
        
        # プロット
        x_pos = np.arange(len(bin_stats_df))
        width = 0.35
        
        ax.bar(x_pos - width/2, bin_stats_df['fp_rate'], width, label='FP Rate', alpha=0.7, color='red')
        ax.bar(x_pos + width/2, bin_stats_df['fn_rate'], width, label='FN Rate', alpha=0.7, color='blue')
        ax.plot(x_pos, bin_stats_df['error_rate'], 'o-', label='Error Rate', color='black', linewidth=2, markersize=8)
        
        ax.set_xlabel(f'{feature} Value Range', fontsize=10)
        ax.set_ylabel('Error Rate', fontsize=10)
        ax.set_title(f'{feature}: Error Rate by QPP Value Range', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_stats_df['bin'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(bin_stats_df[['fp_rate', 'fn_rate', 'error_rate']].max()) * 1.2)
    
    # 余分なサブプロットを非表示
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path = output_dir / f'qpp_threshold_analysis_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"QPP閾値分析を保存: {output_path}")


def plot_correction_effect(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    qpp_features: List[str] = None
):
    """
    QPPで補正した場合の改善率を可視化
    """
    results_df = results_df.copy()
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # 予測を計算
    median_threshold = results_df[f'{base_model}_proba'].median()
    results_df[f'{base_model}_pred'] = (results_df[f'{base_model}_proba'] >= median_threshold).astype(int)
    
    # ベースモデルの誤分類
    base_fp = ((results_df[f'{base_model}_pred'] == 1) & (results_df['label'] == 0)).sum()
    base_fn = ((results_df[f'{base_model}_pred'] == 0) & (results_df['label'] == 1)).sum()
    base_error = base_fp + base_fn
    base_accuracy = (results_df[f'{base_model}_pred'] == results_df['label']).mean()
    
    # QPP特徴量を取得
    if qpp_features is None:
        exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer',
                       'BERT_proba', 'RoBERTa_proba', 'Transfer_proba', 'LogReg_pred',
                       'BERT_pred', 'RoBERTa_pred', 'Transfer_pred', 'threshold', 'group',
                       'score_percentile', 'percentile_bin']
        qpp_features = [col for col in results_df.columns 
                       if col not in exclude_cols and results_df[col].dtype in [np.float64, np.int64]]
    
    # LogReg（QPP統合モデル）の性能
    if 'LogReg' in results_df.columns:
        results_df['LogReg_pred'] = (results_df['LogReg'] >= 0.5).astype(int)
        logreg_fp = ((results_df['LogReg_pred'] == 1) & (results_df['label'] == 0)).sum()
        logreg_fn = ((results_df['LogReg_pred'] == 0) & (results_df['label'] == 1)).sum()
        logreg_error = logreg_fp + logreg_fn
        logreg_accuracy = (results_df['LogReg_pred'] == results_df['label']).mean()
        
        # 改善率を計算
        fp_improvement = (base_fp - logreg_fp) / base_fp * 100 if base_fp > 0 else 0
        fn_improvement = (base_fn - logreg_fn) / base_fn * 100 if base_fn > 0 else 0
        error_improvement = (base_error - logreg_error) / base_error * 100 if base_error > 0 else 0
        accuracy_improvement = (logreg_accuracy - base_accuracy) * 100
        
        # 可視化
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 誤分類数の比較
        ax1 = axes[0]
        categories = ['FP', 'FN', 'Total Error']
        base_values = [base_fp, base_fn, base_error]
        logreg_values = [logreg_fp, logreg_fn, logreg_error]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, base_values, width, label=f'{base_model}', alpha=0.7, color='blue')
        bars2 = ax1.bar(x + width/2, logreg_values, width, label='LogReg (QPP Integrated)', alpha=0.7, color='red')
        
        # 改善率を表示
        for i, (base_val, logreg_val) in enumerate(zip(base_values, logreg_values)):
            if base_val > 0:
                improvement = (base_val - logreg_val) / base_val * 100
                ax1.text(i, max(base_val, logreg_val) * 1.1, f'{improvement:+.1f}%', 
                        ha='center', fontsize=10, fontweight='bold')
        
        ax1.set_ylabel('Number of Errors', fontsize=12)
        ax1.set_title('Error Reduction by QPP Integration', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 精度の比較
        ax2 = axes[1]
        models = [f'{base_model}', 'LogReg (QPP Integrated)']
        accuracies = [base_accuracy * 100, logreg_accuracy * 100]
        colors = ['blue', 'red']
        
        bars = ax2.bar(models, accuracies, alpha=0.7, color=colors)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Accuracy Improvement by QPP Integration', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 改善率を表示
        improvement_text = f'Improvement: {accuracy_improvement:+.2f}%'
        ax2.text(0.5, 0.95, improvement_text, transform=ax2.transAxes, 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # バーの上に値を表示
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        output_path = output_dir / f'correction_effect_{base_model.lower()}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"補正効果を保存: {output_path}")
    else:
        print("LogRegの予測結果が見つかりません。補正効果の可視化をスキップします。")


def generate_decision_rules(
    results_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    group_name: str = "FP",
    significance_threshold: float = 0.05,
    effect_size_threshold: float = 0.2
):
    """
    実用的な判定ルール（if-thenルール）を自動生成
    """
    prefix = group_name.lower()
    
    # 統計的に有意で効果量が大きい特徴を選択
    pvalue_col = f'{prefix}_vs_all_pvalue'
    cohens_d_col = f'{prefix}_vs_all_cohens_d'
    
    # まず統計的有意性でフィルタ
    significant_features = stats_df[
        stats_df[pvalue_col] < significance_threshold
    ].copy()
    
    # 効果量でソートして、上位の特徴を選択
    if len(significant_features) > 0:
        significant_features['abs_cohens_d'] = abs(significant_features[cohens_d_col])
        significant_features = significant_features.sort_values('abs_cohens_d', ascending=False)
        # 効果量が大きいもの、または効果量閾値以上のものを選択
        significant_features = significant_features[
            (significant_features['abs_cohens_d'] > effect_size_threshold) |
            (significant_features['abs_cohens_d'] > significant_features['abs_cohens_d'].quantile(0.5))
        ].copy()
    
    if len(significant_features) == 0:
        print(f"統計的に有意な特徴が見つかりませんでした（p < {significance_threshold}, |Cohen's d| > {effect_size_threshold}）。")
        print(f"閾値を緩めて再試行します...")
        # 閾値を緩める
        significant_features = stats_df[
            stats_df[pvalue_col] < 0.1  # p値を緩める
        ].copy()
        if len(significant_features) > 0:
            significant_features['abs_cohens_d'] = abs(significant_features[cohens_d_col])
            significant_features = significant_features.sort_values('abs_cohens_d', ascending=False)
            significant_features = significant_features.head(5)  # 上位5つを選択
    
    if len(significant_features) == 0:
        print(f"統計的に有意な特徴が見つかりませんでした。")
        return
    
    # データを準備
    results_df = results_df.copy()
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    median_threshold = results_df[f'{base_model}_proba'].median()
    results_df[f'{base_model}_pred'] = (results_df[f'{base_model}_proba'] >= median_threshold).astype(int)
    
    # 各特徴量の閾値を計算
    rules = []
    
    for _, row in significant_features.iterrows():
        feature = row['feature']
        cohens_d = row[cohens_d_col]
        pvalue = row[pvalue_col]
        
        if feature not in results_df.columns:
            continue
        
        # 特徴量の分布を分析
        if group_name == "FP":
            # FPの特徴量分布
            error_mask = ((results_df[f'{base_model}_pred'] == 1) & (results_df['label'] == 0))
            error_values = results_df[error_mask][feature].dropna()
            correct_mask = (results_df[f'{base_model}_pred'] == results_df['label'])
            correct_values = results_df[correct_mask][feature].dropna()
        else:  # FN
            error_mask = ((results_df[f'{base_model}_pred'] == 0) & (results_df['label'] == 1))
            error_values = results_df[error_mask][feature].dropna()
            correct_mask = (results_df[f'{base_model}_pred'] == results_df['label'])
            correct_values = results_df[correct_mask][feature].dropna()
        
        if len(error_values) == 0:
            continue
        
        error_median = error_values.median()
        correct_median = correct_values.median() if len(correct_values) > 0 else results_df[feature].median()
        
        # 方向を判定
        if cohens_d < 0:  # 誤分類グループの方が低い
            threshold = error_median
            direction = "低い"
            correction_direction = "answer" if group_name == "FP" else "clarification"
        else:  # 誤分類グループの方が高い
            threshold = error_median
            direction = "高い"
            correction_direction = "clarification" if group_name == "FP" else "answer"
        
        rules.append({
            'feature': feature,
            'threshold': threshold,
            'direction': direction,
            'correction_direction': correction_direction,
            'cohens_d': cohens_d,
            'pvalue': pvalue,
            'error_median': error_median,
            'correct_median': correct_median
        })
    
    # ルールをテキストファイルに保存
    rules_path = output_dir / f'{group_name.lower()}_decision_rules_{base_model.lower()}.txt'
    with open(rules_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"QPPを使うべき場面の判定ルール ({group_name}分析)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"ベースモデル: {base_model}\n")
        f.write(f"分析対象: {group_name}\n")
        f.write(f"有意水準: p < {significance_threshold}\n")
        f.write(f"効果量閾値: |Cohen's d| > {effect_size_threshold}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        for i, rule in enumerate(rules, 1):
            f.write(f"ルール {i}: {rule['feature']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  条件: {rule['feature']} が {rule['threshold']:.4f} より {rule['direction']}\n")
            f.write(f"  かつ {base_model}が {group_name.lower().replace('fp', 'clarification').replace('fn', 'answer')} と予測\n")
            f.write(f"  → QPPを使って {rule['correction_direction']} 側に補正すべき\n")
            f.write(f"\n  統計情報:\n")
            f.write(f"    - Cohen's d: {rule['cohens_d']:.4f}\n")
            f.write(f"    - p値: {rule['pvalue']:.6f}\n")
            f.write(f"    - 誤分類グループの中央値: {rule['error_median']:.4f}\n")
            f.write(f"    - 正解グループの中央値: {rule['correct_median']:.4f}\n")
            f.write("\n")
        
        # Pythonコード形式でも出力
        f.write("\n" + "=" * 80 + "\n")
        f.write("Pythonコード形式の判定関数\n")
        f.write("=" * 80 + "\n\n")
        f.write("def should_use_qpp_correction(qpp_features, bert_prediction):\n")
        f.write("    \"\"\"\n")
        f.write(f"    {base_model}の予測をQPPで補正すべきかどうかを判定\n")
        f.write("    \n")
        f.write("    Args:\n")
        f.write("        qpp_features: dict, QPP特徴量の辞書\n")
        f.write(f"        bert_prediction: str, {base_model}の予測 ('clarification' または 'answer')\n")
        f.write("    \n")
        f.write("    Returns:\n")
        f.write("        tuple: (should_correct: bool, correction_direction: str or None)\n")
        f.write("    \"\"\"\n")
        f.write("    \n")
        
        for i, rule in enumerate(rules, 1):
            feature = rule['feature']
            threshold = rule['threshold']
            direction = rule['direction']
            correction_direction = rule['correction_direction']
            
            if direction == "低い":
                condition = f"qpp_features.get('{feature}', float('inf')) < {threshold:.4f}"
            else:
                condition = f"qpp_features.get('{feature}', float('-inf')) > {threshold:.4f}"
            
            if group_name == "FP":
                bert_condition = "bert_prediction == 'clarification'"
            else:
                bert_condition = "bert_prediction == 'answer'"
            
            f.write(f"    # ルール {i}: {feature}\n")
            f.write(f"    if {condition} and {bert_condition}:\n")
            f.write(f"        return True, '{correction_direction}'\n")
            f.write("    \n")
        
        f.write("    return False, None  # QPP補正不要\n")
    
    print(f"判定ルールを保存: {rules_path}")
    
    # CSV形式でも保存
    rules_df = pd.DataFrame(rules)
    rules_csv_path = output_dir / f'{group_name.lower()}_decision_rules_{base_model.lower()}.csv'
    rules_df.to_csv(rules_csv_path, index=False)
    print(f"判定ルール（CSV）を保存: {rules_csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="閾値に依存しないFP/FN分析"
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
        "--method",
        type=str,
        default="roc",
        choices=["roc", "percentile"],
        help="分析方法（roc: ROC曲線上の点、percentile: パーセンタイルベース）"
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=20,
        help="ROC曲線上の分析点数（method=rocの場合）"
    )
    parser.add_argument(
        "--n-percentiles",
        type=int,
        default=10,
        help="パーセンタイル区間数（method=percentileの場合）"
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
    
    print(f"=== 閾値に依存しないFP/FN分析 ===")
    print(f"ベースモデル: {base_model}")
    print(f"分析方法: {args.method}")
    print()
    
    # データを読み込む
    results_df = pd.read_csv(results_csv_path)
    print(f"results.csvを読み込み: {len(results_df)} 行")
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # FP/FNを分類
    if args.method == "roc":
        print("ROC曲線上の各点でFP/FNを分類中...")
        classifications = classify_by_roc_points(results_df, base_model, args.n_points)
    else:
        print("パーセンタイルベースでFP/FNを分類中...")
        classifications = classify_by_percentiles(results_df, base_model, args.n_percentiles)
    
    # FPとFNを抽出
    fp_df = classifications[classifications['group'] == 'FP'].copy() if 'group' in classifications.columns and len(classifications) > 0 else pd.DataFrame()
    fn_df = classifications[classifications['group'] == 'FN'].copy() if 'group' in classifications.columns and len(classifications) > 0 else pd.DataFrame()
    
    # FP分析
    if len(fp_df) > 0:
        print(f"FP数: {len(fp_df)}")
        print()
        
        # FP固有の特徴を分析
        print("FP固有の特徴を分析中...")
        fp_stats_df = analyze_fp_specific_features(results_df, fp_df, output_dir, base_model, group_name="FP")
        
        # 可視化
        print("FP可視化を生成中...")
        plot_fp_vs_all_comparison(results_df, fp_df, fp_stats_df, output_dir, base_model, group_name="FP")
        plot_fp_specific_heatmap(fp_stats_df, output_dir, base_model, group_name="FP")
        
        # 判定ルール生成
        print("FP判定ルールを生成中...")
        generate_decision_rules(results_df, fp_stats_df, output_dir, base_model, group_name="FP")
    else:
        print("FPが見つかりませんでした。")
    
    # FN分析
    if len(fn_df) > 0:
        print(f"\nFN数: {len(fn_df)}")
        print()
        
        # FN固有の特徴を分析
        print("FN固有の特徴を分析中...")
        fn_stats_df = analyze_fp_specific_features(results_df, fn_df, output_dir, base_model, group_name="FN")
        
        # 可視化
        print("FN可視化を生成中...")
        plot_fp_vs_all_comparison(results_df, fn_df, fn_stats_df, output_dir, base_model, group_name="FN")
        plot_fp_specific_heatmap(fn_stats_df, output_dir, base_model, group_name="FN")
        
        # 判定ルール生成
        print("FN判定ルールを生成中...")
        generate_decision_rules(results_df, fn_stats_df, output_dir, base_model, group_name="FN")
    else:
        print("FNが見つかりませんでした。")
    
    # RoBERTaの弱い部分とQPP統合による改善を分析
    print("\n=== RoBERTaの弱い部分とQPP統合による改善を分析 ===")
    plot_roberta_weakness_and_qpp_improvement(results_df, output_dir, base_model)
    
    # QPP閾値分析（全体データを使用）
    print("\nQPP閾値分析を生成中...")
    plot_qpp_threshold_analysis(results_df, output_dir, base_model)
    
    # 補正効果の可視化
    print("補正効果を可視化中...")
    plot_correction_effect(results_df, output_dir, base_model)
    
    print()
    print("=== 分析完了 ===")
    print(f"出力ディレクトリ: {output_dir}")

if __name__ == "__main__":
    main()

