"""
一般的な語の多さとRoBERTaの性能の関係を可視化して証拠を提示するスクリプト

BERTが一般的な語を検知できているか、それでもなぜRoBERTaが弱いのかを可視化
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional
import warnings
from scipy import stats
from scipy.stats import pearsonr, spearmanr
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


def plot_roberta_vs_common_words(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    一般的な語の多さ（AvgIDF/AvgICTFが低い）とRoBERTaのlogit値の関係を可視化
    BERTが一般的な語を検知できているかを確認
    """
    # 一般的な語の指標（低い値 = 一般的な語が多い）
    common_word_features = ['AvgIDF', 'AvgICTF', 'MaxIDF']
    
    # RoBERTaのlogit値を取得
    roberta_logit = results_df[base_model].values
    roberta_proba = expit(roberta_logit)
    
    # 予測と誤分類を計算
    median_threshold = np.median(roberta_proba)
    roberta_pred = (roberta_proba >= median_threshold).astype(int)
    roberta_error = (roberta_pred != results_df['label'].values).astype(int)
    
    # 予測確信度（確率が0.5からどれだけ離れているか）
    roberta_confidence = np.abs(roberta_proba - 0.5)
    
    # 利用可能な特徴量を確認
    available_features = [f for f in common_word_features if f in results_df.columns]
    if len(available_features) == 0:
        print("一般的な語の指標が見つかりません")
        return
    
    # 3行×3列のグリッド（各特徴量に対して3つのサブプロット）
    fig, axes = plt.subplots(len(available_features), 3, figsize=(18, 6 * len(available_features)))
    if len(available_features) == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, feature in enumerate(available_features):
        if feature not in results_df.columns:
            continue
        
        feature_values = results_df[feature].values
        valid_mask = ~np.isnan(feature_values) & ~np.isnan(roberta_logit)
        
        if not valid_mask.any():
            continue
        
        feature_vals = feature_values[valid_mask]
        logit_vals = roberta_logit[valid_mask]
        error_vals = roberta_error[valid_mask]
        confidence_vals = roberta_confidence[valid_mask]
        
        # 1. RoBERTa logit vs 一般的な語の多さ（散布図）
        ax = axes[idx * 3]
        scatter = ax.scatter(feature_vals, logit_vals, c=error_vals, 
                           cmap='RdYlGn', alpha=0.5, s=20, edgecolors='none')
        ax.set_xlabel(f'{feature} (lower = more common words)', fontsize=11)
        ax.set_ylabel(f'{base_model} Logit', fontsize=11)
        if USE_JAPANESE:
            ax.set_title(f'{base_model} Logit vs {feature}\n(Red=Error, Green=Correct)', fontsize=12)
        else:
            ax.set_title(f'{base_model} Logit vs {feature}\n(Red=Error, Green=Correct)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 相関を計算
        corr_pearson, p_pearson = pearsonr(feature_vals, logit_vals)
        corr_spearman, p_spearman = spearmanr(feature_vals, logit_vals)
        ax.text(0.05, 0.95, f'Pearson r={corr_pearson:.3f} (p={p_pearson:.3f})\nSpearman ρ={corr_spearman:.3f} (p={p_spearman:.3f})',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
        
        # 2. 誤分類率 vs 一般的な語の多さ（ビンごと）
        ax = axes[idx * 3 + 1]
        # 5つのビンに分割
        bins = np.percentile(feature_vals, [0, 20, 40, 60, 80, 100])
        bin_labels = []
        bin_errors = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                mask = (feature_vals >= bins[i]) & (feature_vals <= bins[i+1])
            else:
                mask = (feature_vals >= bins[i]) & (feature_vals < bins[i+1])
            
            if mask.sum() > 0:
                bin_errors.append(error_vals[mask].mean())
                bin_counts.append(mask.sum())
                bin_labels.append(f'{bins[i]:.2f}\n-\n{bins[i+1]:.2f}')
        
        x_pos = np.arange(len(bin_labels))
        bars = ax.bar(x_pos, bin_errors, alpha=0.7, color='coral')
        ax.set_xlabel(f'{feature} Range (lower = more common words)', fontsize=11)
        ax.set_ylabel('Error Rate', fontsize=11)
        if USE_JAPANESE:
            ax.set_title(f'{base_model} Error Rate by {feature} Range', fontsize=12)
        else:
            ax.set_title(f'{base_model} Error Rate by {feature} Range', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 各バーの上にサンプル数を表示
        for i, (bar, count) in enumerate(zip(bars, bin_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'n={count}', ha='center', va='bottom', fontsize=8)
        
        # 全体の誤分類率の基準線
        overall_error = error_vals.mean()
        ax.axhline(overall_error, color='red', linestyle='--', linewidth=2, 
                  label=f'Overall Error Rate ({overall_error:.3f})')
        ax.legend(fontsize=9)
        
        # 3. 予測確信度 vs 一般的な語の多さ
        ax = axes[idx * 3 + 2]
        # ビンごとの平均確信度
        bin_confidences = []
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                mask = (feature_vals >= bins[i]) & (feature_vals <= bins[i+1])
            else:
                mask = (feature_vals >= bins[i]) & (feature_vals < bins[i+1])
            
            if mask.sum() > 0:
                bin_confidences.append(confidence_vals[mask].mean())
        
        ax.plot(x_pos, bin_confidences, 'o-', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel(f'{feature} Range (lower = more common words)', fontsize=11)
        ax.set_ylabel('Average Confidence\n(|probability - 0.5|)', fontsize=11)
        if USE_JAPANESE:
            ax.set_title(f'{base_model} Prediction Confidence by {feature} Range', fontsize=12)
        else:
            ax.set_title(f'{base_model} Prediction Confidence by {feature} Range', fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # 全体の平均確信度の基準線
        overall_confidence = confidence_vals.mean()
        ax.axhline(overall_confidence, color='red', linestyle='--', linewidth=2,
                  label=f'Overall Confidence ({overall_confidence:.3f})')
        ax.legend(fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f'roberta_vs_common_words_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_common_words_performance_comparison(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    一般的な語が多いケース vs 少ないケースでのRoBERTaの性能を比較
    """
    # 一般的な語の指標を組み合わせて「一般的な語の多さ」スコアを作成
    common_word_features = ['AvgIDF', 'AvgICTF', 'MaxIDF']
    available_features = [f for f in common_word_features if f in results_df.columns]
    
    if len(available_features) == 0:
        print("一般的な語の指標が見つかりません")
        return
    
    # 各特徴量を標準化して平均を取る（低い値 = 一般的な語が多い）
    common_word_scores = []
    for feature in available_features:
        values = results_df[feature].values
        # NaNを中央値で埋める
        median_val = np.nanmedian(values)
        values = np.where(np.isnan(values), median_val, values)
        # 標準化（低い値が一般的な語が多いことを示す）
        normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
        common_word_scores.append(normalized)
    
    # 平均を取る（低い値 = 一般的な語が多い）
    combined_score = np.mean(common_word_scores, axis=0)
    
    # 下位25%を「一般的な語が多い」、上位25%を「一般的な語が少ない」とする
    low_threshold = np.percentile(combined_score, 25)
    high_threshold = np.percentile(combined_score, 75)
    
    low_common_mask = combined_score <= low_threshold  # 一般的な語が多い
    high_common_mask = combined_score >= high_threshold  # 一般的な語が少ない
    
    # RoBERTaの性能を計算
    roberta_proba = expit(results_df[base_model].values)
    median_threshold = np.median(roberta_proba)
    roberta_pred = (roberta_proba >= median_threshold).astype(int)
    roberta_accuracy = (roberta_pred == results_df['label'].values).astype(int)
    roberta_error = 1 - roberta_accuracy
    
    # LogRegの性能も計算
    if 'LogReg' in results_df.columns:
        logreg_proba = expit(results_df['LogReg'].values)
        logreg_pred = (logreg_proba >= 0.5).astype(int)
        logreg_accuracy = (logreg_pred == results_df['label'].values).astype(int)
        logreg_error = 1 - logreg_accuracy
        logreg_improvement = logreg_accuracy - roberta_accuracy
    else:
        logreg_error = None
        logreg_improvement = None
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. 誤分類率の比較
    ax = axes[0, 0]
    categories = ['Many Common\nWords\n(Low Info)', 'Few Common\nWords\n(High Info)']
    roberta_errors = [
        roberta_error[low_common_mask].mean(),
        roberta_error[high_common_mask].mean()
    ]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x_pos - width/2, roberta_errors, width, 
                  label=f'{base_model} Error Rate', alpha=0.7, color='coral')
    
    if logreg_error is not None:
        logreg_errors = [
            logreg_error[low_common_mask].mean(),
            logreg_error[high_common_mask].mean()
        ]
        bars2 = ax.bar(x_pos + width/2, logreg_errors, width,
                      label='LogReg Error Rate', alpha=0.7, color='steelblue')
    
    ax.set_ylabel('Error Rate', fontsize=12)
    if USE_JAPANESE:
        ax.set_title(f'Error Rate: Many Common Words vs Few Common Words', fontsize=13)
    else:
        ax.set_title(f'Error Rate: Many Common Words vs Few Common Words', fontsize=13)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # バーの上に値を表示
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    if logreg_error is not None:
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 統計的検定
    from scipy.stats import mannwhitneyu
    stat_low, p_low = mannwhitneyu(roberta_error[low_common_mask], 
                                   roberta_error[high_common_mask], 
                                   alternative='two-sided')
    ax.text(0.5, 0.95, f'Mann-Whitney U test:\np={p_low:.4f}',
            transform=ax.transAxes, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # 2. 予測確信度の比較
    ax = axes[0, 1]
    roberta_confidence = np.abs(roberta_proba - 0.5)
    confidences = [
        roberta_confidence[low_common_mask].mean(),
        roberta_confidence[high_common_mask].mean()
    ]
    
    bars = ax.bar(categories, confidences, alpha=0.7, color='steelblue')
    ax.set_ylabel('Average Confidence\n(|probability - 0.5|)', fontsize=12)
    if USE_JAPANESE:
        ax.set_title(f'{base_model} Prediction Confidence Comparison', fontsize=13)
    else:
        ax.set_title(f'{base_model} Prediction Confidence Comparison', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    # 統計的検定
    stat_conf, p_conf = mannwhitneyu(roberta_confidence[low_common_mask],
                                    roberta_confidence[high_common_mask],
                                    alternative='two-sided')
    ax.text(0.5, 0.95, f'Mann-Whitney U test:\np={p_conf:.4f}',
            transform=ax.transAxes, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # 3. QPP統合による改善の比較
    if logreg_improvement is not None:
        ax = axes[1, 0]
        improvements = [
            logreg_improvement[low_common_mask].mean(),
            logreg_improvement[high_common_mask].mean()
        ]
        
        bars = ax.bar(categories, improvements, alpha=0.7, color='green')
        ax.set_ylabel('Accuracy Improvement\n(LogReg - RoBERTa)', fontsize=12)
        if USE_JAPANESE:
            ax.set_title('QPP Integration Improvement by Common Words Level', fontsize=13)
        else:
            ax.set_title('QPP Integration Improvement by Common Words Level', fontsize=13)
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., 
                   height + (0.01 if height >= 0 else -0.02),
                   f'{height:+.3f}', ha='center', 
                   va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # 統計的検定
        stat_imp, p_imp = mannwhitneyu(logreg_improvement[low_common_mask],
                                      logreg_improvement[high_common_mask],
                                      alternative='two-sided')
        ax.text(0.5, 0.95, f'Mann-Whitney U test:\np={p_imp:.4f}',
                transform=ax.transAxes, verticalalignment='top', ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # 4. サンプル数の比較
    ax = axes[1, 1]
    counts = [
        low_common_mask.sum(),
        high_common_mask.sum()
    ]
    
    bars = ax.bar(categories, counts, alpha=0.7, color='purple')
    ax.set_ylabel('Number of Samples', fontsize=12)
    if USE_JAPANESE:
        ax.set_title('Sample Size by Common Words Level', fontsize=13)
    else:
        ax.set_title('Sample Size by Common Words Level', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(counts) * 0.01,
               f'n={int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    output_path = output_dir / f'common_words_performance_comparison_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_roberta_logit_distribution_by_common_words(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    一般的な語の多さによるRoBERTa logit値の分布の違いを可視化
    BERTが一般的な語を検知できているかを確認
    """
    # 一般的な語の指標
    common_word_features = ['AvgIDF', 'AvgICTF', 'MaxIDF']
    available_features = [f for f in common_word_features if f in results_df.columns]
    
    if len(available_features) == 0:
        return
    
    # 各特徴量を標準化して平均を取る
    common_word_scores = []
    for feature in available_features:
        values = results_df[feature].values
        median_val = np.nanmedian(values)
        values = np.where(np.isnan(values), median_val, values)
        normalized = (values - np.mean(values)) / (np.std(values) + 1e-8)
        common_word_scores.append(normalized)
    
    combined_score = np.mean(common_word_scores, axis=0)
    
    # 3つのグループに分割
    low_threshold = np.percentile(combined_score, 33.3)
    high_threshold = np.percentile(combined_score, 66.7)
    
    low_common_mask = combined_score <= low_threshold
    mid_common_mask = (combined_score > low_threshold) & (combined_score < high_threshold)
    high_common_mask = combined_score >= high_threshold
    
    roberta_logit = results_df[base_model].values
    roberta_proba = expit(roberta_logit)
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Logit値の分布（ヒストグラム）
    ax = axes[0, 0]
    ax.hist(roberta_logit[low_common_mask], bins=50, alpha=0.5, 
           label='Many Common Words', color='red', density=True)
    ax.hist(roberta_logit[mid_common_mask], bins=50, alpha=0.5,
           label='Medium Common Words', color='orange', density=True)
    ax.hist(roberta_logit[high_common_mask], bins=50, alpha=0.5,
           label='Few Common Words', color='green', density=True)
    ax.set_xlabel(f'{base_model} Logit', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    if USE_JAPANESE:
        ax.set_title(f'{base_model} Logit Distribution by Common Words Level', fontsize=13)
    else:
        ax.set_title(f'{base_model} Logit Distribution by Common Words Level', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 平均値を表示
    ax.axvline(roberta_logit[low_common_mask].mean(), color='red', 
              linestyle='--', linewidth=2, label=f'Mean (Many): {roberta_logit[low_common_mask].mean():.3f}')
    ax.axvline(roberta_logit[high_common_mask].mean(), color='green',
              linestyle='--', linewidth=2, label=f'Mean (Few): {roberta_logit[high_common_mask].mean():.3f}')
    
    # 2. 確率の分布（ヒストグラム）
    ax = axes[0, 1]
    ax.hist(roberta_proba[low_common_mask], bins=50, alpha=0.5,
           label='Many Common Words', color='red', density=True)
    ax.hist(roberta_proba[mid_common_mask], bins=50, alpha=0.5,
           label='Medium Common Words', color='orange', density=True)
    ax.hist(roberta_proba[high_common_mask], bins=50, alpha=0.5,
           label='Few Common Words', color='green', density=True)
    ax.set_xlabel(f'{base_model} Probability', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    if USE_JAPANESE:
        ax.set_title(f'{base_model} Probability Distribution by Common Words Level', fontsize=13)
    else:
        ax.set_title(f'{base_model} Probability Distribution by Common Words Level', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.5, color='black', linestyle=':', linewidth=1)
    
    # 3. ボックスプロット（Logit）
    ax = axes[1, 0]
    data_to_plot = [
        roberta_logit[low_common_mask],
        roberta_logit[mid_common_mask],
        roberta_logit[high_common_mask]
    ]
    bp = ax.boxplot(data_to_plot, labels=['Many\nCommon\nWords', 'Medium\nCommon\nWords', 'Few\nCommon\nWords'],
                   patch_artist=True)
    for patch, color in zip(bp['boxes'], ['red', 'orange', 'green']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(f'{base_model} Logit', fontsize=12)
    if USE_JAPANESE:
        ax.set_title(f'{base_model} Logit by Common Words Level (Boxplot)', fontsize=13)
    else:
        ax.set_title(f'{base_model} Logit by Common Words Level (Boxplot)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 統計的検定
    from scipy.stats import kruskal
    stat, p = kruskal(roberta_logit[low_common_mask],
                     roberta_logit[mid_common_mask],
                     roberta_logit[high_common_mask])
    ax.text(0.5, 0.95, f'Kruskal-Wallis test:\np={p:.4f}',
            transform=ax.transAxes, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    # 4. ボックスプロット（確率）
    ax = axes[1, 1]
    data_to_plot = [
        roberta_proba[low_common_mask],
        roberta_proba[mid_common_mask],
        roberta_proba[high_common_mask]
    ]
    bp = ax.boxplot(data_to_plot, labels=['Many\nCommon\nWords', 'Medium\nCommon\nWords', 'Few\nCommon\nWords'],
                   patch_artist=True)
    for patch, color in zip(bp['boxes'], ['red', 'orange', 'green']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel(f'{base_model} Probability', fontsize=12)
    if USE_JAPANESE:
        ax.set_title(f'{base_model} Probability by Common Words Level (Boxplot)', fontsize=13)
    else:
        ax.set_title(f'{base_model} Probability by Common Words Level (Boxplot)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0.5, color='black', linestyle=':', linewidth=1)
    
    stat, p = kruskal(roberta_proba[low_common_mask],
                     roberta_proba[mid_common_mask],
                     roberta_proba[high_common_mask])
    ax.text(0.5, 0.95, f'Kruskal-Wallis test:\np={p:.4f}',
            transform=ax.transAxes, verticalalignment='top', ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / f'roberta_logit_distribution_by_common_words_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='一般的な語の多さとRoBERTaの性能の関係を可視化')
    parser.add_argument('--results-csv', type=str, required=True,
                       help='Path to results.csv file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--base-model', type=str, default='RoBERTa',
                       help='Base model name (default: RoBERTa)')
    
    args = parser.parse_args()
    
    results_csv_path = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {results_csv_path}")
    results_df = pd.read_csv(results_csv_path)
    print(f"Loaded {len(results_df)} samples")
    
    print("\n=== 一般的な語の多さとRoBERTaの性能の関係を可視化 ===")
    
    # 1. RoBERTa logit vs 一般的な語の多さ
    print("\n1. Plotting RoBERTa logit vs common words...")
    plot_roberta_vs_common_words(results_df, output_dir, args.base_model)
    
    # 2. 一般的な語が多い vs 少ないケースでの性能比較
    print("\n2. Plotting performance comparison...")
    plot_common_words_performance_comparison(results_df, output_dir, args.base_model)
    
    # 3. Logit分布の違い
    print("\n3. Plotting logit distribution by common words...")
    plot_roberta_logit_distribution_by_common_words(results_df, output_dir, args.base_model)
    
    print("\n=== 可視化完了 ===")


if __name__ == '__main__':
    main()

