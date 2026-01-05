"""
高度なQPP証拠分析スクリプト
1. QPPスコアの低・中・高によるAUCの層別分析
2. スコアの「押し上げ量」の分布（Delta Distribution）
3. スコア密度プロット（Score Density Plot）
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# scipy.special.expit (sigmoid関数) を使用するため
try:
    from scipy.special import expit
except ImportError:
    def expit(x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def analyze_auc_by_qpp_strata(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    qpp_feature: str = 'WIG',
    n_strata: int = 3
):
    """
    QPPスコアの低・中・高によるAUCの層別分析
    
    Args:
        results_df: results.csvのDataFrame
        output_dir: 出力ディレクトリ
        base_model: ベースモデル名
        qpp_feature: 層別化に使用するQPP特徴量（デフォルト: WIG）
        n_strata: 層の数（デフォルト: 3）
    """
    if qpp_feature not in results_df.columns:
        print(f"警告: {qpp_feature}が見つかりません。利用可能な特徴量: {results_df.columns.tolist()}")
        return
    
    # ベースモデルのlogit値を確率に変換
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # QPP特徴量でソートして層別化
    results_df = results_df.sort_values(qpp_feature)
    n_per_stratum = len(results_df) // n_strata
    
    strata_results = []
    
    fig, axes = plt.subplots(1, n_strata, figsize=(6 * n_strata, 5))
    if n_strata == 1:
        axes = [axes]
    
    for stratum_idx in range(n_strata):
        start_idx = stratum_idx * n_per_stratum
        if stratum_idx == n_strata - 1:
            # 最後の層は残り全部を含める
            end_idx = len(results_df)
        else:
            end_idx = (stratum_idx + 1) * n_per_stratum
        
        stratum_df = results_df.iloc[start_idx:end_idx].copy()
        
        # 層の範囲を取得
        qpp_min = stratum_df[qpp_feature].min()
        qpp_max = stratum_df[qpp_feature].max()
        qpp_mean = stratum_df[qpp_feature].mean()
        
        # AUCを計算
        try:
            base_auc = roc_auc_score(stratum_df['label'], stratum_df[f'{base_model}_proba'])
        except ValueError:
            base_auc = np.nan
        
        try:
            logreg_auc = roc_auc_score(stratum_df['label'], stratum_df['LogReg'])
        except ValueError:
            logreg_auc = np.nan
        
        # L1, L2, ENetも計算
        aucs = {}
        for model in ['L1', 'L2', 'ENet']:
            if model in stratum_df.columns:
                try:
                    aucs[model] = roc_auc_score(stratum_df['label'], stratum_df[model])
                except ValueError:
                    aucs[model] = np.nan
        
        strata_results.append({
            'Stratum': f'{stratum_idx + 1}',
            'QPP_Range': f'{qpp_min:.3f} - {qpp_max:.3f}',
            'QPP_Mean': qpp_mean,
            'Count': len(stratum_df),
            f'{base_model}_AUC': base_auc,
            'LogReg_AUC': logreg_auc,
            **{f'{k}_AUC': v for k, v in aucs.items()}
        })
        
        # ROC曲線風の可視化（スコア分布）
        ax = axes[stratum_idx]
        
        # 正例と負例のスコア分布
        pos_scores_base = stratum_df[stratum_df['label'] == 1][f'{base_model}_proba']
        neg_scores_base = stratum_df[stratum_df['label'] == 0][f'{base_model}_proba']
        pos_scores_logreg = stratum_df[stratum_df['label'] == 1]['LogReg']
        neg_scores_logreg = stratum_df[stratum_df['label'] == 0]['LogReg']
        
        ax.hist(neg_scores_base, bins=20, alpha=0.5, label=f'{base_model} (negative)', 
               color='blue', density=True)
        ax.hist(pos_scores_base, bins=20, alpha=0.5, label=f'{base_model} (positive)', 
               color='red', density=True)
        ax.hist(neg_scores_logreg, bins=20, alpha=0.3, label='LogReg (negative)', 
               color='cyan', density=True, linestyle='--', edgecolor='cyan')
        ax.hist(pos_scores_logreg, bins=20, alpha=0.3, label='LogReg (positive)', 
               color='orange', density=True, linestyle='--', edgecolor='orange')
        
        ax.set_xlabel('Score (probability)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        # 改善量を計算
        improvement = logreg_auc - base_auc if not np.isnan(base_auc) and not np.isnan(logreg_auc) else np.nan
        improvement_str = f', Improvement: {improvement:+.3f}' if not np.isnan(improvement) else ''
        
        ax.set_title(f'Stratum {stratum_idx + 1}: {qpp_feature} [{qpp_min:.2f}, {qpp_max:.2f}]\n'
                    f'{base_model} AUC: {base_auc:.3f}, LogReg AUC: {logreg_auc:.3f}{improvement_str}', 
                    fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    output_path = output_dir / f'auc_by_{qpp_feature.lower()}_strata_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC層別分析を保存: {output_path}")
    
    # 結果をCSVに保存
    strata_df = pd.DataFrame(strata_results)
    strata_csv_path = output_dir / f'auc_by_{qpp_feature.lower()}_strata_{base_model.lower()}.csv'
    strata_df.to_csv(strata_csv_path, index=False)
    print(f"AUC層別分析結果を保存: {strata_csv_path}")
    
    # サマリーを表示
    print(f"\n=== {qpp_feature}によるAUC層別分析 ===")
    print(strata_df.to_string(index=False))
    
    # 改善量を計算
    if f'{base_model}_AUC' in strata_df.columns and 'LogReg_AUC' in strata_df.columns:
        strata_df['AUC_Improvement'] = strata_df['LogReg_AUC'] - strata_df[f'{base_model}_AUC']
        print(f"\nAUC改善量:")
        for _, row in strata_df.iterrows():
            improvement = row['AUC_Improvement']
            print(f"  Stratum {row['Stratum']}: {improvement:+.4f} ({(improvement/row[f'{base_model}_AUC']*100) if row[f'{base_model}_AUC'] > 0 else 0:.2f}%)")
    print()


def plot_delta_distribution(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    スコアの「押し上げ量」の分布（Delta Distribution）
    正例（Label=1）において、LogReg_score - base_model_confidence の分布
    """
    # ベースモデルのlogit値を確率に変換
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # 正例のみを抽出
    positive_df = results_df[results_df['label'] == 1].copy()
    
    # スコア差を計算
    positive_df['score_delta'] = positive_df['LogReg'] - positive_df[f'{base_model}_proba']
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ヒストグラム
    ax1 = axes[0]
    ax1.hist(positive_df['score_delta'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Delta = 0 (no change)')
    ax1.axvline(positive_df['score_delta'].mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {positive_df["score_delta"].mean():.4f}')
    ax1.axvline(positive_df['score_delta'].median(), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {positive_df["score_delta"].median():.4f}')
    
    ax1.set_xlabel(f'Score Delta (LogReg - {base_model})', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Score Delta Distribution (Positive Examples Only)\n'
                 f'n={len(positive_df)}, Mean={positive_df["score_delta"].mean():.4f}, '
                 f'Positive rate={((positive_df["score_delta"] > 0).sum() / len(positive_df) * 100):.1f}%',
                 fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 累積分布
    ax2 = axes[1]
    sorted_deltas = np.sort(positive_df['score_delta'])
    cumulative = np.arange(1, len(sorted_deltas) + 1) / len(sorted_deltas)
    ax2.plot(sorted_deltas, cumulative, linewidth=2, color='blue')
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Delta = 0')
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # 中央値の位置を表示
    median_delta = positive_df['score_delta'].median()
    median_cdf = (positive_df['score_delta'] <= median_delta).sum() / len(positive_df)
    ax2.plot(median_delta, median_cdf, 'ro', markersize=10, label=f'Median: {median_delta:.4f}')
    
    ax2.set_xlabel(f'Score Delta (LogReg - {base_model})', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Distribution of Score Delta', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / f'delta_distribution_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"スコア押し上げ量分布を保存: {output_path}")
    
    # 統計を表示
    print(f"\n=== スコア押し上げ量の統計（正例のみ） ===")
    print(f"総数: {len(positive_df)}")
    print(f"平均: {positive_df['score_delta'].mean():.4f}")
    print(f"中央値: {positive_df['score_delta'].median():.4f}")
    print(f"標準偏差: {positive_df['score_delta'].std():.4f}")
    print(f"最小値: {positive_df['score_delta'].min():.4f}")
    print(f"最大値: {positive_df['score_delta'].max():.4f}")
    print(f"正の値の割合: {(positive_df['score_delta'] > 0).sum() / len(positive_df) * 100:.1f}%")
    print(f"0.1以上上昇した割合: {(positive_df['score_delta'] >= 0.1).sum() / len(positive_df) * 100:.1f}%")
    print()


def plot_score_density(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    スコア密度プロット（Score Density Plot）
    正例と負例のスコア分布が、統合によってどう「分離」されたかを確認
    """
    # ベースモデルのlogit値を確率に変換
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ベースモデルのスコア分布
    ax1 = axes[0]
    pos_scores_base = results_df[results_df['label'] == 1][f'{base_model}_proba']
    neg_scores_base = results_df[results_df['label'] == 0][f'{base_model}_proba']
    
    ax1.hist(neg_scores_base, bins=50, alpha=0.6, label='Negative (label=0)', 
            color='blue', density=True, edgecolor='black')
    ax1.hist(pos_scores_base, bins=50, alpha=0.6, label='Positive (label=1)', 
            color='red', density=True, edgecolor='black')
    
    # 平均値を表示
    ax1.axvline(neg_scores_base.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Negative mean: {neg_scores_base.mean():.3f}')
    ax1.axvline(pos_scores_base.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Positive mean: {pos_scores_base.mean():.3f}')
    
    # AUCを計算
    try:
        base_auc = roc_auc_score(results_df['label'], results_df[f'{base_model}_proba'])
    except ValueError:
        base_auc = np.nan
    
    ax1.set_xlabel(f'{base_model} Score (probability)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title(f'{base_model} Score Distribution\nAUC: {base_auc:.4f}', 
                 fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    # 統合モデル（LogReg）のスコア分布
    ax2 = axes[1]
    pos_scores_logreg = results_df[results_df['label'] == 1]['LogReg']
    neg_scores_logreg = results_df[results_df['label'] == 0]['LogReg']
    
    ax2.hist(neg_scores_logreg, bins=50, alpha=0.6, label='Negative (label=0)', 
            color='blue', density=True, edgecolor='black')
    ax2.hist(pos_scores_logreg, bins=50, alpha=0.6, label='Positive (label=1)', 
            color='red', density=True, edgecolor='black')
    
    # 平均値を表示
    ax2.axvline(neg_scores_logreg.mean(), color='blue', linestyle='--', linewidth=2,
               label=f'Negative mean: {neg_scores_logreg.mean():.3f}')
    ax2.axvline(pos_scores_logreg.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Positive mean: {pos_scores_logreg.mean():.3f}')
    
    # AUCを計算
    try:
        logreg_auc = roc_auc_score(results_df['label'], results_df['LogReg'])
    except ValueError:
        logreg_auc = np.nan
    
    ax2.set_xlabel('LogReg Score (probability)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title(f'LogReg (Integrated) Score Distribution\nAUC: {logreg_auc:.4f}', 
                 fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    output_path = output_dir / f'score_density_comparison_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"スコア密度プロットを保存: {output_path}")
    
    # 分離度を計算（平均値の差 / 標準偏差の平均）
    base_separation = abs(pos_scores_base.mean() - neg_scores_base.mean()) / \
                     ((pos_scores_base.std() + neg_scores_base.std()) / 2)
    logreg_separation = abs(pos_scores_logreg.mean() - neg_scores_logreg.mean()) / \
                       ((pos_scores_logreg.std() + neg_scores_logreg.std()) / 2)
    
    print(f"\n=== スコア分布の分離度 ===")
    print(f"{base_model} 分離度: {base_separation:.4f}")
    print(f"LogReg 分離度: {logreg_separation:.4f}")
    print(f"改善率: {(logreg_separation / base_separation - 1) * 100:.2f}%")
    print()


def plot_auc_improvement_by_strata(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str,
    qpp_features: List[str] = None
):
    """
    複数のQPP特徴量について、層別AUC改善をまとめて可視化
    """
    if qpp_features is None:
        qpp_features = ['WIG', 'NQC', 'SMV', 'Clarity', 'nSigma']
    
    # 利用可能な特徴量のみを使用
    available_features = [f for f in qpp_features if f in results_df.columns]
    
    if len(available_features) == 0:
        print("警告: 利用可能なQPP特徴量が見つかりません")
        return
    
    # ベースモデルのlogit値を確率に変換
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    # 各特徴量について層別AUCを計算
    all_strata_results = []
    
    for qpp_feature in available_features:
        # 3等分
        results_df_sorted = results_df.sort_values(qpp_feature)
        n_per_stratum = len(results_df_sorted) // 3
        
        for stratum_idx in range(3):
            start_idx = stratum_idx * n_per_stratum
            if stratum_idx == 2:
                end_idx = len(results_df_sorted)
            else:
                end_idx = (stratum_idx + 1) * n_per_stratum
            
            stratum_df = results_df_sorted.iloc[start_idx:end_idx]
            
            try:
                base_auc = roc_auc_score(stratum_df['label'], stratum_df[f'{base_model}_proba'])
            except ValueError:
                base_auc = np.nan
            
            try:
                logreg_auc = roc_auc_score(stratum_df['label'], stratum_df['LogReg'])
            except ValueError:
                logreg_auc = np.nan
            
            auc_improvement = logreg_auc - base_auc if not np.isnan(base_auc) and not np.isnan(logreg_auc) else np.nan
            
            all_strata_results.append({
                'QPP_Feature': qpp_feature,
                'Stratum': ['Low', 'Mid', 'High'][stratum_idx],
                'QPP_Mean': stratum_df[qpp_feature].mean(),
                f'{base_model}_AUC': base_auc,
                'LogReg_AUC': logreg_auc,
                'AUC_Improvement': auc_improvement
            })
    
    strata_df = pd.DataFrame(all_strata_results)
    
    # 可視化
    n_features = len(available_features)
    fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 5))
    if n_features == 1:
        axes = [axes]
    
    for idx, qpp_feature in enumerate(available_features):
        ax = axes[idx]
        feature_data = strata_df[strata_df['QPP_Feature'] == qpp_feature]
        
        x = np.arange(3)
        width = 0.35
        
        base_aucs = [feature_data[feature_data['Stratum'] == s][f'{base_model}_AUC'].values[0] 
                    if len(feature_data[feature_data['Stratum'] == s]) > 0 else np.nan
                    for s in ['Low', 'Mid', 'High']]
        logreg_aucs = [feature_data[feature_data['Stratum'] == s]['LogReg_AUC'].values[0] 
                      if len(feature_data[feature_data['Stratum'] == s]) > 0 else np.nan
                      for s in ['Low', 'Mid', 'High']]
        
        bars1 = ax.bar(x - width/2, base_aucs, width, label=base_model, alpha=0.7, color='blue')
        bars2 = ax.bar(x + width/2, logreg_aucs, width, label='LogReg (Integrated)', alpha=0.7, color='red')
        
        # 改善量をテキストで表示
        for i, (base_auc, logreg_auc) in enumerate(zip(base_aucs, logreg_aucs)):
            if not np.isnan(base_auc) and not np.isnan(logreg_auc):
                improvement = logreg_auc - base_auc
                ax.text(i, max(base_auc, logreg_auc) + 0.01, f'+{improvement:.3f}', 
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('QPP Stratum', fontsize=10)
        ax.set_ylabel('AUC', fontsize=10)
        ax.set_title(f'{qpp_feature}', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Low', 'Mid', 'High'])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = output_dir / f'auc_improvement_by_strata_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC改善の層別比較を保存: {output_path}")
    
    # CSVに保存
    strata_csv_path = output_dir / f'auc_improvement_by_strata_{base_model.lower()}.csv'
    strata_df.to_csv(strata_csv_path, index=False)
    print(f"AUC改善の層別比較結果を保存: {strata_csv_path}")
    print()


def plot_roc_precision_comparison(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    ROC曲線とPrecision-Recall曲線の比較
    特にFPRが低い領域（高精度領域）での改善を強調
    """
    # ベースモデルのlogit値を確率に変換
    results_df[f'{base_model}_proba'] = expit(results_df[base_model])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC曲線
    ax1 = axes[0]
    fpr_base, tpr_base, _ = roc_curve(results_df['label'], results_df[f'{base_model}_proba'])
    fpr_logreg, tpr_logreg, _ = roc_curve(results_df['label'], results_df['LogReg'])
    
    ax1.plot(fpr_base, tpr_base, linewidth=2, label=f'{base_model} (AUC={roc_auc_score(results_df["label"], results_df[f"{base_model}_proba"]):.4f})', color='blue')
    ax1.plot(fpr_logreg, tpr_logreg, linewidth=2, label=f'LogReg (AUC={roc_auc_score(results_df["label"], results_df["LogReg"]):.4f})', color='red')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    # FPRが低い領域（0.0-0.2）を強調
    low_fpr_mask_base = fpr_base <= 0.2
    low_fpr_mask_logreg = fpr_logreg <= 0.2
    if low_fpr_mask_base.any():
        ax1.plot(fpr_base[low_fpr_mask_base], tpr_base[low_fpr_mask_base], 
                linewidth=3, color='blue', alpha=0.7, label='High Precision Region (FPR<0.2)')
    if low_fpr_mask_logreg.any():
        ax1.plot(fpr_logreg[low_fpr_mask_logreg], tpr_logreg[low_fpr_mask_logreg], 
                linewidth=3, color='red', alpha=0.7)
    
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Precision-Recall曲線
    ax2 = axes[1]
    precision_base, recall_base, _ = precision_recall_curve(results_df['label'], results_df[f'{base_model}_proba'])
    precision_logreg, recall_logreg, _ = precision_recall_curve(results_df['label'], results_df['LogReg'])
    
    from sklearn.metrics import average_precision_score
    ap_base = average_precision_score(results_df['label'], results_df[f'{base_model}_proba'])
    ap_logreg = average_precision_score(results_df['label'], results_df['LogReg'])
    
    ax2.plot(recall_base, precision_base, linewidth=2, 
            label=f'{base_model} (AP={ap_base:.4f})', color='blue')
    ax2.plot(recall_logreg, precision_logreg, linewidth=2, 
            label=f'LogReg (AP={ap_logreg:.4f})', color='red')
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = output_dir / f'roc_pr_comparison_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC/PR曲線比較を保存: {output_path}")
    
    # 高精度領域での改善を計算
    # FPR < 0.1でのTPRを比較
    fpr_01_idx_base = np.where(fpr_base <= 0.1)[0]
    fpr_01_idx_logreg = np.where(fpr_logreg <= 0.1)[0]
    
    if len(fpr_01_idx_base) > 0 and len(fpr_01_idx_logreg) > 0:
        tpr_at_fpr01_base = tpr_base[fpr_01_idx_base[-1]] if len(fpr_01_idx_base) > 0 else 0
        tpr_at_fpr01_logreg = tpr_logreg[fpr_01_idx_logreg[-1]] if len(fpr_01_idx_logreg) > 0 else 0
        improvement_fpr01 = tpr_at_fpr01_logreg - tpr_at_fpr01_base
        
        print(f"\n=== 高精度領域（FPR < 0.1）での改善 ===")
        print(f"{base_model} TPR at FPR=0.1: {tpr_at_fpr01_base:.4f}")
        print(f"LogReg TPR at FPR=0.1: {tpr_at_fpr01_logreg:.4f}")
        print(f"改善量: {improvement_fpr01:+.4f} ({(improvement_fpr01/tpr_at_fpr01_base*100) if tpr_at_fpr01_base > 0 else 0:.2f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="高度なQPP証拠分析"
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
        "--qpp-feature",
        type=str,
        default='WIG',
        help="層別分析に使用するQPP特徴量（デフォルト: WIG）"
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
    
    print(f"=== 高度なQPP証拠分析 ===")
    print(f"ベースモデル: {base_model}")
    print()
    
    # データを読み込む
    results_df = pd.read_csv(results_csv_path)
    print(f"results.csvを読み込み: {len(results_df)} 行")
    
    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. QPPスコアの低・中・高によるAUCの層別分析
    print("1. QPPスコアによるAUC層別分析を生成中...")
    analyze_auc_by_qpp_strata(results_df, output_dir, base_model, 
                              qpp_feature=args.qpp_feature, n_strata=3)
    
    # 複数のQPP特徴量について層別AUC改善を可視化
    print("1.5. 複数QPP特徴量によるAUC改善の層別比較を生成中...")
    plot_auc_improvement_by_strata(results_df, output_dir, base_model)
    
    # 2. スコアの「押し上げ量」の分布
    print("2. スコア押し上げ量の分布を生成中...")
    plot_delta_distribution(results_df, output_dir, base_model)
    
    # 3. スコア密度プロット
    print("3. スコア密度プロットを生成中...")
    plot_score_density(results_df, output_dir, base_model)
    
    # 4. ROC/PR曲線の比較（高精度領域での改善を強調）
    print("4. ROC/PR曲線の比較を生成中...")
    plot_roc_precision_comparison(results_df, output_dir, base_model)
    
    print()
    print("=== 分析完了 ===")
    print(f"出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()

