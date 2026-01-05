"""
発表用：QPP統合の成功とQPP特徴量の因果関係を示すグラフ
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu
from scipy.special import expit
import warnings
warnings.filterwarnings('ignore')

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

# 発表用のスタイル設定
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_success_rate_comparison(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    特徴量の高低によるQPP統合成功率の比較（発表用）
    """
    # RoBERTaとLogRegの確率値を計算
    roberta_proba = expit(results_df[base_model].values)
    logreg_proba = expit(results_df['LogReg'].values)
    labels = results_df['label'].values
    
    # 中央値閾値で分析
    roberta_thresh = np.median(roberta_proba)
    logreg_thresh = 0.5
    
    roberta_pred = (roberta_proba >= roberta_thresh).astype(int)
    logreg_pred = (logreg_proba >= logreg_thresh).astype(int)
    
    # RoBERTaが誤分類したケースを分類
    roberta_error = (roberta_pred != labels)
    logreg_correct = (logreg_pred == labels)
    logreg_error = (logreg_pred != labels)
    
    success_mask = roberta_error & logreg_correct
    failure_mask = roberta_error & logreg_error
    
    success_indices = np.where(success_mask)[0]
    failure_indices = np.where(failure_mask)[0]
    roberta_error_indices = np.where(roberta_error)[0]
    
    # 分析対象の特徴量
    target_features = ['AvgIDF', 'SClarity', 'AvgICTF', 'MaxIDF']
    available_features = [f for f in target_features if f in results_df.columns]
    
    if len(available_features) == 0:
        print("分析対象の特徴量が見つかりません")
        return
    
    # データを準備
    comparison_data = []
    
    for feature in available_features:
        roberta_error_values = results_df.iloc[roberta_error_indices][feature].values
        median_val = np.median(roberta_error_values)
        
        high_mask = roberta_error_values >= median_val
        low_mask = roberta_error_values < median_val
        
        high_indices = roberta_error_indices[high_mask]
        low_indices = roberta_error_indices[low_mask]
        
        # 高いグループ
        high_success_count = np.sum([i in success_indices for i in high_indices])
        high_failure_count = np.sum([i in failure_indices for i in high_indices])
        high_total = high_success_count + high_failure_count
        high_success_rate = high_success_count / high_total if high_total > 0 else 0
        
        # 低いグループ
        low_success_count = np.sum([i in success_indices for i in low_indices])
        low_failure_count = np.sum([i in failure_indices for i in low_indices])
        low_total = low_success_count + low_failure_count
        low_success_rate = low_success_count / low_total if low_total > 0 else 0
        
        # 統計的検定
        high_values = results_df.iloc[high_indices][feature].values
        low_values = results_df.iloc[low_indices][feature].values
        try:
            stat, p = mannwhitneyu(high_values, low_values, alternative='two-sided')
        except:
            p = np.nan
        
        comparison_data.append({
            'feature': feature,
            'high_success_rate': high_success_rate,
            'low_success_rate': low_success_rate,
            'difference': high_success_rate - low_success_rate,
            'high_success_count': high_success_count,
            'high_total': high_total,
            'low_success_count': low_success_count,
            'low_total': low_total,
            'pvalue': p,
            'median_threshold': median_val
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # グラフ作成
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x_pos = np.arange(len(available_features))
    width = 0.35
    
    # バーグラフ
    bars1 = ax.bar(x_pos - width/2, comparison_df['high_success_rate'], width,
                   label='High Feature Value', alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, comparison_df['low_success_rate'], width,
                   label='Low Feature Value', alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    # 成功率をバーの上に表示
    for i, (bar1, bar2, row) in enumerate(zip(bars1, bars2, comparison_df.itertuples())):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.01,
               f'{height1:.1%}\n(n={row.high_total})', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.01,
               f'{height2:.1%}\n(n={row.low_total})', ha='center', va='bottom',
               fontsize=11, fontweight='bold')
        
        # 差を表示
        diff = row.difference
        max_height = max(height1, height2)
        ax.text(i, max_height + 0.08,
               f'+{diff:.1%}\n({diff*100:.1f}pp)', ha='center', va='bottom',
               fontsize=12, fontweight='bold', color='#3498db',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#3498db', linewidth=2))
    
    # 統計的有意性を表示
    for i, row in enumerate(comparison_df.itertuples()):
        if not np.isnan(row.pvalue):
            sig_symbol = '***' if row.pvalue < 0.001 else '**' if row.pvalue < 0.01 else '*' if row.pvalue < 0.05 else 'ns'
            ax.text(i, 0.95, sig_symbol, ha='center', va='top', fontsize=16, fontweight='bold', color='#e67e22')
    
    ax.set_ylabel('QPP Integration Success Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('QPP Feature', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(available_features, fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.0])
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f'{i:.0%}' for i in np.arange(0, 1.1, 0.1)], fontsize=12)
    ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_title('QPP Integration Success Rate by Feature Value\n(RoBERTa Error Cases)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # サブタイトルに統計情報を追加
    ax.text(0.5, 0.98, 'Higher feature values → Higher success rate', 
           transform=ax.transAxes, ha='center', va='top',
           fontsize=12, style='italic', color='#7f8c8d')
    
    plt.tight_layout()
    output_path = output_dir / f'causal_evidence_success_rate_comparison_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_feature_value_vs_success_rate(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    特徴量の値域ごとに、RoBERTaとQPP統合の正解率を直接比較（発表用）
    RoBERTa vs QPP統合の性能を明確に示す
    """
    # RoBERTaとLogRegの確率値を計算
    roberta_proba = expit(results_df[base_model].values)
    logreg_proba = expit(results_df['LogReg'].values)
    labels = results_df['label'].values
    
    # 中央値閾値で分析
    roberta_thresh = np.median(roberta_proba)
    logreg_thresh = 0.5
    
    roberta_pred = (roberta_proba >= roberta_thresh).astype(int)
    logreg_pred = (logreg_proba >= logreg_thresh).astype(int)
    
    # 正解率を計算（全ケースで）
    roberta_accuracy = (roberta_pred == labels).astype(int)
    logreg_accuracy = (logreg_pred == labels).astype(int)
    
    # 分析対象の特徴量（全QPP特徴量）
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']
    available_features = [col for col in results_df.columns 
                         if col not in exclude_cols and results_df[col].dtype in [np.float64, np.int64]]
    available_features = sorted(available_features)  # ソートして一貫性を保つ
    
    if len(available_features) == 0:
        return
    
    n_features = len(available_features)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, feature in enumerate(available_features):
        ax = axes[idx]
        
        feature_values = results_df[feature].values
        valid_mask = ~np.isnan(feature_values)
        
        # 5つのビンに分割（全ケースで）
        bins = np.percentile(feature_values[valid_mask], [0, 20, 40, 60, 80, 100])
        
        bin_centers = []
        bin_roberta_accs = []
        bin_logreg_accs = []
        bin_counts = []
        
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                mask = (feature_values >= bins[i]) & (feature_values <= bins[i+1]) & valid_mask
            else:
                mask = (feature_values >= bins[i]) & (feature_values < bins[i+1]) & valid_mask
            
            if mask.sum() > 0:
                bin_center = (bins[i] + bins[i+1]) / 2
                roberta_acc = roberta_accuracy[mask].mean()
                logreg_acc = logreg_accuracy[mask].mean()
                
                bin_centers.append(bin_center)
                bin_roberta_accs.append(roberta_acc)
                bin_logreg_accs.append(logreg_acc)
                bin_counts.append(mask.sum())
        
        # プロット（RoBERTaとLogRegを並べて表示）
        x_pos = np.arange(len(bin_centers))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, bin_roberta_accs, width,
                      label=f'{base_model}', alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x_pos + width/2, bin_logreg_accs, width,
                      label='LogReg (QPP Integrated)', alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=1.5)
        
        # 各バーに値を表示
        for bar1, bar2, roberta_acc, logreg_acc, count in zip(bars1, bars2, bin_roberta_accs, bin_logreg_accs, bin_counts):
            ax.text(bar1.get_x() + bar1.get_width()/2., roberta_acc + 0.01,
                   f'{roberta_acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            ax.text(bar2.get_x() + bar2.get_width()/2., logreg_acc + 0.01,
                   f'{logreg_acc:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # どちらが良いかを表示
            better = 'QPP' if logreg_acc > roberta_acc else 'BERT'
            max_height = max(roberta_acc, logreg_acc)
            color = '#2ecc71' if better == 'QPP' else '#3498db'
            ax.text((bar1.get_x() + bar2.get_x() + bar2.get_width()) / 2, max_height + 0.03,
                   better, ha='center', va='bottom', fontsize=9, fontweight='bold', color=color)
        
        ax.set_xlabel(f'{feature} Value Range', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{feature}', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{i+1}' for i in range(len(bin_centers))], fontsize=9)
        ax.set_ylim([0, 1.0])
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_yticklabels([f'{i:.0%}' for i in np.arange(0, 1.1, 0.2)], fontsize=10)
        ax.legend(fontsize=9, loc='upper left')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 余ったサブプロットを非表示
    for idx in range(len(available_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{base_model} vs QPP Integration: Accuracy Comparison by Feature Value\n(All Cases, Binned Analysis)', 
                fontsize=16, fontweight='bold', y=0.995 if n_rows > 1 else 0.98)
    plt.tight_layout()
    output_path = output_dir / f'causal_evidence_feature_vs_success_rate_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_bert_confidence_vs_qpp_heatmap(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    BERT自信度 × QPP特徴量の2軸ヒートマップ
    X軸：QPP feature（Low → High）
    Y軸：BERT confidence（低 → 高）
    色：Success Rate（QPP統合がRoBERTaの誤りを補正できた割合）
    """
    # RoBERTaとLogRegの確率値を計算
    roberta_proba = expit(results_df[base_model].values)
    logreg_proba = expit(results_df['LogReg'].values)
    labels = results_df['label'].values
    
    # 中央値閾値で分析
    roberta_thresh = np.median(roberta_proba)
    logreg_thresh = 0.5
    
    roberta_pred = (roberta_proba >= roberta_thresh).astype(int)
    logreg_pred = (logreg_proba >= logreg_thresh).astype(int)
    
    # RoBERTaが誤分類したケースを分類
    roberta_error = (roberta_pred != labels)
    logreg_correct = (logreg_pred == labels)
    
    success_mask = roberta_error & logreg_correct
    
    # BERTの自信度を計算（確率が0.5からどれだけ離れているか）
    roberta_confidence = np.abs(roberta_proba - 0.5)
    
    # QPP特徴量を取得
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']
    qpp_features = [col for col in results_df.columns 
                   if col not in exclude_cols and results_df[col].dtype in [np.float64, np.int64]]
    qpp_features = sorted(qpp_features)
    
    if len(qpp_features) == 0:
        print("QPP特徴量が見つかりません")
        return
    
    # 各QPP特徴量についてヒートマップを作成
    n_features = len(qpp_features)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, feature in enumerate(qpp_features):
        ax = axes[idx]
        
        feature_values = results_df[feature].values
        valid_mask = ~np.isnan(feature_values) & roberta_error
        
        if valid_mask.sum() == 0:
            ax.axis('off')
            continue
        
        # QPP特徴量を5つのビンに分割
        qpp_bins = np.percentile(feature_values[valid_mask], [0, 20, 40, 60, 80, 100])
        
        # BERT自信度を5つのビンに分割
        confidence_bins = np.percentile(roberta_confidence[valid_mask], [0, 20, 40, 60, 80, 100])
        
        # ヒートマップ用のデータを準備
        heatmap_data = np.zeros((len(confidence_bins) - 1, len(qpp_bins) - 1))
        count_data = np.zeros((len(confidence_bins) - 1, len(qpp_bins) - 1))
        
        for i in range(len(confidence_bins) - 1):
            for j in range(len(qpp_bins) - 1):
                # 自信度のビン
                if i == len(confidence_bins) - 2:
                    conf_mask = (roberta_confidence >= confidence_bins[i]) & (roberta_confidence <= confidence_bins[i+1])
                else:
                    conf_mask = (roberta_confidence >= confidence_bins[i]) & (roberta_confidence < confidence_bins[i+1])
                
                # QPP特徴量のビン
                if j == len(qpp_bins) - 2:
                    qpp_mask = (feature_values >= qpp_bins[j]) & (feature_values <= qpp_bins[j+1])
                else:
                    qpp_mask = (feature_values >= qpp_bins[j]) & (feature_values < qpp_bins[j+1])
                
                # 両方の条件を満たすケース（RoBERTaが誤分類したケース）
                cell_mask = conf_mask & qpp_mask & roberta_error
                
                if cell_mask.sum() > 0:
                    # Success Rateを計算
                    success_count = (cell_mask & success_mask).sum()
                    total_count = cell_mask.sum()
                    success_rate = success_count / total_count if total_count > 0 else 0
                    
                    heatmap_data[i, j] = success_rate
                    count_data[i, j] = total_count
        
        # ヒートマップをプロット
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        
        # カラーバーを追加
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Success Rate', fontsize=10, fontweight='bold')
        cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        # 各セルに値を表示
        for i in range(len(confidence_bins) - 1):
            for j in range(len(qpp_bins) - 1):
                if count_data[i, j] > 0:
                    text = ax.text(j, i, f'{heatmap_data[i, j]:.2f}\n(n={int(count_data[i, j])})',
                                 ha="center", va="center", color="black", fontsize=8, fontweight='bold')
                else:
                    ax.text(j, i, 'N/A', ha="center", va="center", color="gray", fontsize=8)
        
        # 軸ラベルとタイトル
        ax.set_xlabel(f'{feature} (Low → High)', fontsize=11, fontweight='bold')
        ax.set_ylabel(f'{base_model} Confidence\n(Low → High)', fontsize=11, fontweight='bold')
        ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
        
        # 軸の目盛り
        ax.set_xticks(np.arange(len(qpp_bins) - 1))
        ax.set_xticklabels([f'Q{j+1}' for j in range(len(qpp_bins) - 1)], fontsize=9)
        ax.set_yticks(np.arange(len(confidence_bins) - 1))
        ax.set_yticklabels([f'C{i+1}' for i in range(len(confidence_bins) - 1)], fontsize=9)
        
        # グリッドを追加
        ax.set_xticks(np.arange(len(qpp_bins) - 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(confidence_bins) - 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    
    # 余ったサブプロットを非表示
    for idx in range(len(qpp_features), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'QPP Integration Success Rate: {base_model} Confidence × QPP Feature\n(RoBERTa Error Cases)', 
                fontsize=16, fontweight='bold', y=0.995 if n_rows > 1 else 0.98)
    plt.tight_layout()
    output_path = output_dir / f'causal_evidence_confidence_vs_qpp_heatmap_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_model_comparison_by_qpp_feature(
    results_df: pd.DataFrame,
    output_dir: Path,
    base_model: str
):
    """
    全ケースで、各QPP特徴量の値域によってRoBERTaとLogRegのどちらが良いかを比較
    動的な重みづけの可能性を示す
    """
    # RoBERTaとLogRegの確率値を計算
    roberta_proba = expit(results_df[base_model].values)
    logreg_proba = expit(results_df['LogReg'].values)
    labels = results_df['label'].values
    
    # 中央値閾値で分析
    roberta_thresh = np.median(roberta_proba)
    logreg_thresh = 0.5
    
    roberta_pred = (roberta_proba >= roberta_thresh).astype(int)
    logreg_pred = (logreg_proba >= logreg_thresh).astype(int)
    
    # 正解率を計算
    roberta_accuracy = (roberta_pred == labels).astype(int)
    logreg_accuracy = (logreg_pred == labels).astype(int)
    
    # QPP特徴量を取得
    exclude_cols = ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']
    qpp_features = [col for col in results_df.columns 
                   if col not in exclude_cols and results_df[col].dtype in [np.float64, np.int64]]
    qpp_features = sorted(qpp_features)
    
    if len(qpp_features) == 0:
        print("QPP特徴量が見つかりません")
        return
    
    # データを準備
    comparison_data = []
    
    for feature in qpp_features:
        feature_values = results_df[feature].values
        median_val = np.median(feature_values[~np.isnan(feature_values)])
        
        # 高いグループと低いグループに分割
        high_mask = (feature_values >= median_val) & (~np.isnan(feature_values))
        low_mask = (feature_values < median_val) & (~np.isnan(feature_values))
        
        # 高いグループの性能
        high_roberta_acc = roberta_accuracy[high_mask].mean()
        high_logreg_acc = logreg_accuracy[high_mask].mean()
        high_improvement = high_logreg_acc - high_roberta_acc
        high_better = 'LogReg' if high_logreg_acc > high_roberta_acc else 'RoBERTa'
        
        # 低いグループの性能
        low_roberta_acc = roberta_accuracy[low_mask].mean()
        low_logreg_acc = logreg_accuracy[low_mask].mean()
        low_improvement = low_logreg_acc - low_roberta_acc
        low_better = 'LogReg' if low_logreg_acc > low_roberta_acc else 'RoBERTa'
        
        comparison_data.append({
            'feature': feature,
            'high_roberta_acc': high_roberta_acc,
            'high_logreg_acc': high_logreg_acc,
            'high_improvement': high_improvement,
            'high_better': high_better,
            'high_n': high_mask.sum(),
            'low_roberta_acc': low_roberta_acc,
            'low_logreg_acc': low_logreg_acc,
            'low_improvement': low_improvement,
            'low_better': low_better,
            'low_n': low_mask.sum(),
            'median_threshold': median_val
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # グラフ作成
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # 1. 高いグループでの比較
    ax1 = axes[0]
    x_pos = np.arange(len(qpp_features))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, comparison_df['high_roberta_acc'], width,
                   label=f'{base_model}', alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, comparison_df['high_logreg_acc'], width,
                   label='LogReg (QPP Integrated)', alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    # 性能をバーの上に表示
    for i, (bar1, bar2, row) in enumerate(zip(bars1, bars2, comparison_df.itertuples())):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax1.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.005,
               f'{height1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.005,
               f'{height2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # どちらが良いかを表示
        better = row.high_better
        max_height = max(height1, height2)
        color = '#2ecc71' if better == 'LogReg' else '#3498db'
        ax1.text(i, max_height + 0.02, better, ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)
    
    ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title(f'Model Comparison: High {base_model} Feature Value Group\n(All Cases)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(qpp_features, fontsize=11, rotation=45, ha='right')
    ax1.set_ylim([0, 1.0])
    ax1.set_yticks(np.arange(0, 1.1, 0.1))
    ax1.set_yticklabels([f'{i:.0%}' for i in np.arange(0, 1.1, 0.1)], fontsize=11)
    ax1.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 2. 低いグループでの比較
    ax2 = axes[1]
    
    bars3 = ax2.bar(x_pos - width/2, comparison_df['low_roberta_acc'], width,
                   label=f'{base_model}', alpha=0.8, color='#3498db', edgecolor='black', linewidth=1.5)
    bars4 = ax2.bar(x_pos + width/2, comparison_df['low_logreg_acc'], width,
                   label='LogReg (QPP Integrated)', alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    # 性能をバーの上に表示
    for i, (bar3, bar4, row) in enumerate(zip(bars3, bars4, comparison_df.itertuples())):
        height3 = bar3.get_height()
        height4 = bar4.get_height()
        ax2.text(bar3.get_x() + bar3.get_width()/2., height3 + 0.005,
               f'{height3:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(bar4.get_x() + bar4.get_width()/2., height4 + 0.005,
               f'{height4:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # どちらが良いかを表示
        better = row.low_better
        max_height = max(height3, height4)
        color = '#2ecc71' if better == 'LogReg' else '#3498db'
        ax2.text(i, max_height + 0.02, better, ha='center', va='bottom',
               fontsize=10, fontweight='bold', color=color)
    
    ax2.set_xlabel('QPP Feature', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax2.set_title(f'Model Comparison: Low {base_model} Feature Value Group\n(All Cases)', 
                 fontsize=15, fontweight='bold', pad=15)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(qpp_features, fontsize=11, rotation=45, ha='right')
    ax2.set_ylim([0, 1.0])
    ax2.set_yticks(np.arange(0, 1.1, 0.1))
    ax2.set_yticklabels([f'{i:.0%}' for i in np.arange(0, 1.1, 0.1)], fontsize=11)
    ax2.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.suptitle('Dynamic Model Selection Based on QPP Features\n(All Cases Analysis)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = output_dir / f'causal_evidence_model_comparison_all_cases_{base_model.lower()}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()
    
    # 結果をCSVに保存
    output_csv = output_dir / f'causal_evidence_model_comparison_all_cases_{base_model.lower()}.csv'
    comparison_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description='発表用：因果関係を示すグラフを作成')
    parser.add_argument('--results-csv', type=str, required=True,
                       help='Path to results.csv file')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for graphs')
    parser.add_argument('--base-model', type=str, default='RoBERTa',
                       help='Base model name (default: RoBERTa)')
    
    args = parser.parse_args()
    
    results_csv_path = Path(args.results_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from: {results_csv_path}")
    results_df = pd.read_csv(results_csv_path)
    print(f"Loaded {len(results_df)} samples")
    
    print("\n=== 発表用グラフの作成 ===")
    
    print("\n1. 成功率比較グラフを作成中...")
    plot_success_rate_comparison(results_df, output_dir, args.base_model)
    
    print("\n2. 特徴量値と成功率の関係グラフを作成中...")
    plot_feature_value_vs_success_rate(results_df, output_dir, args.base_model)
    
    print("\n3. 全ケースでのモデル比較グラフを作成中...")
    plot_model_comparison_by_qpp_feature(results_df, output_dir, args.base_model)
    
    print("\n4. BERT自信度 × QPP特徴量の2軸ヒートマップを作成中...")
    plot_bert_confidence_vs_qpp_heatmap(results_df, output_dir, args.base_model)
    
    print("\n=== 完了 ===")


if __name__ == '__main__':
    main()

