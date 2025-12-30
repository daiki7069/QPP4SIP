#!/usr/bin/env python3
"""
AmbigNQ実験結果をまとめるスクリプト
表とグラフを作成して結果を可視化
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import numpy as np

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")

# 結果ディレクトリ
BASE_DIR = Path(__file__).parent

# 実験設定のマッピング
EXPERIMENT_NAMES = {
    'post/clarity_ns50_nqc_smv_wig': 'Post (QPP only)',
    'post_bert/clarity_ns50_nqc_smv_wig_bert': 'Post + BERT',
    'pre/ictf_idf_maxidf_scq_scs': 'Pre (QPP only)',
    'pre_bert/ictf_idf_maxidf_scq_scs_bert': 'Pre + BERT',
    'pre_post/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig': 'Pre+Post (QPP only)',
    'pre_post_bert/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_bert': 'Pre+Post + BERT',
    'pre_post_bert/ictf_idf_maxidf_scq_scs_clarity_ns50_nqc_smv_wig_rob': 'Pre+Post + BERT (rob)',
    'bert/bert': 'BERT only',
    'bert/rob': 'RoBERTa only',
}

def parse_results_file(filepath):
    """results.txtファイルを解析してメトリクスを抽出"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    # デフォルト値を設定（パターンが見つからない場合に備える）
    results['accuracy'] = None
    results['f1'] = None
    results['auc'] = None
    results['ap'] = None
    
    # CVテストデータの結果を抽出
    cv_match = re.search(r'CVテストデータ - Accuracy: ([\d.]+), F1: ([\d.]+), AUC: ([\d.]+), AP: ([\d.]+)', content)
    if cv_match:
        results['accuracy'] = float(cv_match.group(1))
        results['f1'] = float(cv_match.group(2))
        results['auc'] = float(cv_match.group(3))
        results['ap'] = float(cv_match.group(4))
    
    # 平均値と標準偏差を抽出
    mean_match = re.search(
        r'test_acc: ([\d.]+) \(±([\d.]+)\)\s+test_f1: ([\d.]+) \(±([\d.]+)\)\s+test_auc: ([\d.]+) \(±([\d.]+)\)\s+test_ap: ([\d.]+) \(±([\d.]+)\)',
        content, re.MULTILINE
    )
    if mean_match:
        results['accuracy_mean'] = float(mean_match.group(1))
        results['accuracy_std'] = float(mean_match.group(2))
        results['f1_mean'] = float(mean_match.group(3))
        results['f1_std'] = float(mean_match.group(4))
        results['auc_mean'] = float(mean_match.group(5))
        results['auc_std'] = float(mean_match.group(6))
        results['ap_mean'] = float(mean_match.group(7))
        results['ap_std'] = float(mean_match.group(8))
    
    # 単体指標の最高ROCを抽出
    single_metric_match = re.search(
        r'単体指標の最高ROC: (\w+) \(AUC = ([\d.]+)\)',
        content
    )
    if single_metric_match:
        results['single_metric_name'] = single_metric_match.group(1)
        results['single_metric_auc'] = float(single_metric_match.group(2))
    
    # DeLong検定の結果を抽出
    delong_match = re.search(
        r'DeLongの検定結果:.*?AUC差: ([\d.]+).*?p値: ([\d.]+).*?有意差: (あり|なし)',
        content, re.DOTALL
    )
    if delong_match:
        results['delong_auc_diff'] = float(delong_match.group(1))
        results['delong_p_value'] = float(delong_match.group(2))
        results['delong_significant'] = delong_match.group(3) == 'あり'
        
        # 改善度を計算（パーセンテージ）
        if 'single_metric_auc' in results and results['single_metric_auc'] > 0:
            results['improvement_pct'] = (results['delong_auc_diff'] / results['single_metric_auc']) * 100
    
    # 特徴量の数を抽出
    feature_match = re.search(r'特徴量: \[(.*?)\]', content)
    if feature_match:
        features = [f.strip().strip("'\"") for f in feature_match.group(1).split(',')]
        results['num_features'] = len(features)
        results['features'] = features
    
    return results

def collect_all_results():
    """全ての結果ファイルを収集"""
    all_results = []
    
    for exp_path, exp_name in EXPERIMENT_NAMES.items():
        results_file = BASE_DIR / exp_path / 'results.txt'
        if results_file.exists():
            results = parse_results_file(results_file)
            results['experiment'] = exp_name
            results['experiment_path'] = exp_path
            all_results.append(results)
        else:
            print(f"Warning: {results_file} not found")
    
    return pd.DataFrame(all_results)

def create_comparison_table(df):
    """比較表を作成"""
    # メトリクスを整理
    table_data = []
    for _, row in df.iterrows():
        accuracy_str = f"{row['accuracy']:.4f}" if pd.notna(row.get('accuracy')) else 'N/A'
        f1_str = f"{row['f1']:.4f}" if pd.notna(row.get('f1')) else 'N/A'
        auc_str = f"{row['auc']:.4f}" if pd.notna(row.get('auc')) else 'N/A'
        ap_str = f"{row['ap']:.4f}" if pd.notna(row.get('ap')) else 'N/A'
        
        table_data.append({
            'Experiment': row['experiment'],
            'Accuracy': accuracy_str,
            'F1 Score': f1_str,
            'AUC-ROC': auc_str,
            'AP': ap_str,
            'Num Features': row.get('num_features', 'N/A'),
        })
    
    table_df = pd.DataFrame(table_data)
    
    # 表を保存
    table_df.to_csv(BASE_DIR / 'comparison_table.csv', index=False)
    print(f"比較表を保存: {BASE_DIR / 'comparison_table.csv'}")
    
    # 表を表示
    print("\n=== 実験結果比較表 ===")
    print(table_df.to_string(index=False))
    
    return table_df

def create_metrics_comparison_chart(df):
    """メトリクス比較チャートを作成"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('AmbigNQ Experiment Results Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('accuracy', 'Accuracy', axes[0, 0]),
        ('f1', 'F1 Score', axes[0, 1]),
        ('auc', 'AUC-ROC', axes[1, 0]),
        ('ap', 'Average Precision', axes[1, 1]),
    ]
    
    for metric_key, metric_name, ax in metrics:
        # バーグラフを作成
        if metric_key not in df.columns:
            print(f"警告: '{metric_key}'カラムが存在しないため、スキップします")
            continue
        
        x_pos = np.arange(len(df))
        values = df[metric_key].values
        # NaN値を0に置き換え（表示用）
        values = np.nan_to_num(values, nan=0.0)
        colors = ['#2ecc71' if 'BERT' in exp else '#3498db' for exp in df['experiment']]
        
        bars = ax.bar(x_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 値のラベルを追加
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(df['experiment'], rotation=45, ha='right', fontsize=9)
        ax.set_ylim([0, max(values) * 1.15])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 凡例を追加
    qpp_patch = mpatches.Patch(color='#3498db', label='QPP only')
    bert_patch = mpatches.Patch(color='#2ecc71', label='With BERT')
    fig.legend(handles=[qpp_patch, bert_patch], loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"メトリクス比較チャートを保存: {BASE_DIR / 'metrics_comparison.png'}")
    plt.close()

def create_radar_chart(df):
    """レーダーチャートを作成"""
    # メトリクスを正規化（0-1スケール）
    metrics = ['accuracy', 'f1', 'auc', 'ap']
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 角度を設定
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 閉じるために最初の角度を追加
    
    # 各実験のデータをプロット
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[m] if pd.notna(row.get(m)) else 0.0 for m in metrics]
        values += values[:1]  # 閉じるために最初の値を追加
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['experiment'], color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    # 軸ラベルを設定
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Accuracy', 'F1 Score', 'AUC-ROC', 'Average Precision'], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.title('AmbigNQ Experiment Results - Radar Chart', fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'radar_chart.png', dpi=300, bbox_inches='tight')
    print(f"レーダーチャートを保存: {BASE_DIR / 'radar_chart.png'}")
    plt.close()

def create_heatmap(df):
    """ヒートマップを作成"""
    # メトリクスを抽出
    metrics = ['accuracy', 'f1', 'auc', 'ap']
    # 存在するカラムのみを使用
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        print("警告: ヒートマップ用のメトリクスが見つかりません")
        return
    
    heatmap_data = df[['experiment'] + available_metrics].set_index('experiment')
    
    # ヒートマップを作成
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='YlOrRd', 
                cbar_kws={'label': 'Score'}, linewidths=0.5, linecolor='gray',
                vmin=0.5, vmax=0.7, ax=ax)
    
    ax.set_title('AmbigNQ Experiment Results - Heatmap', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Experiment', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'heatmap.png', dpi=300, bbox_inches='tight')
    print(f"ヒートマップを保存: {BASE_DIR / 'heatmap.png'}")
    plt.close()

def create_summary_report(df):
    """サマリーレポートを作成"""
    report = []
    report.append("=" * 80)
    report.append("AmbigNQ 実験結果サマリー")
    report.append("=" * 80)
    report.append("")
    
    # 各メトリクスで最高性能の実験を特定
    metrics = ['accuracy', 'f1', 'auc', 'ap']
    metric_names = ['Accuracy', 'F1 Score', 'AUC-ROC', 'Average Precision']
    
    report.append("【最高性能実験】")
    for metric, name in zip(metrics, metric_names):
        if metric not in df.columns:
            report.append(f"  {name}: データなし")
            continue
        # NaN値を除外
        metric_series = df[metric].dropna()
        if len(metric_series) == 0:
            report.append(f"  {name}: データなし")
            continue
        best_idx = metric_series.idxmax()
        best_row = df.loc[best_idx]
        report.append(f"  {name}: {best_row['experiment']} ({best_row[metric]:.4f})")
    
    report.append("")
    report.append("【BERT有無による比較】")
    
    # BERT有無でグループ化
    bert_experiments = df[df['experiment'].str.contains('BERT', na=False)]
    qpp_only_experiments = df[~df['experiment'].str.contains('BERT', na=False)]
    
    if len(bert_experiments) > 0 and len(qpp_only_experiments) > 0:
        for metric, name in zip(metrics, metric_names):
            if metric not in df.columns:
                report.append(f"  {name}: データなし")
                continue
            bert_avg = bert_experiments[metric].mean()
            qpp_avg = qpp_only_experiments[metric].mean()
            improvement = bert_avg - qpp_avg
            report.append(f"  {name}:")
            report.append(f"    BERT有り平均: {bert_avg:.4f}")
            report.append(f"    QPPのみ平均: {qpp_avg:.4f}")
            if qpp_avg > 0:
                report.append(f"    改善度: {improvement:+.4f} ({improvement/qpp_avg*100:+.2f}%)")
            else:
                report.append(f"    改善度: {improvement:+.4f}")
    
    report.append("")
    report.append("【詳細結果】")
    report.append("-" * 80)
    
    for _, row in df.iterrows():
        report.append(f"\n実験: {row['experiment']}")
        accuracy_val = f"{row['accuracy']:.4f}" if pd.notna(row.get('accuracy')) else 'N/A'
        f1_val = f"{row['f1']:.4f}" if pd.notna(row.get('f1')) else 'N/A'
        auc_val = f"{row['auc']:.4f}" if pd.notna(row.get('auc')) else 'N/A'
        ap_val = f"{row['ap']:.4f}" if pd.notna(row.get('ap')) else 'N/A'
        report.append(f"  Accuracy: {accuracy_val}")
        report.append(f"  F1 Score: {f1_val}")
        report.append(f"  AUC-ROC: {auc_val}")
        report.append(f"  Average Precision: {ap_val}")
        if 'num_features' in row and pd.notna(row['num_features']):
            report.append(f"  特徴量数: {int(row['num_features'])}")
        
        # 単体指標からの改善度を追加
        if 'single_metric_name' in row and pd.notna(row['single_metric_name']):
            single_metric_name = row['single_metric_name']
            single_metric_auc = row.get('single_metric_auc', None)
            regression_auc = row['auc']
            
            if pd.notna(single_metric_auc):
                auc_diff = regression_auc - single_metric_auc
                improvement_pct = (auc_diff / single_metric_auc) * 100 if single_metric_auc > 0 else 0
                report.append(f"  単体指標からの改善:")
                report.append(f"    最高単体指標: {single_metric_name} (AUC = {single_metric_auc:.4f})")
                report.append(f"    回帰モデル: AUC = {regression_auc:.4f}")
                report.append(f"    AUC改善: {auc_diff:+.4f} ({improvement_pct:+.2f}%)")
        
        if 'delong_significant' in row and pd.notna(row['delong_significant']):
            p_value = row.get('delong_p_value', 'N/A')
            if isinstance(p_value, float):
                p_value_str = f"{p_value:.4f}" if p_value >= 0.0001 else f"{p_value:.2e}"
            else:
                p_value_str = str(p_value)
            report.append(f"  DeLong検定: {'有意差あり' if row['delong_significant'] else '有意差なし'} (p={p_value_str})")
    
    report_text = "\n".join(report)
    
    # レポートを保存
    with open(BASE_DIR / 'summary_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nサマリーレポートを保存: {BASE_DIR / 'summary_report.txt'}")
    print("\n" + report_text)

def main():
    """メイン処理"""
    print("AmbigNQ実験結果をまとめています...")
    
    # 結果を収集
    df = collect_all_results()
    
    if df.empty:
        print("エラー: 結果ファイルが見つかりませんでした")
        return
    
    # 必要なカラムが存在するか確認
    required_columns = ['accuracy', 'f1', 'auc', 'ap']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: 以下のカラムが見つかりませんでした: {missing_columns}")
        # 見つからないカラムにNaNを設定
        for col in missing_columns:
            df[col] = None
    
    # 結果をソート（AUC-ROCで降順、NaNを除外）
    if 'auc' in df.columns:
        df = df.sort_values('auc', ascending=False, na_position='last').reset_index(drop=True)
    else:
        print("警告: 'auc'カラムが存在しないため、ソートをスキップします")
    
    # 比較表を作成
    create_comparison_table(df)
    
    # グラフを作成
    create_metrics_comparison_chart(df)
    create_radar_chart(df)
    create_heatmap(df)
    
    # サマリーレポートを作成
    create_summary_report(df)
    
    print("\n完了しました！")

if __name__ == '__main__':
    main()

