#!/usr/bin/env python3
"""
QPP単体指標と統合モデルの比較分析: 新しい仮説検証
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats
from sklearn.metrics import roc_auc_score
from module.bootstrap import delong_test
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_qa_pairs_count(data_path: Path) -> Dict[Tuple[str, int], int]:
    """dev.jsonからqa_pairs_countを読み込む"""
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


def load_qpp_scores(dataset_dir: Path, split: str = 'dev') -> Dict[str, Dict[Tuple[str, int], float]]:
    """QPPスコアを読み込む（10種類）"""
    qpp_metrics = ['clarity', 'wig', 'nqc', 'smv', 'n_sigma_50', 'avgidf', 'avgictf', 'maxidf', 'maxscq', 'simplified_clarity']
    qpp_scores = {}
    
    for metric in qpp_metrics:
        json_path = dataset_dir / f'{split}_{metric}.json'
        if json_path.exists():
            print(f"[INFO] Loading {metric} scores from {json_path}...")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scores_dict = {}
            for conversation in data:
                for turn in conversation:
                    conv_id = str(turn.get('conv_id', ''))
                    turn_id = int(turn.get('turn_id', 1))
                    score = turn.get(metric)
                    if score is not None:
                        key = (conv_id, turn_id)
                        scores_dict[key] = float(score)
            
            qpp_scores[metric] = scores_dict
            print(f"[INFO] Loaded {metric} scores for {len(scores_dict)} entries")
    
    return qpp_scores


def load_integrated_score(logreg_output_dir: Path, data_path: Path, feature_combination: str = None) -> Tuple[Dict[Tuple[str, int], float], Dict[Tuple[str, int], int]]:
    """統合モデルの予測確率と真のラベルを読み込む（dev.jsonの順序と一致させる）"""
    integrated_scores = {}
    true_labels = {}
    
    if feature_combination:
        pred_paths = list(logreg_output_dir.glob(f'**/{feature_combination}/**/predictions_for_delong.csv'))
    else:
        # QPPのみを使用したモデルを探す（pre_postディレクトリ）
        pred_paths = []
        for path in logreg_output_dir.glob('**/pre_post/**/csv/predictions_for_delong.csv'):
            path_str = str(path)
            if '/pre_post/' in path_str:
                feature_dir = path_str.split('/pre_post/')[1].split('/csv/')[0]
                if 'bert' not in feature_dir.lower() and 'roberta' not in feature_dir.lower():
                    pred_paths.append(path)
    
    if not pred_paths:
        print("[WARN] No integrated model predictions found")
        return integrated_scores, true_labels
    
    # 最新のファイルを使用
    pred_paths_with_time = [(p, p.stat().st_mtime) for p in pred_paths]
    pred_paths_with_time.sort(key=lambda x: x[1], reverse=True)
    latest_path = pred_paths_with_time[0][0]
    
    print(f"[INFO] Loading QPP integrated scores from {latest_path}...")
    pred_df = pd.read_csv(latest_path)
    
    # dev.jsonの順序でキーを生成
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    key_list = []
    for conversation in data:
        for turn in conversation:
            conv_id = str(turn.get('conv_id', ''))
            turn_id = int(turn.get('turn_id', 1))
            key = (conv_id, turn_id)
            key_list.append(key)
    
    # predictions_for_delong.csvの順序と一致させる
    if 'y_pred_proba' in pred_df.columns and 'y_true' in pred_df.columns:
        for i, (prob, label) in enumerate(zip(pred_df['y_pred_proba'], pred_df['y_true'])):
            if i < len(key_list) and prob is not None and label is not None:
                key = key_list[i]
                integrated_scores[key] = float(prob)
                true_labels[key] = int(label)
    
    print(f"[INFO] Loaded QPP integrated scores for {len(integrated_scores)} entries")
    print(f"[INFO] Loaded true labels for {len(true_labels)} entries")
    return integrated_scores, true_labels


def perform_hypothesis_testing(
    df: pd.DataFrame,
    qpp_metrics: List[str],
    integrated_name: str,
    output_dir: Path,
    true_labels: Dict[Tuple[str, int], int] = None
):
    """
    統合モデルが優れている点を明らかにする仮説検証分析
    
    仮説1: QPP指標が矛盾する場合、統合モデルは単体QPP指標より正確な予測ができる
    仮説2: 特定のQAペア数範囲で、統合モデルが単体QPP指標より優れている（相関が強い、または統計的有意）
    仮説3: 統合モデルの予測は、単体QPP指標より一貫性が高い（予測のばらつきが小さい）
    仮説4: 複数のQPP指標が「高」または「低」の組み合わせパターンに対して、統合モデルがより適切に反応する
    """
    print("\n[INFO] Performing new hypothesis testing...")
    
    # 1+ QAペアのみを使用
    df_filtered = df[df['qa_pairs_count'] >= 1].copy()
    
    if integrated_name not in df_filtered.columns:
        print(f"[WARN] Integrated model scores not found, skipping hypothesis testing")
        return
    
    results = {}
    df_filtered['integrated_prob'] = df_filtered[integrated_name]
    
    # 各QPP指標を高/低に2分割（中央値で）
    for metric in qpp_metrics:
        if metric in df_filtered.columns:
            median_val = df_filtered[metric].median()
            df_filtered[f'{metric}_high'] = (df_filtered[metric] >= median_val).astype(int)
    
    # 不一致度を計算: 高と判定された指標数と低と判定された指標数の差
    high_counts = df_filtered[[f'{m}_high' for m in qpp_metrics if f'{m}_high' in df_filtered.columns]].sum(axis=1)
    total_metrics = len([m for m in qpp_metrics if f'{m}_high' in df_filtered.columns])
    df_filtered['qpp_inconsistency'] = abs(high_counts - (total_metrics / 2)) / (total_metrics / 2)  # 0-1の範囲
    
    # 仮説1: QPP指標が矛盾する場合、統合モデルは単体QPP指標より正確な予測ができる
    print("\n[INFO] Hypothesis 1: Integrated model excels when QPP metrics are inconsistent")
    
    inconsistency_ranges = [
        (0.0, 0.33, 'Low Inconsistency'),
        (0.33, 0.67, 'Medium Inconsistency'),
        (0.67, 1.0, 'High Inconsistency')
    ]
    
    hypothesis1_results = []
    for low, high, label in inconsistency_ranges:
        range_data = df_filtered[(df_filtered['qpp_inconsistency'] >= low) & (df_filtered['qpp_inconsistency'] < high)].copy()
        if len(range_data) > 10:
            # 統合モデルとQAペア数の相関
            mask = ~(pd.isna(range_data['integrated_prob']) | pd.isna(range_data['qa_pairs_count']))
            if mask.sum() > 1:
                integrated_spearman, integrated_p = spearmanr(
                    range_data.loc[mask, 'qa_pairs_count'],
                    range_data.loc[mask, 'integrated_prob']
                )
            else:
                integrated_spearman = integrated_p = np.nan
            
            # 最良の単体QPP指標との相関
            best_qpp_metric = None
            best_qpp_spearman = -np.inf
            best_qpp_spearman_p = 1.0
            
            for metric in qpp_metrics:
                if metric in range_data.columns:
                    metric_mask = ~(pd.isna(range_data[metric]) | pd.isna(range_data['qa_pairs_count']))
                    if metric_mask.sum() > 1:
                        qpp_spearman, qpp_p = spearmanr(
                            range_data.loc[metric_mask, 'qa_pairs_count'],
                            range_data.loc[metric_mask, metric]
                        )
                        if (qpp_p < 0.05 and best_qpp_spearman_p >= 0.05) or \
                           (qpp_p < 0.05 and best_qpp_spearman_p < 0.05 and abs(qpp_spearman) > abs(best_qpp_spearman)) or \
                           (qpp_p >= 0.05 and best_qpp_spearman_p >= 0.05 and abs(qpp_spearman) > abs(best_qpp_spearman)):
                            best_qpp_metric = metric
                            best_qpp_spearman = qpp_spearman
                            best_qpp_spearman_p = qpp_p
            
            # 改善度: 統合モデルが単体QPPより優れているか
            if not np.isnan(integrated_spearman) and best_qpp_metric:
                if integrated_p < 0.05 and best_qpp_spearman_p < 0.05:
                    improvement = abs(integrated_spearman) - abs(best_qpp_spearman)
                elif integrated_p < 0.05 and best_qpp_spearman_p >= 0.05:
                    improvement = abs(integrated_spearman)  # 統合モデルのみ有意
                elif integrated_p >= 0.05 and best_qpp_spearman_p < 0.05:
                    improvement = -abs(best_qpp_spearman)  # 単体QPPのみ有意
                else:
                    improvement = abs(integrated_spearman) - abs(best_qpp_spearman)
            else:
                improvement = np.nan
            
            hypothesis1_results.append({
                'inconsistency_range': label,
                'n_samples': len(range_data),
                'integrated_spearman_rho': integrated_spearman,
                'integrated_spearman_p': integrated_p,
                'integrated_significant': integrated_p < 0.05 if not np.isnan(integrated_p) else False,
                'best_qpp_metric': best_qpp_metric,
                'best_qpp_spearman_rho': best_qpp_spearman if best_qpp_metric else np.nan,
                'best_qpp_spearman_p': best_qpp_spearman_p if best_qpp_metric else np.nan,
                'best_qpp_significant': best_qpp_spearman_p < 0.05 if best_qpp_metric else False,
                'improvement': improvement,
                'integrated_better': improvement > 0 if not np.isnan(improvement) else False
            })
    
    results['hypothesis1'] = pd.DataFrame(hypothesis1_results)
    
    # 仮説2: 特定のQAペア数範囲で、統合モデルが単体QPP指標より優れている
    print("\n[INFO] Hypothesis 2: Integrated model excels in specific QA pairs count ranges")
    
    qa_ranges = [
        (1, 2, '1-2 QA pairs'),
        (2, 3, '2-3 QA pairs'),
        (3, 4, '3-4 QA pairs'),
        (4, 100, '4+ QA pairs')
    ]
    
    hypothesis2_results = []
    for low, high, label in qa_ranges:
        range_data = df_filtered[(df_filtered['qa_pairs_count'] >= low) & (df_filtered['qa_pairs_count'] < high)].copy()
        if len(range_data) > 10:
            # 統合モデルとQAペア数の相関
            mask = ~(pd.isna(range_data['integrated_prob']) | pd.isna(range_data['qa_pairs_count']))
            if mask.sum() > 1:
                integrated_spearman, integrated_p = spearmanr(
                    range_data.loc[mask, 'qa_pairs_count'],
                    range_data.loc[mask, 'integrated_prob']
                )
            else:
                integrated_spearman = integrated_p = np.nan
            
            # 最良の単体QPP指標との相関
            best_qpp_metric = None
            best_qpp_spearman = -np.inf
            best_qpp_spearman_p = 1.0
            
            for metric in qpp_metrics:
                if metric in range_data.columns:
                    metric_mask = ~(pd.isna(range_data[metric]) | pd.isna(range_data['qa_pairs_count']))
                    if metric_mask.sum() > 1:
                        qpp_spearman, qpp_p = spearmanr(
                            range_data.loc[metric_mask, 'qa_pairs_count'],
                            range_data.loc[metric_mask, metric]
                        )
                        if (qpp_p < 0.05 and best_qpp_spearman_p >= 0.05) or \
                           (qpp_p < 0.05 and best_qpp_spearman_p < 0.05 and abs(qpp_spearman) > abs(best_qpp_spearman)) or \
                           (qpp_p >= 0.05 and best_qpp_spearman_p >= 0.05 and abs(qpp_spearman) > abs(best_qpp_spearman)):
                            best_qpp_metric = metric
                            best_qpp_spearman = qpp_spearman
                            best_qpp_spearman_p = qpp_p
            
            # 改善度
            if not np.isnan(integrated_spearman) and best_qpp_metric:
                if integrated_p < 0.05 and best_qpp_spearman_p < 0.05:
                    improvement = abs(integrated_spearman) - abs(best_qpp_spearman)
                elif integrated_p < 0.05 and best_qpp_spearman_p >= 0.05:
                    improvement = abs(integrated_spearman)
                elif integrated_p >= 0.05 and best_qpp_spearman_p < 0.05:
                    improvement = -abs(best_qpp_spearman)
                else:
                    improvement = abs(integrated_spearman) - abs(best_qpp_spearman)
            else:
                improvement = np.nan
            
            hypothesis2_results.append({
                'qa_range': label,
                'qa_low': low,
                'qa_high': high,
                'n_samples': len(range_data),
                'integrated_spearman_rho': integrated_spearman,
                'integrated_spearman_p': integrated_p,
                'integrated_significant': integrated_p < 0.05 if not np.isnan(integrated_p) else False,
                'best_qpp_metric': best_qpp_metric,
                'best_qpp_spearman_rho': best_qpp_spearman if best_qpp_metric else np.nan,
                'best_qpp_spearman_p': best_qpp_spearman_p if best_qpp_metric else np.nan,
                'best_qpp_significant': best_qpp_spearman_p < 0.05 if best_qpp_metric else False,
                'improvement': improvement,
                'integrated_better': improvement > 0 if not np.isnan(improvement) else False
            })
    
    results['hypothesis2'] = pd.DataFrame(hypothesis2_results)
    
    # 仮説3: 統合モデルは、QAペア数との相関が統計的有意になるケースが多い（予測能力が高い）
    print("\n[INFO] Hypothesis 3: Integrated model achieves statistical significance more often")
    
    # 全体での相関を計算
    mask_all = ~(pd.isna(df_filtered['integrated_prob']) | pd.isna(df_filtered['qa_pairs_count']))
    if mask_all.sum() > 1:
        integrated_spearman_all, integrated_p_all = spearmanr(
            df_filtered.loc[mask_all, 'qa_pairs_count'],
            df_filtered.loc[mask_all, 'integrated_prob']
        )
    else:
        integrated_spearman_all = integrated_p_all = np.nan
    
    # 各単体QPP指標の相関を計算
    qpp_significant_count = 0
    qpp_total_count = 0
    qpp_correlations = []
    
    for metric in qpp_metrics:
        if metric in df_filtered.columns:
            metric_mask = ~(pd.isna(df_filtered[metric]) | pd.isna(df_filtered['qa_pairs_count']))
            if metric_mask.sum() > 1:
                qpp_spearman, qpp_p = spearmanr(
                    df_filtered.loc[metric_mask, 'qa_pairs_count'],
                    df_filtered.loc[metric_mask, metric]
                )
                qpp_correlations.append({
                    'metric': metric,
                    'spearman_rho': qpp_spearman,
                    'spearman_p': qpp_p,
                    'significant': qpp_p < 0.05
                })
                qpp_total_count += 1
                if qpp_p < 0.05:
                    qpp_significant_count += 1
    
    # サブグループ（QAペア数カテゴリ、不一致度グループなど）での統計的有意性をカウント
    categories = ['1 QA pair', '2 QA pairs', '3 QA pairs', '4+ QA pairs']
    integrated_significant_subgroups = 0
    integrated_total_subgroups = 0
    qpp_significant_subgroups = 0
    qpp_total_subgroups = 0
    
    hypothesis3_results = []
    
    # カテゴリごと
    for category in categories:
        category_data = df_filtered[df_filtered['qa_pairs_category'] == category].copy()
        if len(category_data) > 10:
            # 統合モデル
            mask = ~(pd.isna(category_data['integrated_prob']) | pd.isna(category_data['qa_pairs_count']))
            if mask.sum() > 1:
                integrated_spearman, integrated_p = spearmanr(
                    category_data.loc[mask, 'qa_pairs_count'],
                    category_data.loc[mask, 'integrated_prob']
                )
                integrated_total_subgroups += 1
                if integrated_p < 0.05:
                    integrated_significant_subgroups += 1
            else:
                integrated_spearman = integrated_p = np.nan
            
            # 最良の単体QPP指標
            best_qpp_metric = None
            best_qpp_spearman = -np.inf
            best_qpp_spearman_p = 1.0
            
            for metric in qpp_metrics:
                if metric in category_data.columns:
                    metric_mask = ~(pd.isna(category_data[metric]) | pd.isna(category_data['qa_pairs_count']))
                    if metric_mask.sum() > 1:
                        qpp_spearman, qpp_p = spearmanr(
                            category_data.loc[metric_mask, 'qa_pairs_count'],
                            category_data.loc[metric_mask, metric]
                        )
                        if (qpp_p < 0.05 and best_qpp_spearman_p >= 0.05) or \
                           (qpp_p < 0.05 and best_qpp_spearman_p < 0.05 and abs(qpp_spearman) > abs(best_qpp_spearman)) or \
                           (qpp_p >= 0.05 and best_qpp_spearman_p >= 0.05 and abs(qpp_spearman) > abs(best_qpp_spearman)):
                            best_qpp_metric = metric
                            best_qpp_spearman = qpp_spearman
                            best_qpp_spearman_p = qpp_p
            
            if best_qpp_metric:
                qpp_total_subgroups += 1
                if best_qpp_spearman_p < 0.05:
                    qpp_significant_subgroups += 1
            
            hypothesis3_results.append({
                'category': category,
                'n_samples': len(category_data),
                'integrated_spearman_rho': integrated_spearman,
                'integrated_spearman_p': integrated_p,
                'integrated_significant': integrated_p < 0.05 if not np.isnan(integrated_p) else False,
                'best_qpp_metric': best_qpp_metric,
                'best_qpp_spearman_rho': best_qpp_spearman if best_qpp_metric else np.nan,
                'best_qpp_spearman_p': best_qpp_spearman_p if best_qpp_metric else np.nan,
                'best_qpp_significant': best_qpp_spearman_p < 0.05 if best_qpp_metric else False
            })
    
    # 全体での統計的有意性の比較
    integrated_all_significant = integrated_p_all < 0.05 if not np.isnan(integrated_p_all) else False
    qpp_all_significant_ratio = qpp_significant_count / qpp_total_count if qpp_total_count > 0 else 0
    
    # サマリー
    hypothesis3_summary = {
        'integrated_all_significant': integrated_all_significant,
        'integrated_all_spearman_rho': integrated_spearman_all,
        'integrated_all_spearman_p': integrated_p_all,
        'qpp_significant_count': qpp_significant_count,
        'qpp_total_count': qpp_total_count,
        'qpp_significant_ratio': qpp_all_significant_ratio,
        'integrated_subgroups_significant': integrated_significant_subgroups,
        'integrated_subgroups_total': integrated_total_subgroups,
        'integrated_subgroups_ratio': integrated_significant_subgroups / integrated_total_subgroups if integrated_total_subgroups > 0 else 0,
        'qpp_subgroups_significant': qpp_significant_subgroups,
        'qpp_subgroups_total': qpp_total_subgroups,
        'qpp_subgroups_ratio': qpp_significant_subgroups / qpp_total_subgroups if qpp_total_subgroups > 0 else 0,
        'integrated_better': (integrated_all_significant and qpp_all_significant_ratio < 1.0) or \
                            (integrated_significant_subgroups / integrated_total_subgroups if integrated_total_subgroups > 0 else 0) > \
                            (qpp_significant_subgroups / qpp_total_subgroups if qpp_total_subgroups > 0 else 0)
    }
    
    results['hypothesis3'] = pd.DataFrame(hypothesis3_results)
    results['hypothesis3_summary'] = pd.DataFrame([hypothesis3_summary])
    
    # 仮説4: 複数のQPP指標の組み合わせパターンに対して、統合モデルがより適切に反応する
    print("\n[INFO] Hypothesis 4: Integrated model responds better to QPP metric combination patterns")
    
    # QPP指標の組み合わせパターン: 高/低の組み合わせ
    # 例: 多くの指標が「高」、多くの指標が「低」、混在
    hypothesis4_results = []
    
    # 高指標数でグループ化
    high_count_ranges = [
        (0, 3, 'Few High (0-3)'),
        (3, 7, 'Mixed (3-7)'),
        (7, 11, 'Many High (7-10)')
    ]
    
    for low, high, label in high_count_ranges:
        range_data = df_filtered[(high_counts >= low) & (high_counts < high)].copy()
        if len(range_data) > 10:
            # 統合モデルとQAペア数の相関
            mask = ~(pd.isna(range_data['integrated_prob']) | pd.isna(range_data['qa_pairs_count']))
            if mask.sum() > 1:
                integrated_spearman, integrated_p = spearmanr(
                    range_data.loc[mask, 'qa_pairs_count'],
                    range_data.loc[mask, 'integrated_prob']
                )
            else:
                integrated_spearman = integrated_p = np.nan
            
            # 最良の単体QPP指標との相関
            best_qpp_metric = None
            best_qpp_spearman = -np.inf
            best_qpp_spearman_p = 1.0
            
            for metric in qpp_metrics:
                if metric in range_data.columns:
                    metric_mask = ~(pd.isna(range_data[metric]) | pd.isna(range_data['qa_pairs_count']))
                    if metric_mask.sum() > 1:
                        qpp_spearman, qpp_p = spearmanr(
                            range_data.loc[metric_mask, 'qa_pairs_count'],
                            range_data.loc[metric_mask, metric]
                        )
                        if (qpp_p < 0.05 and best_qpp_spearman_p >= 0.05) or \
                           (qpp_p < 0.05 and best_qpp_spearman_p < 0.05 and abs(qpp_spearman) > abs(best_qpp_spearman)) or \
                           (qpp_p >= 0.05 and best_qpp_spearman_p >= 0.05 and abs(qpp_spearman) > abs(best_qpp_spearman)):
                            best_qpp_metric = metric
                            best_qpp_spearman = qpp_spearman
                            best_qpp_spearman_p = qpp_p
            
            # 改善度
            if not np.isnan(integrated_spearman) and best_qpp_metric:
                if integrated_p < 0.05 and best_qpp_spearman_p < 0.05:
                    improvement = abs(integrated_spearman) - abs(best_qpp_spearman)
                elif integrated_p < 0.05 and best_qpp_spearman_p >= 0.05:
                    improvement = abs(integrated_spearman)
                elif integrated_p >= 0.05 and best_qpp_spearman_p < 0.05:
                    improvement = -abs(best_qpp_spearman)
                else:
                    improvement = abs(integrated_spearman) - abs(best_qpp_spearman)
            else:
                improvement = np.nan
            
            hypothesis4_results.append({
                'pattern': label,
                'high_count_low': low,
                'high_count_high': high,
                'n_samples': len(range_data),
                'integrated_spearman_rho': integrated_spearman,
                'integrated_spearman_p': integrated_p,
                'integrated_significant': integrated_p < 0.05 if not np.isnan(integrated_p) else False,
                'best_qpp_metric': best_qpp_metric,
                'best_qpp_spearman_rho': best_qpp_spearman if best_qpp_metric else np.nan,
                'best_qpp_spearman_p': best_qpp_spearman_p if best_qpp_metric else np.nan,
                'best_qpp_significant': best_qpp_spearman_p < 0.05 if best_qpp_metric else False,
                'improvement': improvement,
                'integrated_better': improvement > 0 if not np.isnan(improvement) else False
            })
    
    results['hypothesis4'] = pd.DataFrame(hypothesis4_results)
    
    # 仮説5（新規）: 統合モデルは予測タスク（曖昧性判定）で単体QPP指標より優れている
    if true_labels:
        print("\n[INFO] Hypothesis 5: Integrated model excels in prediction task (ambiguity classification)")
        
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
        
        # 真のラベルをデータフレームに追加
        df_with_labels = df_filtered.copy()
        df_with_labels['true_label'] = df_with_labels.apply(
            lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
        )
        df_with_labels = df_with_labels.dropna(subset=['true_label', 'integrated_prob'])
        
        if len(df_with_labels) > 10:
            # 統合モデルの性能
            integrated_pred = (df_with_labels['integrated_prob'] >= 0.5).astype(int)
            integrated_auc = roc_auc_score(df_with_labels['true_label'], df_with_labels['integrated_prob'])
            integrated_ap = average_precision_score(df_with_labels['true_label'], df_with_labels['integrated_prob'])
            integrated_f1 = f1_score(df_with_labels['true_label'], integrated_pred)
            integrated_acc = accuracy_score(df_with_labels['true_label'], integrated_pred)
            
            # 各単体QPP指標の性能（閾値は中央値）
            qpp_performances = []
            for metric in qpp_metrics:
                if metric in df_with_labels.columns:
                    metric_data = df_with_labels.dropna(subset=[metric, 'true_label'])
                    if len(metric_data) > 10:
                        # QPP指標を0-1に正規化（予測確率として扱う）
                        metric_min = metric_data[metric].min()
                        metric_max = metric_data[metric].max()
                        if metric_max > metric_min:
                            metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                        else:
                            metric_normalized = metric_data[metric] * 0  # すべて同じ値の場合
                        
                        metric_pred = (metric_normalized >= 0.5).astype(int)
                        try:
                            metric_auc = roc_auc_score(metric_data['true_label'], metric_normalized)
                            metric_ap = average_precision_score(metric_data['true_label'], metric_normalized)
                            metric_f1 = f1_score(metric_data['true_label'], metric_pred)
                            metric_acc = accuracy_score(metric_data['true_label'], metric_pred)
                            
                            qpp_performances.append({
                                'metric': metric,
                                'auc': metric_auc,
                                'ap': metric_ap,
                                'f1': metric_f1,
                                'accuracy': metric_acc
                            })
                        except:
                            pass
            
            if qpp_performances:
                qpp_perf_df = pd.DataFrame(qpp_performances)
                best_qpp_auc = qpp_perf_df.loc[qpp_perf_df['auc'].idxmax()]
                mean_qpp_auc = qpp_perf_df['auc'].mean()
                mean_qpp_ap = qpp_perf_df['ap'].mean()
                mean_qpp_f1 = qpp_perf_df['f1'].mean()
                mean_qpp_acc = qpp_perf_df['accuracy'].mean()
                
                hypothesis5_results = {
                    'integrated_auc': integrated_auc,
                    'integrated_ap': integrated_ap,
                    'integrated_f1': integrated_f1,
                    'integrated_accuracy': integrated_acc,
                    'best_qpp_metric': best_qpp_auc['metric'],
                    'best_qpp_auc': best_qpp_auc['auc'],
                    'best_qpp_ap': best_qpp_auc['ap'],
                    'best_qpp_f1': best_qpp_auc['f1'],
                    'best_qpp_accuracy': best_qpp_auc['accuracy'],
                    'mean_qpp_auc': mean_qpp_auc,
                    'mean_qpp_ap': mean_qpp_ap,
                    'mean_qpp_f1': mean_qpp_f1,
                    'mean_qpp_accuracy': mean_qpp_acc,
                    'auc_improvement_vs_best': integrated_auc - best_qpp_auc['auc'],
                    'auc_improvement_vs_mean': integrated_auc - mean_qpp_auc,
                    'ap_improvement_vs_best': integrated_ap - best_qpp_auc['ap'],
                    'ap_improvement_vs_mean': integrated_ap - mean_qpp_ap,
                    'f1_improvement_vs_best': integrated_f1 - best_qpp_auc['f1'],
                    'f1_improvement_vs_mean': integrated_f1 - mean_qpp_f1,
                    'integrated_better_auc': integrated_auc > best_qpp_auc['auc'],
                    'integrated_better_ap': integrated_ap > best_qpp_auc['ap'],
                    'integrated_better_f1': integrated_f1 > best_qpp_auc['f1']
                }
                
                results['hypothesis5'] = pd.DataFrame([hypothesis5_results])
    
    # 仮説6: 予測確率の範囲ごとに、統合モデルが単体QPP指標より優れている（予測タスクでの性能）
    if true_labels:
        print("\n[INFO] Hypothesis 6: Integrated model excels in specific prediction probability ranges")
        
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
        
        df_with_labels = df_filtered.copy()
        df_with_labels['true_label'] = df_with_labels.apply(
            lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
        )
        df_with_labels = df_with_labels.dropna(subset=['true_label', 'integrated_prob'])
        
        # 予測確率を範囲に分割
        prob_ranges = [
            (0.0, 0.4, 'Low (0.0-0.4)'),
            (0.4, 0.6, 'Medium (0.4-0.6)'),
            (0.6, 1.0, 'High (0.6-1.0)')
        ]
        
        hypothesis6_results = []
        for low, high, label in prob_ranges:
            range_data = df_with_labels[(df_with_labels['integrated_prob'] >= low) & (df_with_labels['integrated_prob'] < high)].copy()
            if len(range_data) > 10:
                # 統合モデルの性能
                integrated_pred = (range_data['integrated_prob'] >= 0.5).astype(int)
                try:
                    integrated_auc = roc_auc_score(range_data['true_label'], range_data['integrated_prob'])
                    integrated_ap = average_precision_score(range_data['true_label'], range_data['integrated_prob'])
                    integrated_f1 = f1_score(range_data['true_label'], integrated_pred)
                    integrated_acc = accuracy_score(range_data['true_label'], integrated_pred)
                except:
                    integrated_auc = integrated_ap = integrated_f1 = integrated_acc = np.nan
                
                # 最良の単体QPP指標の性能
                best_qpp_metric = None
                best_qpp_auc = -np.inf
                best_qpp_ap = -np.inf
                best_qpp_f1 = -np.inf
                
                for metric in qpp_metrics:
                    if metric in range_data.columns:
                        metric_data = range_data.dropna(subset=[metric, 'true_label'])
                        if len(metric_data) > 10:
                            metric_min = metric_data[metric].min()
                            metric_max = metric_data[metric].max()
                            if metric_max > metric_min:
                                metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                            else:
                                metric_normalized = metric_data[metric] * 0
                            
                            metric_pred = (metric_normalized >= 0.5).astype(int)
                            try:
                                metric_auc = roc_auc_score(metric_data['true_label'], metric_normalized)
                                metric_ap = average_precision_score(metric_data['true_label'], metric_normalized)
                                metric_f1 = f1_score(metric_data['true_label'], metric_pred)
                                
                                if metric_auc > best_qpp_auc:
                                    best_qpp_metric = metric
                                    best_qpp_auc = metric_auc
                                    best_qpp_ap = metric_ap
                                    best_qpp_f1 = metric_f1
                            except:
                                pass
                
                if not np.isnan(integrated_auc) and best_qpp_metric:
                    auc_improvement = integrated_auc - best_qpp_auc
                    ap_improvement = integrated_ap - best_qpp_ap
                    f1_improvement = integrated_f1 - best_qpp_f1
                    
                    hypothesis6_results.append({
                        'prob_range': label,
                        'prob_low': low,
                        'prob_high': high,
                        'n_samples': len(range_data),
                        'integrated_auc': integrated_auc,
                        'integrated_ap': integrated_ap,
                        'integrated_f1': integrated_f1,
                        'integrated_accuracy': integrated_acc,
                        'best_qpp_metric': best_qpp_metric,
                        'best_qpp_auc': best_qpp_auc,
                        'best_qpp_ap': best_qpp_ap,
                        'best_qpp_f1': best_qpp_f1,
                        'auc_improvement': auc_improvement,
                        'ap_improvement': ap_improvement,
                        'f1_improvement': f1_improvement,
                        'integrated_better_auc': auc_improvement > 0,
                        'integrated_better_ap': ap_improvement > 0,
                        'integrated_better_f1': f1_improvement > 0
                    })
        
        if hypothesis6_results:
            results['hypothesis6'] = pd.DataFrame(hypothesis6_results)
    
    # 仮説7: 統合モデルは「不確実性が高いケース」（予測確率が0.4-0.6）で、単体QPP指標より優れている
    if true_labels:
        print("\n[INFO] Hypothesis 7: Integrated model excels in uncertain cases (prediction probability 0.4-0.6)")
        
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
        
        df_with_labels = df_filtered.copy()
        df_with_labels['true_label'] = df_with_labels.apply(
            lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
        )
        df_with_labels = df_with_labels.dropna(subset=['true_label', 'integrated_prob'])
        
        # 不確実性が高いケース: 0.4-0.6
        uncertain = df_with_labels[(df_with_labels['integrated_prob'] >= 0.4) & (df_with_labels['integrated_prob'] < 0.6)].copy()
        # 確信度が高いケース: <0.4 または >=0.6
        certain = df_with_labels[~((df_with_labels['integrated_prob'] >= 0.4) & (df_with_labels['integrated_prob'] < 0.6))].copy()
        
        hypothesis7_results = []
        
        for case_type, case_data in [('Uncertain (0.4-0.6)', uncertain), ('Certain (<0.4 or >=0.6)', certain)]:
            if len(case_data) > 10:
                # 統合モデルの性能
                integrated_pred = (case_data['integrated_prob'] >= 0.5).astype(int)
                try:
                    integrated_auc = roc_auc_score(case_data['true_label'], case_data['integrated_prob'])
                    integrated_ap = average_precision_score(case_data['true_label'], case_data['integrated_prob'])
                    integrated_f1 = f1_score(case_data['true_label'], integrated_pred)
                    integrated_acc = accuracy_score(case_data['true_label'], integrated_pred)
                except:
                    integrated_auc = integrated_ap = integrated_f1 = integrated_acc = np.nan
                
                # 最良の単体QPP指標の性能
                best_qpp_metric = None
                best_qpp_auc = -np.inf
                best_qpp_ap = -np.inf
                best_qpp_f1 = -np.inf
                
                for metric in qpp_metrics:
                    if metric in case_data.columns:
                        metric_data = case_data.dropna(subset=[metric, 'true_label'])
                        if len(metric_data) > 10:
                            metric_min = metric_data[metric].min()
                            metric_max = metric_data[metric].max()
                            if metric_max > metric_min:
                                metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                            else:
                                metric_normalized = metric_data[metric] * 0
                            
                            metric_pred = (metric_normalized >= 0.5).astype(int)
                            try:
                                metric_auc = roc_auc_score(metric_data['true_label'], metric_normalized)
                                metric_ap = average_precision_score(metric_data['true_label'], metric_normalized)
                                metric_f1 = f1_score(metric_data['true_label'], metric_pred)
                                
                                if metric_auc > best_qpp_auc:
                                    best_qpp_metric = metric
                                    best_qpp_auc = metric_auc
                                    best_qpp_ap = metric_ap
                                    best_qpp_f1 = metric_f1
                            except:
                                pass
                
                if not np.isnan(integrated_auc) and best_qpp_metric:
                    auc_improvement = integrated_auc - best_qpp_auc
                    ap_improvement = integrated_ap - best_qpp_ap
                    f1_improvement = integrated_f1 - best_qpp_f1
                    
                    hypothesis7_results.append({
                        'case_type': case_type,
                        'n_samples': len(case_data),
                        'integrated_auc': integrated_auc,
                        'integrated_ap': integrated_ap,
                        'integrated_f1': integrated_f1,
                        'integrated_accuracy': integrated_acc,
                        'best_qpp_metric': best_qpp_metric,
                        'best_qpp_auc': best_qpp_auc,
                        'best_qpp_ap': best_qpp_ap,
                        'best_qpp_f1': best_qpp_f1,
                        'auc_improvement': auc_improvement,
                        'ap_improvement': ap_improvement,
                        'f1_improvement': f1_improvement,
                        'integrated_better_auc': auc_improvement > 0,
                        'integrated_better_ap': ap_improvement > 0,
                        'integrated_better_f1': f1_improvement > 0
                    })
        
        if hypothesis7_results:
            results['hypothesis7'] = pd.DataFrame(hypothesis7_results)
    
    # 仮説8: 統合モデルは、複数のQPP指標が「極端な値」（非常に高い/低い）を示す場合に優れている
    if true_labels:
        print("\n[INFO] Hypothesis 8: Integrated model excels when QPP metrics show extreme values")
        
        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
        
        df_with_labels = df_filtered.copy()
        df_with_labels['true_label'] = df_with_labels.apply(
            lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
        )
        df_with_labels = df_with_labels.dropna(subset=['true_label', 'integrated_prob'])
        
        # 各QPP指標の極端な値をカウント（上位25%または下位25%）
        extreme_counts = []
        for idx, row in df_with_labels.iterrows():
            count = 0
            for metric in qpp_metrics:
                if metric in df_with_labels.columns:
                    metric_val = row[metric]
                    if not pd.isna(metric_val):
                        q25 = df_with_labels[metric].quantile(0.25)
                        q75 = df_with_labels[metric].quantile(0.75)
                        if metric_val <= q25 or metric_val >= q75:
                            count += 1
            extreme_counts.append(count)
        
        df_with_labels['extreme_qpp_count'] = extreme_counts
        
        # 極端な値の数でグループ化
        extreme_ranges = [
            (0, 3, 'Few Extreme (0-3)'),
            (3, 7, 'Medium Extreme (3-7)'),
            (7, 11, 'Many Extreme (7-10)')
        ]
        
        hypothesis8_results = []
        for low, high, label in extreme_ranges:
            range_data = df_with_labels[(df_with_labels['extreme_qpp_count'] >= low) & (df_with_labels['extreme_qpp_count'] < high)].copy()
            if len(range_data) > 10:
                # 統合モデルの性能
                integrated_pred = (range_data['integrated_prob'] >= 0.5).astype(int)
                try:
                    integrated_auc = roc_auc_score(range_data['true_label'], range_data['integrated_prob'])
                    integrated_ap = average_precision_score(range_data['true_label'], range_data['integrated_prob'])
                    integrated_f1 = f1_score(range_data['true_label'], integrated_pred)
                    integrated_acc = accuracy_score(range_data['true_label'], integrated_pred)
                except:
                    integrated_auc = integrated_ap = integrated_f1 = integrated_acc = np.nan
                
                # 最良の単体QPP指標の性能
                best_qpp_metric = None
                best_qpp_auc = -np.inf
                best_qpp_ap = -np.inf
                best_qpp_f1 = -np.inf
                
                for metric in qpp_metrics:
                    if metric in range_data.columns:
                        metric_data = range_data.dropna(subset=[metric, 'true_label'])
                        if len(metric_data) > 10:
                            metric_min = metric_data[metric].min()
                            metric_max = metric_data[metric].max()
                            if metric_max > metric_min:
                                metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                            else:
                                metric_normalized = metric_data[metric] * 0
                            
                            metric_pred = (metric_normalized >= 0.5).astype(int)
                            try:
                                metric_auc = roc_auc_score(metric_data['true_label'], metric_normalized)
                                metric_ap = average_precision_score(metric_data['true_label'], metric_normalized)
                                metric_f1 = f1_score(metric_data['true_label'], metric_pred)
                                
                                if metric_auc > best_qpp_auc:
                                    best_qpp_metric = metric
                                    best_qpp_auc = metric_auc
                                    best_qpp_ap = metric_ap
                                    best_qpp_f1 = metric_f1
                            except:
                                pass
                
                if not np.isnan(integrated_auc) and best_qpp_metric:
                    auc_improvement = integrated_auc - best_qpp_auc
                    ap_improvement = integrated_ap - best_qpp_ap
                    f1_improvement = integrated_f1 - best_qpp_f1
                    
                    hypothesis8_results.append({
                        'extreme_range': label,
                        'extreme_low': low,
                        'extreme_high': high,
                        'n_samples': len(range_data),
                        'integrated_auc': integrated_auc,
                        'integrated_ap': integrated_ap,
                        'integrated_f1': integrated_f1,
                        'integrated_accuracy': integrated_acc,
                        'best_qpp_metric': best_qpp_metric,
                        'best_qpp_auc': best_qpp_auc,
                        'best_qpp_ap': best_qpp_ap,
                        'best_qpp_f1': best_qpp_f1,
                        'auc_improvement': auc_improvement,
                        'ap_improvement': ap_improvement,
                        'f1_improvement': f1_improvement,
                        'integrated_better_auc': auc_improvement > 0,
                        'integrated_better_ap': ap_improvement > 0,
                        'integrated_better_f1': f1_improvement > 0
                    })
        
        if hypothesis8_results:
            results['hypothesis8'] = pd.DataFrame(hypothesis8_results)
    
    # 仮説9: 統合モデルは、全体で統計的有意に優れている（DeLong検定）
    if true_labels:
        print("\n[INFO] Hypothesis 9: Integrated model is statistically significantly better overall (DeLong test)")
        
        from sklearn.metrics import roc_auc_score
        
        df_with_labels = df_filtered.copy()
        df_with_labels['true_label'] = df_with_labels.apply(
            lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
        )
        df_with_labels = df_with_labels.dropna(subset=['true_label', 'integrated_prob'])
        
        if len(df_with_labels) > 20:
            # 統合モデルのAUC
            integrated_auc = roc_auc_score(df_with_labels['true_label'], df_with_labels['integrated_prob'])
            
            # 最良の単体QPP指標を探す
            best_qpp_metric = None
            best_qpp_auc = -np.inf
            best_qpp_normalized = None
            
            for metric in qpp_metrics:
                if metric in df_with_labels.columns:
                    metric_data = df_with_labels.dropna(subset=[metric, 'true_label'])
                    if len(metric_data) > 20:
                        metric_min = metric_data[metric].min()
                        metric_max = metric_data[metric].max()
                        if metric_max > metric_min:
                            metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                            try:
                                metric_auc = roc_auc_score(metric_data['true_label'], metric_normalized)
                                if metric_auc > best_qpp_auc:
                                    best_qpp_metric = metric
                                    best_qpp_auc = metric_auc
                                    best_qpp_normalized = metric_normalized.values
                            except:
                                pass
            
            if best_qpp_metric and best_qpp_normalized is not None:
                # DeLong検定（全体のデータで）
                try:
                    delong_result = delong_test(
                        y_true=df_with_labels['true_label'].values,
                        y_pred_proba_a=df_with_labels['integrated_prob'].values,
                        y_pred_proba_b=best_qpp_normalized
                    )
                    
                    hypothesis9_results = {
                        'n_samples': len(df_with_labels),
                        'integrated_auc': integrated_auc,
                        'best_qpp_metric': best_qpp_metric,
                        'best_qpp_auc': best_qpp_auc,
                        'auc_improvement': integrated_auc - best_qpp_auc,
                        'delong_z_stat': delong_result.get('z_stat', np.nan),
                        'delong_p_value': delong_result.get('p_value', np.nan),
                        'delong_significant': delong_result.get('significant', False),
                        'integrated_better': integrated_auc > best_qpp_auc,
                        'integrated_significantly_better': (integrated_auc > best_qpp_auc) and delong_result.get('significant', False)
                    }
                    
                    results['hypothesis9'] = pd.DataFrame([hypothesis9_results])
                except Exception as e:
                    print(f"[WARN] Hypothesis 9 DeLong test failed: {e}")
    
    # 仮説10: 統合モデルは、QPP指標の分散が大きい場合に優れている（複数の情報源を統合する利点）
    if true_labels:
        print("\n[INFO] Hypothesis 10: Integrated model excels when QPP metrics have high variance")
        
        from sklearn.metrics import roc_auc_score
        
        df_with_labels = df_filtered.copy()
        df_with_labels['true_label'] = df_with_labels.apply(
            lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
        )
        df_with_labels = df_with_labels.dropna(subset=['true_label', 'integrated_prob'])
        
        # 各サンプルについて、QPP指標の分散を計算
        qpp_vars = []
        for idx, row in df_with_labels.iterrows():
            qpp_values = []
            for metric in qpp_metrics:
                if metric in df_with_labels.columns:
                    val = row[metric]
                    if not pd.isna(val):
                        qpp_values.append(val)
            if len(qpp_values) > 1:
                qpp_vars.append(np.var(qpp_values))
            else:
                qpp_vars.append(np.nan)
        
        df_with_labels['qpp_variance'] = qpp_vars
        
        # 分散でグループ化（低、中、高）
        var_ranges = [
            (0, np.percentile([v for v in qpp_vars if not np.isnan(v)], 33), 'Low Variance'),
            (np.percentile([v for v in qpp_vars if not np.isnan(v)], 33), 
             np.percentile([v for v in qpp_vars if not np.isnan(v)], 67), 'Medium Variance'),
            (np.percentile([v for v in qpp_vars if not np.isnan(v)], 67), 
             np.max([v for v in qpp_vars if not np.isnan(v)]), 'High Variance')
        ]
        
        hypothesis10_results = []
        for low, high, label in var_ranges:
            range_data = df_with_labels[
                (df_with_labels['qpp_variance'] >= low) & 
                (df_with_labels['qpp_variance'] < high)
            ].copy()
            if len(range_data) > 20:
                # 統合モデルの性能
                try:
                    integrated_auc = roc_auc_score(range_data['true_label'], range_data['integrated_prob'])
                except:
                    integrated_auc = np.nan
                
                # 最良の単体QPP指標の性能
                best_qpp_metric = None
                best_qpp_auc = -np.inf
                best_qpp_proba = None
                
                for metric in qpp_metrics:
                    if metric in range_data.columns:
                        metric_data = range_data.dropna(subset=[metric, 'true_label'])
                        if len(metric_data) > 20:
                            metric_min = metric_data[metric].min()
                            metric_max = metric_data[metric].max()
                            if metric_max > metric_min:
                                metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                                try:
                                    metric_auc = roc_auc_score(metric_data['true_label'], metric_normalized)
                                    if metric_auc > best_qpp_auc:
                                        best_qpp_metric = metric
                                        best_qpp_auc = metric_auc
                                        best_qpp_proba = metric_normalized.values
                                except:
                                    pass
                
                if not np.isnan(integrated_auc) and best_qpp_metric and best_qpp_proba is not None:
                    # DeLong検定
                    try:
                        delong_result = delong_test(
                            y_true=range_data['true_label'].values,
                            y_pred_proba_a=range_data['integrated_prob'].values,
                            y_pred_proba_b=best_qpp_proba
                        )
                    except:
                        delong_result = {'z_stat': np.nan, 'p_value': np.nan, 'significant': False}
                    
                    hypothesis10_results.append({
                        'variance_range': label,
                        'variance_low': low,
                        'variance_high': high,
                        'n_samples': len(range_data),
                        'integrated_auc': integrated_auc,
                        'best_qpp_metric': best_qpp_metric,
                        'best_qpp_auc': best_qpp_auc,
                        'auc_improvement': integrated_auc - best_qpp_auc,
                        'delong_z_stat': delong_result.get('z_stat', np.nan),
                        'delong_p_value': delong_result.get('p_value', np.nan),
                        'delong_significant': delong_result.get('significant', False),
                        'integrated_better': integrated_auc > best_qpp_auc,
                        'integrated_significantly_better': (integrated_auc > best_qpp_auc) and delong_result.get('significant', False)
                    })
        
        if hypothesis10_results:
            results['hypothesis10'] = pd.DataFrame(hypothesis10_results)
    
    # 仮説11: 統合モデルは、予測確率の分布の分離度が高い（正例と負例の分布がより分離されている）
    # 閾値に依存せず、分布の形状を直接評価
    if true_labels:
        print("\n[INFO] Hypothesis 11: Integrated model has better separation between positive and negative distributions")
        
        df_with_labels = df_filtered.copy()
        df_with_labels['true_label'] = df_with_labels.apply(
            lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
        )
        df_with_labels = df_with_labels.dropna(subset=['true_label', 'integrated_prob'])
        
        if len(df_with_labels) > 20:
            # 統合モデルの分布分離度
            pos_proba = df_with_labels[df_with_labels['true_label'] == 1]['integrated_prob'].values
            neg_proba = df_with_labels[df_with_labels['true_label'] == 0]['integrated_prob'].values
            
            # 分離度の指標: 正例の平均 - 負例の平均（大きいほど良い）
            integrated_separation = pos_proba.mean() - neg_proba.mean()
            # 重複度: 分布の重なり（小さいほど良い）- Kolmogorov-Smirnov統計量
            from scipy.stats import ks_2samp
            integrated_ks_stat, _ = ks_2samp(pos_proba, neg_proba)
            integrated_overlap = 1 - integrated_ks_stat  # 1に近いほど重なりが少ない
            
            # 最良の単体QPP指標の分布分離度
            best_qpp_metric = None
            best_qpp_separation = -np.inf
            best_qpp_overlap = 1.0
            
            for metric in qpp_metrics:
                if metric in df_with_labels.columns:
                    metric_data = df_with_labels.dropna(subset=[metric, 'true_label'])
                    if len(metric_data) > 20:
                        metric_min = metric_data[metric].min()
                        metric_max = metric_data[metric].max()
                        if metric_max > metric_min:
                            metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                            pos_metric = metric_normalized[metric_data['true_label'] == 1].values
                            neg_metric = metric_normalized[metric_data['true_label'] == 0].values
                            
                            if len(pos_metric) > 0 and len(neg_metric) > 0:
                                metric_separation = pos_metric.mean() - neg_metric.mean()
                                metric_ks_stat, _ = ks_2samp(pos_metric, neg_metric)
                                metric_overlap = 1 - metric_ks_stat
                                
                                # 分離度が高いものを選ぶ
                                if metric_separation > best_qpp_separation:
                                    best_qpp_metric = metric
                                    best_qpp_separation = metric_separation
                                    best_qpp_overlap = metric_overlap
            
            if best_qpp_metric:
                hypothesis11_results = {
                    'n_samples': len(df_with_labels),
                    'integrated_separation': integrated_separation,
                    'integrated_overlap': integrated_overlap,
                    'best_qpp_metric': best_qpp_metric,
                    'best_qpp_separation': best_qpp_separation,
                    'best_qpp_overlap': best_qpp_overlap,
                    'separation_improvement': integrated_separation - best_qpp_separation,
                    'overlap_improvement': best_qpp_overlap - integrated_overlap,  # 統合モデルの重なりが小さいほど良い
                    'integrated_better_separation': integrated_separation > best_qpp_separation,
                    'integrated_better_overlap': integrated_overlap < best_qpp_overlap  # 重なりが小さいほど良い
                }
                
                results['hypothesis11'] = pd.DataFrame([hypothesis11_results])
    
    # 結果を保存
    for name, df_result in results.items():
        if len(df_result) > 0:
            df_result.to_csv(output_dir / f'{name}_results.csv', index=False)
            print(f"[INFO] Saved {name} results to {output_dir / f'{name}_results.csv'}")
    
    # 可視化
    create_hypothesis_visualizations(df_filtered, results, output_dir, qpp_metrics, integrated_name, true_labels)
    
    # 仮説12: エラー分析と成功パターン（仮説検証形式）
    if true_labels:
        hypothesis12_results = create_error_and_success_pattern_analysis(
            df_filtered, qpp_metrics, integrated_name, true_labels, output_dir
        )
        if hypothesis12_results:
            results['hypothesis12'] = pd.DataFrame(hypothesis12_results)
    
    # 仮説13: 1クエリ1プロット分析（個々のクエリの特徴を可視化・仮説検証）
    if true_labels:
        hypothesis13_results = create_individual_query_analysis(
            df_filtered, qpp_metrics, integrated_name, true_labels, output_dir
        )
        if hypothesis13_results:
            results['hypothesis13'] = pd.DataFrame(hypothesis13_results)
    
    # サマリーを表示
    print_hypothesis_summary(results)


def create_hypothesis_visualizations(
    df: pd.DataFrame,
    results: Dict[str, pd.DataFrame],
    output_dir: Path,
    qpp_metrics: List[str],
    integrated_name: str,
    true_labels: Dict[Tuple[str, int], int] = None
):
    """統合モデルの優位性を示す可視化"""
    
    # 仮説の数に応じてグリッドサイズを調整
    n_hypotheses = sum(1 for name in ['hypothesis1', 'hypothesis2', 'hypothesis3', 'hypothesis4', 'hypothesis5', 'hypothesis6', 'hypothesis7', 'hypothesis8'] 
                       if name in results and len(results[name]) > 0)
    
    if n_hypotheses <= 4:
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(20, 28))
        gs = fig.add_gridspec(7, 2, hspace=0.3, wspace=0.3)
    
    # 仮説1: QPP指標が矛盾する場合、統合モデルが優れている
    if 'hypothesis1' in results and len(results['hypothesis1']) > 0:
        ax1 = fig.add_subplot(gs[0, 0])
        h1_df = results['hypothesis1']
        x = np.arange(len(h1_df))
        width = 0.35
        
        colors_integrated = ['green' if better else 'red' for better in h1_df['integrated_better']]
        colors_single = ['orange' if sig else 'lightcoral' for sig in h1_df['best_qpp_significant']]
        
        bars1 = ax1.bar(x - width/2, h1_df['integrated_spearman_rho'], width,
                       label='Integrated Model', alpha=0.8, color=colors_integrated, 
                       edgecolor='black', linewidth=1)
        bars2 = ax1.bar(x + width/2, h1_df['best_qpp_spearman_rho'], width,
                       label='Best Single QPP', alpha=0.8, color=colors_single,
                       edgecolor='black', linewidth=1)
        
        for i, (idx, row) in enumerate(h1_df.iterrows()):
            improvement_text = f"Δ={row['improvement']:+.3f}" if not np.isnan(row['improvement']) else "N/A"
            max_val = max(abs(row['integrated_spearman_rho']) if not np.isnan(row['integrated_spearman_rho']) else 0,
                         abs(row['best_qpp_spearman_rho']) if not np.isnan(row['best_qpp_spearman_rho']) else 0)
            ax1.text(i, max_val + 0.05, improvement_text, ha='center', fontsize=9, fontweight='bold',
                    color='green' if row['integrated_better'] else 'red')
        
        ax1.set_xlabel('QPP Inconsistency Range', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Spearman ρ with QA Pairs', fontsize=11, fontweight='bold')
        ax1.set_title('Hypothesis 1: Integrated Model Excels When QPP Metrics Are Inconsistent\n(Green = Integrated Better)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(h1_df['inconsistency_range'], rotation=45, ha='right')
        ax1.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # 仮説2: 特定のQAペア数範囲で統合モデルが優れている
    if 'hypothesis2' in results and len(results['hypothesis2']) > 0:
        ax2 = fig.add_subplot(gs[0, 1])
        h2_df = results['hypothesis2']
        x = np.arange(len(h2_df))
        width = 0.35
        
        colors_integrated = ['green' if better else 'red' for better in h2_df['integrated_better']]
        colors_single = ['orange' if sig else 'lightcoral' for sig in h2_df['best_qpp_significant']]
        
        bars1 = ax2.bar(x - width/2, h2_df['integrated_spearman_rho'], width,
                       label='Integrated Model', alpha=0.8, color=colors_integrated, 
                       edgecolor='black', linewidth=1)
        bars2 = ax2.bar(x + width/2, h2_df['best_qpp_spearman_rho'], width,
                       label='Best Single QPP', alpha=0.8, color=colors_single,
                       edgecolor='black', linewidth=1)
        
        for i, (idx, row) in enumerate(h2_df.iterrows()):
            improvement_text = f"Δ={row['improvement']:+.3f}" if not np.isnan(row['improvement']) else "N/A"
            max_val = max(abs(row['integrated_spearman_rho']) if not np.isnan(row['integrated_spearman_rho']) else 0,
                         abs(row['best_qpp_spearman_rho']) if not np.isnan(row['best_qpp_spearman_rho']) else 0)
            ax2.text(i, max_val + 0.05, improvement_text, ha='center', fontsize=9, fontweight='bold',
                    color='green' if row['integrated_better'] else 'red')
        
        ax2.set_xlabel('QA Pairs Count Range', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Spearman ρ with QA Pairs', fontsize=11, fontweight='bold')
        ax2.set_title('Hypothesis 2: Integrated Model Excels in Specific QA Pairs Ranges\n(Green = Integrated Better)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(h2_df['qa_range'], rotation=45, ha='right')
        ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
    
    # 仮説3: 統合モデルが統計的有意性を達成するケースが多い
    if 'hypothesis3' in results and len(results['hypothesis3']) > 0:
        ax3 = fig.add_subplot(gs[1, :])
        h3_df = results['hypothesis3']
        x = np.arange(len(h3_df))
        width = 0.35
        
        colors_integrated = ['green' if sig else 'gray' for sig in h3_df['integrated_significant']]
        colors_single = ['orange' if sig else 'lightcoral' for sig in h3_df['best_qpp_significant']]
        
        bars1 = ax3.bar(x - width/2, h3_df['integrated_spearman_rho'], width,
                       label='Integrated Model', alpha=0.8, color=colors_integrated, 
                       edgecolor='black', linewidth=1)
        bars2 = ax3.bar(x + width/2, h3_df['best_qpp_spearman_rho'], width,
                       label='Best Single QPP', alpha=0.8, color=colors_single,
                       edgecolor='black', linewidth=1)
        
        # サマリー情報を表示
        if 'hypothesis3_summary' in results and len(results['hypothesis3_summary']) > 0:
            summary = results['hypothesis3_summary'].iloc[0]
            summary_text = f"Overall: Integrated={summary['integrated_all_significant']}, " \
                          f"QPP={summary['qpp_significant_ratio']:.1%} significant\n" \
                          f"Subgroups: Integrated={summary['integrated_subgroups_ratio']:.1%}, " \
                          f"QPP={summary['qpp_subgroups_ratio']:.1%} significant"
            ax3.text(0.5, 0.95, summary_text, transform=ax3.transAxes, 
                    ha='center', va='top', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        for i, (idx, row) in enumerate(h3_df.iterrows()):
            int_mark = "*" if row['integrated_significant'] else ""
            qpp_mark = "*" if row['best_qpp_significant'] else ""
            ax3.text(i - width/2, row['integrated_spearman_rho'] + (0.02 if row['integrated_spearman_rho'] >= 0 else -0.02),
                    f"p={row['integrated_spearman_p']:.3f}{int_mark}", ha='center', fontsize=7, fontweight='bold',
                    color='green' if row['integrated_significant'] else 'gray')
            if not pd.isna(row['best_qpp_spearman_rho']):
                ax3.text(i + width/2, row['best_qpp_spearman_rho'] + (0.02 if row['best_qpp_spearman_rho'] >= 0 else -0.02),
                        f"p={row['best_qpp_spearman_p']:.3f}{qpp_mark}", ha='center', fontsize=7, fontweight='bold',
                        color='orange' if row['best_qpp_significant'] else 'gray')
        
        ax3.set_xlabel('QA Pairs Category', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Spearman ρ with QA Pairs', fontsize=11, fontweight='bold')
        ax3.set_title('Hypothesis 3: Integrated Model Achieves Statistical Significance More Often\n(Green/Orange = significant, * = p < 0.05)', 
                     fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(h3_df['category'], rotation=45, ha='right')
        ax3.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
    
    # 仮説4: QPP指標の組み合わせパターンに対して統合モデルが優れている
    if 'hypothesis4' in results and len(results['hypothesis4']) > 0:
        ax4 = fig.add_subplot(gs[2, :])
        h4_df = results['hypothesis4']
        x = np.arange(len(h4_df))
        width = 0.35
        
        colors_integrated = ['green' if better else 'red' for better in h4_df['integrated_better']]
        colors_single = ['orange' if sig else 'lightcoral' for sig in h4_df['best_qpp_significant']]
        
        bars1 = ax4.bar(x - width/2, h4_df['integrated_spearman_rho'], width,
                       label='Integrated Model', alpha=0.8, color=colors_integrated, 
                       edgecolor='black', linewidth=1)
        bars2 = ax4.bar(x + width/2, h4_df['best_qpp_spearman_rho'], width,
                       label='Best Single QPP', alpha=0.8, color=colors_single,
                       edgecolor='black', linewidth=1)
        
        for i, (idx, row) in enumerate(h4_df.iterrows()):
            improvement_text = f"Δ={row['improvement']:+.3f}" if not np.isnan(row['improvement']) else "N/A"
            max_val = max(abs(row['integrated_spearman_rho']) if not np.isnan(row['integrated_spearman_rho']) else 0,
                         abs(row['best_qpp_spearman_rho']) if not np.isnan(row['best_qpp_spearman_rho']) else 0)
            ax4.text(i, max_val + 0.05, improvement_text, ha='center', fontsize=9, fontweight='bold',
                    color='green' if row['integrated_better'] else 'red')
        
        ax4.set_xlabel('QPP Metric Combination Pattern', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Spearman ρ with QA Pairs', fontsize=11, fontweight='bold')
        ax4.set_title('Hypothesis 4: Integrated Model Responds Better to QPP Combination Patterns\n(Green = Integrated Better)', 
                     fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(h4_df['pattern'], rotation=45, ha='right')
        ax4.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    
    # 仮説5: 予測タスクでの性能比較
    if 'hypothesis5' in results and len(results['hypothesis5']) > 0:
        ax5 = fig.add_subplot(gs[3, :])
        h5_df = results['hypothesis5'].iloc[0]
        
        metrics = ['AUC', 'AP', 'F1', 'Accuracy']
        integrated_values = [
            h5_df['integrated_auc'],
            h5_df['integrated_ap'],
            h5_df['integrated_f1'],
            h5_df['integrated_accuracy']
        ]
        best_qpp_values = [
            h5_df['best_qpp_auc'],
            h5_df['best_qpp_ap'],
            h5_df['best_qpp_f1'],
            h5_df['best_qpp_accuracy']
        ]
        mean_qpp_values = [
            h5_df['mean_qpp_auc'],
            h5_df['mean_qpp_ap'],
            h5_df['mean_qpp_f1'],
            h5_df['mean_qpp_accuracy']
        ]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        colors_integrated = ['green' if h5_df['integrated_better_auc'] or h5_df['integrated_better_ap'] or h5_df['integrated_better_f1'] else 'red'] * len(metrics)
        
        bars1 = ax5.bar(x - width, integrated_values, width,
                       label='Integrated Model', alpha=0.8, color='green', 
                       edgecolor='black', linewidth=1)
        bars2 = ax5.bar(x, best_qpp_values, width,
                       label=f"Best Single QPP ({h5_df['best_qpp_metric']})", alpha=0.8, color='orange',
                       edgecolor='black', linewidth=1)
        bars3 = ax5.bar(x + width, mean_qpp_values, width,
                       label='Mean Single QPP', alpha=0.8, color='steelblue',
                       edgecolor='black', linewidth=1)
        
        # 改善度を表示
        improvements = [
            h5_df['auc_improvement_vs_best'],
            h5_df['ap_improvement_vs_best'],
            h5_df['f1_improvement_vs_best'],
            h5_df['integrated_accuracy'] - h5_df['best_qpp_accuracy']
        ]
        for i, (val, imp) in enumerate(zip(integrated_values, improvements)):
            ax5.text(i - width, val + 0.01, f"{imp:+.4f}", ha='center', fontsize=9, fontweight='bold',
                    color='green' if imp > 0 else 'red')
        
        ax5.set_xlabel('Metric', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax5.set_title('Hypothesis 5: Integrated Model Performance in Prediction Task\n(Green = Better than Best Single QPP)', 
                     fontsize=12, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics)
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_ylim([0, 1.1])
    
    # 仮説6: 予測確率の範囲ごとの性能比較
    if 'hypothesis6' in results and len(results['hypothesis6']) > 0:
        ax6 = fig.add_subplot(gs[4, 0])
        h6_df = results['hypothesis6']
        x = np.arange(len(h6_df))
        width = 0.25
        
        integrated_aucs = h6_df['integrated_auc'].values
        best_qpp_aucs = h6_df['best_qpp_auc'].values
        colors = ['green' if better else 'red' for better in h6_df['integrated_better_auc']]
        
        bars1 = ax6.bar(x - width, integrated_aucs, width, label='Integrated Model', 
                       alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        bars2 = ax6.bar(x, best_qpp_aucs, width, label='Best Single QPP', 
                       alpha=0.8, color='orange', edgecolor='black', linewidth=1)
        
        for i, (idx, row) in enumerate(h6_df.iterrows()):
            ax6.text(i - width, row['integrated_auc'] + 0.01, f"{row['auc_improvement']:+.3f}", 
                    ha='center', fontsize=8, fontweight='bold',
                    color='green' if row['integrated_better_auc'] else 'red')
        
        ax6.set_xlabel('Prediction Probability Range', fontsize=11, fontweight='bold')
        ax6.set_ylabel('AUC', fontsize=11, fontweight='bold')
        ax6.set_title('Hypothesis 6: Performance by Prediction Probability Range', 
                     fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(h6_df['prob_range'], rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim([0, 1.1])
    
    # 仮説7: 不確実性が高いケースでの性能比較
    if 'hypothesis7' in results and len(results['hypothesis7']) > 0:
        ax7 = fig.add_subplot(gs[4, 1])
        h7_df = results['hypothesis7']
        x = np.arange(len(h7_df))
        width = 0.25
        
        integrated_aucs = h7_df['integrated_auc'].values
        best_qpp_aucs = h7_df['best_qpp_auc'].values
        colors = ['green' if better else 'red' for better in h7_df['integrated_better_auc']]
        
        bars1 = ax7.bar(x - width, integrated_aucs, width, label='Integrated Model', 
                       alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        bars2 = ax7.bar(x, best_qpp_aucs, width, label='Best Single QPP', 
                       alpha=0.8, color='orange', edgecolor='black', linewidth=1)
        
        for i, (idx, row) in enumerate(h7_df.iterrows()):
            ax7.text(i - width, row['integrated_auc'] + 0.01, f"{row['auc_improvement']:+.3f}", 
                    ha='center', fontsize=8, fontweight='bold',
                    color='green' if row['integrated_better_auc'] else 'red')
        
        ax7.set_xlabel('Case Type', fontsize=11, fontweight='bold')
        ax7.set_ylabel('AUC', fontsize=11, fontweight='bold')
        ax7.set_title('Hypothesis 7: Performance in Uncertain vs Certain Cases', 
                     fontsize=12, fontweight='bold')
        ax7.set_xticks(x)
        ax7.set_xticklabels(h7_df['case_type'], rotation=45, ha='right')
        ax7.legend()
        ax7.grid(True, alpha=0.3, axis='y')
        ax7.set_ylim([0, 1.1])
    
    # 仮説8: 極端なQPP値の数による性能比較
    if 'hypothesis8' in results and len(results['hypothesis8']) > 0:
        ax8 = fig.add_subplot(gs[5, :])
        h8_df = results['hypothesis8']
        x = np.arange(len(h8_df))
        width = 0.25
        
        integrated_aucs = h8_df['integrated_auc'].values
        best_qpp_aucs = h8_df['best_qpp_auc'].values
        colors = ['green' if better else 'red' for better in h8_df['integrated_better_auc']]
        
        bars1 = ax8.bar(x - width, integrated_aucs, width, label='Integrated Model', 
                       alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        bars2 = ax8.bar(x, best_qpp_aucs, width, label='Best Single QPP', 
                       alpha=0.8, color='orange', edgecolor='black', linewidth=1)
        
        for i, (idx, row) in enumerate(h8_df.iterrows()):
            ax8.text(i - width, row['integrated_auc'] + 0.01, f"{row['auc_improvement']:+.3f}", 
                    ha='center', fontsize=8, fontweight='bold',
                    color='green' if row['integrated_better_auc'] else 'red')
        
        ax8.set_xlabel('Number of Extreme QPP Values', fontsize=11, fontweight='bold')
        ax8.set_ylabel('AUC', fontsize=11, fontweight='bold')
        ax8.set_title('Hypothesis 8: Performance by Number of Extreme QPP Values', 
                     fontsize=12, fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels(h8_df['extreme_range'], rotation=45, ha='right')
        ax8.legend()
        ax8.grid(True, alpha=0.3, axis='y')
        ax8.set_ylim([0, 1.1])
    
    # 仮説9: 全体での統計的有意性（DeLong検定）
    if 'hypothesis9' in results and len(results['hypothesis9']) > 0:
        ax9 = fig.add_subplot(gs[6, 0])
        h9_df = results['hypothesis9'].iloc[0]
        
        metrics = ['AUC']
        integrated_values = [h9_df['integrated_auc']]
        best_qpp_values = [h9_df['best_qpp_auc']]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        color = 'darkgreen' if h9_df['delong_significant'] else ('green' if h9_df['integrated_better'] else 'red')
        
        bars1 = ax9.bar(x - width, integrated_values, width, label='Integrated Model', 
                       alpha=0.8, color=color, edgecolor='black', linewidth=1)
        bars2 = ax9.bar(x, best_qpp_values, width, label=f"Best Single QPP ({h9_df['best_qpp_metric']})", 
                       alpha=0.8, color='orange', edgecolor='black', linewidth=1)
        
        sig_mark = "*" if h9_df['delong_significant'] else ""
        ax9.text(0 - width, h9_df['integrated_auc'] + 0.01, 
                f"{h9_df['auc_improvement']:+.4f}{sig_mark}", 
                ha='center', fontsize=10, fontweight='bold',
                color='darkgreen' if h9_df['delong_significant'] else 
                      ('green' if h9_df['integrated_better'] else 'red'))
        if h9_df['delong_significant']:
            ax9.text(0 - width, h9_df['integrated_auc'] + 0.05, 
                    f"p={h9_df['delong_p_value']:.4f}", 
                    ha='center', fontsize=9, fontweight='bold')
        
        ax9.set_xlabel('Metric', fontsize=11, fontweight='bold')
        ax9.set_ylabel('AUC', fontsize=11, fontweight='bold')
        ax9.set_title('Hypothesis 9: Overall Statistical Significance (DeLong Test)\n(Dark Green = Statistically Significant)', 
                     fontsize=12, fontweight='bold')
        ax9.set_xticks(x)
        ax9.set_xticklabels(metrics)
        ax9.legend()
        ax9.grid(True, alpha=0.3, axis='y')
        ax9.set_ylim([0, 1.1])
    
    # 仮説10: QPP指標の分散による性能比較（統計的有意性を考慮）
    if 'hypothesis10' in results and len(results['hypothesis10']) > 0:
        ax10 = fig.add_subplot(gs[6, 1])
        h10_df = results['hypothesis10']
        x = np.arange(len(h10_df))
        width = 0.25
        
        integrated_aucs = h10_df['integrated_auc'].values
        best_qpp_aucs = h10_df['best_qpp_auc'].values
        colors = ['darkgreen' if sig else ('green' if better else 'red') 
                 for better, sig in zip(h10_df['integrated_better'], h10_df['delong_significant'])]
        
        bars1 = ax10.bar(x - width, integrated_aucs, width, label='Integrated Model', 
                        alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        bars2 = ax10.bar(x, best_qpp_aucs, width, label='Best Single QPP', 
                        alpha=0.8, color='orange', edgecolor='black', linewidth=1)
        
        for i, (idx, row) in enumerate(h10_df.iterrows()):
            sig_mark = "*" if row['delong_significant'] else ""
            ax10.text(i - width, row['integrated_auc'] + 0.01, 
                     f"{row['auc_improvement']:+.3f}{sig_mark}", 
                     ha='center', fontsize=8, fontweight='bold',
                     color='darkgreen' if row['delong_significant'] else 
                           ('green' if row['integrated_better'] else 'red'))
            if row['delong_significant']:
                ax10.text(i - width, row['integrated_auc'] + 0.05, 
                         f"p={row['delong_p_value']:.3f}", 
                         ha='center', fontsize=7, fontweight='bold')
        
        ax10.set_xlabel('QPP Variance Range', fontsize=11, fontweight='bold')
        ax10.set_ylabel('AUC', fontsize=11, fontweight='bold')
        ax10.set_title('Hypothesis 10: Performance by QPP Variance\n(Dark Green = Statistically Significant)', 
                      fontsize=12, fontweight='bold')
        ax10.set_xticks(x)
        ax10.set_xticklabels(h10_df['variance_range'], rotation=45, ha='right')
        ax10.legend()
        ax10.grid(True, alpha=0.3, axis='y')
        ax10.set_ylim([0, 1.1])
    
    # 仮説11: 予測確率の分布の分離度
    if 'hypothesis11' in results and len(results['hypothesis11']) > 0:
        ax11 = fig.add_subplot(gs[6, 1])
        h11_df = results['hypothesis11'].iloc[0]
        
        metrics = ['Separation\n(Mean Diff)', 'Overlap\n(1-KS)']
        integrated_values = [h11_df['integrated_separation'], h11_df['integrated_overlap']]
        best_qpp_values = [h11_df['best_qpp_separation'], h11_df['best_qpp_overlap']]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        colors = ['green' if h11_df['integrated_better_separation'] else 'red',
                 'green' if h11_df['integrated_better_overlap'] else 'red']
        
        bars1 = ax11.bar(x - width, integrated_values, width, label='Integrated Model', 
                        alpha=0.8, color=colors, edgecolor='black', linewidth=1)
        bars2 = ax11.bar(x, best_qpp_values, width, label=f"Best Single QPP ({h11_df['best_qpp_metric']})", 
                        alpha=0.8, color='orange', edgecolor='black', linewidth=1)
        
        improvements = [h11_df['separation_improvement'], h11_df['overlap_improvement']]
        for i, (val, imp) in enumerate(zip(integrated_values, improvements)):
            ax11.text(i - width, val + 0.01, 
                     f"{imp:+.4f}", 
                     ha='center', fontsize=8, fontweight='bold',
                     color='green' if colors[i] == 'green' else 'red')
        
        ax11.set_xlabel('Metric', fontsize=11, fontweight='bold')
        ax11.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax11.set_title('Hypothesis 11: Distribution Separation (Threshold-Free)\n(Green = Better Separation/Less Overlap)', 
                      fontsize=12, fontweight='bold')
        ax11.set_xticks(x)
        ax11.set_xticklabels(metrics)
        ax11.legend()
        ax11.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hypothesis_testing_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved hypothesis testing visualizations to {output_dir / 'hypothesis_testing_results.png'}")


def create_error_and_success_pattern_analysis(
    df: pd.DataFrame,
    qpp_metrics: List[str],
    integrated_name: str,
    true_labels: Dict[Tuple[str, int], int],
    output_dir: Path
):
    """
    エラー分析と成功パターンを視覚的かつ定量的に示す（仮説検証形式）
    
    仮説12: 統合モデルは、特定の条件（予測確率範囲、QAペア数、QPP値域）で単体QPP指標よりエラー率が低い
    """
    print("\n[INFO] Hypothesis 12: Integrated model has lower error rate in specific conditions")
    
    df_with_labels = df.copy()
    df_with_labels['true_label'] = df_with_labels.apply(
        lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
    )
    df_with_labels = df_with_labels.dropna(subset=['true_label', integrated_name])
    
    if len(df_with_labels) == 0:
        print("[WARN] No data with labels, skipping error analysis")
        return
    
    # 統合モデルの予測
    df_with_labels['integrated_pred'] = (df_with_labels[integrated_name] >= 0.5).astype(int)
    df_with_labels['integrated_correct'] = (df_with_labels['integrated_pred'] == df_with_labels['true_label']).astype(int)
    df_with_labels['integrated_tp'] = ((df_with_labels['integrated_pred'] == 1) & (df_with_labels['true_label'] == 1)).astype(int)
    df_with_labels['integrated_tn'] = ((df_with_labels['integrated_pred'] == 0) & (df_with_labels['true_label'] == 0)).astype(int)
    df_with_labels['integrated_fp'] = ((df_with_labels['integrated_pred'] == 1) & (df_with_labels['true_label'] == 0)).astype(int)
    df_with_labels['integrated_fn'] = ((df_with_labels['integrated_pred'] == 0) & (df_with_labels['true_label'] == 1)).astype(int)
    
    # 最良の単体QPP指標を特定
    best_qpp_metric = None
    best_qpp_auc = -np.inf
    
    for metric in qpp_metrics:
        if metric in df_with_labels.columns:
            metric_data = df_with_labels.dropna(subset=[metric, 'true_label'])
            if len(metric_data) > 20:
                metric_min = metric_data[metric].min()
                metric_max = metric_data[metric].max()
                if metric_max > metric_min:
                    metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                    try:
                        metric_auc = roc_auc_score(metric_data['true_label'], metric_normalized)
                        if metric_auc > best_qpp_auc:
                            best_qpp_metric = metric
                            best_qpp_auc = metric_auc
                    except:
                        pass
    
    if not best_qpp_metric:
        print("[WARN] No valid QPP metric found, skipping error analysis")
        return
    
    # 最良の単体QPP指標の予測を計算
    best_qpp_data = df_with_labels.dropna(subset=[best_qpp_metric, 'true_label'])
    if len(best_qpp_data) > 20:
        best_qpp_min = best_qpp_data[best_qpp_metric].min()
        best_qpp_max = best_qpp_data[best_qpp_metric].max()
        if best_qpp_max > best_qpp_min:
            best_qpp_normalized = (best_qpp_data[best_qpp_metric] - best_qpp_min) / (best_qpp_max - best_qpp_min)
            best_qpp_data = best_qpp_data.copy()
            best_qpp_data['qpp_pred'] = (best_qpp_normalized >= 0.5).astype(int)
            best_qpp_data['qpp_correct'] = (best_qpp_data['qpp_pred'] == best_qpp_data['true_label']).astype(int)
            best_qpp_data['qpp_tp'] = ((best_qpp_data['qpp_pred'] == 1) & (best_qpp_data['true_label'] == 1)).astype(int)
            best_qpp_data['qpp_tn'] = ((best_qpp_data['qpp_pred'] == 0) & (best_qpp_data['true_label'] == 0)).astype(int)
            best_qpp_data['qpp_fp'] = ((best_qpp_data['qpp_pred'] == 1) & (best_qpp_data['true_label'] == 0)).astype(int)
            best_qpp_data['qpp_fn'] = ((best_qpp_data['qpp_pred'] == 0) & (best_qpp_data['true_label'] == 1)).astype(int)
    
    # 1. 予測確率の範囲ごとのエラー率と成功率
    prob_bins = np.linspace(0, 1, 11)  # 0.0-1.0を10分割
    prob_ranges = [(prob_bins[i], prob_bins[i+1]) for i in range(len(prob_bins)-1)]
    
    integrated_error_by_prob = []
    qpp_error_by_prob = []
    
    for low, high in prob_ranges:
        # 統合モデル
        integrated_range = df_with_labels[
            (df_with_labels[integrated_name] >= low) & (df_with_labels[integrated_name] < high)
        ]
        if len(integrated_range) > 10:
            error_rate = 1 - integrated_range['integrated_correct'].mean()
            integrated_error_by_prob.append({
                'prob_range': f'{low:.1f}-{high:.1f}',
                'prob_mid': (low + high) / 2,
                'n_samples': len(integrated_range),
                'error_rate': error_rate,
                'tp_rate': integrated_range['integrated_tp'].mean(),
                'tn_rate': integrated_range['integrated_tn'].mean(),
                'fp_rate': integrated_range['integrated_fp'].mean(),
                'fn_rate': integrated_range['integrated_fn'].mean()
            })
        
        # 単体QPP指標
        if len(best_qpp_data) > 0:
            qpp_range = best_qpp_data[
                (best_qpp_data[best_qpp_metric] >= best_qpp_min + low * (best_qpp_max - best_qpp_min)) &
                (best_qpp_data[best_qpp_metric] < best_qpp_min + high * (best_qpp_max - best_qpp_min))
            ]
            if len(qpp_range) > 10:
                error_rate = 1 - qpp_range['qpp_correct'].mean()
                qpp_error_by_prob.append({
                    'prob_range': f'{low:.1f}-{high:.1f}',
                    'prob_mid': (low + high) / 2,
                    'n_samples': len(qpp_range),
                    'error_rate': error_rate,
                    'tp_rate': qpp_range['qpp_tp'].mean(),
                    'tn_rate': qpp_range['qpp_tn'].mean(),
                    'fp_rate': qpp_range['qpp_fp'].mean(),
                    'fn_rate': qpp_range['qpp_fn'].mean()
                })
    
    # 2. QAペア数ごとのエラー率と成功率
    qa_ranges = [
        (0, 1, '0 QA pair'),
        (1, 2, '1 QA pair'),
        (2, 3, '2 QA pairs'),
        (3, 4, '3 QA pairs'),
        (4, 100, '4+ QA pairs')
    ]
    
    integrated_error_by_qa = []
    qpp_error_by_qa = []
    
    for low, high, label in qa_ranges:
        # 統合モデル
        qa_range = df_with_labels[
            (df_with_labels['qa_pairs_count'] >= low) & (df_with_labels['qa_pairs_count'] < high)
        ]
        if len(qa_range) > 10:
            error_rate = 1 - qa_range['integrated_correct'].mean()
            integrated_error_by_qa.append({
                'qa_range': label,
                'n_samples': len(qa_range),
                'error_rate': error_rate,
                'tp_rate': qa_range['integrated_tp'].mean(),
                'tn_rate': qa_range['integrated_tn'].mean(),
                'fp_rate': qa_range['integrated_fp'].mean(),
                'fn_rate': qa_range['integrated_fn'].mean()
            })
        
        # 単体QPP指標
        if len(best_qpp_data) > 0:
            qpp_qa_range = best_qpp_data[
                (best_qpp_data['qa_pairs_count'] >= low) & (best_qpp_data['qa_pairs_count'] < high)
            ]
            if len(qpp_qa_range) > 10:
                error_rate = 1 - qpp_qa_range['qpp_correct'].mean()
                qpp_error_by_qa.append({
                    'qa_range': label,
                    'n_samples': len(qpp_qa_range),
                    'error_rate': error_rate,
                    'tp_rate': qpp_qa_range['qpp_tp'].mean(),
                    'tn_rate': qpp_qa_range['qpp_tn'].mean(),
                    'fp_rate': qpp_qa_range['qpp_fp'].mean(),
                    'fn_rate': qpp_qa_range['qpp_fn'].mean()
                })
    
    # 3. 最良のQPP指標の値域ごとのエラー率と成功率
    if len(best_qpp_data) > 0:
        qpp_bins = np.percentile(best_qpp_data[best_qpp_metric].dropna(), [0, 20, 40, 60, 80, 100])
        qpp_ranges = [(qpp_bins[i], qpp_bins[i+1]) for i in range(len(qpp_bins)-1)]
        
        integrated_error_by_qpp = []
        qpp_error_by_qpp_val = []
        
        for low, high in qpp_ranges:
            # 統合モデル
            qpp_val_range = df_with_labels[
                (df_with_labels[best_qpp_metric] >= low) & (df_with_labels[best_qpp_metric] < high)
            ]
            if len(qpp_val_range) > 10:
                error_rate = 1 - qpp_val_range['integrated_correct'].mean()
                integrated_error_by_qpp.append({
                    'qpp_range': f'{low:.3f}-{high:.3f}',
                    'qpp_mid': (low + high) / 2,
                    'n_samples': len(qpp_val_range),
                    'error_rate': error_rate,
                    'tp_rate': qpp_val_range['integrated_tp'].mean(),
                    'tn_rate': qpp_val_range['integrated_tn'].mean(),
                    'fp_rate': qpp_val_range['integrated_fp'].mean(),
                    'fn_rate': qpp_val_range['integrated_fn'].mean()
                })
            
            # 単体QPP指標
            qpp_val_range_single = best_qpp_data[
                (best_qpp_data[best_qpp_metric] >= low) & (best_qpp_data[best_qpp_metric] < high)
            ]
            if len(qpp_val_range_single) > 10:
                error_rate = 1 - qpp_val_range_single['qpp_correct'].mean()
                qpp_error_by_qpp_val.append({
                    'qpp_range': f'{low:.3f}-{high:.3f}',
                    'qpp_mid': (low + high) / 2,
                    'n_samples': len(qpp_val_range_single),
                    'error_rate': error_rate,
                    'tp_rate': qpp_val_range_single['qpp_tp'].mean(),
                    'tn_rate': qpp_val_range_single['qpp_tn'].mean(),
                    'fp_rate': qpp_val_range_single['qpp_fp'].mean(),
                    'fn_rate': qpp_val_range_single['qpp_fn'].mean()
                })
    
    # 可視化
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. 予測確率の範囲ごとのエラー率
    ax1 = fig.add_subplot(gs[0, :])
    if integrated_error_by_prob and qpp_error_by_prob:
        int_df = pd.DataFrame(integrated_error_by_prob)
        qpp_df = pd.DataFrame(qpp_error_by_prob)
        
        ax1.plot(int_df['prob_mid'], int_df['error_rate'], 'o-', label='Integrated Model', 
                linewidth=2, markersize=8, color='green')
        ax1.plot(qpp_df['prob_mid'], qpp_df['error_rate'], 's-', label=f'Single QPP ({best_qpp_metric})', 
                linewidth=2, markersize=8, color='orange')
        ax1.set_xlabel('Prediction Probability Range', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Error Rate by Prediction Probability Range', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1])
    
    # 2. QAペア数ごとのエラー率
    ax2 = fig.add_subplot(gs[1, :])
    if integrated_error_by_qa and qpp_error_by_qa:
        int_qa_df = pd.DataFrame(integrated_error_by_qa)
        qpp_qa_df = pd.DataFrame(qpp_error_by_qa)
        
        x = np.arange(len(int_qa_df))
        width = 0.35
        
        ax2.bar(x - width/2, int_qa_df['error_rate'], width, label='Integrated Model', 
               alpha=0.8, color='green', edgecolor='black')
        ax2.bar(x + width/2, qpp_qa_df['error_rate'], width, label=f'Single QPP ({best_qpp_metric})', 
               alpha=0.8, color='orange', edgecolor='black')
        ax2.set_xlabel('QA Pairs Count', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax2.set_title('Error Rate by QA Pairs Count', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(int_qa_df['qa_range'], rotation=45, ha='right')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])
    
    # 3. QPP指標の値域ごとのエラー率
    ax3 = fig.add_subplot(gs[2, :])
    if integrated_error_by_qpp and qpp_error_by_qpp_val:
        int_qpp_df = pd.DataFrame(integrated_error_by_qpp)
        qpp_val_df = pd.DataFrame(qpp_error_by_qpp_val)
        
        ax3.plot(int_qpp_df['qpp_mid'], int_qpp_df['error_rate'], 'o-', label='Integrated Model', 
                linewidth=2, markersize=8, color='green')
        ax3.plot(qpp_val_df['qpp_mid'], qpp_val_df['error_rate'], 's-', label=f'Single QPP ({best_qpp_metric})', 
                linewidth=2, markersize=8, color='orange')
        ax3.set_xlabel(f'{best_qpp_metric} Value Range', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax3.set_title(f'Error Rate by {best_qpp_metric} Value Range', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1])
    
    # 4. TP/TN/FP/FNの分布（統合モデル）
    ax4 = fig.add_subplot(gs[3, 0])
    if len(df_with_labels) > 0:
        tp_count = df_with_labels['integrated_tp'].sum()
        tn_count = df_with_labels['integrated_tn'].sum()
        fp_count = df_with_labels['integrated_fp'].sum()
        fn_count = df_with_labels['integrated_fn'].sum()
        
        categories = ['TP', 'TN', 'FP', 'FN']
        counts = [tp_count, tn_count, fp_count, fn_count]
        colors = ['green', 'blue', 'orange', 'red']
        
        bars = ax4.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax4.set_title('Integrated Model: TP/TN/FP/FN Distribution', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(df_with_labels)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 5. TP/TN/FP/FNの分布（単体QPP指標）
    ax5 = fig.add_subplot(gs[3, 1])
    if len(best_qpp_data) > 0:
        tp_count = best_qpp_data['qpp_tp'].sum()
        tn_count = best_qpp_data['qpp_tn'].sum()
        fp_count = best_qpp_data['qpp_fp'].sum()
        fn_count = best_qpp_data['qpp_fn'].sum()
        
        categories = ['TP', 'TN', 'FP', 'FN']
        counts = [tp_count, tn_count, fp_count, fn_count]
        colors = ['green', 'blue', 'orange', 'red']
        
        bars = ax5.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax5.set_title(f'Single QPP ({best_qpp_metric}): TP/TN/FP/FN Distribution', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/len(best_qpp_data)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. エラー率の比較
    ax6 = fig.add_subplot(gs[3, 2])
    if len(df_with_labels) > 0 and len(best_qpp_data) > 0:
        integrated_error = 1 - df_with_labels['integrated_correct'].mean()
        qpp_error = 1 - best_qpp_data['qpp_correct'].mean()
        
        categories = ['Integrated Model', f'Single QPP\n({best_qpp_metric})']
        error_rates = [integrated_error, qpp_error]
        colors = ['green' if integrated_error < qpp_error else 'red', 'orange']
        
        bars = ax6.bar(categories, error_rates, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Error Rate', fontsize=12, fontweight='bold')
        ax6.set_title('Overall Error Rate Comparison', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        ax6.set_ylim([0, 1])
        
        for bar, rate in zip(bars, error_rates):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.3f}\n({rate*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'error_and_success_pattern_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved error and success pattern analysis to {output_path}")
    
    # 結果をCSVに保存
    if integrated_error_by_prob:
        pd.DataFrame(integrated_error_by_prob).to_csv(
            output_dir / 'integrated_error_by_probability.csv', index=False
        )
    if qpp_error_by_prob:
        pd.DataFrame(qpp_error_by_prob).to_csv(
            output_dir / 'qpp_error_by_probability.csv', index=False
        )
    if integrated_error_by_qa:
        pd.DataFrame(integrated_error_by_qa).to_csv(
            output_dir / 'integrated_error_by_qa_pairs.csv', index=False
        )
    if qpp_error_by_qa:
        pd.DataFrame(qpp_error_by_qa).to_csv(
            output_dir / 'qpp_error_by_qa_pairs.csv', index=False
        )
    
    # 仮説検証形式で結果を返す
    hypothesis12_results = []
    
    # 1. 予測確率の範囲ごとのエラー率の比較（統計的有意性を検定）
    if integrated_error_by_prob and qpp_error_by_prob:
        int_prob_df = pd.DataFrame(integrated_error_by_prob)
        qpp_prob_df = pd.DataFrame(qpp_error_by_prob)
        
        # 同じ範囲で比較
        for _, int_row in int_prob_df.iterrows():
            prob_range = int_row['prob_range']
            qpp_row = qpp_prob_df[qpp_prob_df['prob_range'] == prob_range]
            
            if len(qpp_row) > 0:
                qpp_row = qpp_row.iloc[0]
                
                # 統合モデルと単体QPP指標のエラー率を比較
                integrated_error = int_row['error_rate']
                qpp_error = qpp_row['error_rate']
                error_improvement = qpp_error - integrated_error  # 統合モデルのエラー率が低いほど良い
                
                # 統計的有意性の検定（カイ二乗検定）
                # 同じ範囲のデータで比較するため、実際のデータから計算
                int_range_data = df_with_labels[
                    (df_with_labels[integrated_name] >= float(prob_range.split('-')[0])) &
                    (df_with_labels[integrated_name] < float(prob_range.split('-')[1]))
                ]
                qpp_range_data = best_qpp_data[
                    (best_qpp_data[best_qpp_metric] >= best_qpp_min + float(prob_range.split('-')[0]) * (best_qpp_max - best_qpp_min)) &
                    (best_qpp_data[best_qpp_metric] < best_qpp_min + float(prob_range.split('-')[1]) * (best_qpp_max - best_qpp_min))
                ]
                
                if len(int_range_data) > 20 and len(qpp_range_data) > 20:
                    # エラー率の差の統計的有意性を検定
                    int_errors = (1 - int_range_data['integrated_correct']).sum()
                    int_total = len(int_range_data)
                    qpp_errors = (1 - qpp_range_data['qpp_correct']).sum()
                    qpp_total = len(qpp_range_data)
                    
                    # 比率の差の検定（Z検定）
                    p1 = int_errors / int_total if int_total > 0 else 0
                    p2 = qpp_errors / qpp_total if qpp_total > 0 else 0
                    p_pooled = (int_errors + qpp_errors) / (int_total + qpp_total) if (int_total + qpp_total) > 0 else 0
                    
                    if p_pooled > 0 and p_pooled < 1:
                        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/int_total + 1/qpp_total))
                        if se > 0:
                            z_stat = (p1 - p2) / se
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        else:
                            z_stat = np.nan
                            p_value = np.nan
                    else:
                        z_stat = np.nan
                        p_value = np.nan
                    
                    hypothesis12_results.append({
                        'condition_type': 'Prediction Probability Range',
                        'condition': prob_range,
                        'n_samples_integrated': int_total,
                        'n_samples_qpp': qpp_total,
                        'integrated_error_rate': integrated_error,
                        'qpp_error_rate': qpp_error,
                        'error_improvement': error_improvement,
                        'z_stat': z_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05 if not np.isnan(p_value) else False,
                        'integrated_better': error_improvement > 0,
                        'integrated_significantly_better': (error_improvement > 0) and (p_value < 0.05 if not np.isnan(p_value) else False)
                    })
    
    # 2. QAペア数ごとのエラー率の比較
    if integrated_error_by_qa and qpp_error_by_qa:
        int_qa_df = pd.DataFrame(integrated_error_by_qa)
        qpp_qa_df = pd.DataFrame(qpp_error_by_qa)
        
        for _, int_row in int_qa_df.iterrows():
            qa_range = int_row['qa_range']
            qpp_row = qpp_qa_df[qpp_qa_df['qa_range'] == qa_range]
            
            if len(qpp_row) > 0:
                qpp_row = qpp_row.iloc[0]
                
                integrated_error = int_row['error_rate']
                qpp_error = qpp_row['error_rate']
                error_improvement = qpp_error - integrated_error
                
                # 統計的有意性の検定
                qa_low = 0 if '0' in qa_range else (1 if '1' in qa_range else (2 if '2' in qa_range else (3 if '3' in qa_range else 4)))
                qa_high = 1 if '0' in qa_range else (2 if '1' in qa_range else (3 if '2' in qa_range else (4 if '3' in qa_range else 100)))
                
                int_qa_data = df_with_labels[
                    (df_with_labels['qa_pairs_count'] >= qa_low) & (df_with_labels['qa_pairs_count'] < qa_high)
                ]
                qpp_qa_data = best_qpp_data[
                    (best_qpp_data['qa_pairs_count'] >= qa_low) & (best_qpp_data['qa_pairs_count'] < qa_high)
                ]
                
                if len(int_qa_data) > 20 and len(qpp_qa_data) > 20:
                    int_errors = (1 - int_qa_data['integrated_correct']).sum()
                    int_total = len(int_qa_data)
                    qpp_errors = (1 - qpp_qa_data['qpp_correct']).sum()
                    qpp_total = len(qpp_qa_data)
                    
                    p1 = int_errors / int_total if int_total > 0 else 0
                    p2 = qpp_errors / qpp_total if qpp_total > 0 else 0
                    p_pooled = (int_errors + qpp_errors) / (int_total + qpp_total) if (int_total + qpp_total) > 0 else 0
                    
                    if p_pooled > 0 and p_pooled < 1:
                        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/int_total + 1/qpp_total))
                        if se > 0:
                            z_stat = (p1 - p2) / se
                            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                        else:
                            z_stat = np.nan
                            p_value = np.nan
                    else:
                        z_stat = np.nan
                        p_value = np.nan
                    
                    hypothesis12_results.append({
                        'condition_type': 'QA Pairs Count',
                        'condition': qa_range,
                        'n_samples_integrated': int_total,
                        'n_samples_qpp': qpp_total,
                        'integrated_error_rate': integrated_error,
                        'qpp_error_rate': qpp_error,
                        'error_improvement': error_improvement,
                        'z_stat': z_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05 if not np.isnan(p_value) else False,
                        'integrated_better': error_improvement > 0,
                        'integrated_significantly_better': (error_improvement > 0) and (p_value < 0.05 if not np.isnan(p_value) else False)
                    })
    
    return hypothesis12_results


def create_individual_query_analysis(
    df: pd.DataFrame,
    qpp_metrics: List[str],
    integrated_name: str,
    true_labels: Dict[Tuple[str, int], int],
    output_dir: Path
):
    """
    1クエリ1プロット分析: 個々のクエリの特徴を可視化
    - 統合モデル vs 単体QPP指標の予測確率の散布図
    - エラー/成功パターンの可視化
    - QPP指標の値やQAペア数での色分け
    """
    print("\n[INFO] Creating individual query analysis (1 query = 1 plot)...")
    
    df_with_labels = df.copy()
    df_with_labels['true_label'] = df_with_labels.apply(
        lambda row: true_labels.get((row['conv_id'], row['turn_id']), np.nan), axis=1
    )
    df_with_labels = df_with_labels.dropna(subset=['true_label', integrated_name])
    
    if len(df_with_labels) == 0:
        print("[WARN] No data with labels, skipping individual query analysis")
        return
    
    # 統合モデルの予測
    df_with_labels['integrated_pred'] = (df_with_labels[integrated_name] >= 0.5).astype(int)
    df_with_labels['integrated_correct'] = (df_with_labels['integrated_pred'] == df_with_labels['true_label']).astype(int)
    
    # 最良の単体QPP指標を特定
    best_qpp_metric = None
    best_qpp_auc = -np.inf
    
    for metric in qpp_metrics:
        if metric in df_with_labels.columns:
            metric_data = df_with_labels.dropna(subset=[metric, 'true_label'])
            if len(metric_data) > 20:
                metric_min = metric_data[metric].min()
                metric_max = metric_data[metric].max()
                if metric_max > metric_min:
                    metric_normalized = (metric_data[metric] - metric_min) / (metric_max - metric_min)
                    try:
                        metric_auc = roc_auc_score(metric_data['true_label'], metric_normalized)
                        if metric_auc > best_qpp_auc:
                            best_qpp_metric = metric
                            best_qpp_auc = metric_auc
                    except:
                        pass
    
    if not best_qpp_metric:
        print("[WARN] No valid QPP metric found, skipping individual query analysis")
        return
    
    # 最良の単体QPP指標の予測を計算
    best_qpp_data = df_with_labels.dropna(subset=[best_qpp_metric, 'true_label'])
    if len(best_qpp_data) > 20:
        best_qpp_min = best_qpp_data[best_qpp_metric].min()
        best_qpp_max = best_qpp_data[best_qpp_metric].max()
        if best_qpp_max > best_qpp_min:
            best_qpp_normalized = (best_qpp_data[best_qpp_metric] - best_qpp_min) / (best_qpp_max - best_qpp_min)
            best_qpp_data = best_qpp_data.copy()
            best_qpp_data['qpp_pred'] = (best_qpp_normalized >= 0.5).astype(int)
            best_qpp_data['qpp_correct'] = (best_qpp_data['qpp_pred'] == best_qpp_data['true_label']).astype(int)
            best_qpp_data['qpp_prob_normalized'] = best_qpp_normalized
    
    # 統合モデルと単体QPP指標の両方のデータがあるクエリのみ
    merged_data = df_with_labels.merge(
        best_qpp_data[['conv_id', 'turn_id', 'qpp_prob_normalized', 'qpp_correct']],
        on=['conv_id', 'turn_id'],
        how='inner'
    )
    
    if len(merged_data) == 0:
        print("[WARN] No overlapping data, skipping individual query analysis")
        return
    
    # パターン分類: 統合モデルと単体QPP指標の正解/不正解の組み合わせ
    merged_data['pattern'] = 'Other'
    merged_data.loc[
        (merged_data['integrated_correct'] == 1) & (merged_data['qpp_correct'] == 1),
        'pattern'
    ] = 'Both Correct'
    merged_data.loc[
        (merged_data['integrated_correct'] == 1) & (merged_data['qpp_correct'] == 0),
        'pattern'
    ] = 'Integrated Wins'
    merged_data.loc[
        (merged_data['integrated_correct'] == 0) & (merged_data['qpp_correct'] == 1),
        'pattern'
    ] = 'QPP Wins'
    merged_data.loc[
        (merged_data['integrated_correct'] == 0) & (merged_data['qpp_correct'] == 0),
        'pattern'
    ] = 'Both Wrong'
    
    # 可視化
    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. 統合モデル vs 単体QPP指標の予測確率（パターンで色分け）
    ax1 = fig.add_subplot(gs[0, 0])
    pattern_colors = {
        'Both Correct': 'green',
        'Integrated Wins': 'blue',
        'QPP Wins': 'orange',
        'Both Wrong': 'red',
        'Other': 'gray'
    }
    for pattern, color in pattern_colors.items():
        pattern_data = merged_data[merged_data['pattern'] == pattern]
        if len(pattern_data) > 0:
            ax1.scatter(
                pattern_data['qpp_prob_normalized'],
                pattern_data[integrated_name],
                alpha=0.6, s=30, label=f'{pattern} (n={len(pattern_data)})',
                color=color, edgecolors='black', linewidth=0.5
            )
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='y=x')
    ax1.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel(f'Single QPP ({best_qpp_metric}) Prediction Probability', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Integrated Model Prediction Probability', fontsize=11, fontweight='bold')
    ax1.set_title('Integrated vs Single QPP: Prediction Probability\n(Colored by Correct/Wrong Pattern)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # 2. 統合モデル vs 単体QPP指標の予測確率（QAペア数で色分け）
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(
        merged_data['qpp_prob_normalized'],
        merged_data[integrated_name],
        c=merged_data['qa_pairs_count'],
        cmap='viridis', alpha=0.6, s=30,
        edgecolors='black', linewidth=0.5
    )
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('QA Pairs Count', fontsize=10)
    ax2.set_xlabel(f'Single QPP ({best_qpp_metric}) Prediction Probability', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Integrated Model Prediction Probability', fontsize=11, fontweight='bold')
    ax2.set_title('Integrated vs Single QPP: Prediction Probability\n(Colored by QA Pairs Count)', 
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # 3. 統合モデル vs 単体QPP指標の予測確率（パターンと真のラベルで色分け）
    ax3 = fig.add_subplot(gs[0, 2])
    # Integrated WinsとQPP Winsを強調
    for pattern, color, marker in [('Integrated Wins', 'blue', 'o'), ('QPP Wins', 'orange', 's'), 
                                    ('Both Correct', 'green', '^'), ('Both Wrong', 'red', 'x')]:
        pattern_data = merged_data[merged_data['pattern'] == pattern]
        if len(pattern_data) > 0:
            # 真のラベルでさらに色分け（透明度で）
            for label_val, alpha_val in [(1, 0.8), (0, 0.4)]:
                label_data = pattern_data[pattern_data['true_label'] == label_val]
                if len(label_data) > 0:
                    ax3.scatter(
                        label_data['qpp_prob_normalized'],
                        label_data[integrated_name],
                        alpha=alpha_val, s=50, label=f'{pattern} (label={label_val}, n={len(label_data)})',
                        color=color, marker=marker, edgecolors='black', linewidth=0.5
                    )
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax3.set_xlabel(f'Single QPP ({best_qpp_metric}) Prediction Probability', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Integrated Model Prediction Probability', fontsize=11, fontweight='bold')
    ax3.set_title('Integrated vs Single QPP: Prediction Probability\n(Pattern + True Label, Dark=Ambiguous, Light=Non-ambiguous)', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper left', ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # 4. 予測確率の差（統合 - 単体QPP）の分布（パターンで色分け）
    ax4 = fig.add_subplot(gs[1, 0])
    merged_data['prob_diff'] = merged_data[integrated_name] - merged_data['qpp_prob_normalized']
    for pattern, color in pattern_colors.items():
        pattern_data = merged_data[merged_data['pattern'] == pattern]
        if len(pattern_data) > 0:
            ax4.hist(pattern_data['prob_diff'], bins=50, alpha=0.6, label=f'{pattern} (n={len(pattern_data)})',
                    color=color, edgecolor='black', linewidth=0.5)
    ax4.axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
    ax4.set_xlabel('Prediction Probability Difference (Integrated - Single QPP)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Distribution of Prediction Probability Difference\n(Colored by Correct/Wrong Pattern)', 
                 fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. 予測確率の差 vs QAペア数
    ax5 = fig.add_subplot(gs[1, 1])
    scatter = ax5.scatter(
        merged_data['qa_pairs_count'],
        merged_data['prob_diff'],
        c=[pattern_colors.get(p, 'gray') for p in merged_data['pattern']],
        alpha=0.6, s=30, edgecolors='black', linewidth=0.5
    )
    ax5.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax5.set_xlabel('QA Pairs Count', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Prediction Probability Difference (Integrated - Single QPP)', fontsize=11, fontweight='bold')
    ax5.set_title('Prediction Difference vs QA Pairs Count\n(Colored by Pattern)', 
                 fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 予測確率の差 vs 最良のQPP指標の値
    ax6 = fig.add_subplot(gs[1, 2])
    scatter = ax6.scatter(
        merged_data[best_qpp_metric],
        merged_data['prob_diff'],
        c=[pattern_colors.get(p, 'gray') for p in merged_data['pattern']],
        alpha=0.6, s=30, edgecolors='black', linewidth=0.5
    )
    ax6.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax6.set_xlabel(f'{best_qpp_metric} Value', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Prediction Probability Difference (Integrated - Single QPP)', fontsize=11, fontweight='bold')
    ax6.set_title(f'Prediction Difference vs {best_qpp_metric} Value\n(Colored by Pattern)', 
                 fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. 統合モデルが勝つケースの特徴（Integrated Wins）- 真のラベルで色分け
    ax7 = fig.add_subplot(gs[2, 0])
    integrated_wins = merged_data[merged_data['pattern'] == 'Integrated Wins']
    if len(integrated_wins) > 0:
        # 真のラベルで色分け（曖昧=赤、非曖昧=青）
        for label_val, color, label_name in [(1, 'red', 'Ambiguous (1)'), (0, 'blue', 'Non-ambiguous (0)')]:
            label_data = integrated_wins[integrated_wins['true_label'] == label_val]
            if len(label_data) > 0:
                ax7.scatter(
                    label_data['qpp_prob_normalized'],
                    label_data[integrated_name],
                    alpha=0.7, s=50, label=f'{label_name} (n={len(label_data)})',
                    color=color, edgecolors='black', linewidth=0.5
                )
        ax7.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax7.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax7.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        # 統計情報を表示
        ambiguous_rate = integrated_wins['true_label'].mean()
        ax7.text(0.05, 0.95, f'Ambiguity Rate: {ambiguous_rate:.1%}\n(n={len(integrated_wins)})',
                transform=ax7.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax7.set_xlabel(f'Single QPP ({best_qpp_metric}) Prediction Probability', fontsize=11, fontweight='bold')
        ax7.set_ylabel('Integrated Model Prediction Probability', fontsize=11, fontweight='bold')
        ax7.set_title(f'Integrated Wins Cases (n={len(integrated_wins)})\n(Colored by True Label)', 
                     fontsize=12, fontweight='bold')
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim([0, 1])
        ax7.set_ylim([0, 1])
    
    # 8. 単体QPP指標が勝つケースの特徴（QPP Wins）- 真のラベルで色分け
    ax8 = fig.add_subplot(gs[2, 1])
    qpp_wins = merged_data[merged_data['pattern'] == 'QPP Wins']
    if len(qpp_wins) > 0:
        # 真のラベルで色分け（曖昧=赤、非曖昧=青）
        for label_val, color, label_name in [(1, 'red', 'Ambiguous (1)'), (0, 'blue', 'Non-ambiguous (0)')]:
            label_data = qpp_wins[qpp_wins['true_label'] == label_val]
            if len(label_data) > 0:
                ax8.scatter(
                    label_data['qpp_prob_normalized'],
                    label_data[integrated_name],
                    alpha=0.7, s=50, label=f'{label_name} (n={len(label_data)})',
                    color=color, edgecolors='black', linewidth=0.5
                )
        ax8.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
        ax8.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
        ax8.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
        
        # 統計情報を表示
        ambiguous_rate = qpp_wins['true_label'].mean()
        ax8.text(0.05, 0.95, f'Ambiguity Rate: {ambiguous_rate:.1%}\n(n={len(qpp_wins)})',
                transform=ax8.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax8.set_xlabel(f'Single QPP ({best_qpp_metric}) Prediction Probability', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Integrated Model Prediction Probability', fontsize=11, fontweight='bold')
        ax8.set_title(f'QPP Wins Cases (n={len(qpp_wins)})\n(Colored by True Label)', 
                     fontsize=12, fontweight='bold')
        ax8.legend(fontsize=9)
        ax8.grid(True, alpha=0.3)
        ax8.set_xlim([0, 1])
        ax8.set_ylim([0, 1])
    
    # 9. パターン別の曖昧性率の比較（統合モデルが勝つケース vs 単体QPP指標が勝つケース）
    ax9 = fig.add_subplot(gs[2, 2])
    integrated_wins = merged_data[merged_data['pattern'] == 'Integrated Wins']
    qpp_wins = merged_data[merged_data['pattern'] == 'QPP Wins']
    
    if len(integrated_wins) > 0 and len(qpp_wins) > 0:
        categories = ['Integrated\nWins', 'QPP\nWins']
        ambiguity_rates = [
            integrated_wins['true_label'].mean(),
            qpp_wins['true_label'].mean()
        ]
        counts = [len(integrated_wins), len(qpp_wins)]
        colors = ['blue', 'orange']
        
        bars = ax9.bar(categories, ambiguity_rates, color=colors, alpha=0.7, edgecolor='black')
        ax9.set_ylabel('Ambiguity Rate (True Label = 1)', fontsize=11, fontweight='bold')
        ax9.set_title('Ambiguity Rate: Integrated Wins vs QPP Wins\n(Key Finding: Integrated Wins = Ambiguous Queries)', 
                     fontsize=12, fontweight='bold')
        ax9.set_ylim([0, 1])
        ax9.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, rate, count) in enumerate(zip(bars, ambiguity_rates, counts)):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{rate:.1%}\n(n={count})',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 統計的有意性を表示
        from scipy.stats import mannwhitneyu
        try:
            int_labels = integrated_wins['true_label'].values
            qpp_labels = qpp_wins['true_label'].values
            # 比率の差の検定
            p1 = int_labels.mean()
            p2 = qpp_labels.mean()
            n1 = len(int_labels)
            n2 = len(qpp_labels)
            p_pooled = (int_labels.sum() + qpp_labels.sum()) / (n1 + n2) if (n1 + n2) > 0 else 0
            if p_pooled > 0 and p_pooled < 1:
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                if se > 0:
                    z_stat = (p1 - p2) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    sig_text = f"*** p < 0.001" if p_value < 0.001 else f"** p = {p_value:.4f}" if p_value < 0.05 else ""
                    if sig_text:
                        ax9.text(0.5, 0.95, sig_text, transform=ax9.transAxes,
                                ha='center', va='top', fontsize=12, fontweight='bold',
                                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        except:
            pass
    
    plt.tight_layout()
    output_path = output_dir / 'individual_query_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved individual query analysis to {output_path}")
    
    # パターン別の統計情報をCSVに保存
    pattern_counts = merged_data['pattern'].value_counts()
    pattern_stats = []
    for pattern in pattern_counts.index:
        pattern_data = merged_data[merged_data['pattern'] == pattern]
        if len(pattern_data) > 0:
            pattern_stats.append({
                'pattern': pattern,
                'count': len(pattern_data),
                'percentage': len(pattern_data) / len(merged_data) * 100,
                'mean_integrated_prob': pattern_data[integrated_name].mean(),
                'mean_qpp_prob': pattern_data['qpp_prob_normalized'].mean(),
                'mean_prob_diff': pattern_data['prob_diff'].mean(),
                'mean_qa_pairs': pattern_data['qa_pairs_count'].mean(),
                'mean_true_label': pattern_data['true_label'].mean()
            })
    
    if pattern_stats:
        pd.DataFrame(pattern_stats).to_csv(
            output_dir / 'individual_query_pattern_stats.csv', index=False
        )
    
    # 閾値依存性の検証: 複数の閾値でパターンを確認
    threshold_analysis_results = []
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in thresholds:
        # 各閾値で予測を再計算
        merged_data_thresh = merged_data.copy()
        merged_data_thresh['integrated_pred_thresh'] = (merged_data_thresh[integrated_name] >= threshold).astype(int)
        merged_data_thresh['qpp_pred_thresh'] = (merged_data_thresh['qpp_prob_normalized'] >= threshold).astype(int)
        merged_data_thresh['integrated_correct_thresh'] = (merged_data_thresh['integrated_pred_thresh'] == merged_data_thresh['true_label']).astype(int)
        merged_data_thresh['qpp_correct_thresh'] = (merged_data_thresh['qpp_pred_thresh'] == merged_data_thresh['true_label']).astype(int)
        
        # パターン分類
        integrated_wins_thresh = merged_data_thresh[
            (merged_data_thresh['integrated_correct_thresh'] == 1) & 
            (merged_data_thresh['qpp_correct_thresh'] == 0)
        ]
        qpp_wins_thresh = merged_data_thresh[
            (merged_data_thresh['integrated_correct_thresh'] == 0) & 
            (merged_data_thresh['qpp_correct_thresh'] == 1)
        ]
        
        if len(integrated_wins_thresh) > 0 and len(qpp_wins_thresh) > 0:
            int_ambiguity_rate = integrated_wins_thresh['true_label'].mean()
            qpp_ambiguity_rate = qpp_wins_thresh['true_label'].mean()
            
            threshold_analysis_results.append({
                'threshold': threshold,
                'n_integrated_wins': len(integrated_wins_thresh),
                'n_qpp_wins': len(qpp_wins_thresh),
                'integrated_wins_ambiguity_rate': int_ambiguity_rate,
                'qpp_wins_ambiguity_rate': qpp_ambiguity_rate,
                'ambiguity_rate_difference': int_ambiguity_rate - qpp_ambiguity_rate
            })
    
    if threshold_analysis_results:
        threshold_df = pd.DataFrame(threshold_analysis_results)
        threshold_df.to_csv(
            output_dir / 'threshold_dependency_analysis.csv', index=False
        )
        print(f"[INFO] Saved threshold dependency analysis to {output_dir / 'threshold_dependency_analysis.csv'}")
    
    # 予測確率の分布を確認（閾値に依存しない分析）
    # Integrated WinsとQPP Winsの予測確率分布を比較
    integrated_wins = merged_data[merged_data['pattern'] == 'Integrated Wins']
    qpp_wins = merged_data[merged_data['pattern'] == 'QPP Wins']
    
    # 予測確率の分布を可視化（閾値非依存）
    fig_thresh = plt.figure(figsize=(16, 10))
    gs_thresh = fig_thresh.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    if len(integrated_wins) > 0 and len(qpp_wins) > 0:
        # 1. 統合モデルの予測確率分布（Integrated Wins vs QPP Wins）
        ax1 = fig_thresh.add_subplot(gs_thresh[0, 0])
        ax1.hist(integrated_wins[integrated_name], bins=30, alpha=0.6, label=f'Integrated Wins (n={len(integrated_wins)})', 
                color='blue', edgecolor='black')
        ax1.hist(qpp_wins[integrated_name], bins=30, alpha=0.6, label=f'QPP Wins (n={len(qpp_wins)})', 
                color='orange', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
        ax1.set_xlabel('Integrated Model Prediction Probability', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Integrated Model Probability Distribution\n(by Win Pattern)', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. 単体QPP指標の予測確率分布（Integrated Wins vs QPP Wins）
        ax2 = fig_thresh.add_subplot(gs_thresh[0, 1])
        ax2.hist(integrated_wins['qpp_prob_normalized'], bins=30, alpha=0.6, label=f'Integrated Wins (n={len(integrated_wins)})', 
                color='blue', edgecolor='black')
        ax2.hist(qpp_wins['qpp_prob_normalized'], bins=30, alpha=0.6, label=f'QPP Wins (n={len(qpp_wins)})', 
                color='orange', edgecolor='black')
        ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
        ax2.set_xlabel(f'Single QPP ({best_qpp_metric}) Prediction Probability', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Single QPP Probability Distribution\n(by Win Pattern)', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. 真のラベルの分布（Integrated Wins vs QPP Wins）
        ax3 = fig_thresh.add_subplot(gs_thresh[0, 2])
        int_labels = integrated_wins['true_label'].value_counts().sort_index()
        qpp_labels = qpp_wins['true_label'].value_counts().sort_index()
        x = np.arange(len([0, 1]))
        width = 0.35
        ax3.bar(x - width/2, [int_labels.get(0, 0), int_labels.get(1, 0)], width, 
               label='Integrated Wins', color='blue', alpha=0.7, edgecolor='black')
        ax3.bar(x + width/2, [qpp_labels.get(0, 0), qpp_labels.get(1, 0)], width,
               label='QPP Wins', color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('True Label (0=Non-ambiguous, 1=Ambiguous)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax3.set_title('True Label Distribution\n(by Win Pattern)', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Non-ambiguous (0)', 'Ambiguous (1)'])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 閾値依存性の可視化（曖昧性率の変化）
        ax4 = fig_thresh.add_subplot(gs_thresh[1, 0])
        if threshold_analysis_results:
            threshold_df = pd.DataFrame(threshold_analysis_results)
            ax4.plot(threshold_df['threshold'], threshold_df['integrated_wins_ambiguity_rate'], 
                    'o-', label='Integrated Wins', color='blue', linewidth=2, markersize=8)
            ax4.plot(threshold_df['threshold'], threshold_df['qpp_wins_ambiguity_rate'], 
                    's-', label='QPP Wins', color='orange', linewidth=2, markersize=8)
            ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
            ax4.set_xlabel('Prediction Threshold', fontsize=11, fontweight='bold')
            ax4.set_ylabel('Ambiguity Rate (True Label = 1)', fontsize=11, fontweight='bold')
            ax4.set_title('Threshold Dependency: Ambiguity Rate\n(Does pattern depend on threshold?)', 
                         fontsize=12, fontweight='bold')
            ax4.legend(fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
        
        # 5. 閾値依存性の可視化（勝率の変化）
        ax5 = fig_thresh.add_subplot(gs_thresh[1, 1])
        if threshold_analysis_results:
            threshold_df = pd.DataFrame(threshold_analysis_results)
            total_n = len(merged_data)
            int_win_rate = threshold_df['n_integrated_wins'] / total_n
            qpp_win_rate = threshold_df['n_qpp_wins'] / total_n
            ax5.plot(threshold_df['threshold'], int_win_rate, 
                    'o-', label='Integrated Wins Rate', color='blue', linewidth=2, markersize=8)
            ax5.plot(threshold_df['threshold'], qpp_win_rate, 
                    's-', label='QPP Wins Rate', color='orange', linewidth=2, markersize=8)
            ax5.set_xlabel('Prediction Threshold', fontsize=11, fontweight='bold')
            ax5.set_ylabel('Win Rate', fontsize=11, fontweight='bold')
            ax5.set_title('Threshold Dependency: Win Rate\n(How does win rate change with threshold?)', 
                         fontsize=12, fontweight='bold')
            ax5.legend(fontsize=9)
            ax5.grid(True, alpha=0.3)
        
        # 6. 予測確率の差の分布（閾値非依存）
        ax6 = fig_thresh.add_subplot(gs_thresh[1, 2])
        merged_data['prob_diff'] = merged_data[integrated_name] - merged_data['qpp_prob_normalized']
        ax6.hist(integrated_wins['prob_diff'], bins=30, alpha=0.6, label=f'Integrated Wins (n={len(integrated_wins)})', 
                color='blue', edgecolor='black')
        ax6.hist(qpp_wins['prob_diff'], bins=30, alpha=0.6, label=f'QPP Wins (n={len(qpp_wins)})', 
                color='orange', edgecolor='black')
        ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
        ax6.set_xlabel('Probability Difference (Integrated - Single QPP)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax6.set_title('Probability Difference Distribution\n(Threshold-free analysis)', 
                     fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 統計情報をテキストで表示
        fig_thresh.text(0.5, 0.02, 
                       f'Key Finding: Integrated Wins ambiguity rate = {integrated_wins["true_label"].mean():.1%}, '
                       f'QPP Wins ambiguity rate = {qpp_wins["true_label"].mean():.1%}\n'
                       f'This pattern is {"threshold-dependent" if len(threshold_analysis_results) > 0 and abs(threshold_df["ambiguity_rate_difference"].std()) > 0.1 else "relatively threshold-independent"}',
                       ha='center', fontsize=11, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path_thresh = output_dir / 'threshold_dependency_analysis.png'
    plt.savefig(output_path_thresh, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved threshold dependency analysis to {output_path_thresh}")
    
    # 仮説検証形式で結果を返す
    hypothesis13_results = []
    
    # 統合モデルが勝つケース（Integrated Wins）と単体QPP指標が勝つケース（QPP Wins）の特徴の違いを統計的に検定
    integrated_wins = merged_data[merged_data['pattern'] == 'Integrated Wins']
    qpp_wins = merged_data[merged_data['pattern'] == 'QPP Wins']
    
    if len(integrated_wins) > 20 and len(qpp_wins) > 20:
        # QAペア数の違いを検定
        from scipy.stats import mannwhitneyu
        
        # QAペア数
        qa_wins_int = integrated_wins['qa_pairs_count'].dropna()
        qa_wins_qpp = qpp_wins['qa_pairs_count'].dropna()
        if len(qa_wins_int) > 0 and len(qa_wins_qpp) > 0:
            try:
                u_stat, p_value_qa = mannwhitneyu(qa_wins_int, qa_wins_qpp, alternative='two-sided')
                hypothesis13_results.append({
                    'comparison': 'Integrated Wins vs QPP Wins',
                    'feature': 'QA Pairs Count',
                    'integrated_wins_mean': qa_wins_int.mean(),
                    'qpp_wins_mean': qa_wins_qpp.mean(),
                    'mean_difference': qa_wins_int.mean() - qa_wins_qpp.mean(),
                    'u_statistic': u_stat,
                    'p_value': p_value_qa,
                    'significant': p_value_qa < 0.05,
                    'n_integrated_wins': len(qa_wins_int),
                    'n_qpp_wins': len(qa_wins_qpp)
                })
            except:
                pass
        
        # 真のラベルの違いを検定（比率の差の検定）
        true_label_int = integrated_wins['true_label'].mean()
        true_label_qpp = qpp_wins['true_label'].mean()
        n_int = len(integrated_wins)
        n_qpp = len(qpp_wins)
        
        p1 = true_label_int
        p2 = true_label_qpp
        p_pooled = (integrated_wins['true_label'].sum() + qpp_wins['true_label'].sum()) / (n_int + n_qpp) if (n_int + n_qpp) > 0 else 0
        
        if p_pooled > 0 and p_pooled < 1:
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_int + 1/n_qpp))
            if se > 0:
                z_stat = (p1 - p2) / se
                p_value_label = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                
                hypothesis13_results.append({
                    'comparison': 'Integrated Wins vs QPP Wins',
                    'feature': 'True Label (Ambiguity Rate)',
                    'integrated_wins_mean': true_label_int,
                    'qpp_wins_mean': true_label_qpp,
                    'mean_difference': true_label_int - true_label_qpp,
                    'z_statistic': z_stat,
                    'p_value': p_value_label,
                    'significant': p_value_label < 0.05,
                    'n_integrated_wins': n_int,
                    'n_qpp_wins': n_qpp
                })
        
        # 予測確率の差の違いを検定
        prob_diff_int = integrated_wins['prob_diff'].dropna()
        prob_diff_qpp = qpp_wins['prob_diff'].dropna()
        if len(prob_diff_int) > 0 and len(prob_diff_qpp) > 0:
            try:
                u_stat, p_value_diff = mannwhitneyu(prob_diff_int, prob_diff_qpp, alternative='two-sided')
                hypothesis13_results.append({
                    'comparison': 'Integrated Wins vs QPP Wins',
                    'feature': 'Prediction Probability Difference',
                    'integrated_wins_mean': prob_diff_int.mean(),
                    'qpp_wins_mean': prob_diff_qpp.mean(),
                    'mean_difference': prob_diff_int.mean() - prob_diff_qpp.mean(),
                    'u_statistic': u_stat,
                    'p_value': p_value_diff,
                    'significant': p_value_diff < 0.05,
                    'n_integrated_wins': len(prob_diff_int),
                    'n_qpp_wins': len(prob_diff_qpp)
                })
            except:
                pass
        
        # 統合モデルが勝つケースの割合が統計的に有意に多いか
        n_total = len(merged_data)
        n_int_wins = len(integrated_wins)
        n_qpp_wins = len(qpp_wins)
        
        # 統合モデルが勝つケースの割合
        p_int_wins = n_int_wins / n_total if n_total > 0 else 0
        p_qpp_wins = n_qpp_wins / n_total if n_total > 0 else 0
        
        # 比率の差の検定
        p_pooled_wins = (n_int_wins + n_qpp_wins) / (2 * n_total) if n_total > 0 else 0
        if p_pooled_wins > 0 and p_pooled_wins < 1:
            se_wins = np.sqrt(p_pooled_wins * (1 - p_pooled_wins) * (2 / n_total)) if n_total > 0 else 0
            if se_wins > 0:
                z_stat_wins = (p_int_wins - p_qpp_wins) / se_wins
                p_value_wins = 2 * (1 - stats.norm.cdf(abs(z_stat_wins)))
                
                hypothesis13_results.append({
                    'comparison': 'Overall',
                    'feature': 'Win Rate (Integrated Wins vs QPP Wins)',
                    'integrated_wins_rate': p_int_wins,
                    'qpp_wins_rate': p_qpp_wins,
                    'rate_difference': p_int_wins - p_qpp_wins,
                    'z_statistic': z_stat_wins,
                    'p_value': p_value_wins,
                    'significant': p_value_wins < 0.05,
                    'n_integrated_wins': n_int_wins,
                    'n_qpp_wins': n_qpp_wins,
                    'n_total': n_total
                })
    
    return hypothesis13_results


def print_hypothesis_summary(results: Dict[str, pd.DataFrame]):
    """統合モデルの優位性を示すサマリーを表示"""
    print("\n" + "="*80)
    print("INTEGRATED MODEL ADVANTAGES - HYPOTHESIS TESTING SUMMARY")
    print("="*80)
    
    if 'hypothesis1' in results and len(results['hypothesis1']) > 0:
        print("\n[Hypothesis 1] Integrated model excels when QPP metrics are inconsistent:")
        h1_df = results['hypothesis1']
        for _, row in h1_df.iterrows():
            better_mark = "✓" if row['integrated_better'] else "✗"
            int_sig = "*" if row['integrated_significant'] else ""
            qpp_sig = "*" if row['best_qpp_significant'] else ""
            print(f"  {row['inconsistency_range']} {better_mark}:")
            print(f"    - Integrated: ρ = {row['integrated_spearman_rho']:.4f}, p = {row['integrated_spearman_p']:.4f}{int_sig}")
            if row['best_qpp_metric']:
                print(f"    - Best QPP ({row['best_qpp_metric']}): ρ = {row['best_qpp_spearman_rho']:.4f}, "
                      f"p = {row['best_qpp_spearman_p']:.4f}{qpp_sig}")
                print(f"    - Improvement: {row['improvement']:+.4f}")
    
    if 'hypothesis2' in results and len(results['hypothesis2']) > 0:
        print("\n[Hypothesis 2] Integrated model excels in specific QA pairs count ranges:")
        h2_df = results['hypothesis2']
        for _, row in h2_df.iterrows():
            better_mark = "✓" if row['integrated_better'] else "✗"
            int_sig = "*" if row['integrated_significant'] else ""
            qpp_sig = "*" if row['best_qpp_significant'] else ""
            print(f"  {row['qa_range']} {better_mark}:")
            print(f"    - Integrated: ρ = {row['integrated_spearman_rho']:.4f}, p = {row['integrated_spearman_p']:.4f}{int_sig}")
            if row['best_qpp_metric']:
                print(f"    - Best QPP ({row['best_qpp_metric']}): ρ = {row['best_qpp_spearman_rho']:.4f}, "
                      f"p = {row['best_qpp_spearman_p']:.4f}{qpp_sig}")
                print(f"    - Improvement: {row['improvement']:+.4f}")
    
    if 'hypothesis3' in results and len(results['hypothesis3']) > 0:
        print("\n[Hypothesis 3] Integrated model achieves statistical significance more often:")
        h3_df = results['hypothesis3']
        for _, row in h3_df.iterrows():
            int_sig = "*" if row['integrated_significant'] else ""
            qpp_sig = "*" if row['best_qpp_significant'] else ""
            print(f"  {row['category']}:")
            print(f"    - Integrated: ρ = {row['integrated_spearman_rho']:.4f}, p = {row['integrated_spearman_p']:.4f}{int_sig}")
            if row['best_qpp_metric']:
                print(f"    - Best QPP ({row['best_qpp_metric']}): ρ = {row['best_qpp_spearman_rho']:.4f}, "
                      f"p = {row['best_qpp_spearman_p']:.4f}{qpp_sig}")
        
        if 'hypothesis3_summary' in results and len(results['hypothesis3_summary']) > 0:
            summary = results['hypothesis3_summary'].iloc[0]
            print(f"\n  Summary:")
            print(f"    - Overall: Integrated significant = {summary['integrated_all_significant']}, "
                  f"QPP significant = {summary['qpp_significant_count']}/{summary['qpp_total_count']} ({summary['qpp_significant_ratio']:.1%})")
            print(f"    - Subgroups: Integrated = {summary['integrated_subgroups_significant']}/{summary['integrated_subgroups_total']} "
                  f"({summary['integrated_subgroups_ratio']:.1%}), "
                  f"QPP = {summary['qpp_subgroups_significant']}/{summary['qpp_subgroups_total']} ({summary['qpp_subgroups_ratio']:.1%})")
            better_mark = "✓" if summary['integrated_better'] else "✗"
            print(f"    - Integrated better: {better_mark}")
    
    if 'hypothesis4' in results and len(results['hypothesis4']) > 0:
        print("\n[Hypothesis 4] Integrated model responds better to QPP combination patterns:")
        h4_df = results['hypothesis4']
        for _, row in h4_df.iterrows():
            better_mark = "✓" if row['integrated_better'] else "✗"
            int_sig = "*" if row['integrated_significant'] else ""
            qpp_sig = "*" if row['best_qpp_significant'] else ""
            print(f"  {row['pattern']} {better_mark}:")
            print(f"    - Integrated: ρ = {row['integrated_spearman_rho']:.4f}, p = {row['integrated_spearman_p']:.4f}{int_sig}")
            if row['best_qpp_metric']:
                print(f"    - Best QPP ({row['best_qpp_metric']}): ρ = {row['best_qpp_spearman_rho']:.4f}, "
                      f"p = {row['best_qpp_spearman_p']:.4f}{qpp_sig}")
                print(f"    - Improvement: {row['improvement']:+.4f}")
    
    if 'hypothesis5' in results and len(results['hypothesis5']) > 0:
        print("\n[Hypothesis 5] Integrated model performance in prediction task (ambiguity classification):")
        h5_df = results['hypothesis5'].iloc[0]
        better_auc = "✓" if h5_df['integrated_better_auc'] else "✗"
        better_ap = "✓" if h5_df['integrated_better_ap'] else "✗"
        better_f1 = "✓" if h5_df['integrated_better_f1'] else "✗"
        print(f"  Integrated Model {better_auc}:")
        print(f"    - AUC: {h5_df['integrated_auc']:.4f} (vs Best QPP: {h5_df['best_qpp_auc']:.4f}, "
              f"Improvement: {h5_df['auc_improvement_vs_best']:+.4f})")
        print(f"    - AP: {h5_df['integrated_ap']:.4f} (vs Best QPP: {h5_df['best_qpp_ap']:.4f}, "
              f"Improvement: {h5_df['ap_improvement_vs_best']:+.4f})")
        print(f"    - F1: {h5_df['integrated_f1']:.4f} (vs Best QPP: {h5_df['best_qpp_f1']:.4f}, "
              f"Improvement: {h5_df['f1_improvement_vs_best']:+.4f})")
        print(f"    - Accuracy: {h5_df['integrated_accuracy']:.4f} (vs Best QPP: {h5_df['best_qpp_accuracy']:.4f})")
        print(f"  Best Single QPP: {h5_df['best_qpp_metric']}")
    
    if 'hypothesis6' in results and len(results['hypothesis6']) > 0:
        print("\n[Hypothesis 6] Integrated model excels in specific prediction probability ranges:")
        h6_df = results['hypothesis6']
        for _, row in h6_df.iterrows():
            better_mark = "✓" if row['integrated_better_auc'] else "✗"
            print(f"  {row['prob_range']} {better_mark}:")
            print(f"    - Integrated AUC: {row['integrated_auc']:.4f} (vs Best QPP: {row['best_qpp_auc']:.4f}, "
                  f"Improvement: {row['auc_improvement']:+.4f})")
    
    if 'hypothesis7' in results and len(results['hypothesis7']) > 0:
        print("\n[Hypothesis 7] Integrated model excels in uncertain cases:")
        h7_df = results['hypothesis7']
        for _, row in h7_df.iterrows():
            better_mark = "✓" if row['integrated_better_auc'] else "✗"
            print(f"  {row['case_type']} {better_mark}:")
            print(f"    - Integrated AUC: {row['integrated_auc']:.4f} (vs Best QPP: {row['best_qpp_auc']:.4f}, "
                  f"Improvement: {row['auc_improvement']:+.4f})")
    
    if 'hypothesis8' in results and len(results['hypothesis8']) > 0:
        print("\n[Hypothesis 8] Integrated model excels when QPP metrics show extreme values:")
        h8_df = results['hypothesis8']
        for _, row in h8_df.iterrows():
            better_mark = "✓" if row['integrated_better_auc'] else "✗"
            print(f"  {row['extreme_range']} {better_mark}:")
            print(f"    - Integrated AUC: {row['integrated_auc']:.4f} (vs Best QPP: {row['best_qpp_auc']:.4f}, "
                  f"Improvement: {row['auc_improvement']:+.4f})")
    
    if 'hypothesis9' in results and len(results['hypothesis9']) > 0:
        print("\n[Hypothesis 9] Integrated model overall statistical significance (DeLong test):")
        h9_df = results['hypothesis9'].iloc[0]
        better_mark = "✓" if h9_df['integrated_better'] else "✗"
        sig_mark = "*" if h9_df['delong_significant'] else ""
        print(f"  Overall {better_mark}{sig_mark}:")
        print(f"    - Integrated AUC: {h9_df['integrated_auc']:.4f} (vs Best QPP ({h9_df['best_qpp_metric']}): {h9_df['best_qpp_auc']:.4f}, "
              f"Improvement: {h9_df['auc_improvement']:+.4f})")
        print(f"    - DeLong test: Z = {h9_df['delong_z_stat']:.4f}, p = {h9_df['delong_p_value']:.4f}")
        if h9_df['delong_significant']:
            print(f"    *STATISTICALLY SIGNIFICANT* (p < 0.05)")
    
    if 'hypothesis10' in results and len(results['hypothesis10']) > 0:
        print("\n[Hypothesis 10] Integrated model excels when QPP metrics have high variance (with statistical significance):")
        h10_df = results['hypothesis10']
        for _, row in h10_df.iterrows():
            better_mark = "✓" if row['integrated_better'] else "✗"
            sig_mark = "*" if row['delong_significant'] else ""
            print(f"  {row['variance_range']} {better_mark}{sig_mark}:")
            print(f"    - Integrated AUC: {row['integrated_auc']:.4f} (vs Best QPP: {row['best_qpp_auc']:.4f}, "
                  f"Improvement: {row['auc_improvement']:+.4f})")
            if row['delong_significant']:
                print(f"    - DeLong test: p = {row['delong_p_value']:.4f} (Z = {row['delong_z_stat']:.4f}) *STATISTICALLY SIGNIFICANT*")
    
    if 'hypothesis11' in results and len(results['hypothesis11']) > 0:
        print("\n[Hypothesis 11] Integrated model has better distribution separation (threshold-free):")
        h11_df = results['hypothesis11'].iloc[0]
        sep_mark = "✓" if h11_df['integrated_better_separation'] else "✗"
        overlap_mark = "✓" if h11_df['integrated_better_overlap'] else "✗"
        print(f"  Distribution Separation {sep_mark}:")
        print(f"    - Integrated: {h11_df['integrated_separation']:.4f} (vs Best QPP ({h11_df['best_qpp_metric']}): {h11_df['best_qpp_separation']:.4f}, "
              f"Improvement: {h11_df['separation_improvement']:+.4f})")
        print(f"  Distribution Overlap {overlap_mark}:")
        print(f"    - Integrated: {h11_df['integrated_overlap']:.4f} (vs Best QPP: {h11_df['best_qpp_overlap']:.4f}, "
              f"Improvement: {h11_df['overlap_improvement']:+.4f}, lower is better)")
    
    if 'hypothesis12' in results and len(results['hypothesis12']) > 0:
        print("\n[Hypothesis 12] Integrated model has lower error rate in specific conditions:")
        h12_df = results['hypothesis12']
        
        # 条件タイプごとにグループ化
        for condition_type in h12_df['condition_type'].unique():
            type_df = h12_df[h12_df['condition_type'] == condition_type]
            print(f"\n  {condition_type}:")
            for _, row in type_df.iterrows():
                better_mark = "✓" if row['integrated_better'] else "✗"
                sig_mark = "*" if row['significant'] else ""
                print(f"    {row['condition']} {better_mark}{sig_mark}:")
                print(f"      - Integrated error rate: {row['integrated_error_rate']:.4f} (vs QPP: {row['qpp_error_rate']:.4f}, "
                      f"Improvement: {row['error_improvement']:+.4f})")
                if row['significant']:
                    print(f"      - Z-test: Z = {row['z_stat']:.4f}, p = {row['p_value']:.4f} *STATISTICALLY SIGNIFICANT*")
        
        # サマリー
        total_cases = len(h12_df)
        better_cases = h12_df['integrated_better'].sum()
        significant_cases = h12_df['integrated_significantly_better'].sum()
        print(f"\n  Summary: Integrated better in {better_cases}/{total_cases} cases ({better_cases/total_cases*100:.1f}%)")
        if significant_cases > 0:
            print(f"  Statistically significant: {significant_cases}/{total_cases} cases ({significant_cases/total_cases*100:.1f}%)")
    
    if 'hypothesis13' in results and len(results['hypothesis13']) > 0:
        print("\n[Hypothesis 13] Individual query analysis: Integrated Wins vs QPP Wins characteristics:")
        h13_df = results['hypothesis13']
        
        for _, row in h13_df.iterrows():
            sig_mark = "*" if row['significant'] else ""
            print(f"  {row['feature']} {sig_mark}:")
            print(f"    - Integrated Wins: {row['integrated_wins_mean']:.4f} (n={row['n_integrated_wins']})")
            print(f"    - QPP Wins: {row['qpp_wins_mean']:.4f} (n={row['n_qpp_wins']})")
            print(f"    - Difference: {row['mean_difference']:+.4f}")
            if 'z_statistic' in row and not pd.isna(row['z_statistic']):
                print(f"    - Z-test: Z = {row['z_statistic']:.4f}, p = {row['p_value']:.4f}")
            elif 'u_statistic' in row and not pd.isna(row['u_statistic']):
                print(f"    - Mann-Whitney U: U = {row['u_statistic']:.4f}, p = {row['p_value']:.4f}")
            if row['significant']:
                print(f"    *STATISTICALLY SIGNIFICANT*")
        
        # サマリー
        total_comparisons = len(h13_df)
        significant_comparisons = h13_df['significant'].sum()
        print(f"\n  Summary: Statistically significant differences in {significant_comparisons}/{total_comparisons} features "
              f"({significant_comparisons/total_comparisons*100:.1f}%)")
        
        # 統合モデルが勝つケースの割合
        wins_row = h13_df[h13_df['feature'] == 'Win Rate (Integrated Wins vs QPP Wins)']
        if len(wins_row) > 0:
            wins_row = wins_row.iloc[0]
            print(f"  Integrated Wins rate: {wins_row['integrated_wins_rate']:.1%} vs QPP Wins rate: {wins_row['qpp_wins_rate']:.1%}")
            if wins_row['significant']:
                print(f"  * Integrated model wins significantly more often (p = {wins_row['p_value']:.4f})")
    
    print("\n" + "="*80)
    print("SUMMARY: Where Integrated Model Excels")
    print("="*80)
    
    # 統合モデルが優れているケースをカウント
    total_cases = 0
    better_cases = 0
    
    for name, df_result in results.items():
        if name in ['hypothesis1', 'hypothesis2', 'hypothesis4'] and len(df_result) > 0:
            if 'integrated_better' in df_result.columns:
                total_cases += len(df_result)
                better_cases += df_result['integrated_better'].sum()
        elif name == 'hypothesis3_summary' and len(df_result) > 0:
            if 'integrated_better' in df_result.columns:
                total_cases += 1
                if df_result.iloc[0]['integrated_better']:
                    better_cases += 1
        elif name == 'hypothesis5' and len(df_result) > 0:
            h5 = df_result.iloc[0]
            total_cases += 3  # AUC, AP, F1
            if h5['integrated_better_auc']:
                better_cases += 1
            if h5['integrated_better_ap']:
                better_cases += 1
            if h5['integrated_better_f1']:
                better_cases += 1
        elif name in ['hypothesis6', 'hypothesis7', 'hypothesis8'] and len(df_result) > 0:
            if 'integrated_better_auc' in df_result.columns:
                total_cases += len(df_result)
                better_cases += df_result['integrated_better_auc'].sum()
        elif name == 'hypothesis9' and len(df_result) > 0:
            h9 = df_result.iloc[0]
            total_cases += 1
            if h9['integrated_better']:
                better_cases += 1
            if h9['delong_significant']:
                print(f"  * Hypothesis 9: Statistically significant (p = {h9['delong_p_value']:.4f})")
        elif name == 'hypothesis10' and len(df_result) > 0:
            if 'integrated_better' in df_result.columns:
                total_cases += len(df_result)
                better_cases += df_result['integrated_better'].sum()
            # 統計的有意なケースもカウント
            if 'integrated_significantly_better' in df_result.columns:
                significant_cases = df_result['integrated_significantly_better'].sum()
                if significant_cases > 0:
                    print(f"  * Hypothesis 10: Statistically significant improvements: {significant_cases}/{len(df_result)} cases")
        elif name == 'hypothesis11' and len(df_result) > 0:
            h11 = df_result.iloc[0]
            total_cases += 2  # Separation and Overlap
            if h11['integrated_better_separation']:
                better_cases += 1
            if h11['integrated_better_overlap']:
                better_cases += 1
        elif name == 'hypothesis12' and len(df_result) > 0:
            if 'integrated_better' in df_result.columns:
                total_cases += len(df_result)
                better_cases += df_result['integrated_better'].sum()
            if 'integrated_significantly_better' in df_result.columns:
                significant_cases = df_result['integrated_significantly_better'].sum()
                if significant_cases > 0:
                    print(f"  * Hypothesis 12: Statistically significant improvements: {significant_cases}/{len(df_result)} cases")
        elif name == 'hypothesis13' and len(df_result) > 0:
            # 統合モデルが勝つケースの割合が統計的有意に多いか
            wins_row = df_result[df_result['feature'] == 'Win Rate (Integrated Wins vs QPP Wins)']
            if len(wins_row) > 0:
                wins_row = wins_row.iloc[0]
                total_cases += 1
                if wins_row['rate_difference'] > 0:
                    better_cases += 1
                if wins_row['significant']:
                    print(f"  * Hypothesis 13: Integrated model wins significantly more often "
                          f"({wins_row['integrated_wins_rate']:.1%} vs {wins_row['qpp_wins_rate']:.1%}, p = {wins_row['p_value']:.4f})")
    
    if total_cases > 0:
        print(f"\nIntegrated model is better in {better_cases}/{total_cases} cases ({better_cases/total_cases*100:.1f}%)")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QPP metrics vs integrated model with new hypothesis testing"
    )
    parser.add_argument('--dataset_dir', type=Path, required=True, help='Dataset directory')
    parser.add_argument('--logreg_output_dir', type=Path, required=True, help='LogReg output directory')
    parser.add_argument('--output_dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'train'], help='Dataset split')
    parser.add_argument('--feature_combination', type=str, default=None, help='Feature combination directory name')
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # データを読み込む
    data_path = args.dataset_dir / f'{args.split}.json'
    qa_pairs_dict = load_qa_pairs_count(data_path)
    qpp_scores_dict = load_qpp_scores(args.dataset_dir, split=args.split)
    integrated_scores, true_labels = load_integrated_score(args.logreg_output_dir, data_path, args.feature_combination)
    
    # データフレームに結合
    qpp_metric_names = list(qpp_scores_dict.keys())
    data_list = []
    all_keys = set(qa_pairs_dict.keys()) | set(integrated_scores.keys())
    for key in all_keys:
        row = {
            'conv_id': key[0],
            'turn_id': key[1],
            'qa_pairs_count': qa_pairs_dict.get(key, 0),
            'QPP Integrated': integrated_scores.get(key, np.nan)
        }
        for metric in qpp_metric_names:
            row[metric] = qpp_scores_dict[metric].get(key, np.nan)
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    # QAペア数カテゴリを作成
    def create_category(count):
        if count == 0:
            return '0 QA pair'
        elif count == 1:
            return '1 QA pair'
        elif count == 2:
            return '2 QA pairs'
        elif count == 3:
            return '3 QA pairs'
        else:
            return '4+ QA pairs'
    
    df['qa_pairs_category'] = df['qa_pairs_count'].apply(create_category)
    
    print(f"\n[INFO] Total entries: {len(df)}")
    print(f"[INFO] QA pairs count distribution:\n{df['qa_pairs_category'].value_counts()}")
    
    # 仮説検証を実行
    perform_hypothesis_testing(df, qpp_metric_names, 'QPP Integrated', args.output_dir, true_labels)
    
    print(f"\n[INFO] Analysis complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
