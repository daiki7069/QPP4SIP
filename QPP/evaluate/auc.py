"""
AUC算出とROC曲線の可視化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score
from typing import Dict, List, Optional, Tuple


def calculate_roc_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    ROC曲線を計算する
    
    Args:
        y_true: 真のラベル（0または1）
        y_scores: 予測スコア（高いほど正例である可能性が高い）
    
    Returns:
        fpr: False Positive Rateの配列
        tpr: True Positive Rateの配列
        thresholds: 閾値の配列
        auc_score: AUC値
    """
    # NaN値を除外
    mask = ~(np.isnan(y_scores) | np.isnan(y_true))
    y_true_clean = y_true[mask]
    y_scores_clean = y_scores[mask]
    
    if len(y_true_clean) == 0:
        return np.array([]), np.array([]), np.array([]), 0.0
    
    # ROC曲線を計算
    fpr, tpr, thresholds = roc_curve(y_true_clean, y_scores_clean)
    
    # AUCを計算
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, auc_score


def create_binary_labels(response_types: pd.Series, positive_class: str = 'clarification') -> np.ndarray:
    """
    response_typeから二値ラベルを作成する
    
    Args:
        response_types: response_typeのSeries
        positive_class: 正例とするクラス（デフォルト: 'clarification'）
    
    Returns:
        二値ラベルの配列（1: 正例, 0: 負例）
    """
    # [SEP]で分割されている場合があるので、最初の部分を取得
    response_types_clean = response_types.str.split(' [SEP] ', regex=False).str[0].str.strip()
    
    # 正例を1、それ以外を0に変換
    binary_labels = (response_types_clean == positive_class).astype(int).values
    
    return binary_labels


def plot_roc_curves(
    csv_path: str,
    metric_csv_paths: Dict[str, str],
    output_path: str,
    positive_class: str = 'clarification',
    merge_config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    複数のQPPメトリクスについてROC曲線を描画し、AUCを計算する
    
    Args:
        csv_path: csvのパス
        metric_csv_paths: メトリクス名とCSVパスの辞書（例: {'nqc': 'path/to/nqc.csv', 'entropy': 'path/to/entropy.csv'}）
        output_path: 出力画像のパス
        positive_class: 正例とするresponse_type（デフォルト: 'clarification'）
        merge_config: マージ設定（Noneの場合はデフォルト値を使用）
    
    Returns:
        メトリクス名とAUC値の辞書
    """
    # デフォルトのマージ設定
    if merge_config is None:
        merge_config = {
            'left_on': ['dialogue_id', 'turn_id'],
            'right_on': ['conv_id', 'turn_id'],
            'how': 'inner'
        }
    
    # dev.csvを読み込む
    df = pd.read_csv(csv_path)
    
    # 二値ラベルを作成
    if 'response_type' not in df.columns:
        raise ValueError("dev.csvに'response_type'カラムが見つかりません")
    
    y_true = create_binary_labels(df['response_type'], positive_class=positive_class)
    
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 図を作成（凡例用に余白を確保）
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 各メトリクスについてROC曲線を計算・描画
    auc_scores = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_csv_paths)))
    
    for i, (metric_name, csv_path) in enumerate(metric_csv_paths.items()):
        # メトリクスCSVを読み込む
        metric_df = pd.read_csv(csv_path)
        
        # データをマージ
        merged_df = df.merge(
            metric_df,
            left_on=merge_config['left_on'],
            right_on=merge_config['right_on'],
            how=merge_config['how']
        )
        
        # スコアカラムを取得（メトリクス名に応じて）
        score_col = None
        if metric_name == 'nqc':
            score_col = 'nqc'
        elif metric_name == 'entropy':
            score_col = 'entropy'
        elif metric_name == 'unique_titles':
            score_col = 'num_unique_titles'
        elif metric_name == 'lci':
            score_col = 'lci'
        elif metric_name == 'similarity':
            score_col = 'mean_similarity'
        else:
            # メトリクス名がカラム名と一致する場合
            if metric_name in merged_df.columns:
                score_col = metric_name
            else:
                raise ValueError(f"メトリクス '{metric_name}' に対応するカラムが見つかりません")
        
        if score_col not in merged_df.columns:
            print(f"警告: {csv_path}に'{score_col}'カラムが見つかりません。スキップします。")
            continue
        
        # マージ後のラベルを取得
        merged_y_true = create_binary_labels(merged_df['response_type'], positive_class=positive_class)
        y_scores = merged_df[score_col].values
        
        # ROC曲線を計算
        fpr, tpr, thresholds, auc_score = calculate_roc_curve(merged_y_true, y_scores)
        
        if len(fpr) > 0:
            # ROC曲線を描画
            ax.plot(
                fpr, tpr,
                color=colors[i],
                lw=2,
                label=f'{metric_name.upper()} (AUC = {auc_score:.3f})'
            )
            auc_scores[metric_name] = auc_score
        else:
            print(f"警告: {metric_name}のROC曲線を計算できませんでした（有効なデータがありません）")
    
    # 対角線（ランダム分類器）を描画
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
    
    # グラフの設定
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(f'ROC Curves for QPP Metrics\n(Positive Class: {positive_class})', fontsize=14, fontweight='bold')
    
    # 凡例をグラフの中に配置
    num_metrics = len(auc_scores) + 1  # +1 for Random
    ncol = min(3, max(1, (num_metrics + 1) // 2))  # 最大3列、必要に応じて調整
    ax.legend(
        loc='lower right',
        fontsize=9,
        ncol=1,
        framealpha=0.3  # 半透明の背景
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"ROC曲線を保存しました: {output_path}")
    
    # AUC値を表示
    print("\n=== AUC値 ===")
    for metric_name, auc_value in auc_scores.items():
        print(f"{metric_name.upper()}: {auc_value:.4f}")
    
    plt.show()
    
    return auc_scores


def calculate_pr_curve(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Precision-Recall曲線を計算する
    
    Args:
        y_true: 真のラベル（0または1）
        y_scores: 予測スコア（高いほど正例である可能性が高い）
    
    Returns:
        precision: Precisionの配列
        recall: Recallの配列
        thresholds: 閾値の配列
        ap_score: Average Precision値
    """
    # NaN値を除外
    mask = ~(np.isnan(y_scores) | np.isnan(y_true))
    y_true_clean = y_true[mask]
    y_scores_clean = y_scores[mask]
    
    if len(y_true_clean) == 0:
        return np.array([]), np.array([]), np.array([]), 0.0
    
    # Precision-Recall曲線を計算
    precision, recall, thresholds = precision_recall_curve(y_true_clean, y_scores_clean)
    
    # Average Precisionを計算
    ap_score = average_precision_score(y_true_clean, y_scores_clean)
    
    return precision, recall, thresholds, ap_score


def plot_pr_curves(
    csv_path: str,
    metric_csv_paths: Dict[str, str],
    output_path: str,
    positive_class: str = 'clarification',
    merge_config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    複数のQPPメトリクスについてPrecision-Recall曲線を描画し、Average Precisionを計算する
    
    Args:
        csv_path: csvのパス
        metric_csv_paths: メトリクス名とCSVパスの辞書（例: {'nqc': 'path/to/nqc.csv', 'entropy': 'path/to/entropy.csv'}）
        output_path: 出力画像のパス
        positive_class: 正例とするresponse_type（デフォルト: 'clarification'）
        merge_config: マージ設定（Noneの場合はデフォルト値を使用）
    
    Returns:
        メトリクス名とAverage Precision値の辞書
    """
    # デフォルトのマージ設定
    if merge_config is None:
        merge_config = {
            'left_on': ['dialogue_id', 'turn_id'],
            'right_on': ['conv_id', 'turn_id'],
            'how': 'inner'
        }
    
    # dev.csvを読み込む
    df = pd.read_csv(csv_path)
    
    # 二値ラベルを作成
    if 'response_type' not in df.columns:
        raise ValueError("dev.csvに'response_type'カラムが見つかりません")
    
    y_true = create_binary_labels(df['response_type'], positive_class=positive_class)
    
    # ベースライン（ランダム分類器）を計算
    baseline = np.sum(y_true) / len(y_true)
    
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # 図を作成（凡例用に余白を確保）
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 各メトリクスについてPR曲線を計算・描画
    ap_scores = {}
    colors = plt.cm.tab10(np.linspace(0, 1, len(metric_csv_paths)))
    
    for i, (metric_name, csv_path) in enumerate(metric_csv_paths.items()):
        # メトリクスCSVを読み込む
        metric_df = pd.read_csv(csv_path)
        
        # データをマージ
        merged_df = df.merge(
            metric_df,
            left_on=merge_config['left_on'],
            right_on=merge_config['right_on'],
            how=merge_config['how']
        )
        
        # スコアカラムを取得（メトリクス名に応じて）
        score_col = None
        if metric_name == 'nqc':
            score_col = 'nqc'
        elif metric_name == 'entropy':
            score_col = 'entropy'
        elif metric_name == 'unique_titles':
            score_col = 'num_unique_titles'
        elif metric_name == 'lci':
            score_col = 'lci'
        elif metric_name == 'similarity':
            score_col = 'mean_similarity'
        else:
            # メトリクス名がカラム名と一致する場合
            if metric_name in merged_df.columns:
                score_col = metric_name
            else:
                raise ValueError(f"メトリクス '{metric_name}' に対応するカラムが見つかりません")
        
        if score_col not in merged_df.columns:
            print(f"警告: {csv_path}に'{score_col}'カラムが見つかりません。スキップします。")
            continue
        
        # マージ後のラベルを取得
        merged_y_true = create_binary_labels(merged_df['response_type'], positive_class=positive_class)
        y_scores = merged_df[score_col].values
        
        # PR曲線を計算
        precision, recall, thresholds, ap_score = calculate_pr_curve(merged_y_true, y_scores)
        
        if len(precision) > 0:
            # PR曲線を描画
            ax.plot(
                recall, precision,
                color=colors[i],
                lw=2,
                label=f'{metric_name.upper()} (AP = {ap_score:.3f})'
            )
            ap_scores[metric_name] = ap_score
        else:
            print(f"警告: {metric_name}のPR曲線を計算できませんでした（有効なデータがありません）")
    
    # ベースライン（ランダム分類器）を描画
    ax.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Random (AP = {baseline:.3f})')
    
    # グラフの設定
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curves for QPP Metrics\n(Positive Class: {positive_class})', fontsize=14, fontweight='bold')
    
    # 凡例をグラフの中に配置
    num_metrics = len(ap_scores) + 1  # +1 for Random
    ncol = min(3, max(1, (num_metrics + 1) // 2))  # 最大3列、必要に応じて調整
    ax.legend(
        loc='lower left',
        fontsize=9,
        ncol=1,
        framealpha=0.3  # 半透明の背景
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PR曲線を保存しました: {output_path}")
    
    # Average Precision値を表示
    print("\n=== Average Precision値 ===")
    for metric_name, ap_value in ap_scores.items():
        print(f"{metric_name.upper()}: {ap_value:.4f}")
    
    plt.show()
    
    return ap_scores


def plot_individual_roc_curves(
    csv_path: str,
    metric_csv_paths: Dict[str, str],
    output_dir: str,
    positive_class: str = 'clarification',
    merge_config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    各メトリクスについて個別にROC曲線を描画する
    
    Args:
        csv_path: csvのパス
        metric_csv_paths: メトリクス名とCSVパスの辞書
        output_dir: 出力ディレクトリ
        positive_class: 正例とするresponse_type
        merge_config: マージ設定
    
    Returns:
        メトリクス名とAUC値の辞書
    """
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # デフォルトのマージ設定
    if merge_config is None:
        merge_config = {
            'left_on': ['dialogue_id', 'turn_id'],
            'right_on': ['conv_id', 'turn_id'],
            'how': 'inner'
        }
    
    # dev.csvを読み込む
    df = pd.read_csv(csv_path)
    y_true = create_binary_labels(dev_df['response_type'], positive_class=positive_class)
    
    # 日本語フォントの設定
    plt.rcParams['font.family'] = 'DejaVu Sans'
    
    auc_scores = {}
    
    for metric_name, csv_path in metric_csv_paths.items():
        # メトリクスCSVを読み込む
        metric_df = pd.read_csv(csv_path)
        
        # データをマージ
        merged_df = df.merge(
            metric_df,
            left_on=merge_config['left_on'],
            right_on=merge_config['right_on'],
            how=merge_config['how']
        )
        
        # スコアカラムを取得
        score_col = None
        if metric_name == 'nqc':
            score_col = 'nqc'
        elif metric_name == 'entropy':
            score_col = 'entropy'
        elif metric_name == 'unique_titles':
            score_col = 'num_unique_titles'
        elif metric_name == 'lci':
            score_col = 'lci'
        elif metric_name == 'similarity':
            score_col = 'mean_similarity'
        else:
            if metric_name in merged_df.columns:
                score_col = metric_name
            else:
                print(f"警告: {metric_name}のスコアカラムが見つかりません。スキップします。")
                continue
        
        if score_col not in merged_df.columns:
            print(f"警告: {csv_path}に'{score_col}'カラムが見つかりません。スキップします。")
            continue
        
        # マージ後のラベルを取得
        merged_y_true = create_binary_labels(merged_df['response_type'], positive_class=positive_class)
        y_scores = merged_df[score_col].values
        
        # ROC曲線を計算
        fpr, tpr, thresholds, auc_score = calculate_roc_curve(merged_y_true, y_scores)
        
        if len(fpr) == 0:
            print(f"警告: {metric_name}のROC曲線を計算できませんでした")
            continue
        
        auc_scores[metric_name] = auc_score
        
        # 個別のROC曲線を描画
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{metric_name.upper()} (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=12)
        plt.ylabel('True Positive Rate (TPR)', fontsize=12)
        plt.title(f'ROC Curve: {metric_name.upper()}\n(Positive Class: {positive_class})', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10, framealpha=0.3)  # 半透明の背景
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存
        output_path = output_dir / f'roc_curve_{metric_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲線を保存しました: {output_path}")
        plt.close()
    
    return auc_scores

