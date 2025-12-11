"""
可視化関連のモジュール
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, f1_score


def get_feature_dir_name(feature_names: Optional[List[str]]) -> str:
    """
    特徴量名のリストからディレクトリ名を生成（階層構造）
    1階層目: カテゴリの組み合わせ（pre, post, pre_post, pre_bert, post_bert, pre_post_bertなど）
    2階層目以降: 各カテゴリの指標名
    
    Args:
        feature_names: 特徴量名のリスト（Noneの場合は'all'を返す）
    
    Returns:
        ディレクトリ名（例: 'pre/idf_ictf', 'pre_post/idf_ictf/nqc_wig', 'pre_post_bert/idf_ictf/nqc_wig/rob'）
    """
    if feature_names is None or len(feature_names) == 0:
        return 'all'
    
    # 特徴量をカテゴリ別に分類
    pre_features = []
    post_features = []
    base_features = []
    nsp_features = []
    other_features = []
    
    for name in sorted(feature_names):
        if name.startswith('pre_'):
            pre_features.append(name.replace('pre_', ''))
        elif name.startswith('nsp_'):
            nsp_features.append(name.replace('nsp_', ''))
        elif 'logit_clarification' in name:
            # base特徴量（モデル名を抽出）
            if 'roberta' in name:
                base_features.append('roberta')
            elif 'bert' in name:
                base_features.append('bert')
            else:
                base_features.append('base')
        elif name in ['nqc', 'wig', 'smv', 'nsv', 'clarity', 'similarity', 'entropy', 'lci', 'unique_titles', 'acc', 'wacc', 'n_sigma_50']:
            post_features.append(name)
        else:
            other_features.append(name)
    
    # カテゴリの有無を記録
    has_pre = len(pre_features) > 0
    has_post = len(post_features) > 0
    has_base = len(base_features) > 0
    has_nsp = len(nsp_features) > 0
    has_other = len(other_features) > 0
    
    # カテゴリがない場合は'all'
    if not any([has_pre, has_post, has_base, has_nsp, has_other]):
        return 'all'
    
    # 1階層目: カテゴリの組み合わせを決定
    category_parts = []
    if has_pre:
        category_parts.append('pre')
    if has_post:
        category_parts.append('post')
    if has_base:
        category_parts.append('bert')
    if has_nsp:
        category_parts.append('nsp')
    if has_other:
        category_parts.append('other')
    
    first_level = '_'.join(category_parts)
    
    # 2階層目: 全ての指標名を1つにまとめる
    all_metrics = []
    
    # pre特徴量
    if has_pre:
        # 指標名を短縮
        for f in pre_features:
            if f == 'simplified_clarity':
                all_metrics.append('scs')
            elif f == 'avgictf':
                all_metrics.append('ictf')
            elif f == 'avgidf':
                all_metrics.append('idf')
            elif f == 'maxscq':
                all_metrics.append('scq')
            else:
                all_metrics.append(f)
    
    # post特徴量
    if has_post:
        # 指標名を短縮
        for f in post_features:
            if f == 'n_sigma_50':
                all_metrics.append('ns50')
            elif f == 'unique_titles':
                all_metrics.append('ut')
            else:
                all_metrics.append(f)
    
    # base特徴量
    if has_base:
        unique_base = sorted(set(base_features))
        for model_name in unique_base:
            # モデル名を短縮（roberta -> rob, bert -> bert）
            if model_name == 'roberta':
                all_metrics.append('rob')
            else:
                all_metrics.append(model_name[:4])  # 最大4文字
    
    # nsp特徴量
    if has_nsp:
        all_metrics.extend(nsp_features)
    
    # other特徴量
    if has_other:
        all_metrics.extend(other_features)
    
    # ディレクトリ名を構築: 1階層目/2階層目（全指標を_で連結）
    if all_metrics:
        second_level = '_'.join(all_metrics)
        return f"{first_level}/{second_level}"
    else:
        return first_level


def plot_roc_curves(
    y_train: pd.Series,
    y_train_proba: np.ndarray,
    y_test: pd.Series,
    y_test_proba: np.ndarray,
    train_auc: float,
    test_auc: float,
    output_dir: Path,
    hide_train: bool = False,
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    show_single_metrics: bool = False,
    feature_names: Optional[List[str]] = None,
    y_test_single: pd.Series = None
) -> None:
    """ROC曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練データのROC曲線（存在する場合のみ）
    if y_train is not None and y_train_proba is not None:
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    
    # テストデータのROC曲線
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    # プロット
    figsize = (12, 10) if show_single_metrics else (10, 8)
    plt.figure(figsize=figsize)
    
    # モデルのROC曲線
    if not hide_train and y_train is not None and y_train_proba is not None and train_auc is not None:
        plt.plot(fpr_train, tpr_train, label=f'Model Train (AUC = {train_auc:.4f})', linewidth=2.5, color='blue')
    if test_auc is not None:
        label_prefix = "CV" if y_test_single is not None else "Test"
        plt.plot(fpr_test, tpr_test, label=f'Model {label_prefix} (AUC = {test_auc:.4f})', linewidth=2.5, color='red')
    
    # 各特徴量単体のROC曲線（オプション）
    if show_single_metrics and X_test is not None:
        # 単一指標のラベル（CV結果の場合はy_test_singleを使用）
        y_single_labels = y_test_single if y_test_single is not None else y_test
        
        for feature_name in X_test.columns:
            # テストデータ（またはCV結果）
            test_scores = X_test[feature_name].values
            fpr_test_single, tpr_test_single, _ = roc_curve(y_single_labels, test_scores)
            test_auc_single = roc_auc_score(y_single_labels, test_scores)
            label_prefix = "CV" if y_test_single is not None else "Test"
            
            # AUCが0.5未満の場合は反転させたバージョンも描画
            if test_auc_single < 0.5:
                # 元のROC曲線（点線で表示）
                plt.plot(fpr_test_single, tpr_test_single, 
                        label=f'{feature_name} ({label_prefix}, AUC = {test_auc_single:.4f}) [Original]', 
                        linewidth=1.5, alpha=0.5, linestyle=':', color='gray')
                # 反転させたROC曲線
                flipped_scores = 1.0 - test_scores
                fpr_flipped, tpr_flipped, _ = roc_curve(y_single_labels, flipped_scores)
                flipped_auc = roc_auc_score(y_single_labels, flipped_scores)
                plt.plot(fpr_flipped, tpr_flipped, 
                        label=f'{feature_name} ({label_prefix}, AUC = {flipped_auc:.4f}) [Flipped]', 
                        linewidth=1.5, alpha=0.7, linestyle='-')
            else:
                plt.plot(fpr_test_single, tpr_test_single, 
                        label=f'{feature_name} ({label_prefix}, AUC = {test_auc_single:.4f})', 
                        linewidth=1.5, alpha=0.7)
    
    # ランダム分類器
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    title = 'ROC Curves - Model and Single Metrics' if show_single_metrics else 'ROC Curves - L1 Regularized Logistic Regression'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9 if show_single_metrics else 11, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'roc_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def plot_pr_curves(
    y_train: pd.Series,
    y_train_proba: np.ndarray,
    y_test: pd.Series,
    y_test_proba: np.ndarray,
    train_ap: float,
    test_ap: float,
    output_dir: Path,
    hide_train: bool = False,
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    show_single_metrics: bool = False,
    feature_names: Optional[List[str]] = None,
    y_test_single: pd.Series = None
) -> None:
    """Precision-Recall曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練データのPR曲線（存在する場合のみ）
    if y_train is not None and y_train_proba is not None:
        precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    
    # テストデータのPR曲線
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)
    
    # ベースライン（ランダム分類器）
    baseline = len(y_test[y_test == 1]) / len(y_test)
    
    # プロット
    figsize = (12, 10) if show_single_metrics else (10, 8)
    plt.figure(figsize=figsize)
    
    # モデルのPR曲線
    if not hide_train and y_train is not None and y_train_proba is not None and train_ap is not None:
        plt.plot(recall_train, precision_train, label=f'Model Train (AP = {train_ap:.4f})', linewidth=2.5, color='blue')
    if test_ap is not None:
        label_prefix = "CV" if y_test_single is not None else "Test"
        plt.plot(recall_test, precision_test, label=f'Model {label_prefix} (AP = {test_ap:.4f})', linewidth=2.5, color='red')
    
    # 各特徴量単体のPR曲線（オプション）
    if show_single_metrics and X_test is not None:
        # 単一指標のラベル（CV結果の場合はy_test_singleを使用）
        y_single_labels = y_test_single if y_test_single is not None else y_test
        
        for feature_name in X_test.columns:
            # テストデータ（またはCV結果）
            test_scores = X_test[feature_name].values
            # ROCのAUCを計算して、0.5未満かどうかを判定
            test_auc_single = roc_auc_score(y_single_labels, test_scores)
            label_prefix = "CV" if y_test_single is not None else "Test"
            
            # AUCが0.5未満の場合は反転させたバージョンも描画
            if test_auc_single < 0.5:
                # 元のPR曲線（点線で表示）
                precision_test_single, recall_test_single, _ = precision_recall_curve(y_single_labels, test_scores)
                test_ap_single = average_precision_score(y_single_labels, test_scores)
                plt.plot(recall_test_single, precision_test_single, 
                        label=f'{feature_name} ({label_prefix}, AP = {test_ap_single:.4f}) [Original]', 
                        linewidth=1.5, alpha=0.5, linestyle=':', color='gray')
                # 反転させたPR曲線
                flipped_scores = 1.0 - test_scores
                precision_flipped, recall_flipped, _ = precision_recall_curve(y_single_labels, flipped_scores)
                flipped_ap = average_precision_score(y_single_labels, flipped_scores)
                plt.plot(recall_flipped, precision_flipped, 
                        label=f'{feature_name} ({label_prefix}, AP = {flipped_ap:.4f}) [Flipped]', 
                        linewidth=1.5, alpha=0.7, linestyle='-')
            else:
                precision_test_single, recall_test_single, _ = precision_recall_curve(y_single_labels, test_scores)
                test_ap_single = average_precision_score(y_single_labels, test_scores)
                plt.plot(recall_test_single, precision_test_single, 
                        label=f'{feature_name} ({label_prefix}, AP = {test_ap_single:.4f})', 
                        linewidth=1.5, alpha=0.7)
    
    # ランダム分類器
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Random (AP = {baseline:.4f})', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    title = 'Precision-Recall Curves - Model and Single Metrics' if show_single_metrics else 'Precision-Recall Curves - L1 Regularized Logistic Regression'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9 if show_single_metrics else 11, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'pr_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def plot_single_metric_roc_curves(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    hide_train: bool = False,
    feature_names: Optional[List[str]] = None
) -> None:
    """各特徴量単体でのROC曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # 各特徴量についてROC曲線を計算
    for feature_name in X_train.columns:
        # 訓練データ
        train_scores = X_train[feature_name].values
        if not hide_train:
            fpr_train, tpr_train, _ = roc_curve(y_train, train_scores)
            train_auc = roc_auc_score(y_train, train_scores)
            if train_auc < 0.5:
                # 反転させたROC曲線
                flipped_train_scores = 1.0 - train_scores
                fpr_train_flipped, tpr_train_flipped, _ = roc_curve(y_train, flipped_train_scores)
                train_auc_flipped = roc_auc_score(y_train, flipped_train_scores)
                plt.plot(fpr_train, tpr_train, label=f'{feature_name} (Train, AUC = {train_auc:.4f}) [Original]', 
                        linewidth=1.5, linestyle=':', alpha=0.5, color='gray')
                plt.plot(fpr_train_flipped, tpr_train_flipped, 
                        label=f'{feature_name} (Train, AUC = {train_auc_flipped:.4f}) [Flipped]', 
                        linewidth=1.5, linestyle='--', alpha=0.7)
            else:
                plt.plot(fpr_train, tpr_train, label=f'{feature_name} (Train, AUC = {train_auc:.4f})', 
                        linewidth=1.5, linestyle='--', alpha=0.7)
        
        # テストデータ
        test_scores = X_test[feature_name].values
        fpr_test, tpr_test, _ = roc_curve(y_test, test_scores)
        test_auc = roc_auc_score(y_test, test_scores)
        if test_auc < 0.5:
            # 反転させたROC曲線
            flipped_test_scores = 1.0 - test_scores
            fpr_test_flipped, tpr_test_flipped, _ = roc_curve(y_test, flipped_test_scores)
            test_auc_flipped = roc_auc_score(y_test, flipped_test_scores)
            plt.plot(fpr_test, tpr_test, label=f'{feature_name} (Test, AUC = {test_auc:.4f}) [Original]', 
                    linewidth=1.5, linestyle=':', alpha=0.5, color='gray')
            plt.plot(fpr_test_flipped, tpr_test_flipped, 
                    label=f'{feature_name} (Test, AUC = {test_auc_flipped:.4f}) [Flipped]', 
                    linewidth=2, alpha=0.9, linestyle='-')
        else:
            plt.plot(fpr_test, tpr_test, label=f'{feature_name} (Test, AUC = {test_auc:.4f})', 
                    linewidth=2, alpha=0.9)
    
    # ランダム分類器
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Single Metric Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'roc_curves_single_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def plot_single_metric_pr_curves(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    hide_train: bool = False,
    feature_names: Optional[List[str]] = None
) -> None:
    """各特徴量単体でのPrecision-Recall曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 10))
    
    # ベースライン（ランダム分類器）
    baseline = len(y_test[y_test == 1]) / len(y_test)
    
    # 各特徴量についてPR曲線を計算
    for feature_name in X_train.columns:
        # 訓練データ
        train_scores = X_train[feature_name].values
        if not hide_train:
            # ROCのAUCを計算して、0.5未満かどうかを判定
            train_auc = roc_auc_score(y_train, train_scores)
            precision_train, recall_train, _ = precision_recall_curve(y_train, train_scores)
            train_ap = average_precision_score(y_train, train_scores)
            if train_auc < 0.5:
                # 反転させたPR曲線
                flipped_train_scores = 1.0 - train_scores
                precision_train_flipped, recall_train_flipped, _ = precision_recall_curve(y_train, flipped_train_scores)
                train_ap_flipped = average_precision_score(y_train, flipped_train_scores)
                plt.plot(recall_train, precision_train, 
                        label=f'{feature_name} (Train, AP = {train_ap:.4f}) [Original]', 
                        linewidth=1.5, linestyle=':', alpha=0.5, color='gray')
                plt.plot(recall_train_flipped, precision_train_flipped, 
                        label=f'{feature_name} (Train, AP = {train_ap_flipped:.4f}) [Flipped]', 
                        linewidth=1.5, linestyle='--', alpha=0.7)
            else:
                plt.plot(recall_train, precision_train, 
                        label=f'{feature_name} (Train, AP = {train_ap:.4f})', 
                        linewidth=1.5, linestyle='--', alpha=0.7)
        
        # テストデータ
        test_scores = X_test[feature_name].values
        # ROCのAUCを計算して、0.5未満かどうかを判定
        test_auc = roc_auc_score(y_test, test_scores)
        precision_test, recall_test, _ = precision_recall_curve(y_test, test_scores)
        test_ap = average_precision_score(y_test, test_scores)
        if test_auc < 0.5:
            # 反転させたPR曲線
            flipped_test_scores = 1.0 - test_scores
            precision_test_flipped, recall_test_flipped, _ = precision_recall_curve(y_test, flipped_test_scores)
            test_ap_flipped = average_precision_score(y_test, flipped_test_scores)
            plt.plot(recall_test, precision_test, 
                    label=f'{feature_name} (Test, AP = {test_ap:.4f}) [Original]', 
                    linewidth=1.5, linestyle=':', alpha=0.5, color='gray')
            plt.plot(recall_test_flipped, precision_test_flipped, 
                    label=f'{feature_name} (Test, AP = {test_ap_flipped:.4f}) [Flipped]', 
                    linewidth=2, alpha=0.9, linestyle='-')
        else:
            plt.plot(recall_test, precision_test, 
                    label=f'{feature_name} (Test, AP = {test_ap:.4f})', 
                    linewidth=2, alpha=0.9)
    
    # ランダム分類器
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Random (AP = {baseline:.4f})', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Single Metric Performance', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'pr_curves_single_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def plot_threshold_f1_curves(
    y_train: pd.Series,
    y_train_proba: np.ndarray,
    y_test: pd.Series,
    y_test_proba: np.ndarray,
    output_dir: Path,
    hide_train: bool = False,
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    show_single_metrics: bool = False,
    feature_names: Optional[List[str]] = None,
    y_test_single: pd.Series = None
) -> None:
    """閾値とF1スコアの関係を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 閾値の範囲を設定（0から1まで0.01刻み）
    thresholds = np.arange(0.0, 1.01, 0.01)
    
    # プロット
    figsize = (12, 10) if show_single_metrics else (10, 8)
    plt.figure(figsize=figsize)
    
    # モデルのF1スコアを計算
    train_f1_scores = []
    test_f1_scores = []
    
    for threshold in thresholds:
        # 訓練データ（存在する場合のみ）
        if y_train is not None and y_train_proba is not None:
            y_train_pred = (y_train_proba >= threshold).astype(int)
            train_f1 = f1_score(y_train, y_train_pred)
            train_f1_scores.append(train_f1)
        
        # テストデータ
        y_test_pred = (y_test_proba >= threshold).astype(int)
        test_f1 = f1_score(y_test, y_test_pred)
        test_f1_scores.append(test_f1)
    
    # モデルのF1曲線
    if not hide_train and y_train is not None and y_train_proba is not None and len(train_f1_scores) > 0:
        plt.plot(thresholds, train_f1_scores, label=f'Model Train (Max F1 = {max(train_f1_scores):.4f})', 
                linewidth=2.5, color='blue')
    if len(test_f1_scores) > 0:
        label_prefix = "CV" if y_test_single is not None else "Test"
        plt.plot(thresholds, test_f1_scores, label=f'Model {label_prefix} (Max F1 = {max(test_f1_scores):.4f})', 
            linewidth=2.5, color='red')
    
    # 最適な閾値をマーク
    best_test_threshold_idx = np.argmax(test_f1_scores)
    best_test_threshold = thresholds[best_test_threshold_idx]
    best_test_f1 = test_f1_scores[best_test_threshold_idx]
    plt.plot(best_test_threshold, best_test_f1, 'ro', markersize=10, 
            label=f'Best Test Threshold = {best_test_threshold:.2f} (F1 = {best_test_f1:.4f})')
    
    if not hide_train:
        best_train_threshold_idx = np.argmax(train_f1_scores)
        best_train_threshold = thresholds[best_train_threshold_idx]
        best_train_f1 = train_f1_scores[best_train_threshold_idx]
        plt.plot(best_train_threshold, best_train_f1, 'bo', markersize=10, 
                label=f'Best Train Threshold = {best_train_threshold:.2f} (F1 = {best_train_f1:.4f})')
    
    # 各特徴量単体のF1曲線（オプション）
    if show_single_metrics and X_test is not None:
        # 単一指標のラベル（CV結果の場合はy_test_singleを使用）
        y_single_labels = y_test_single if y_test_single is not None else y_test
        
        for feature_name in X_test.columns:
            # テストデータ（またはCV結果）
            test_scores = X_test[feature_name].values
            test_f1_single_scores = []
            for threshold in thresholds:
                y_test_pred_single = (test_scores >= threshold).astype(int)
                test_f1_single = f1_score(y_single_labels, y_test_pred_single)
                test_f1_single_scores.append(test_f1_single)
            
            max_test_f1_single = max(test_f1_single_scores)
            label_prefix = "CV" if y_test_single is not None else "Test"
            plt.plot(thresholds, test_f1_single_scores, 
                    label=f'{feature_name} ({label_prefix}, Max F1 = {max_test_f1_single:.4f})', 
                    linewidth=1.5, alpha=0.7)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    title = 'Threshold vs F1 Score - Model and Single Metrics' if show_single_metrics else 'Threshold vs F1 Score - L1 Regularized Logistic Regression'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9 if show_single_metrics else 11, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'threshold_f1_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def plot_feature_distributions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: Path,
    hide_train: bool = False,
    feature_names: Optional[List[str]] = None,
    y_train: pd.Series = None,
    y_test: pd.Series = None,
    y_train_proba: np.ndarray = None,
    y_test_proba: np.ndarray = None
) -> None:
    """各特徴量のスコア分布と統合モデルの予測確率分布を描画して保存（全体分布とラベルごとの分布を別々のグラフに）"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # カラーマップを生成（特徴量が多い場合に備えて）
    try:
        from matplotlib import cm
        colormap = cm.get_cmap('tab20')
    except:
        colormap = plt.cm.get_cmap('tab20')
    
    # 1. 全体の分布を描画
    plt.figure(figsize=(14, 8))
    
    # 各特徴量の分布をプロット
    for idx, feature_name in enumerate(X_train.columns):
        color = colormap(idx / max(len(X_train.columns) - 1, 1))
        
        # 訓練データ
        if not hide_train:
            train_values = X_train[feature_name].values
            plt.hist(train_values, bins=50, alpha=0.4, label=f'{feature_name} (Train)', 
                    color=color, linestyle='--', linewidth=1.5, histtype='step', density=True)
        
        # テストデータ
        test_values = X_test[feature_name].values
        plt.hist(test_values, bins=50, alpha=0.6, label=f'{feature_name} (Test)', 
                color=color, linewidth=2, histtype='step', density=True)
    
    # 統合モデルの予測確率分布を追加
    if y_train_proba is not None and y_test_proba is not None:
        model_color = 'black'
        
        # 訓練データの予測確率
        if not hide_train:
            plt.hist(y_train_proba, bins=50, alpha=0.5, label='Combined Model (Train)', 
                    color=model_color, linestyle='--', linewidth=2.5, histtype='step', density=True)
        
        # テストデータの予測確率
        plt.hist(y_test_proba, bins=50, alpha=0.7, label='Combined Model (Test)', 
                color=model_color, linewidth=3, histtype='step', density=True)
    
    plt.xlabel('Feature Value / Prediction Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    title = 'Feature Score and Combined Model Distributions (All)' if not hide_train else 'Feature Score and Combined Model Distributions (All, Test Only)'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=8, ncol=2 if len(X_train.columns) <= 5 else 3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'feature_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()
    
    # 2. ラベルごとの分布を描画（y_trainとy_testが存在する場合のみ）
    if y_train is not None and y_test is not None:
        # サブプロットの数を計算（特徴量数 + 統合モデル）
        n_features = len(X_train.columns)
        n_plots = n_features + (1 if y_train_proba is not None and y_test_proba is not None else 0)
        
        # サブプロットのレイアウトを決定（2列または3列）
        n_cols = 3 if n_plots > 6 else 2
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
        # axesを常に1次元配列に変換（squeeze=Falseにより常に2次元配列として返される）
        axes = axes.flatten()
        
        # 各特徴量の分布をプロット（ラベルごと）
        for idx, feature_name in enumerate(X_train.columns):
            ax = axes[idx]
            color = colormap(idx / max(len(X_train.columns) - 1, 1))
            gray_color = 'gray'
            
            # 訓練データ（ラベルごと）
            if not hide_train:
                train_values_pos = X_train[y_train == 1][feature_name].values
                train_values_neg = X_train[y_train == 0][feature_name].values
                if len(train_values_pos) > 0:
                    ax.hist(train_values_pos, bins=50, alpha=0.5, 
                            label='Train (Pos)', 
                            color=color, linestyle='--', linewidth=1.5, histtype='step', density=True)
                if len(train_values_neg) > 0:
                    ax.hist(train_values_neg, bins=50, alpha=0.5, 
                            label='Train (Neg)', 
                            color=gray_color, linestyle='--', linewidth=1.5, histtype='step', density=True)
            
            # テストデータ（ラベルごと）
            test_values_pos = X_test[y_test == 1][feature_name].values
            test_values_neg = X_test[y_test == 0][feature_name].values
            if len(test_values_pos) > 0:
                ax.hist(test_values_pos, bins=50, alpha=0.7, 
                        label='Test (Pos)', 
                        color=color, linestyle='-', linewidth=2, histtype='step', density=True)
            if len(test_values_neg) > 0:
                ax.hist(test_values_neg, bins=50, alpha=0.7, 
                        label='Test (Neg)', 
                        color=gray_color, linestyle='-', linewidth=2, histtype='step', density=True)
            
            ax.set_xlabel('Feature Value', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(feature_name, fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 統合モデルの予測確率分布（ラベルごと）
        if y_train_proba is not None and y_test_proba is not None:
            ax = axes[n_features]
            model_color = 'black'
            gray_color = 'gray'
            
            # 訓練データの予測確率（ラベルごと）
            if not hide_train:
                train_proba_pos = y_train_proba[y_train == 1]
                train_proba_neg = y_train_proba[y_train == 0]
                if len(train_proba_pos) > 0:
                    ax.hist(train_proba_pos, bins=50, alpha=0.5, 
                            label='Train (Pos)', 
                            color=model_color, linestyle='--', linewidth=2.5, histtype='step', density=True)
                if len(train_proba_neg) > 0:
                    ax.hist(train_proba_neg, bins=50, alpha=0.5, 
                            label='Train (Neg)', 
                            color=gray_color, linestyle='--', linewidth=2.5, histtype='step', density=True)
            
            # テストデータの予測確率（ラベルごと）
            test_proba_pos = y_test_proba[y_test == 1]
            test_proba_neg = y_test_proba[y_test == 0]
            if len(test_proba_pos) > 0:
                ax.hist(test_proba_pos, bins=50, alpha=0.7, 
                        label='Test (Pos)', 
                        color=model_color, linestyle='-', linewidth=3, histtype='step', density=True)
            if len(test_proba_neg) > 0:
                ax.hist(test_proba_neg, bins=50, alpha=0.7, 
                        label='Test (Neg)', 
                        color=gray_color, linestyle='-', linewidth=3, histtype='step', density=True)
            
            ax.set_xlabel('Prediction Probability', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title('Combined Model', fontsize=11, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 余分なサブプロットを非表示
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle('Feature Score and Combined Model Distributions (by Label)', 
                    fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        
        # 保存
        output_path = feature_output_dir / 'feature_distributions_by_label.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  - 保存先: {output_path}")
        plt.close()


def plot_correlation_heatmaps(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_dir: Path,
    feature_names: Optional[List[str]] = None,
    use_combined: bool = False
) -> None:
    """
    選択された指標（特徴量）同士のピアソン相関係数をヒートマップで可視化する
    
    Args:
        X_train: 訓練データの特徴量DataFrame
        X_test: テストデータの特徴量DataFrame
        y_train: 訓練データのラベル
        y_test: テストデータのラベル
        output_dir: 出力ディレクトリ
        feature_names: 特徴量名のリスト（Noneの場合は全特徴量）
        use_combined: trainとtestを統合して相関係数を計算するかどうか
    """
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用する特徴量を決定
    if feature_names is None:
        metric_columns = list(X_train.columns)
    else:
        metric_columns = [col for col in feature_names if col in X_train.columns]
    
    if len(metric_columns) == 0:
        print("警告: 可視化する特徴量がありません。")
        return
    
    # データを統合（use_combinedの場合）
    if use_combined:
        X_combined = pd.concat([X_train[metric_columns], X_test[metric_columns]], axis=0, ignore_index=True)
        y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
        merged_df_clean = X_combined.copy()
        # インデックスをリセットしてからラベルを追加
        merged_df_clean = merged_df_clean.reset_index(drop=True)
        y_combined = y_combined.reset_index(drop=True)
        merged_df_clean['label'] = y_combined
        print(f"trainとtestを統合: train={len(X_train)}サンプル, test={len(X_test)}サンプル, 合計={len(merged_df_clean)}サンプル")
    else:
        # テストデータのみを使用
        merged_df_clean = X_test[metric_columns].copy()
        merged_df_clean = merged_df_clean.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)
        merged_df_clean['label'] = y_test_reset
        print(f"testデータのみ使用: {len(merged_df_clean)}サンプル")
    
    # NaN値を含む行を削除
    merged_df_clean = merged_df_clean.dropna()
    if len(merged_df_clean) == 0:
        print("警告: 有効なデータがありません。")
        return
    
    # 相関係数を計算
    # 1. 全体の相関係数（ピアソン）
    correlation_all_pearson = merged_df_clean[metric_columns].corr(method='pearson')
    
    # 2. 全体の相関係数（スピアマン）
    correlation_all_spearman = merged_df_clean[metric_columns].corr(method='spearman')
    
    # 3. ラベル別の相関係数
    labels = sorted(merged_df_clean['label'].unique())
    correlations_by_label_pearson = {}
    correlations_by_label_spearman = {}
    
    for label in labels:
        df_label = merged_df_clean[merged_df_clean['label'] == label]
        if len(df_label) > 1:  # 相関係数を計算するには最低2行必要
            correlations_by_label_pearson[label] = df_label[metric_columns].corr(method='pearson')
            correlations_by_label_spearman[label] = df_label[metric_columns].corr(method='spearman')
    
    # ヒートマップを作成
    num_metrics = len(metric_columns)
    num_labels = len(labels)
    
    # 図のサイズを調整
    fig_width = max(20, num_metrics * 1.2)
    fig_height = max(16, num_metrics * 1.8)
    
    # サブプロットのレイアウト: 2行（ピアソン、スピアマン）、最大3列（全体、ラベル1、ラベル2）
    n_cols = min(3, 1 + num_labels)
    fig, axes = plt.subplots(2, n_cols, figsize=(fig_width, fig_height))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # カラーマップの範囲を統一
    vmin, vmax = -1, 1
    
    # フォントサイズを調整
    annot_fontsize = max(6, min(10, 100 // num_metrics))
    label_fontsize = max(7, min(10, 120 // num_metrics))
    title_fontsize = max(8, min(12, 140 // num_metrics))
    
    # ピアソン相関係数のヒートマップ（1行目）
    # 1. 全体の相関係数ヒートマップ（ピアソン）
    sns.heatmap(
        correlation_all_pearson,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'shrink': 0.8},
        ax=axes[0, 0],
        annot_kws={'size': annot_fontsize}
    )
    axes[0, 0].set_title('All Labels (Pearson)', fontsize=title_fontsize, fontweight='bold')
    axes[0, 0].set_xlabel('Metrics', fontsize=label_fontsize)
    axes[0, 0].set_ylabel('Metrics', fontsize=label_fontsize)
    axes[0, 0].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
    axes[0, 0].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
    
    # 2-3. ラベル別の相関係数ヒートマップ（ピアソン）
    label_idx = 1
    for label in labels[:2]:  # 最大2つのラベルを表示
        if label in correlations_by_label_pearson:
            sns.heatmap(
                correlations_by_label_pearson[label],
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'shrink': 0.8},
                ax=axes[0, label_idx],
                annot_kws={'size': annot_fontsize}
            )
            label_name = 'Positive' if label == 1 else 'Negative'
            axes[0, label_idx].set_title(f'Label: {label_name} (Pearson)', fontsize=title_fontsize, fontweight='bold')
            axes[0, label_idx].set_xlabel('Metrics', fontsize=label_fontsize)
            axes[0, label_idx].set_ylabel('Metrics', fontsize=label_fontsize)
            axes[0, label_idx].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
            axes[0, label_idx].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
            label_idx += 1
    
    # 3つ目のラベルがない場合は非表示（ピアソン）
    if label_idx < n_cols:
        for idx in range(label_idx, n_cols):
            axes[0, idx].set_visible(False)
    
    # スピアマン相関係数のヒートマップ（2行目）
    # 1. 全体の相関係数ヒートマップ（スピアマン）
    sns.heatmap(
        correlation_all_spearman,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={'shrink': 0.8},
        ax=axes[1, 0],
        annot_kws={'size': annot_fontsize}
    )
    axes[1, 0].set_title('All Labels (Spearman)', fontsize=title_fontsize, fontweight='bold')
    axes[1, 0].set_xlabel('Metrics', fontsize=label_fontsize)
    axes[1, 0].set_ylabel('Metrics', fontsize=label_fontsize)
    axes[1, 0].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
    axes[1, 0].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
    
    # 2-3. ラベル別の相関係数ヒートマップ（スピアマン）
    label_idx = 1
    for label in labels[:2]:  # 最大2つのラベルを表示
        if label in correlations_by_label_spearman:
            sns.heatmap(
                correlations_by_label_spearman[label],
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                vmin=vmin,
                vmax=vmax,
                square=True,
                cbar_kws={'shrink': 0.8},
                ax=axes[1, label_idx],
                annot_kws={'size': annot_fontsize}
            )
            label_name = 'Positive' if label == 1 else 'Negative'
            axes[1, label_idx].set_title(f'Label: {label_name} (Spearman)', fontsize=title_fontsize, fontweight='bold')
            axes[1, label_idx].set_xlabel('Metrics', fontsize=label_fontsize)
            axes[1, label_idx].set_ylabel('Metrics', fontsize=label_fontsize)
            axes[1, label_idx].tick_params(axis='x', rotation=45, labelsize=label_fontsize)
            axes[1, label_idx].tick_params(axis='y', rotation=0, labelsize=label_fontsize)
            label_idx += 1
    
    # 3つ目のラベルがない場合は非表示（スピアマン）
    if label_idx < n_cols:
        for idx in range(label_idx, n_cols):
            axes[1, idx].set_visible(False)
    
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'correlation_heatmaps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 相関係数ヒートマップを保存しました: {output_path}")
    plt.close()
    
    # 統計情報を表示
    print("\n=== 相関係数の統計情報 ===")
    print(f"全体のデータ数: {len(merged_df_clean)}")
    print("\n全体の相関係数（ピアソン）:")
    print(correlation_all_pearson)
    print("\n全体の相関係数（スピアマン）:")
    print(correlation_all_spearman)
    
    for label in labels[:2]:
        label_name = 'Positive' if label == 1 else 'Negative'
        if label in correlations_by_label_pearson:
            print(f"\n{label_name}ラベルの相関係数（ピアソン） (データ数: {len(merged_df_clean[merged_df_clean['label'] == label])}):")
            print(correlations_by_label_pearson[label])
        if label in correlations_by_label_spearman:
            print(f"\n{label_name}ラベルの相関係数（スピアマン） (データ数: {len(merged_df_clean[merged_df_clean['label'] == label])}):")
            print(correlations_by_label_spearman[label])


def plot_confidence_analysis(
    confidence_ranges: List[tuple],
    range_results: List[dict],
    overall_auc_improvement: float,
    output_dir: Path,
    feature_names: Optional[List[str]] = None
) -> None:
    """
    Confidence Analysisの結果を可視化
    
    Args:
        confidence_ranges: 確率範囲のリスト [(low, high), ...]
        range_results: 各範囲の評価結果のリスト [{'auc_regression': ..., 'auc_single': ..., 'n_samples': ...}, ...]
        overall_auc_improvement: 全体データでのAUC改善度
        output_dir: 出力ディレクトリ
        feature_names: 特徴量名のリスト（ディレクトリ名生成用）
    """
    # 日本語フォントの設定
    use_english = False
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        
        # 利用可能な日本語フォントを検索
        jp_font_candidates = [
            'Noto Sans CJK JP', 'Noto Sans Japanese', 'Takao', 'TakaoGothic', 'TakaoPGothic',
            'IPAexGothic', 'IPAPGothic', 'IPAPMincho', 'VL PGothic', 'VL Gothic',
            'Yu Gothic', 'YuGothic', 'Meiryo', 'MS PGothic', 'MS Gothic',
            'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Osaka'
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
            # 日本語フォントが見つからない場合、英語表記にフォールバック
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
            matplotlib.rcParams['axes.unicode_minus'] = False
            # 日本語ラベルを英語に変更
            use_english = True
    except Exception as e:
        # エラーが発生した場合も英語表記にフォールバック
        use_english = True
        try:
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
            matplotlib.rcParams['axes.unicode_minus'] = False
        except:
            pass
    
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # データの準備
    range_labels = [f"[{low:.1f}-{high:.1f}]" for low, high in confidence_ranges]
    auc_improvements = [r['auc_regression'] - r['auc_single'] if r.get('auc_single') is not None else None for r in range_results]
    n_samples_list = [r['n_samples'] for r in range_results]
    auc_regression_list = [r['auc_regression'] for r in range_results]
    auc_single_list = [r.get('auc_single') for r in range_results]
    
    # ラベルを準備（日本語フォントがない場合は英語）
    if use_english:
        xlabel1 = 'BERT Prediction Probability Range'
        ylabel1 = 'AUC Improvement (Regression - Single Metric)'
        title1 = 'Confidence Analysis: AUC Improvement by Probability Range'
        legend_label1 = f'Overall Improvement ({overall_auc_improvement:+.4f})'
        xlabel2 = 'BERT Prediction Probability Range'
        ylabel2 = 'AUC-ROC'
        title2 = 'AUC-ROC Comparison by Probability Range'
        label_regression = 'Regression Model (QPP Integrated)'
        label_single = 'Single Metric'
    else:
        xlabel1 = 'BERT予測確率範囲'
        ylabel1 = 'AUC改善度 (回帰モデル - 単体指標)'
        title1 = 'Confidence Analysis: 確率範囲ごとのQPP統合によるAUC改善度'
        legend_label1 = f'全体データでの改善度 ({overall_auc_improvement:+.4f})'
        xlabel2 = 'BERT予測確率範囲'
        ylabel2 = 'AUC-ROC'
        title2 = '確率範囲ごとのAUC-ROC比較'
        label_regression = '回帰モデル (QPP統合)'
        label_single = '単体指標'
    
    # プロット1: AUC改善度のバープロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # サブプロット1: AUC改善度
    colors = ['green' if imp > overall_auc_improvement else 'orange' if imp > 0 else 'red' 
              for imp in auc_improvements]
    bars = ax1.bar(range(len(range_labels)), auc_improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # 全体データでの改善度を参考線として表示
    ax1.axhline(y=overall_auc_improvement, color='blue', linestyle='--', linewidth=2, 
                label=legend_label1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # バーの上に値を表示
    for i, (bar, imp, n) in enumerate(zip(bars, auc_improvements, n_samples_list)):
        if imp is not None:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.3f}\n(n={n})',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel(xlabel1, fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel1, fontsize=12, fontweight='bold')
    ax1.set_title(title1, fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(range_labels)))
    ax1.set_xticklabels(range_labels, rotation=45, ha='right')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([min(min([i for i in auc_improvements if i is not None]), overall_auc_improvement) - 0.05,
                  max(max([i for i in auc_improvements if i is not None]), overall_auc_improvement) + 0.05])
    
    # サブプロット2: 各範囲でのAUC比較（回帰モデル vs 単体指標）
    x_pos = np.arange(len(range_labels))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, auc_regression_list, width, label=label_regression, 
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    
    if any(auc_single_list):
        bars2 = ax2.bar(x_pos + width/2, [a if a is not None else 0 for a in auc_single_list], width,
                        label=label_single, color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
        
        # バーの上に値を表示
        for i, (bar1, bar2, auc_r, auc_s, n) in enumerate(zip(bars1, bars2, auc_regression_list, auc_single_list, n_samples_list)):
            ax2.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                    f'{auc_r:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
            if auc_s is not None:
                ax2.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                        f'{auc_s:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    else:
        # 単体指標がない場合
        for i, (bar1, auc_r, n) in enumerate(zip(bars1, auc_regression_list, n_samples_list)):
            ax2.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                    f'{auc_r:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel(xlabel2, fontsize=12, fontweight='bold')
    ax2.set_ylabel(ylabel2, fontsize=12, fontweight='bold')
    ax2.set_title(title2, fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(range_labels, rotation=45, ha='right')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'confidence_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - Confidence Analysis可視化を保存: {output_path}")
    plt.close()


def plot_overconfidence_analysis(
    overconfidence_results: dict,
    output_dir: Path,
    feature_names: Optional[List[str]] = None
) -> None:
    """
    Overconfidence Analysisの結果を可視化
    
    Args:
        overconfidence_results: Overconfidence分析の結果辞書
            {
                'high_confidence_fp': {'mean_regression': ..., 'mean_single': ..., 'improvement': ..., 'n_samples': ...},
                'high_confidence_fn': {'mean_regression': ..., 'mean_single': ..., 'improvement': ..., 'n_samples': ...},
                'overall_auc_improvement': ...
            }
        output_dir: 出力ディレクトリ
        feature_names: 特徴量名のリスト（ディレクトリ名生成用）
    """
    # 日本語フォントの設定
    use_english = False
    try:
        import matplotlib
        import matplotlib.font_manager as fm
        
        jp_font_candidates = [
            'Noto Sans CJK JP', 'Noto Sans Japanese', 'Takao', 'TakaoGothic', 'TakaoPGothic',
            'IPAexGothic', 'IPAPGothic', 'IPAPMincho', 'VL PGothic', 'VL Gothic',
            'Yu Gothic', 'YuGothic', 'Meiryo', 'MS PGothic', 'MS Gothic',
            'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Osaka'
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
            use_english = True
    except Exception as e:
        use_english = True
        try:
            matplotlib.rcParams['font.family'] = 'DejaVu Sans'
            matplotlib.rcParams['axes.unicode_minus'] = False
        except:
            pass
    
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # ラベルを準備
    if use_english:
        title = 'Overconfidence Analysis: QPP Effect on BERT Misclassifications'
        xlabel = 'Error Type'
        ylabel1 = 'Prediction Probability Improvement\n(Closer to Correct Label)'
        ylabel2 = 'Average Prediction Probability'
        label_fp = 'False Positive\n(High Confidence, Wrong)'
        label_fn = 'False Negative\n(High Confidence, Wrong)'
        label_regression = 'Regression Model (QPP Integrated)'
        label_single = 'Single Metric'
        ideal_fp = 'Ideal (0.0)'
        ideal_fn = 'Ideal (1.0)'
    else:
        title = 'Overconfidence Analysis: BERTの過信による誤分類でのQPP効果'
        xlabel = '誤分類タイプ'
        ylabel1 = '予測確率の改善度\n(正解ラベルに近づく)'
        ylabel2 = '平均予測確率'
        label_fp = 'False Positive\n(高確信度だが誤分類)'
        label_fn = 'False Negative\n(高確信度だが誤分類)'
        label_regression = '回帰モデル (QPP統合)'
        label_single = '単体指標'
        ideal_fp = '理想値 (0.0)'
        ideal_fn = '理想値 (1.0)'
    
    # データの準備
    fp_result = overconfidence_results.get('high_confidence_fp', {})
    fn_result = overconfidence_results.get('high_confidence_fn', {})
    
    categories = []
    improvements = []
    mean_regression_list = []
    mean_single_list = []
    n_samples_list = []
    ideal_values = []
    
    if fp_result.get('n_samples', 0) > 0:
        categories.append(label_fp)
        fp_mean_r = fp_result.get('mean_regression')
        fp_mean_s = fp_result.get('mean_single')
        mean_regression_list.append(fp_mean_r)
        mean_single_list.append(fp_mean_s)
        improvements.append(fp_result.get('improvement'))
        n_samples_list.append(fp_result.get('n_samples', 0))
        ideal_values.append(0.0)  # FPの理想値は0
    
    if fn_result.get('n_samples', 0) > 0:
        categories.append(label_fn)
        fn_mean_r = fn_result.get('mean_regression')
        fn_mean_s = fn_result.get('mean_single')
        mean_regression_list.append(fn_mean_r)
        mean_single_list.append(fn_mean_s)
        improvements.append(fn_result.get('improvement'))
        n_samples_list.append(fn_result.get('n_samples', 0))
        ideal_values.append(1.0)  # FNの理想値は1
    
    if len(categories) == 0:
        print("  ⚠️  Overconfidence事例が見つかりませんでした")
        return
    
    # プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # サブプロット1: 改善度（正解ラベルに近づく度合い）
    colors = ['green' if imp is not None and imp > 0 else 'red' if imp is not None and imp < 0 else 'gray'
              for imp in improvements]
    bars = ax1.bar(range(len(categories)), improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # バーの上に値を表示
    for i, (bar, imp, n) in enumerate(zip(bars, improvements, n_samples_list)):
        if imp is not None:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.3f}\n(n={n})',
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel1, fontsize=12, fontweight='bold')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels(categories, rotation=0, ha='center')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # サブプロット2: 平均予測確率の比較
    x_pos = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, mean_regression_list, width, label=label_regression,
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    
    if any(mean_single_list):
        bars2 = ax2.bar(x_pos + width/2, [m if m is not None else 0 for m in mean_single_list], width,
                        label=label_single, color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
        
        # 理想値を点線で表示
        for i, ideal in enumerate(ideal_values):
            ax2.axhline(y=ideal, xmin=(x_pos[i] - width)/len(categories), xmax=(x_pos[i] + width)/len(categories),
                       color='green', linestyle='--', linewidth=2, alpha=0.5)
        
        for i, (bar1, bar2, mean_r, mean_s) in enumerate(zip(bars1, bars2, mean_regression_list, mean_single_list)):
            if mean_r is not None:
                ax2.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                        f'{mean_r:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            if mean_s is not None:
                ax2.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height(),
                        f'{mean_s:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        for i, ideal in enumerate(ideal_values):
            ax2.axhline(y=ideal, xmin=(x_pos[i] - width)/len(categories), xmax=(x_pos[i] + width)/len(categories),
                       color='green', linestyle='--', linewidth=2, alpha=0.5, label=ideal_fp if ideal == 0.0 else ideal_fn)
        
        for i, (bar1, mean_r) in enumerate(zip(bars1, mean_regression_list)):
            if mean_r is not None:
                ax2.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height(),
                        f'{mean_r:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax2.set_ylabel(ylabel2, fontsize=12, fontweight='bold')
    if use_english:
        ax2.set_title('Average Prediction Probability for Overconfidence Errors', fontsize=14, fontweight='bold')
    else:
        ax2.set_title('過信誤分類での平均予測確率', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories, rotation=0, ha='center')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'overconfidence_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - Overconfidence Analysis可視化を保存: {output_path}")
    plt.close()

