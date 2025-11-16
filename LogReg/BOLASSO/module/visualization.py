"""
可視化関連のモジュール
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score


def get_feature_dir_name(feature_names: Optional[List[str]]) -> str:
    """
    特徴量名のリストからディレクトリ名を生成
    
    Args:
        feature_names: 特徴量名のリスト（Noneの場合は'all'を返す）
    
    Returns:
        ディレクトリ名（例: 'features_similarity', 'features_logit_clarification_similarity'）
    """
    if feature_names is None or len(feature_names) == 0:
        return 'features_all'
    
    # 特徴量名をソートして一貫性を保つ
    sorted_features = sorted(feature_names)
    # ディレクトリ名に使えない文字を置換
    safe_features = [f.replace(' ', '_').replace('/', '_') for f in sorted_features]
    return 'features_' + '_'.join(safe_features)


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
    feature_names: Optional[List[str]] = None
) -> None:
    """ROC曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練データのROC曲線
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    
    # テストデータのROC曲線
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    # プロット
    figsize = (12, 10) if show_single_metrics else (10, 8)
    plt.figure(figsize=figsize)
    
    # モデルのROC曲線
    if not hide_train:
        plt.plot(fpr_train, tpr_train, label=f'Model Train (AUC = {train_auc:.4f})', linewidth=2.5, color='blue')
    plt.plot(fpr_test, tpr_test, label=f'Model Test (AUC = {test_auc:.4f})', linewidth=2.5, color='red')
    
    # 各特徴量単体のROC曲線（オプション）
    if show_single_metrics and X_train is not None and X_test is not None:
        for feature_name in X_train.columns:
            # 訓練データ
            train_scores = X_train[feature_name].values
            if not hide_train:
                fpr_train_single, tpr_train_single, _ = roc_curve(y_train, train_scores)
                train_auc_single = roc_auc_score(y_train, train_scores)
                plt.plot(fpr_train_single, tpr_train_single, 
                        label=f'{feature_name} (Train, AUC = {train_auc_single:.4f})', 
                        linewidth=1.5, linestyle='--', alpha=0.6)
            
            # テストデータ
            test_scores = X_test[feature_name].values
            fpr_test_single, tpr_test_single, _ = roc_curve(y_test, test_scores)
            test_auc_single = roc_auc_score(y_test, test_scores)
            plt.plot(fpr_test_single, tpr_test_single, 
                    label=f'{feature_name} (Test, AUC = {test_auc_single:.4f})', 
                    linewidth=1.5, alpha=0.7)
    
    # ランダム分類器
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    title = 'ROC Curves - Model and Single Metrics' if show_single_metrics else 'ROC Curves - BOLASSO Logistic Regression'
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
    feature_names: Optional[List[str]] = None
) -> None:
    """Precision-Recall曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練データのPR曲線
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    
    # テストデータのPR曲線
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)
    
    # ベースライン（ランダム分類器）
    baseline = len(y_test[y_test == 1]) / len(y_test)
    
    # プロット
    figsize = (12, 10) if show_single_metrics else (10, 8)
    plt.figure(figsize=figsize)
    
    # モデルのPR曲線
    if not hide_train:
        plt.plot(recall_train, precision_train, label=f'Model Train (AP = {train_ap:.4f})', linewidth=2.5, color='blue')
    plt.plot(recall_test, precision_test, label=f'Model Test (AP = {test_ap:.4f})', linewidth=2.5, color='red')
    
    # 各特徴量単体のPR曲線（オプション）
    if show_single_metrics and X_train is not None and X_test is not None:
        for feature_name in X_train.columns:
            # 訓練データ
            train_scores = X_train[feature_name].values
            if not hide_train:
                precision_train_single, recall_train_single, _ = precision_recall_curve(y_train, train_scores)
                train_ap_single = average_precision_score(y_train, train_scores)
                plt.plot(recall_train_single, precision_train_single, 
                        label=f'{feature_name} (Train, AP = {train_ap_single:.4f})', 
                        linewidth=1.5, linestyle='--', alpha=0.6)
            
            # テストデータ
            test_scores = X_test[feature_name].values
            precision_test_single, recall_test_single, _ = precision_recall_curve(y_test, test_scores)
            test_ap_single = average_precision_score(y_test, test_scores)
            plt.plot(recall_test_single, precision_test_single, 
                    label=f'{feature_name} (Test, AP = {test_ap_single:.4f})', 
                    linewidth=1.5, alpha=0.7)
    
    # ランダム分類器
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Random (AP = {baseline:.4f})', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    title = 'Precision-Recall Curves - Model and Single Metrics' if show_single_metrics else 'Precision-Recall Curves - BOLASSO Logistic Regression'
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
    hide_train: bool = False
) -> None:
    """各特徴量単体でのROC曲線を描画して保存"""
    plt.figure(figsize=(12, 10))
    
    # 各特徴量についてROC曲線を計算
    for feature_name in X_train.columns:
        # 訓練データ
        train_scores = X_train[feature_name].values
        if not hide_train:
            fpr_train, tpr_train, _ = roc_curve(y_train, train_scores)
            train_auc = roc_auc_score(y_train, train_scores)
            plt.plot(fpr_train, tpr_train, label=f'{feature_name} (Train, AUC = {train_auc:.4f})', 
                    linewidth=1.5, linestyle='--', alpha=0.7)
        
        # テストデータ
        test_scores = X_test[feature_name].values
        fpr_test, tpr_test, _ = roc_curve(y_test, test_scores)
        test_auc = roc_auc_score(y_test, test_scores)
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
    output_path = output_dir / 'roc_curves_single_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def plot_single_metric_pr_curves(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path,
    hide_train: bool = False
) -> None:
    """各特徴量単体でのPrecision-Recall曲線を描画して保存"""
    plt.figure(figsize=(12, 10))
    
    # ベースライン（ランダム分類器）
    baseline = len(y_test[y_test == 1]) / len(y_test)
    
    # 各特徴量についてPR曲線を計算
    for feature_name in X_train.columns:
        # 訓練データ
        train_scores = X_train[feature_name].values
        if not hide_train:
            precision_train, recall_train, _ = precision_recall_curve(y_train, train_scores)
            train_ap = average_precision_score(y_train, train_scores)
            plt.plot(recall_train, precision_train, 
                    label=f'{feature_name} (Train, AP = {train_ap:.4f})', 
                    linewidth=1.5, linestyle='--', alpha=0.7)
        
        # テストデータ
        test_scores = X_test[feature_name].values
        precision_test, recall_test, _ = precision_recall_curve(y_test, test_scores)
        test_ap = average_precision_score(y_test, test_scores)
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
    output_path = output_dir / 'pr_curves_single_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()

