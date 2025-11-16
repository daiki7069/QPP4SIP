"""
前処理関連のモジュール
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


def balance_label_distribution(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_positive_rate: float,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    訓練データのラベル分布を調整して、指定された正例率に近づける
    
    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        target_positive_rate: 目標とする正例率（0.0-1.0）
        random_state: 乱数シード
    
    Returns:
        調整された訓練データ (X_train_balanced, y_train_balanced)
    """
    current_positive_rate = y_train.mean()
    
    # 既に目標の正例率に近い場合はそのまま返す
    if abs(current_positive_rate - target_positive_rate) < 0.01:
        return X_train.copy(), y_train.copy()
    
    # クラスごとにデータを分離
    positive_indices = y_train[y_train == 1].index
    negative_indices = y_train[y_train == 0].index
    
    n_positive = len(positive_indices)
    n_negative = len(negative_indices)
    
    # 目標の正例率に基づいて、必要なサンプル数を計算
    if target_positive_rate > current_positive_rate:
        # 正例を増やす必要がある場合
        target_n_positive = int(n_negative * target_positive_rate / (1 - target_positive_rate))
        target_n_negative = n_negative
    else:
        # 負例を増やす必要がある場合
        target_n_positive = n_positive
        target_n_negative = int(n_positive * (1 - target_positive_rate) / target_positive_rate)
    
    # サンプリング
    np.random.seed(random_state)
    
    if target_n_positive > n_positive:
        # 正例をオーバーサンプリング（復元抽出）
        positive_sampled_indices = np.random.choice(positive_indices, size=target_n_positive, replace=True)
    else:
        # 正例をアンダーサンプリング
        positive_sampled_indices = np.random.choice(positive_indices, size=target_n_positive, replace=False)
    
    if target_n_negative > n_negative:
        # 負例をオーバーサンプリング（復元抽出）
        negative_sampled_indices = np.random.choice(negative_indices, size=target_n_negative, replace=True)
    else:
        # 負例をアンダーサンプリング
        negative_sampled_indices = np.random.choice(negative_indices, size=target_n_negative, replace=False)
    
    # インデックスを結合してシャッフル
    all_indices = np.concatenate([positive_sampled_indices, negative_sampled_indices])
    np.random.shuffle(all_indices)
    
    # データを抽出
    X_train_balanced = X_train.loc[all_indices].reset_index(drop=True)
    y_train_balanced = y_train.loc[all_indices].reset_index(drop=True)
    
    return X_train_balanced, y_train_balanced


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, StandardScaler]]:
    """
    各特徴量を個別にz-score正規化
    訓練データの統計量でテストデータも正規化
    戻り値: (X_train_norm, X_test_norm, scalers)
    """
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    
    scalers = {}
    for column in X_train.columns:
        scaler = StandardScaler()
        X_train_norm[column] = scaler.fit_transform(X_train[[column]]).flatten()
        X_test_norm[column] = scaler.transform(X_test[[column]]).flatten()
        scalers[column] = scaler
    
    return X_train_norm, X_test_norm, scalers
