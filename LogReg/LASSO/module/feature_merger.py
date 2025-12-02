"""
特徴量マージ関連のモジュール
"""
import pandas as pd
from typing import Dict, Tuple


def merge_features(
    all_scores: Dict[str, Dict[Tuple[str, int], float]],
    labels: Dict[Tuple[str, int], int]
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    全ての特徴量をマージしてDataFrameを作成
    戻り値: (features_df, labels_series)
    
    Args:
        all_scores: 全てのスコア（base、post、nspを含む）の辞書
                    {feature_name: {(conv_id, turn_id): score}}
        labels: ラベルの辞書
    """
    # 全てのキーを収集
    all_keys = set()
    for metric_scores in all_scores.values():
        all_keys.update(metric_scores.keys())
    all_keys = all_keys.intersection(set(labels.keys()))
    
    # データを収集
    data = []
    for key in all_keys:
        conv_id, turn_id = key
        row = {
            'conv_id': conv_id,
            'turn_id': turn_id,
        }
        
        # 全てのスコアを追加
        for feature_name in all_scores.keys():
            row[feature_name] = all_scores[feature_name].get(key)
        
        # ラベルを追加
        row['label'] = labels.get(key)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # データが空の場合はエラー
    if len(df) == 0:
        raise ValueError("No data found after merging features. Check if keys match between scores and labels.")
    
    # labelカラムが存在するか確認
    if 'label' not in df.columns:
        raise ValueError(f"'label' column not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # 特徴量とラベルを分離
    # 理論的な特徴量カラム
    theoretical_feature_columns = list(all_scores.keys())
    
    # 実際にDataFrameに存在する特徴量カラムのみを使用
    # conv_id, turn_id, labelは除外
    exclude_columns = {'conv_id', 'turn_id', 'label'}
    available_feature_columns = [col for col in df.columns 
                                 if col not in exclude_columns 
                                 and col in theoretical_feature_columns]
    
    if len(available_feature_columns) == 0:
        raise ValueError(f"No feature columns found. Available columns: {df.columns.tolist()}, "
                         f"Theoretical columns: {theoretical_feature_columns}")
    
    features_df = df[available_feature_columns].copy()
    labels_series = df['label'].copy()
    
    # NaN値のチェックと処理
    if features_df.isna().any().any():
        nan_counts = features_df.isna().sum()
        print("Warning: NaN values found in features:")
        print(nan_counts[nan_counts > 0])
        
        # NaN値を含む行を削除
        valid_mask = ~features_df.isna().any(axis=1)
        num_removed = len(features_df) - valid_mask.sum()
        
        if num_removed > 0:
            print(f"Removing {num_removed} rows with NaN values (keeping {valid_mask.sum()} rows)")
            features_df = features_df[valid_mask].copy()
            labels_series = labels_series[valid_mask].copy()
        
        # 削除後もNaNが残っている場合はエラー
        if features_df.isna().any().any():
            remaining_nan_counts = features_df.isna().sum()
            print("Error: NaN values still remain after removal:")
            print(remaining_nan_counts[remaining_nan_counts > 0])
            raise ValueError("NaN values are not allowed. Please check the data.")
    
    return features_df, labels_series

