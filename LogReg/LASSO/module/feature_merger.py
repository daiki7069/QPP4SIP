"""
特徴量マージ関連のモジュール
"""
import pandas as pd
from typing import Dict, Tuple


def merge_features(
    base_scores: Dict[Tuple[str, int], float],
    qpp_scores: Dict[str, Dict[Tuple[str, int], float]],
    labels: Dict[Tuple[str, int], int],
    use_base_score: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    全ての特徴量をマージしてDataFrameを作成
    戻り値: (features_df, labels_series)
    
    Args:
        use_base_score: Trueの場合、ベーススコア（logit_clarification）も使用する
    """
    # 全てのキーを収集
    if use_base_score:
        all_keys = set(base_scores.keys())
    else:
        all_keys = set()
    
    for metric_scores in qpp_scores.values():
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
        
        # ベーススコアを追加（オプション）
        if use_base_score:
            row['logit_clarification'] = base_scores.get(key)
        
        # QPPスコアを追加
        for metric_name in qpp_scores.keys():
            row[metric_name] = qpp_scores[metric_name].get(key)
        
        # ラベルを追加
        row['label'] = labels.get(key)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # データが空の場合はエラー
    if len(df) == 0:
        raise ValueError("No data found after merging features. Check if keys match between QPP scores and labels.")
    
    # labelカラムが存在するか確認
    if 'label' not in df.columns:
        raise ValueError(f"'label' column not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # 特徴量とラベルを分離
    # 理論的な特徴量カラム
    if use_base_score:
        theoretical_feature_columns = ['logit_clarification'] + list(qpp_scores.keys())
    else:
        theoretical_feature_columns = list(qpp_scores.keys())
    
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

