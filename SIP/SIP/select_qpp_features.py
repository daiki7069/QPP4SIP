#!/usr/bin/env python3
"""
QPP特徴量を選択するための設定ファイル
"""

# 利用可能なQPP特徴量
AVAILABLE_QPP_FEATURES = {
    'mrr': 'Mean Reciprocal Rank',
    'found_ratio': 'Found Ratio',
    'mean_rank': 'Mean Rank',
    'hit@1': 'Hit@1',
    'hit@5': 'Hit@5', 
    'hit@10': 'Hit@10',
    'hit@20': 'Hit@20',
    'hit@50': 'Hit@50',
    'precision@1': 'Precision@1',
    'precision@5': 'Precision@5',
    'precision@10': 'Precision@10',
    'precision@20': 'Precision@20',
    'precision@50': 'Precision@50',
    'recall@1': 'Recall@1',
    'recall@5': 'Recall@5',
    'recall@10': 'Recall@10',
    'recall@20': 'Recall@20',
    'recall@50': 'Recall@50',
    'f1@1': 'F1@1',
    'f1@5': 'F1@5',
    'f1@10': 'F1@10',
    'f1@20': 'F1@20',
    'f1@50': 'F1@50',
    'ndcg@1': 'NDCG@1',
    'ndcg@3': 'NDCG@3',
    'ndcg@5': 'NDCG@5',
    'ndcg@10': 'NDCG@10',
    'ndcg@20': 'NDCG@20',
    'ndcg@50': 'NDCG@50'
}

# 推奨されるQPP特徴量セット
RECOMMENDED_FEATURE_SETS = {
    'minimal': ['ndcg@3', 'precision@1', 'recall@1'],
    'balanced': ['ndcg@1', 'ndcg@3', 'ndcg@5', 'precision@1', 'precision@5', 'recall@1', 'recall@5'],
    'comprehensive': ['ndcg@1', 'ndcg@3', 'ndcg@5', 'precision@1', 'precision@3', 'precision@5', 'recall@1', 'recall@3', 'recall@5'],
    'ranking_focused': ['mrr', 'ndcg@1', 'ndcg@3', 'ndcg@5', 'hit@1', 'hit@5'],
    'precision_focused': ['precision@1', 'precision@3', 'precision@5', 'precision@10', 'precision@20'],
    'recall_focused': ['recall@1', 'recall@3', 'recall@5', 'recall@10', 'recall@20'],
    'f1_focused': ['f1@1', 'f1@3', 'f1@5', 'f1@10', 'f1@20']
}

def get_qpp_features(feature_set='comprehensive'):
    """
    指定された特徴量セットを取得
    
    Args:
        feature_set: 特徴量セット名またはカスタム特徴量リスト
        
    Returns:
        list: 選択されたQPP特徴量のリスト
    """
    if isinstance(feature_set, list):
        # カスタム特徴量リスト
        return feature_set
    elif feature_set in RECOMMENDED_FEATURE_SETS:
        # 推奨セット
        return RECOMMENDED_FEATURE_SETS[feature_set]
    else:
        # デフォルトは包括的セット
        return RECOMMENDED_FEATURE_SETS['comprehensive']

def print_available_features():
    """利用可能な特徴量を表示"""
    print("=== 利用可能なQPP特徴量 ===")
    for feature, description in AVAILABLE_QPP_FEATURES.items():
        print(f"{feature}: {description}")
    
    print("\n=== 推奨特徴量セット ===")
    for name, features in RECOMMENDED_FEATURE_SETS.items():
        print(f"{name}: {features}")

if __name__ == "__main__":
    print_available_features()
