"""
QPP特徴量実験の設定を管理するモジュール
"""
import torch


class QPPExperimentConfig:
    """
    QPP特徴量実験の設定を管理するクラス
    """
    
    # QPP特徴量の定義（インデックスと名前のマッピング）
    QPP_FEATURE_NAMES = {
        0: "ndcg@1",
        1: "ndcg@5", 
        2: "ndcg@10",
        3: "precision@1",
        4: "precision@5",
        5: "precision@10",
        6: "recall@1",
        7: "recall@5",
        8: "recall@10",
    }
    
    # よく使用される特徴量セットの定義
    FEATURE_SETS = {
        "all": list(range(9)),  # 全ての特徴量
        "ndcg_only": [0, 1, 2],  # NDCGのみ
        "precision_only": [3, 4, 5],  # Precisionのみ
        "recall_only": [6, 7, 8],  # Recallのみ
        "at1": [0, 3, 6],  # @1の特徴量のみ
        "at3": [1, 4, 7],  # @3の特徴量のみ
        "at5": [2, 5, 8],  # @5の特徴量のみ
        "ndcg3_only": [1],  # NDCG@3のみ
        "precision3_only": [4],  # Precision@3のみ
        "recall3_only": [7],  # Recall@3のみ
    }
    
    @classmethod
    def get_feature_indices(cls, feature_set_name):
        """
        特徴量セット名からインデックスリストを取得
        
        Args:
            feature_set_name (str): 特徴量セット名
            
        Returns:
            list: 特徴量インデックスのリスト
        """
        if feature_set_name in cls.FEATURE_SETS:
            return cls.FEATURE_SETS[feature_set_name]
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}. Available sets: {list(cls.FEATURE_SETS.keys())}")
    
    @classmethod
    def get_feature_names(cls, indices):
        """
        インデックスリストから特徴量名リストを取得
        
        Args:
            indices (list): 特徴量インデックスのリスト
            
        Returns:
            list: 特徴量名のリスト
        """
        return [cls.QPP_FEATURE_NAMES[i] for i in indices if i in cls.QPP_FEATURE_NAMES]
    
    @classmethod
    def create_experiment_args(cls, base_args, feature_set_name, target_feature_index=1, gate_feature_index=1):
        """
        実験用のargsを作成
        
        Args:
            base_args: ベースとなるargsオブジェクト
            feature_set_name (str): 使用する特徴量セット名
            target_feature_index (int): auxiliary_headで予測する特徴量のインデックス
            gate_feature_index (int): policy_gatingで使用する特徴量のインデックス
            
        Returns:
            args: 実験設定が追加されたargsオブジェクト
        """
        # 特徴量インデックスを取得
        feature_indices = cls.get_feature_indices(feature_set_name)
        
        # argsに設定を追加
        base_args.qpp_feature_dim = 9  # 元の特徴量次元
        base_args.qpp_feature_indices = feature_indices  # 使用する特徴量のインデックス
        base_args.qpp_target_feature_index = target_feature_index  # auxiliary_headの予測対象
        base_args.qpp_gate_feature_index = gate_feature_index  # policy_gatingの使用特徴量
        
        return base_args
    
    @classmethod
    def validate_qpp_features(cls, qpp_features, expected_dim=9):
        """
        TSVから変換されたPKL形式のQPP特徴量の妥当性を検証
        
        Args:
            qpp_features (dict): TSVから変換されたPKL形式のQPP特徴量辞書
            expected_dim (int): 期待される特徴量次元
            
        Returns:
            bool: 妥当性の結果
        """
        if qpp_features is None:
            return True  # Noneは有効（QPP特徴量を使用しない場合）
        
        # TSVから変換されたPKL形式（辞書形式）のみをサポート
        if not isinstance(qpp_features, dict):
            print(f"警告: QPP特徴量は辞書形式である必要があります。実際の型: {type(qpp_features)}")
            return False
        
        feature_names = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'precision@1', 'precision@5', 'precision@10', 'recall@1', 'recall@5', 'recall@10']
        actual_dim = len([name for name in feature_names if name in qpp_features])
        if actual_dim != expected_dim:
            print(f"警告: QPP特徴量の次元が期待値と異なります。期待: {expected_dim}, 実際: {actual_dim}")
            return False
        
        return True
    
    @classmethod
    def list_available_feature_sets(cls):
        """
        利用可能な特徴量セットの一覧を表示
        
        Returns:
            dict: 特徴量セット名とその説明の辞書
        """
        descriptions = {
            "all": "全ての特徴量 (ndcg@1, ndcg@3, ndcg@5, precision@1, precision@3, precision@5, recall@1, recall@3, recall@5)",
            "ndcg_only": "NDCGのみ (ndcg@1, ndcg@3, ndcg@5)",
            "precision_only": "Precisionのみ (precision@1, precision@3, precision@5)",
            "recall_only": "Recallのみ (recall@1, recall@3, recall@5)",
            "at1": "@1の特徴量のみ (ndcg@1, precision@1, recall@1)",
            "at3": "@3の特徴量のみ (ndcg@3, precision@3, recall@3)",
            "at5": "@5の特徴量のみ (ndcg@5, precision@5, recall@5)",
            "ndcg3_only": "NDCG@3のみ",
            "precision3_only": "Precision@3のみ",
            "recall3_only": "Recall@3のみ",
        }
        
        for name, desc in descriptions.items():
            print(f"- {name}: {desc}")
        
        return descriptions
