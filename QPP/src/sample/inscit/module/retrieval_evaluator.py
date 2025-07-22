from typing import List, Dict, Any, Optional
import numpy as np


class RetrievalEvaluator:
    """
    検索結果の評価指標を算出するクラス
    """
    
    def __init__(self):
        """検索評価指標を初期化"""
        pass
    
    def _get_gold_ids(self, gold_evidence: List[Dict[str, Any]]) -> set:
        """
        ゴールド文書のIDセットを取得
        
        Args:
            gold_evidence: ゴールド文書のリスト
            
        Returns:
            ゴールド文書IDのセット
        """
        gold_ids = set()
        for evidence in gold_evidence:
            if isinstance(evidence, dict) and 'passage_id' in evidence:
                gold_ids.add(evidence['passage_id'])
            elif isinstance(evidence, str):
                gold_ids.add(evidence)
        return gold_ids
    
    def calculate_precision_at_k(self, retrieved_evidence: List[Dict[str, Any]], 
                                gold_evidence: List[Dict[str, Any]], 
                                k: Optional[int] = None) -> float:
        """
        Precision@kを算出
        
        Args:
            retrieved_evidence: 検索で取得された文書のリスト
            gold_evidence: ゴールド文書のリスト
            k: 評価する上位k件（Noneの場合は全て）
            
        Returns:
            Precision@kスコア
        """
        if k is None:
            k = len(retrieved_evidence)
        
        gold_ids = self._get_gold_ids(gold_evidence)
        
        # 上位k件のうち、ゴールド文書の数をカウント
        relevant_count = 0
        for doc in retrieved_evidence[:k]:
            if doc['passage_id'] in gold_ids:
                relevant_count += 1
        
        return relevant_count / k if k > 0 else 0.0
    
    def calculate_recall_at_k(self, retrieved_evidence: List[Dict[str, Any]], 
                             gold_evidence: List[Dict[str, Any]], 
                             k: Optional[int] = None) -> float:
        """
        Recall@kを算出
        
        Args:
            retrieved_evidence: 検索で取得された文書のリスト
            gold_evidence: ゴールド文書のリスト
            k: 評価する上位k件（Noneの場合は全て）
            
        Returns:
            Recall@kスコア
        """
        if k is None:
            k = len(retrieved_evidence)
        
        gold_ids = self._get_gold_ids(gold_evidence)
        
        if len(gold_ids) == 0:
            return 0.0
        
        # 上位k件のうち、ゴールド文書の数をカウント
        relevant_count = 0
        for doc in retrieved_evidence[:k]:
            if doc['passage_id'] in gold_ids:
                relevant_count += 1
        
        return relevant_count / len(gold_ids)
    
    def calculate_ndcg(self, retrieved_evidence: List[Dict[str, Any]], 
                      gold_evidence: List[Dict[str, Any]], 
                      k: Optional[int] = None) -> float:
        """
        nDCG（Normalized Discounted Cumulative Gain）を算出
        
        Args:
            retrieved_evidence: 検索で取得された文書のリスト
            gold_evidence: ゴールド文書のリスト
            k: 評価する上位k件（Noneの場合は全て）
            
        Returns:
            nDCGスコア
        """
        if k is None:
            k = len(retrieved_evidence)
        
        gold_ids = self._get_gold_ids(gold_evidence)
        
        dcg = 0.0
        for i, doc in enumerate(retrieved_evidence[:k]):
            if doc['passage_id'] in gold_ids:
                relevance = 1.0
            else:
                relevance = 0.0
            
            dcg += relevance / np.log2(i + 2)
        
        # 理想的なDCG（IDCG）を計算
        idcg = 0.0
        num_gold = min(len(gold_ids), k)
        for i in range(num_gold):
            idcg += 1.0 / np.log2(i + 2)
        
        # nDCGを計算
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def calculate_all_metrics(self, retrieved_evidence: List[Dict[str, Any]], 
                             gold_evidence: List[Dict[str, Any]], 
                             k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        全ての評価指標を一度に算出
        
        Args:
            retrieved_evidence: 検索で取得された文書のリスト
            gold_evidence: ゴールド文書のリスト
            k_values: 評価するk値のリスト
            
        Returns:
            各評価指標のスコアを含む辞書
        """
        results = {}
        
        # nDCG@k
        for k in k_values:
            results[f'ndcg@{k}'] = self.calculate_ndcg(retrieved_evidence, gold_evidence, k)
        
        # Precision@k
        for k in k_values:
            results[f'precision@{k}'] = self.calculate_precision_at_k(retrieved_evidence, gold_evidence, k)
        
        # Recall@k
        for k in k_values:
            results[f'recall@{k}'] = self.calculate_recall_at_k(retrieved_evidence, gold_evidence, k)
        
        return results
    
    def calculate_ndcg_multiple_k(self, retrieved_evidence: List[Dict[str, Any]], 
                                 gold_evidence: List[Dict[str, Any]], 
                                 k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        複数のk値でnDCGを算出（後方互換性のため残す）
        
        Args:
            retrieved_evidence: 検索で取得された文書のリスト
            gold_evidence: ゴールド文書のリスト
            k_values: 評価するk値のリスト
            
        Returns:
            各k値でのnDCGスコアを含む辞書
        """
        results = {}
        for k in k_values:
            results[f'ndcg@{k}'] = self.calculate_ndcg(retrieved_evidence, gold_evidence, k)
        return results 