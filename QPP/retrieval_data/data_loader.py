"""
Retrieval結果と対話データを合体するデータローダー
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    conv_id: str
    turn_id: str
    question: str
    documents: List[Dict]
    evidence_passage_ids: List[str]
    prev_evidence_passage_ids: List[str]


class RetrievalDataLoader:
    def __init__(self, dialogue_data_dir: str, retrieval_results_dir: str):
        self.dialogue_data_dir = Path(dialogue_data_dir)
        self.retrieval_results_dir = Path(retrieval_results_dir)
    
    def load_dialogue_data(self, split: str = "dev") -> Dict[str, Dict]:
        """対話データを読み込む"""
        file_path = self.dialogue_data_dir / f"{split}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_retrieval_results(self, split: str = "dev") -> List[Dict]:
        """retrieval結果を読み込む"""
        file_path = self.retrieval_results_dir / f"dpr_{split}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_evidence_passage_ids(self, turn: Dict) -> Tuple[List[str], List[str]]:
        """evidenceとprevEvidenceからpassage_idを抽出（最初のラベルのみ）"""
        evidence_ids = []
        prev_evidence_ids = []
        
        # evidenceの抽出（最初のラベルのみ）
        labels = turn.get('labels', [])
        if labels:
            first_label = labels[0]  # 最初のラベルのみを使用
            evidence = first_label.get('evidence', [])
            for ev in evidence:
                passage_id = ev.get('passage_id', '')
                if passage_id:
                    evidence_ids.append(passage_id)
        
        # prevEvidenceの抽出
        prev_evidence = turn.get('prevEvidence', [])
        for prev_ev_list in prev_evidence:
            for prev_ev in prev_ev_list:
                passage_id = prev_ev.get('passage_id', '')
                if passage_id:
                    prev_evidence_ids.append(passage_id)
        
        return evidence_ids, prev_evidence_ids
    
    def calculate_ndcg(self, relevance_scores: List[float], k: int) -> float:
        """NDCG@kを計算"""
        if not relevance_scores:
            return 0.0
        
        # 最初のk個のスコアを使用
        scores = relevance_scores[:k]
        
        # DCGの計算
        dcg = 0.0
        for i, score in enumerate(scores):
            dcg += score / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # IDCGの計算（理想的な順序）
        ideal_scores = sorted(scores, reverse=True)
        idcg = 0.0
        for i, score in enumerate(ideal_scores):
            idcg += score / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_mrr(self, retrieved_docs: List[Dict], relevant_passage_ids: List[str]) -> float:
        """MRRを計算"""
        if not retrieved_docs or not relevant_passage_ids:
            return 0.0
        
        for rank, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc.get('id', '')
            if doc_id in relevant_passage_ids:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_precision_at_k(self, relevance_scores: List[float], k: int) -> float:
        """Precision@kを計算"""
        if not relevance_scores or k == 0:
            return 0.0
        
        top_k = relevance_scores[:k]
        relevant_count = sum(top_k)
        return relevant_count / k if k > 0 else 0.0
    
    def calculate_recall_at_k(self, relevance_scores: List[float], k: int, total_relevant: int) -> float:
        """Recall@kを計算"""
        if not relevance_scores or total_relevant == 0:
            return 0.0
        
        top_k = relevance_scores[:k]
        relevant_count = sum(top_k)
        return relevant_count / total_relevant if total_relevant > 0 else 0.0
    
    def calculate_hit_rate_at_k(self, relevance_scores: List[float], k: int) -> float:
        """Hit Rate@kを計算（少なくとも1つの関連ドキュメントがあるかどうか）"""
        if not relevance_scores or k == 0:
            return 0.0
        
        top_k = relevance_scores[:k]
        return 1.0 if any(top_k) else 0.0
    
    def calculate_map(self, relevance_scores: List[float], total_relevant: int) -> float:
        """MAP (Mean Average Precision)を計算"""
        if not relevance_scores or total_relevant == 0:
            return 0.0
        
        precision_sum = 0.0
        relevant_found = 0
        
        for i, score in enumerate(relevance_scores):
            if score > 0:  # 関連ドキュメントが見つかった
                relevant_found += 1
                precision_at_i = relevant_found / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / total_relevant if total_relevant > 0 else 0.0
    
    def process_data(self, split: str = "dev") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """データを処理して2種類のCSVを生成"""
        # データの読み込み
        dialogue_data = self.load_dialogue_data(split)
        retrieval_results = self.load_retrieval_results(split)
        
        # retrieval結果を辞書に変換（conv_id+turn_idをキーとして）
        retrieval_dict = {}
        for item in retrieval_results:
            conv_id = item['conv_id']
            turn_id = item['turn_id']
            key = f"{conv_id}_{turn_id}"
            retrieval_dict[key] = item
        
        # 結果を格納するリスト
        evidence_only_results = []
        evidence_prev_evidence_results = []
        
        for conv_id, dialogue_info in dialogue_data.items():
            turns = dialogue_info.get('turns', [])
            
            for turn_idx, turn in enumerate(turns):
                turn_id = str(turn_idx + 1)
                key = f"{conv_id}_{turn_id}"
                
                # retrieval結果を取得
                if key not in retrieval_dict:
                    continue
                
                retrieval_item = retrieval_dict[key]
                retrieved_docs = retrieval_item.get('ctxs', [])
                
                # evidenceの抽出
                evidence_ids, prev_evidence_ids = self.extract_evidence_passage_ids(turn)
                
                # 共通の処理
                base_result = {
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'num_retrieved_docs': len(retrieved_docs),
                    'num_evidence_docs': len(evidence_ids),
                    'num_prev_evidence_docs': len(prev_evidence_ids)
                }
                
                # evidenceのみの場合（全てのターン、evidenceのみで計算）
                evidence_result = base_result.copy()
                if evidence_ids:
                    evidence_result.update(self._calculate_metrics(retrieved_docs, evidence_ids))
                else:
                    # evidenceがない場合は、全て0で記録
                    evidence_result.update({
                        'found_ratio': 0,
                        'mrr': 0.0,
                        'map': 0.0,
                        'ndcg@1': 0.0,
                        'ndcg@5': 0.0,
                        'ndcg@10': 0.0,
                        'ndcg@20': 0.0,
                        'ndcg@50': 0.0,
                        'ndcg@100': 0.0,
                        'precision@1': 0.0,
                        'precision@5': 0.0,
                        'precision@10': 0.0,
                        'precision@20': 0.0,
                        'precision@50': 0.0,
                        'precision@100': 0.0,
                        'recall@1': 0.0,
                        'recall@5': 0.0,
                        'recall@10': 0.0,
                        'recall@20': 0.0,
                        'recall@50': 0.0,
                        'recall@100': 0.0,
                        'hit_rate@1': 0.0,
                        'hit_rate@5': 0.0,
                        'hit_rate@10': 0.0,
                        'hit_rate@20': 0.0,
                        'hit_rate@50': 0.0,
                        'hit_rate@100': 0.0
                    })
                evidence_only_results.append(evidence_result)
                
                # evidence + prevEvidenceの場合（全てのターン、evidenceとprevEvidence両方で計算）
                all_turns_result = base_result.copy()
                if evidence_ids:
                    if prev_evidence_ids:
                        # prevEvidenceがある場合は、evidence + prevEvidenceで計算
                        combined_evidence_ids = evidence_ids + prev_evidence_ids
                        all_turns_result.update(self._calculate_metrics(retrieved_docs, combined_evidence_ids))
                    else:
                        # prevEvidenceがない場合は、evidenceのみで計算
                        all_turns_result.update(self._calculate_metrics(retrieved_docs, evidence_ids))
                else:
                    # evidenceがない場合は、全て0で記録
                    all_turns_result.update({
                        'found_ratio': 0,
                        'mrr': 0.0,
                        'map': 0.0,
                        'ndcg@1': 0.0,
                        'ndcg@5': 0.0,
                        'ndcg@10': 0.0,
                        'ndcg@20': 0.0,
                        'ndcg@50': 0.0,
                        'ndcg@100': 0.0,
                        'precision@1': 0.0,
                        'precision@5': 0.0,
                        'precision@10': 0.0,
                        'precision@20': 0.0,
                        'precision@50': 0.0,
                        'precision@100': 0.0,
                        'recall@1': 0.0,
                        'recall@5': 0.0,
                        'recall@10': 0.0,
                        'recall@20': 0.0,
                        'recall@50': 0.0,
                        'recall@100': 0.0,
                        'hit_rate@1': 0.0,
                        'hit_rate@5': 0.0,
                        'hit_rate@10': 0.0,
                        'hit_rate@20': 0.0,
                        'hit_rate@50': 0.0,
                        'hit_rate@100': 0.0
                    })
                evidence_prev_evidence_results.append(all_turns_result)
        
        # DataFrameに変換
        evidence_only_df = pd.DataFrame(evidence_only_results)
        evidence_prev_evidence_df = pd.DataFrame(evidence_prev_evidence_results)
        
        return evidence_only_df, evidence_prev_evidence_df
    
    def _calculate_metrics(self, retrieved_docs: List[Dict], relevant_passage_ids: List[str]) -> Dict:
        """メトリクスを計算"""
        # 関連性スコアの計算
        relevance_scores = []
        found_ratio = 0
        total_relevant = len(relevant_passage_ids)
        
        for doc in retrieved_docs:
            doc_id = doc.get('id', '')
            if doc_id in relevant_passage_ids:
                relevance_scores.append(1.0)
                found_ratio = 1
            else:
                relevance_scores.append(0.0)
        
        # MRRの計算
        mrr = self.calculate_mrr(retrieved_docs, relevant_passage_ids)
        
        # NDCG@kの計算
        ndcg_1 = self.calculate_ndcg(relevance_scores, 1)
        ndcg_5 = self.calculate_ndcg(relevance_scores, 5)
        ndcg_10 = self.calculate_ndcg(relevance_scores, 10)
        ndcg_20 = self.calculate_ndcg(relevance_scores, 20)
        ndcg_50 = self.calculate_ndcg(relevance_scores, 50)
        ndcg_100 = self.calculate_ndcg(relevance_scores, 100)
        
        # Precision@kの計算
        precision_1 = self.calculate_precision_at_k(relevance_scores, 1)
        precision_5 = self.calculate_precision_at_k(relevance_scores, 5)
        precision_10 = self.calculate_precision_at_k(relevance_scores, 10)
        precision_20 = self.calculate_precision_at_k(relevance_scores, 20)
        precision_50 = self.calculate_precision_at_k(relevance_scores, 50)
        precision_100 = self.calculate_precision_at_k(relevance_scores, 100)
        
        # Recall@kの計算
        recall_1 = self.calculate_recall_at_k(relevance_scores, 1, total_relevant)
        recall_5 = self.calculate_recall_at_k(relevance_scores, 5, total_relevant)
        recall_10 = self.calculate_recall_at_k(relevance_scores, 10, total_relevant)
        recall_20 = self.calculate_recall_at_k(relevance_scores, 20, total_relevant)
        recall_50 = self.calculate_recall_at_k(relevance_scores, 50, total_relevant)
        recall_100 = self.calculate_recall_at_k(relevance_scores, 100, total_relevant)
        
        # Hit Rate@kの計算
        hit_rate_1 = self.calculate_hit_rate_at_k(relevance_scores, 1)
        hit_rate_5 = self.calculate_hit_rate_at_k(relevance_scores, 5)
        hit_rate_10 = self.calculate_hit_rate_at_k(relevance_scores, 10)
        hit_rate_20 = self.calculate_hit_rate_at_k(relevance_scores, 20)
        hit_rate_50 = self.calculate_hit_rate_at_k(relevance_scores, 50)
        hit_rate_100 = self.calculate_hit_rate_at_k(relevance_scores, 100)
        
        # MAPの計算
        map_score = self.calculate_map(relevance_scores, total_relevant)
        
        return {
            'found_ratio': found_ratio,
            'mrr': mrr,
            'map': map_score,
            'ndcg@1': ndcg_1,
            'ndcg@5': ndcg_5,
            'ndcg@10': ndcg_10,
            'ndcg@20': ndcg_20,
            'ndcg@50': ndcg_50,
            'ndcg@100': ndcg_100,
            'precision@1': precision_1,
            'precision@5': precision_5,
            'precision@10': precision_10,
            'precision@20': precision_20,
            'precision@50': precision_50,
            'precision@100': precision_100,
            'recall@1': recall_1,
            'recall@5': recall_5,
            'recall@10': recall_10,
            'recall@20': recall_20,
            'recall@50': recall_50,
            'recall@100': recall_100,
            'hit_rate@1': hit_rate_1,
            'hit_rate@5': hit_rate_5,
            'hit_rate@10': hit_rate_10,
            'hit_rate@20': hit_rate_20,
            'hit_rate@50': hit_rate_50,
            'hit_rate@100': hit_rate_100
        }
