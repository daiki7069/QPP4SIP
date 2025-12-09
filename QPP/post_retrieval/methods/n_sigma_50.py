"""
n(σ_50%) 指標の実装
Ronan Cummins, Joemon Jose, and Colm O'Riordan. 2011. 
Improved query performance prediction using standard deviation. 
In Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval. 1089–1090.
"""
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

from .base import BaseQPPAnalyzer
from data_loader import DPRResultLoader, TurnData


class NSigma50(BaseQPPAnalyzer):
    """
    n(σ_50%) を計算するクラス
    
    最大スコアの50%以上のスコアを持つドキュメントの標準偏差を計算し、
    クエリ長の平方根で正規化する
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_n_sigma_50(self, scores: np.ndarray, query_length: int) -> float:
        """
        n(σ_50%) を計算する
        
        Args:
            scores: ドキュメントのスコア配列
            query_length: クエリ長（単語数）
        
        Returns:
            n(σ_50%) の値
        """
        if scores.size == 0 or query_length <= 0:
            return 0.0
        
        # 1. 最大スコアを特定
        top_score = np.max(scores)
        
        # 2. カットオフ閾値を設定 (50%)
        cutoff_threshold = top_score * 0.5
        
        # 3. 動的カットオフでスコアを抽出
        filtered_scores = scores[scores >= cutoff_threshold]
        
        # 4. 標準偏差 (σ_50%) を計算
        if filtered_scores.size > 1:
            sigma_50 = float(np.std(filtered_scores, ddof=0))  # 母標準偏差
        else:
            # スコアが1つ以下の場合は標準偏差を0とする
            sigma_50 = 0.0
        
        # 5. プレディクタ n(σ_50%) を計算（クエリ長の平方根で正規化）
        n_sigma_50 = sigma_50 / np.sqrt(query_length)
        
        return n_sigma_50

    def _get_query_length(self, turn: TurnData) -> int:
        """
        クエリ長（単語数）を取得
        
        Args:
            turn: TurnDataオブジェクト
        
        Returns:
            クエリ長（単語数）
        """
        question = turn.question.strip()
        if not question:
            return 1  # 空のクエリの場合は1を返す（ゼロ除算を避ける）
        # 空白で分割して単語数をカウント
        query_length = len(question.split())
        return max(1, query_length)  # 最小値は1

    def compute(
        self,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        各ターンについてn(σ_50%)を計算

        Args:
            top_k: 上位k件の文書のみを対象にする
        """
        results: List[Dict[str, Any]] = []

        for turn in tqdm(self.turn_data_list, desc="Computing n(σ_50%)", unit="turn"):
            documents = self._get_documents(turn, top_k=top_k)
            if not documents:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'n_sigma_50': np.nan,
                    'sigma_50': np.nan,
                    'query_length': np.nan,
                    'num_documents': 0,
                    'num_filtered_documents': 0
                })
                continue

            scores = np.array([doc.score for doc in documents], dtype=float)
            query_length = self._get_query_length(turn)
            
            # n(σ_50%) を計算
            n_sigma_50 = self._calculate_n_sigma_50(scores, query_length)
            
            # デバッグ情報用にσ_50%も計算
            top_score = np.max(scores)
            cutoff_threshold = top_score * 0.5
            filtered_scores = scores[scores >= cutoff_threshold]
            sigma_50 = float(np.std(filtered_scores, ddof=0)) if filtered_scores.size > 1 else 0.0

            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'n_sigma_50': n_sigma_50,
                'sigma_50': sigma_50,
                'query_length': query_length,
                'num_documents': len(scores),
                'num_filtered_documents': len(filtered_scores)
            })

        return pd.DataFrame(results)

    @classmethod
    def compute_from_files(
        cls,
        dpr_json_path: str,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: str,
        top_k: Optional[int] = None
    ) -> None:
        """
        DPR結果からn(σ_50%)を計算して保存
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")

        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        n_sigma_50_stats = analyzer.compute(top_k=top_k)

        csv_df = pd.DataFrame([{
            'conv_id': row['conv_id'],
            'turn_id': row['turn_id'],
            'n_sigma_50': None if np.isnan(row['n_sigma_50']) else float(row['n_sigma_50']),
            'sigma_50': None if np.isnan(row['sigma_50']) else float(row['sigma_50']),
            'query_length': None if np.isnan(row['query_length']) else int(row['query_length']),
            'num_retrieved_documents': int(row['num_documents']),
            'num_filtered_documents': int(row['num_filtered_documents'])
        } for _, row in n_sigma_50_stats.iterrows()])

        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")

        stats_dict: Dict[tuple, Dict[str, Any]] = {}
        for _, row in n_sigma_50_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            stats_dict[key] = {
                'n_sigma_50': None if np.isnan(row['n_sigma_50']) else float(row['n_sigma_50']),
                'sigma_50': None if np.isnan(row['sigma_50']) else float(row['sigma_50']),
                'query_length': None if np.isnan(row['query_length']) else int(row['query_length']),
                'num_retrieved_documents': int(row['num_documents']),
                'num_filtered_documents': int(row['num_filtered_documents']),
            }

        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)

        for conversation in base_data:
            for turn in conversation:
                key = (turn['conv_id'], str(turn['turn_id']))
                stats = stats_dict.get(key, None)
                if stats:
                    turn['n_sigma_50'] = stats['n_sigma_50']
                    turn['sigma_50'] = stats['sigma_50']
                    turn['query_length'] = stats['query_length']
                    turn['num_retrieved_documents'] = stats['num_retrieved_documents']
                    turn['num_filtered_documents'] = stats['num_filtered_documents']
                else:
                    turn['n_sigma_50'] = None
                    turn['sigma_50'] = None
                    turn['query_length'] = None
                    turn['num_retrieved_documents'] = 0
                    turn['num_filtered_documents'] = 0

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)

        print(f"JSON saved to: {output_json_path}")

