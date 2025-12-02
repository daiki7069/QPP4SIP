"""
NSV (Normalized Score Variance / N(σ)) 指標の実装
"""
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

from .base import BaseQPPAnalyzer
from data_loader import DPRResultLoader, TurnData


class NSV(BaseQPPAnalyzer):
    """
    Normalized Score Variance (N(σ)) を計算するクラス

    標準偏差をスコアのレンジで正規化し、スコアスケールの違いを補正する
    """

    def __init__(self, *args, epsilon: float = 1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def compute(
        self,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        各ターンについてNSVを計算

        Args:
            top_k: 上位k件の文書のみを対象にする
        """
        results: List[Dict[str, Any]] = []

        for turn in tqdm(self.turn_data_list, desc="Computing NSV", unit="turn"):
            documents = self._get_documents(turn, top_k=top_k)
            if not documents:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'nsv': np.nan,
                    'score_mean': np.nan,
                    'score_std': np.nan,
                    'score_range': np.nan,
                    'num_documents': 0
                })
                continue

            scores = np.array([doc.score for doc in documents], dtype=float)
            std = float(np.std(scores, ddof=0))
            score_range = float(np.max(scores) - np.min(scores))

            if score_range <= self.epsilon:
                nsv_value = np.nan
            else:
                nsv_value = std / score_range

            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'nsv': nsv_value,
                'score_mean': float(np.mean(scores)),
                'score_std': std,
                'score_range': score_range,
                'num_documents': len(scores)
            })

        return pd.DataFrame(results)

    @classmethod
    def compute_from_files(
        cls,
        dpr_json_path: str,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: str,
        top_k: Optional[int] = None,
        epsilon: float = 1e-8
    ) -> None:
        """
        DPR結果からNSVを計算して保存
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")

        analyzer = cls(turn_data_list, device=None, cache_dir=None, epsilon=epsilon)
        nsv_stats = analyzer.compute(top_k=top_k)

        csv_df = pd.DataFrame([{
            'conv_id': row['conv_id'],
            'turn_id': row['turn_id'],
            'nsv': None if np.isnan(row['nsv']) else float(row['nsv']),
            'score_mean': None if np.isnan(row['score_mean']) else float(row['score_mean']),
            'score_std': None if np.isnan(row['score_std']) else float(row['score_std']),
            'score_range': None if np.isnan(row['score_range']) else float(row['score_range']),
            'num_retrieved_documents': int(row['num_documents'])
        } for _, row in nsv_stats.iterrows()])

        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")

        stats_dict: Dict[tuple, Dict[str, Any]] = {}
        for _, row in nsv_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            stats_dict[key] = {
                'nsv': None if np.isnan(row['nsv']) else float(row['nsv']),
                'score_mean': None if np.isnan(row['score_mean']) else float(row['score_mean']),
                'score_std': None if np.isnan(row['score_std']) else float(row['score_std']),
                'score_range': None if np.isnan(row['score_range']) else float(row['score_range']),
                'num_retrieved_documents': int(row['num_documents']),
            }

        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)

        for conversation in base_data:
            for turn in conversation:
                key = (turn['conv_id'], str(turn['turn_id']))
                stats = stats_dict.get(key, None)
                if stats:
                    turn['nsv'] = stats['nsv']
                    turn['score_mean'] = stats['score_mean']
                    turn['score_std'] = stats['score_std']
                    turn['score_range'] = stats['score_range']
                    turn['num_retrieved_documents'] = stats['num_retrieved_documents']
                else:
                    turn['nsv'] = None
                    turn['score_mean'] = None
                    turn['score_std'] = None
                    turn['score_range'] = None
                    turn['num_retrieved_documents'] = 0

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)

        print(f"JSON saved to: {output_json_path}")

