"""
SMV (Score Magnitude Variance) 指標の実装
"""
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

from .base import BaseQPPAnalyzer
from data_loader import DPRResultLoader, TurnData


class SMV(BaseQPPAnalyzer):
    """
    Score Magnitude Variance を計算するクラス

    SMVは上位文書スコアの分散 (variance) を用いてクエリ性能を推定する
    """

    def compute(
        self,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        各ターンについてSMVを計算

        Args:
            top_k: 上位k件の文書のみを対象にする（Noneの場合は全件、最大100件）

        Returns:
            DataFrame: conv_id, turn_id, smv, score_mean, score_std, score_variance, num_documents
        """
        results: List[Dict[str, Any]] = []

        for turn in tqdm(self.turn_data_list, desc="Computing SMV", unit="turn"):
            documents = self._get_documents(turn, top_k=top_k)

            if not documents:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'smv': np.nan,
                    'score_mean': np.nan,
                    'score_std': np.nan,
                    'score_variance': np.nan,
                    'num_documents': 0,
                })
                continue

            scores = np.array([doc.score for doc in documents], dtype=float)
            variance = float(np.var(scores, ddof=0))
            std = float(np.sqrt(variance))
            mean = float(np.mean(scores))

            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'smv': variance,
                'score_mean': mean,
                'score_std': std,
                'score_variance': variance,
                'num_documents': len(scores),
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
        ファイルからDPR結果を読み込み、SMVを計算して保存
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")

        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        smv_stats = analyzer.compute(top_k=top_k)

        csv_df = pd.DataFrame([{
            'conv_id': row['conv_id'],
            'turn_id': row['turn_id'],
            'smv': None if np.isnan(row['smv']) else float(row['smv']),
            'score_mean': None if np.isnan(row['score_mean']) else float(row['score_mean']),
            'score_std': None if np.isnan(row['score_std']) else float(row['score_std']),
            'score_variance': None if np.isnan(row['score_variance']) else float(row['score_variance']),
            'num_retrieved_documents': int(row['num_documents'])
        } for _, row in smv_stats.iterrows()])

        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")

        stats_dict: Dict[tuple, Dict[str, Any]] = {}
        for _, row in smv_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            stats_dict[key] = {
                'smv': None if np.isnan(row['smv']) else float(row['smv']),
                'score_mean': None if np.isnan(row['score_mean']) else float(row['score_mean']),
                'score_std': None if np.isnan(row['score_std']) else float(row['score_std']),
                'score_variance': None if np.isnan(row['score_variance']) else float(row['score_variance']),
                'num_retrieved_documents': int(row['num_documents']),
            }

        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)

        for conversation in base_data:
            for turn in conversation:
                key = (turn['conv_id'], str(turn['turn_id']))
                stats = stats_dict.get(key, None)
                if stats:
                    turn['smv'] = stats['smv']
                    turn['score_mean'] = stats['score_mean']
                    turn['score_std'] = stats['score_std']
                    turn['score_variance'] = stats['score_variance']
                    turn['num_retrieved_documents'] = stats['num_retrieved_documents']
                else:
                    turn['smv'] = None
                    turn['score_mean'] = None
                    turn['score_std'] = None
                    turn['score_variance'] = None
                    turn['num_retrieved_documents'] = 0

        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)

        print(f"JSON saved to: {output_json_path}")

