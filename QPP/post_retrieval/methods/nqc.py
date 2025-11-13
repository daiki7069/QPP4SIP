"""
NQC (Normalized Query Clarity) 手法の実装
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm
import json
from pathlib import Path
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class NQC(BaseQPPAnalyzer):
    """NQC (Normalized Query Clarity) 計算クラス"""
    
    def compute(
        self,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        各ターンについて、上位k件のスコアからNQC（Normalized Query Clarity）を計算
        
        NQC = √(Σ (si - μ)^2 / k) / μ
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
        
        Returns:
            DataFrame with columns: conv_id, turn_id, nqc, num_documents, score_mean, score_std
        """
        results = []
        
        for turn in tqdm(self.turn_data_list, desc="Computing NQC", unit="turn"):
            documents = self._get_documents(turn, top_k=top_k)
            
            if len(documents) == 0:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'nqc': np.nan,
                    'num_documents': 0,
                    'score_mean': np.nan,
                    'score_std': np.nan
                })
                continue
            
            # スコアを取得
            scores = np.array([doc.score for doc in documents])
            k = len(scores)
            
            # 平均値を計算
            mu = np.mean(scores)
            
            if mu == 0 or k == 0:
                # 平均が0またはkが0の場合はNQCを計算できない
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'nqc': np.nan,
                    'num_documents': k,
                    'score_mean': mu,
                    'score_std': np.nan
                })
                continue
            
            # 標準偏差を計算（母標準偏差、ddof=0）
            # √(Σ (si - μ)^2 / k)
            std = np.std(scores, ddof=0)
            
            # NQC = std / μ
            nqc = std / mu
            
            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'nqc': nqc,
                'num_documents': k,
                'score_mean': mu,
                'score_std': std
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
        DPR結果ファイルから上位k件のスコアからNQC（Normalized Query Clarity）を計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing NQC (Normalized Query Clarity)...")
        # NQC計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # NQCを計算
        nqc_stats = analyzer.compute(top_k=top_k)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in nqc_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'nqc': float(row['nqc']) if not np.isnan(row['nqc']) else None,
                'num_retrieved_documents': int(row['num_documents']),
                'score_mean': float(row['score_mean']) if not np.isnan(row['score_mean']) else None,
                'score_std': float(row['score_std']) if not np.isnan(row['score_std']) else None
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        nqc_dict = {}
        for _, row in nqc_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            nqc_dict[key] = {
                'nqc': float(row['nqc']) if not np.isnan(row['nqc']) else None,
                'num_documents': int(row['num_documents']),
                'score_mean': float(row['score_mean']) if not np.isnan(row['score_mean']) else None,
                'score_std': float(row['score_std']) if not np.isnan(row['score_std']) else None
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンにNQCを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in nqc_dict:
                    stats = nqc_dict[key]
                    turn['nqc'] = stats['nqc']
                    turn['num_retrieved_documents'] = stats['num_documents']
                    turn['score_mean'] = stats['score_mean']
                    turn['score_std'] = stats['score_std']
                else:
                    turn['nqc'] = None
                    turn['num_retrieved_documents'] = 0
                    turn['score_mean'] = None
                    turn['score_std'] = None
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

