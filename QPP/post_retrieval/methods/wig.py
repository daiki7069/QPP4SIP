"""
WIG (Weighted Information Gain) 手法の実装（DPR版）
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm
import json
from pathlib import Path
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class WIG(BaseQPPAnalyzer):
    """WIG (Weighted Information Gain) 計算クラス（DPR版）"""
    
    def compute(
        self,
        top_k: Optional[int] = None,
        k: int = 50,
        bg_ratio: float = 0.2
    ) -> pd.DataFrame:
        """
        各ターンについて、DPRスコアからWIG（Weighted Information Gain）を計算
        
        WIG_DPR(q) = (1/k) * Σ[sim(q,d) - sim(q,D)]
        
        ここで：
        - sim(q,d): DPRのdot score（データ内のスコア）
        - sim(q,D): 背景スコア（下位ランクの平均）
        - k: WIGに使うtop-k文書数
        
        Args:
            top_k: 処理する上位k件の文書数（Noneの場合は全件、最大100件）
            k: WIG計算に使うtop-k文書数（デフォルト: 50）
            bg_ratio: 背景モデルとして使う割合（下位の文書、デフォルト: 0.2）
        
        Returns:
            DataFrame with columns: conv_id, turn_id, wig, num_documents, 
                                   topk_mean_score, bg_score, k, bg_count
        """
        results = []
        
        for turn in tqdm(self.turn_data_list, desc="Computing WIG", unit="turn"):
            documents = self._get_documents(turn, top_k=top_k)
            
            if len(documents) == 0:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'wig': np.nan,
                    'num_documents': 0,
                    'topk_mean_score': np.nan,
                    'bg_score': np.nan,
                    'k': k,
                    'bg_count': 0
                })
                continue
            
            # スコアを取得（DPRのdot score）
            scores = np.array([doc.score for doc in documents])
            n = len(scores)
            
            # kが利用可能な文書数より大きい場合は調整
            actual_k = min(k, n)
            
            if actual_k == 0 or n == 0:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'wig': np.nan,
                    'num_documents': n,
                    'topk_mean_score': np.nan,
                    'bg_score': np.nan,
                    'k': actual_k,
                    'bg_count': 0
                })
                continue
            
            # ① 上位k件のスコア
            topk_scores = np.sort(scores)[::-1][:actual_k]  # 降順ソートして上位k件
            
            # ② 背景モデル（bottom-m の平均）
            # bg_ratioに基づいて下位m件を取得
            m = max(1, int(n * bg_ratio))  # 最低1件は確保
            m = min(m, n - actual_k)  # top-kと重ならないように調整
            
            if m > 0:
                tail_scores = np.sort(scores)[:m]  # 昇順ソートして下位m件
                bg_score = np.mean(tail_scores)
            else:
                # 背景スコアが計算できない場合は0とする（方法B）
                bg_score = 0.0
            
            # ③ WIG = (1/k) * Σ(topk_scores - bg_score)
            wig = np.mean(topk_scores - bg_score)
            
            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'wig': wig,
                'num_documents': n,
                'topk_mean_score': np.mean(topk_scores),
                'bg_score': bg_score,
                'k': actual_k,
                'bg_count': m
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
        k: int = 50,
        bg_ratio: float = 0.2
    ) -> None:
        """
        DPR結果ファイルからWIG（Weighted Information Gain）を計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス
            top_k: 処理する上位k件の文書数（Noneの場合は全件、最大100件）
            k: WIG計算に使うtop-k文書数（デフォルト: 50）
            bg_ratio: 背景モデルとして使う割合（下位の文書、デフォルト: 0.2）
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print(f"Computing WIG (Weighted Information Gain) with k={k}, bg_ratio={bg_ratio}...")
        # WIG計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # WIGを計算
        wig_stats = analyzer.compute(top_k=top_k, k=k, bg_ratio=bg_ratio)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in wig_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'wig': float(row['wig']) if not np.isnan(row['wig']) else None,
                'num_retrieved_documents': int(row['num_documents']),
                'topk_mean_score': float(row['topk_mean_score']) if not np.isnan(row['topk_mean_score']) else None,
                'bg_score': float(row['bg_score']) if not np.isnan(row['bg_score']) else None,
                'k': int(row['k']),
                'bg_count': int(row['bg_count'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        wig_dict = {}
        for _, row in wig_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            wig_dict[key] = {
                'wig': float(row['wig']) if not np.isnan(row['wig']) else None,
                'num_documents': int(row['num_documents']),
                'topk_mean_score': float(row['topk_mean_score']) if not np.isnan(row['topk_mean_score']) else None,
                'bg_score': float(row['bg_score']) if not np.isnan(row['bg_score']) else None,
                'k': int(row['k']),
                'bg_count': int(row['bg_count'])
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンにWIGを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in wig_dict:
                    stats = wig_dict[key]
                    turn['wig'] = stats['wig']
                    turn['num_retrieved_documents'] = stats['num_documents']
                    turn['topk_mean_score'] = stats['topk_mean_score']
                    turn['bg_score'] = stats['bg_score']
                    turn['wig_k'] = stats['k']
                    turn['wig_bg_count'] = stats['bg_count']
                else:
                    turn['wig'] = None
                    turn['num_retrieved_documents'] = 0
                    turn['topk_mean_score'] = None
                    turn['bg_score'] = None
                    turn['wig_k'] = k
                    turn['wig_bg_count'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

