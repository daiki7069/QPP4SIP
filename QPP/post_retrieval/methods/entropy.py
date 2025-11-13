"""
Entropy (Title Distribution Entropy) 手法の実装
"""
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
import json
from pathlib import Path
from collections import Counter
from math import log
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class Entropy(BaseQPPAnalyzer):
    """Entropy (Title Distribution Entropy) 計算クラス"""
    
    def _title_entropy(self, titles: List[str]) -> float:
        """
        タイトル分布の正規化エントロピーを計算
        
        Args:
            titles: タイトルのリスト
        
        Returns:
            正規化エントロピー値（0.0 ~ 1.0）
        """
        n = len(titles)
        if n == 0:
            return 0.0
        
        # タイトルの出現頻度をカウント
        counts = Counter(titles)
        m = len(counts)  # タイトルの種類数
        
        if m <= 1:
            # タイトルが1種類以下ならエントロピーは0
            return 0.0
        
        # 各タイトルの出現確率を計算
        probs = [c / n for c in counts.values()]
        
        # シャノンエントロピーを計算
        H = -sum(p * log(p) for p in probs if p > 0)
        
        # 正規化（最大エントロピー log(m) で割る）
        H_max = log(m)
        H_norm = H / H_max if H_max > 0 else 0.0
        
        return H_norm
    
    def compute(
        self,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        各ターンについて、タイトル分布の正規化エントロピーを計算
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
        
        Returns:
            DataFrame with columns: conv_id, turn_id, entropy, num_documents, num_unique_titles
        """
        results = []
        
        for turn in tqdm(self.turn_data_list, desc="Computing entropy", unit="turn"):
            documents = self._get_documents(turn, top_k=top_k)
            
            if len(documents) == 0:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'entropy': 0.0,
                    'num_documents': 0,
                    'num_unique_titles': 0
                })
                continue
            
            # タイトル列を抽出
            titles = [self._extract_title_from_id(doc.id) for doc in documents]
            
            # エントロピーを計算
            entropy = self._title_entropy(titles)
            
            # ユニークなタイトル数を計算
            num_unique_titles = len(set(titles))
            
            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'entropy': entropy,
                'num_documents': len(documents),
                'num_unique_titles': num_unique_titles
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
        DPR結果ファイルからタイトル分布の正規化エントロピーを計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing Title Distribution Entropy...")
        # エントロピー計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # エントロピーを計算
        entropy_stats = analyzer.compute(top_k=top_k)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in entropy_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'entropy': float(row['entropy']),
                'num_retrieved_documents': int(row['num_documents']),
                'num_unique_titles': int(row['num_unique_titles'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        entropy_dict = {}
        for _, row in entropy_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            entropy_dict[key] = {
                'entropy': float(row['entropy']),
                'num_documents': int(row['num_documents']),
                'num_unique_titles': int(row['num_unique_titles'])
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンにエントロピーを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in entropy_dict:
                    stats = entropy_dict[key]
                    turn['entropy'] = stats['entropy']
                    turn['num_retrieved_documents'] = stats['num_documents']
                    turn['num_unique_titles'] = stats['num_unique_titles']
                else:
                    turn['entropy'] = 0.0
                    turn['num_retrieved_documents'] = 0
                    turn['num_unique_titles'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

