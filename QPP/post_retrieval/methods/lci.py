"""
LCI (Local Concentration Index) 手法の実装
"""
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
import json
from pathlib import Path
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class LCI(BaseQPPAnalyzer):
    """LCI (Local Concentration Index) 計算クラス"""
    
    def _local_concentration_index(self, titles: List[str], window: int = 3) -> float:
        """
        局所的集中度（Local Concentration Index）を計算
        
        Args:
            titles: タイトルのリスト
            window: 半径（デフォルト: 3）
        
        Returns:
            LCI値（0.0 ~ 1.0）
        """
        n = len(titles)
        if n < 2:
            return 0.0
        
        total = 0
        for i in range(n):
            # ウィンドウ内の他のタイトルをチェック
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if i != j and titles[i] == titles[j]:
                    total += 1
        
        # 正規化: total / (n * (2 * window))
        return total / (n * (2 * window))
    
    def compute(
        self,
        top_k: Optional[int] = None,
        window: int = 3
    ) -> pd.DataFrame:
        """
        各ターンについて、タイトル列の局所的集中度（LCI）を計算
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            window: LCI計算の半径（デフォルト: 3）
        
        Returns:
            DataFrame with columns: conv_id, turn_id, lci, num_documents
        """
        results = []
        
        for turn in tqdm(self.turn_data_list, desc="Computing LCI", unit="turn"):
            documents = self._get_documents(turn, top_k=top_k)
            
            if len(documents) < 2:
                # 文書が2件未満の場合はLCIを計算できない
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'lci': 0.0,
                    'num_documents': len(documents)
                })
                continue
            
            # タイトル列を抽出
            titles = [self._extract_title_from_id(doc.id) for doc in documents]
            
            # LCIを計算
            lci = self._local_concentration_index(titles, window=window)
            
            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'lci': lci,
                'num_documents': len(documents)
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
        window: int = 3
    ) -> None:
        """
        DPR結果ファイルからタイトル列の局所的集中度（LCI）を計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            window: LCI計算の半径（デフォルト: 3）
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing Local Concentration Index (LCI)...")
        # LCI計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # LCIを計算
        lci_stats = analyzer.compute(top_k=top_k, window=window)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in lci_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'lci': float(row['lci']),
                'num_retrieved_documents': int(row['num_documents'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        lci_dict = {}
        for _, row in lci_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            lci_dict[key] = {
                'lci': float(row['lci']),
                'num_documents': int(row['num_documents'])
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンにLCIを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in lci_dict:
                    stats = lci_dict[key]
                    turn['lci'] = stats['lci']
                    turn['num_retrieved_documents'] = stats['num_documents']
                else:
                    turn['lci'] = 0.0
                    turn['num_retrieved_documents'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

