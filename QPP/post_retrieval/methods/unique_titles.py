"""
Unique Titles 手法の実装
"""
import pandas as pd
from typing import List, Optional
from tqdm import tqdm
import json
from pathlib import Path
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class UniqueTitles(BaseQPPAnalyzer):
    """Unique Titles 計算クラス"""
    
    def compute(
        self,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        各ターンについて、top_kに含まれるユニークなタイトルの種類数を計算
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
        
        Returns:
            DataFrame with columns: conv_id, turn_id, num_unique_titles, num_documents
        """
        results = []
        
        for turn in tqdm(self.turn_data_list, desc="Computing unique titles", unit="turn"):
            documents = self._get_documents(turn, top_k=top_k)
            
            if len(documents) == 0:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'num_unique_titles': 0,
                    'num_documents': 0
                })
                continue
            
            # タイトル列を抽出
            titles = [self._extract_title_from_id(doc.id) for doc in documents]
            
            # ユニークなタイトル数を計算
            num_unique_titles = len(set(titles))
            
            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'num_unique_titles': num_unique_titles,
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
        top_k: Optional[int] = None
    ) -> None:
        """
        DPR結果ファイルからtop_kに含まれるユニークなタイトルの種類数を計算し、ベースJSONに追記して出力
        
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
        
        print("Computing unique titles count...")
        # ユニークタイトル数計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # ユニークタイトル数を計算
        unique_titles_stats = analyzer.compute(top_k=top_k)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in unique_titles_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'num_unique_titles': int(row['num_unique_titles']),
                'num_retrieved_documents': int(row['num_documents'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        unique_titles_dict = {}
        for _, row in unique_titles_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            unique_titles_dict[key] = {
                'num_unique_titles': int(row['num_unique_titles']),
                'num_documents': int(row['num_documents'])
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンにユニークタイトル数を追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in unique_titles_dict:
                    stats = unique_titles_dict[key]
                    turn['num_unique_titles'] = stats['num_unique_titles']
                    turn['num_retrieved_documents'] = stats['num_documents']
                else:
                    turn['num_unique_titles'] = 0
                    turn['num_retrieved_documents'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

