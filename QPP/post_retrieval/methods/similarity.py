"""
Similarity Statistics 手法の実装
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class Similarity(BaseQPPAnalyzer):
    """Similarity Statistics 計算クラス"""
    
    def compute(
        self, 
        top_k: Optional[int] = None,
        return_all_pairs: bool = False
    ) -> pd.DataFrame:
        """
        各ターンについて、全文書ペアのコサイン類似度を計算し、統計値を返す
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            return_all_pairs: Trueの場合、全ペアの類似度も含める
        
        Returns:
            DataFrame with columns: conv_id, turn_id, num_documents, 
                                    mean_similarity, sum_similarity, num_pairs
                                    (return_all_pairs=Trueの場合、追加でpair_dataが含まれる)
        """
        results = []
        total_turns = len(self.turn_data_list)
        cache_hits = 0
        cache_misses = 0
        
        for idx, turn in enumerate(tqdm(self.turn_data_list, desc="Processing turns", unit="turn"), 1):
            documents = self._get_documents(turn, top_k=top_k)
            
            if len(documents) < 2:
                # 文書が2件未満の場合は類似度を計算できない
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'num_documents': len(documents),
                    'mean_similarity': np.nan,
                    'sum_similarity': np.nan,
                    'num_pairs': 0
                })
                continue
            
            # 各文書の埋め込みを計算
            embeddings = []
            num_docs = len(documents)
            for doc in tqdm(documents, desc=f"  Encoding docs (turn {idx}/{total_turns})", leave=False, unit="doc"):
                text_hash = self._get_text_hash(doc.text)
                was_cached = text_hash in self.embedding_cache
                if was_cached:
                    cache_hits += 1
                else:
                    cache_misses += 1
                embedding = self._encode_document(doc.text)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)  # (num_docs, 768)
            
            # 全ペアのコサイン類似度を計算
            # cosine_similarityは(n, n)の行列を返す（対角成分は1.0）
            similarity_matrix = cosine_similarity(embeddings)
            
            # 上三角行列（対角成分を除く）から全ペアの類似度を取得
            pair_similarities = []
            pair_data = [] if return_all_pairs else None
            
            for i in range(num_docs):
                for j in range(i + 1, num_docs):
                    similarity = similarity_matrix[i][j]
                    pair_similarities.append(similarity)
                    
                    if return_all_pairs:
                        pair_data.append({
                                    'doc1_rank': documents[i].rank,
                                    'doc2_rank': documents[j].rank,
                                    'similarity': similarity
                        })
            
            # 統計値を計算
            mean_similarity = np.mean(pair_similarities)
            sum_similarity = np.sum(pair_similarities)
            num_pairs = len(pair_similarities)
            
            result = {
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'num_documents': num_docs,
                'mean_similarity': mean_similarity,
                'sum_similarity': sum_similarity,
                'num_pairs': num_pairs
            }
            
            if return_all_pairs and pair_data:
                result['pair_data'] = pair_data
            
            results.append(result)
        
        # キャッシュ統計を表示
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            hit_rate = cache_hits / total_requests * 100
            print(f"\nCache statistics: {cache_hits}/{total_requests} hits ({hit_rate:.1f}%)")
        
        return pd.DataFrame(results)
    
    @classmethod
    def compute_from_files(
        cls,
        dpr_json_path: str,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: Optional[str] = None,
        top_k: Optional[int] = None,
        device: Optional[str] = None
    ) -> None:
        """
        DPR結果ファイルから類似度統計を計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス（Noneの場合は自動生成）
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            device: 使用するデバイス（Noneの場合は自動選択）
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing document embeddings and similarity statistics...")
        # キャッシュディレクトリを設定（デフォルトは.embedding_cache）
        cache_dir = Path(".embedding_cache")
        analyzer = cls(turn_data_list, device=device, cache_dir=str(cache_dir))
        
        # 類似度統計を計算
        print("Computing similarity statistics and merging with base JSON...")
        similarity_stats = analyzer.compute(top_k=top_k)
        
        # CSV出力（output_csv_pathが指定されていない場合は自動生成）
        if output_csv_path is None:
            output_csv_path = Path(output_json_path).with_suffix('.csv')
        else:
            output_csv_path = Path(output_csv_path)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in similarity_stats.iterrows():
            # conv_idを文字列として保存（科学記数法を避けるため）
            conv_id = str(row['conv_id'])
            csv_data.append({
                'conv_id': conv_id,
                'turn_id': int(row['turn_id']),
                'mean_similarity': row['mean_similarity'] if not np.isnan(row['mean_similarity']) else None,
                'sum_similarity': row['sum_similarity'] if not np.isnan(row['sum_similarity']) else None,
                'num_similarity_pairs': int(row['num_pairs']),
                'num_retrieved_documents': int(row['num_documents'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        # conv_idを文字列として保存するため、dtypeを指定
        csv_df['conv_id'] = csv_df['conv_id'].astype(str)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        similarity_dict = {}
        for _, row in similarity_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            similarity_dict[key] = {
                'mean_similarity': float(row['mean_similarity']) if not np.isnan(row['mean_similarity']) else None,
                'sum_similarity': float(row['sum_similarity']) if not np.isnan(row['sum_similarity']) else None,
                'num_pairs': int(row['num_pairs']),
                'num_documents': int(row['num_documents'])
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンに類似度統計を追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])  # turn_idは数値なので文字列に変換
                key = (conv_id, turn_id)
                
                if key in similarity_dict:
                    stats = similarity_dict[key]
                    turn['mean_similarity'] = stats['mean_similarity']
                    turn['sum_similarity'] = stats['sum_similarity']
                    turn['num_similarity_pairs'] = stats['num_pairs']
                    turn['num_retrieved_documents'] = stats['num_documents']
                else:
                    # 見つからない場合はNoneを設定
                    turn['mean_similarity'] = None
                    turn['sum_similarity'] = None
                    turn['num_similarity_pairs'] = 0
                    turn['num_retrieved_documents'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        # キャッシュを保存
        analyzer.save_cache()
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

