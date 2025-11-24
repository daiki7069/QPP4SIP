"""
Coherency (ACC/WACC) 手法の実装
文書間ネットワークを構築し、クラスタリング係数を計算
"""
import pandas as pd
import numpy as np
from typing import List, Optional
from tqdm import tqdm
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class Coherency(BaseQPPAnalyzer):
    """Coherency (ACC/WACC) 計算クラス"""
    
    def _normalize_similarity(self, similarity: float, min_val: float, max_val: float) -> float:
        """
        類似度を0~1の範囲に正規化
        
        Args:
            similarity: 正規化する類似度
            min_val: 最小値
            max_val: 最大値
        
        Returns:
            正規化された類似度（0~1）
        """
        if max_val == min_val:
            return 1.0
        return (similarity - min_val) / (max_val - min_val)
    
    def _build_network(
        self,
        similarity_matrix: np.ndarray,
        mean_similarity: float,
        use_weighted: bool = True
    ) -> nx.Graph:
        """
        類似度行列からネットワークを構築
        
        Args:
            similarity_matrix: 類似度行列 (n, n)
            mean_similarity: 平均類似度（この値より高い類似度のペアにエッジを張る）
            use_weighted: 重み付きエッジを使用するかどうか
        
        Returns:
            NetworkXグラフオブジェクト
        """
        n = similarity_matrix.shape[0]
        G = nx.Graph()
        
        # ノードを追加
        for i in range(n):
            G.add_node(i)
        
        # エッジを追加（平均より高い類似度のペアのみ）
        if use_weighted:
            # 重み付きエッジの場合、類似度を0~1に正規化
            # 類似度行列の最小値と最大値を取得（対角成分を除く）
            triu_indices = np.triu_indices(n, k=1)
            similarities = similarity_matrix[triu_indices]
            min_sim = np.min(similarities)
            max_sim = np.max(similarities)
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = similarity_matrix[i][j]
                if similarity > mean_similarity:
                    if use_weighted:
                        # 0~1に正規化した値を重みとする
                        weight = self._normalize_similarity(similarity, min_sim, max_sim)
                        G.add_edge(i, j, weight=weight)
                    else:
                        # 重みなしエッジ
                        G.add_edge(i, j)
        
        return G
    
    def _compute_clustering_coefficients(self, G: nx.Graph, use_weighted: bool = True) -> tuple:
        """
        クラスタリング係数を計算
        
        Args:
            G: NetworkXグラフオブジェクト
            use_weighted: 重み付きクラスタリング係数を計算するかどうか
        
        Returns:
            (ACC, WACC) のタプル
        """
        if G.number_of_nodes() == 0:
            return (np.nan, np.nan)
        
        if G.number_of_edges() == 0:
            # エッジがない場合はクラスタリング係数は0
            return (0.0, 0.0)
        
        # ACC: 通常のクラスタリング係数の平均
        # NetworkXのclustering関数は各ノードのクラスタリング係数を返す
        clustering_dict = nx.clustering(G)
        if len(clustering_dict) == 0:
            acc = 0.0
        else:
            acc = np.mean(list(clustering_dict.values()))
        
        # WACC: 重み付きクラスタリング係数の平均
        if use_weighted and nx.is_weighted(G):
            # 重み付きクラスタリング係数を計算
            weighted_clustering_dict = nx.clustering(G, weight='weight')
            if len(weighted_clustering_dict) == 0:
                wacc = 0.0
            else:
                wacc = np.mean(list(weighted_clustering_dict.values()))
        else:
            # 重みがない場合はACCと同じ
            wacc = acc
        
        return (acc, wacc)
    
    def compute(
        self,
        top_k: Optional[int] = None,
        top_t: Optional[int] = None,
        use_weighted: bool = True
    ) -> pd.DataFrame:
        """
        各ターンについて、文書間ネットワークを構築し、ACC/WACCを計算
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            top_t: ネットワーク構築の対象とする上位t文書（Noneの場合はtop_kと同じ）
            use_weighted: 重み付きエッジを使用するかどうか
        
        Returns:
            DataFrame with columns: conv_id, turn_id, num_documents, 
                                    mean_similarity, acc, wacc, num_edges, num_nodes
        """
        results = []
        total_turns = len(self.turn_data_list)
        cache_hits = 0
        cache_misses = 0
        
        # top_tが指定されていない場合はtop_kと同じにする
        if top_t is None:
            top_t = top_k
        
        for idx, turn in enumerate(tqdm(self.turn_data_list, desc="Processing turns", unit="turn"), 1):
            documents = self._get_documents(turn, top_k=top_k)
            
            if len(documents) < 2:
                # 文書が2件未満の場合は計算できない
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'num_documents': len(documents),
                    'mean_similarity': np.nan,
                    'acc': np.nan,
                    'wacc': np.nan,
                    'num_edges': 0,
                    'num_nodes': len(documents)
                })
                continue
            
            # ネットワーク構築の対象とする文書数（上位t件）
            num_docs_for_network = len(documents) if top_t is None else min(top_t, len(documents))
            documents_for_network = documents[:num_docs_for_network]
            
            # 各文書の埋め込みを計算
            embeddings = []
            for doc in tqdm(documents_for_network, desc=f"  Encoding docs (turn {idx}/{total_turns})", leave=False, unit="doc"):
                text_hash = self._get_text_hash(doc.text)
                was_cached = text_hash in self.embedding_cache
                if was_cached:
                    cache_hits += 1
                else:
                    cache_misses += 1
                embedding = self._encode_document(doc.text)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)  # (num_docs_for_network, 768)
            
            # 全ペアのコサイン類似度を計算
            similarity_matrix = cosine_similarity(embeddings)
            
            # 全ペアの類似度の平均を計算（対角成分を除く）
            triu_indices = np.triu_indices(num_docs_for_network, k=1)
            pair_similarities = similarity_matrix[triu_indices]
            mean_similarity = np.mean(pair_similarities)
            
            # ネットワークを構築
            G = self._build_network(similarity_matrix, mean_similarity, use_weighted=use_weighted)
            
            # クラスタリング係数を計算
            acc, wacc = self._compute_clustering_coefficients(G, use_weighted=use_weighted)
            
            result = {
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'num_documents': len(documents),
                'num_documents_for_network': num_docs_for_network,
                'mean_similarity': mean_similarity,
                'acc': acc,
                'wacc': wacc,
                'num_edges': G.number_of_edges(),
                'num_nodes': G.number_of_nodes()
            }
            
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
        top_t: Optional[int] = None,
        use_weighted: bool = True,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        DPR結果ファイルからCoherency (ACC/WACC)を計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス（Noneの場合は自動生成）
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            top_t: ネットワーク構築の対象とする上位t文書（Noneの場合はtop_kと同じ）
            use_weighted: 重み付きエッジを使用するかどうか
            device: 使用するデバイス（Noneの場合は自動選択）
            cache_dir: キャッシュディレクトリのパス（Noneの場合はキャッシュを使用しない）
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing document embeddings and coherency metrics...")
        # キャッシュディレクトリを設定
        if cache_dir is None:
            cache_dir = Path(".embedding_cache")
        else:
            cache_dir = Path(cache_dir)
        analyzer = cls(turn_data_list, device=device, cache_dir=str(cache_dir))
        
        # Coherencyを計算
        print("Computing coherency metrics and merging with base JSON...")
        coherency_stats = analyzer.compute(top_k=top_k, top_t=top_t, use_weighted=use_weighted)
        
        # CSV出力（output_csv_pathが指定されていない場合は自動生成）
        if output_csv_path is None:
            output_csv_path = Path(output_json_path).with_suffix('.csv')
        else:
            output_csv_path = Path(output_csv_path)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in coherency_stats.iterrows():
            # conv_idを文字列として保存（科学記数法を避けるため）
            conv_id = str(row['conv_id'])
            csv_data.append({
                'conv_id': conv_id,
                'turn_id': int(row['turn_id']),
                'acc': row['acc'] if not np.isnan(row['acc']) else None,
                'wacc': row['wacc'] if not np.isnan(row['wacc']) else None,
                'mean_similarity': row['mean_similarity'] if not np.isnan(row['mean_similarity']) else None,
                'num_edges': int(row['num_edges']),
                'num_nodes': int(row['num_nodes']),
                'num_documents': int(row['num_documents']),
                'num_documents_for_network': int(row['num_documents_for_network'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        # conv_idを文字列として保存するため、dtypeを指定
        csv_df['conv_id'] = csv_df['conv_id'].astype(str)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        coherency_dict = {}
        for _, row in coherency_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            coherency_dict[key] = {
                'acc': float(row['acc']) if not np.isnan(row['acc']) else None,
                'wacc': float(row['wacc']) if not np.isnan(row['wacc']) else None,
                'mean_similarity': float(row['mean_similarity']) if not np.isnan(row['mean_similarity']) else None,
                'num_edges': int(row['num_edges']),
                'num_nodes': int(row['num_nodes']),
                'num_documents': int(row['num_documents']),
                'num_documents_for_network': int(row['num_documents_for_network'])
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンにCoherency統計を追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])  # turn_idは数値なので文字列に変換
                key = (conv_id, turn_id)
                
                if key in coherency_dict:
                    stats = coherency_dict[key]
                    turn['acc'] = stats['acc']
                    turn['wacc'] = stats['wacc']
                    turn['mean_similarity'] = stats['mean_similarity']
                    turn['num_edges'] = stats['num_edges']
                    turn['num_nodes'] = stats['num_nodes']
                    turn['num_documents'] = stats['num_documents']
                    turn['num_documents_for_network'] = stats['num_documents_for_network']
                else:
                    # 見つからない場合はNoneを設定
                    turn['acc'] = None
                    turn['wacc'] = None
                    turn['mean_similarity'] = None
                    turn['num_edges'] = 0
                    turn['num_nodes'] = 0
                    turn['num_documents'] = 0
                    turn['num_documents_for_network'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        # キャッシュを保存
        analyzer.save_cache()
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

