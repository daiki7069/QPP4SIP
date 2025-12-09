"""
MaxSCQ (Maximum SCQ) 手法の実装
クエリに含まれる各語のSCQ（Similarity of Collection and Query）の最大値
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from tqdm import tqdm
import json
from pathlib import Path
import re
import pickle
import math

# pyseriniのAnalyzerを使用する場合（オプション）
try:
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False


class MaxSCQ:
    """MaxSCQ (Maximum SCQ) 計算クラス"""
    
    def __init__(
        self,
        query_data_list: List[Dict],
        document_freq: Optional[Dict[str, int]] = None,
        total_documents: Optional[int] = None,
        collection_freq: Optional[Dict[str, int]] = None,
        document_freq_cache_path: Optional[str] = None,
        collection_freq_cache_path: Optional[str] = None,
        use_pyserini: bool = False
    ):
        """
        Args:
            query_data_list: クエリデータのリスト（各要素は{'conv_id', 'turn_id', 'question'}を持つ）
            document_freq: 語を含む文書数の辞書（語 -> 文書数）
            total_documents: コレクション全体の文書数
            collection_freq: コレクション全体の語頻度（語 -> 出現回数）
            document_freq_cache_path: 文書頻度のキャッシュファイルパス（AvgIDFと同じキャッシュを使用）
            collection_freq_cache_path: コレクション語頻度のキャッシュファイルパス（AvgICTFと同じキャッシュを使用）
            use_pyserini: pyseriniのAnalyzerを使用するかどうか（デフォルト: False）
        """
        self.query_data_list = query_data_list
        self.document_freq = document_freq
        self.total_documents = total_documents
        self.collection_freq = collection_freq
        self.document_freq_cache_path = document_freq_cache_path
        self.collection_freq_cache_path = collection_freq_cache_path
        self.use_pyserini = use_pyserini and PYSERINI_AVAILABLE
        
        # pyseriniのAnalyzerを初期化
        if self.use_pyserini:
            try:
                # BM25インデックス作成時と同じAnalyzerを使用
                self.analyzer = Analyzer(get_lucene_analyzer())
                print("Using pyserini Analyzer for tokenization")
            except Exception as e:
                print(f"Warning: Failed to initialize pyserini Analyzer: {e}")
                self.use_pyserini = False
                self.analyzer = None
        else:
            self.analyzer = None
        
        # 文書頻度が提供されていない場合はキャッシュから読み込む
        if self.document_freq is None and document_freq_cache_path:
            cache_path = Path(document_freq_cache_path)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        self.document_freq = cache_data.get('document_freq', {})
                        self.total_documents = cache_data.get('total_documents', 0)
                    print(f"Loaded document frequency from cache: {cache_path}")
                except Exception as e:
                    print(f"Warning: Failed to load document frequency cache: {e}")
                    self.document_freq = {}
                    self.total_documents = 0
        
        # コレクション語頻度が提供されていない場合はキャッシュから読み込む
        if self.collection_freq is None and collection_freq_cache_path:
            cache_path = Path(collection_freq_cache_path)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        # 新しい形式（辞書+total_terms）または古い形式（辞書のみ）に対応
                        if isinstance(cache_data, dict) and 'collection_freq' in cache_data:
                            self.collection_freq = cache_data.get('collection_freq', {})
                        else:
                            # clarity.pyの形式（辞書のみ）
                            self.collection_freq = cache_data
                    print(f"Loaded collection frequency from cache: {cache_path}")
                except Exception as e:
                    print(f"Warning: Failed to load collection frequency cache: {e}")
                    self.collection_freq = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """
        テキストをトークン化（単語に分割）
        
        Args:
            text: トークン化するテキスト
        
        Returns:
            トークンのリスト
        """
        if self.use_pyserini and self.analyzer:
            # pyseriniのAnalyzerを使用（BM25インデックス作成時と同じ）
            tokens = self.analyzer.analyze(text)
            return [token for token in tokens]
        else:
            # 正規表現ベースのトークナイザー
            text = text.lower()
            # 英数字とアポストロフィのみを保持
            tokens = re.findall(r"\b[a-z0-9']+\b", text)
            return tokens
    
    def _compute_scq(self, term: str) -> float:
        """
        語のSCQを計算
        
        SCQ(t) = (1 + log tf(t, D)) * log(N / Nt)
        
        Args:
            term: 語
        
        Returns:
            SCQ値
        """
        if not self.document_freq or not self.total_documents or not self.collection_freq:
            return 0.0
        
        # 語を含む文書数
        Nt = self.document_freq.get(term, 0)
        
        # 語の出現回数
        tf = self.collection_freq.get(term, 0)
        
        if Nt == 0 or tf == 0:
            # 語がコレクションに存在しない場合
            return 0.0
        
        # IDF = log(N / Nt)
        idf = math.log(self.total_documents / Nt)
        
        # SCQ = (1 + log tf(t, D)) * log(N / Nt)
        scq = (1 + math.log(tf)) * idf
        
        return scq
    
    def compute(self) -> pd.DataFrame:
        """
        各クエリについて、MaxSCQを計算
        
        MaxSCQ(q) = max_{t in q} SCQ(t)
        
        Returns:
            DataFrame with columns: conv_id, turn_id, maxscq, num_terms
        """
        results = []
        
        for query_data in tqdm(self.query_data_list, desc="Computing MaxSCQ", unit="query"):
            question = query_data.get('question', query_data.get('query', ''))
            conv_id = query_data.get('conv_id', '')
            turn_id = query_data.get('turn_id', '')
            
            if not question:
                results.append({
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'maxscq': np.nan,
                    'num_terms': 0
                })
                continue
            
            # クエリをトークン化
            tokens = self._tokenize(question)
            
            if len(tokens) == 0:
                results.append({
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'maxscq': np.nan,
                    'num_terms': 0
                })
                continue
            
            # 各トークンのSCQを計算
            scq_values = []
            for token in tokens:
                scq = self._compute_scq(token)
                scq_values.append(scq)
            
            # MaxSCQ = 最大SCQ
            maxscq = np.max(scq_values) if scq_values else np.nan
            
            results.append({
                'conv_id': conv_id,
                'turn_id': turn_id,
                'maxscq': maxscq,
                'num_terms': len(tokens)
            })
        
        return pd.DataFrame(results)
    
    @classmethod
    def compute_from_files(
        cls,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: str,
        dataset: str = "INSCIT",
        document_freq_cache_path: Optional[str] = None,
        collection_freq_cache_path: Optional[str] = None,
        use_pyserini: bool = False
    ) -> None:
        """
        ベースJSONファイルからMaxSCQスコアを計算し、ベースJSONに追記して出力
        
        Args:
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス
            dataset: データセット名（"INSCIT" または "AmbigNQ"）
            document_freq_cache_path: 文書頻度のキャッシュファイルパス（AvgIDFと同じキャッシュを使用）
            collection_freq_cache_path: コレクション語頻度のキャッシュファイルパス（AvgICTFと同じキャッシュを使用）
            use_pyserini: pyseriniのAnalyzerを使用するかどうか
        """
        print(f"Loading queries from: {base_json_path}")
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # クエリデータを抽出
        query_data_list = []
        for conversation in base_data:
            for turn in conversation:
                query_data = {
                    'conv_id': turn.get('conv_id', ''),
                    'turn_id': str(turn.get('turn_id', '')),
                    'question': turn.get('query', turn.get('question', ''))
                }
                query_data_list.append(query_data)
        
        print(f"Loaded {len(query_data_list)} queries")
        
        # 文書頻度を計算または読み込む（AvgIDFと同じキャッシュを使用）
        if document_freq_cache_path is None:
            base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
            document_freq_cache_path = str(base_dir / "QPP" / "pre_retrieval" / ".document_freq_cache" / f"{dataset}_document_freq.pkl")
        
        from .avgidf import AvgIDF
        document_freq, total_documents = AvgIDF.compute_document_frequency(
            dataset=dataset,
            document_freq_cache_path=document_freq_cache_path
        )
        
        # コレクション語頻度を計算または読み込む（AvgICTFと同じキャッシュを使用）
        if collection_freq_cache_path is None:
            base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
            collection_freq_cache_path = str(base_dir / "QPP" / "pre_retrieval" / ".collection_freq_cache" / f"{dataset}_collection_freq.pkl")
        
        from .avgictf import AvgICTF
        collection_freq, _ = AvgICTF.compute_collection_frequency(
            dataset=dataset,
            collection_freq_cache_path=collection_freq_cache_path
        )
        
        print("Computing MaxSCQ scores...")
        analyzer = cls(
            query_data_list,
            document_freq=document_freq,
            total_documents=total_documents,
            collection_freq=collection_freq,
            document_freq_cache_path=document_freq_cache_path,
            collection_freq_cache_path=collection_freq_cache_path,
            use_pyserini=use_pyserini
        )
        
        # MaxSCQを計算
        maxscq_stats = analyzer.compute()
        
        # CSV出力
        csv_data = []
        for _, row in maxscq_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'maxscq': float(row['maxscq']) if not np.isnan(row['maxscq']) else None,
                'num_terms': int(row['num_terms'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        maxscq_dict = {}
        for _, row in maxscq_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            maxscq_dict[key] = {
                'maxscq': float(row['maxscq']) if not np.isnan(row['maxscq']) else None,
                'num_terms': int(row['num_terms'])
            }
        
        # 各ターンにMaxSCQを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in maxscq_dict:
                    stats = maxscq_dict[key]
                    turn['maxscq'] = stats['maxscq']
                    turn['num_terms'] = stats['num_terms']
                else:
                    turn['maxscq'] = None
                    turn['num_terms'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

