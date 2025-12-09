"""
Simplified Clarity (SCS) 手法の実装
クエリ言語モデルとコレクション言語モデルのKLダイバージェンスの簡略版
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from tqdm import tqdm
import json
from pathlib import Path
import re
from collections import Counter
import pickle
import math

# pyseriniのAnalyzerを使用する場合（オプション）
try:
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False


class SimplifiedClarity:
    """Simplified Clarity (SCS) 計算クラス"""
    
    def __init__(
        self,
        query_data_list: List[Dict],
        collection_freq: Optional[Dict[str, int]] = None,
        total_terms: Optional[int] = None,
        collection_freq_cache_path: Optional[str] = None,
        use_pyserini: bool = False
    ):
        """
        Args:
            query_data_list: クエリデータのリスト（各要素は{'conv_id', 'turn_id', 'question'}を持つ）
            collection_freq: コレクション全体の語頻度（語 -> 出現回数）
            total_terms: コレクション全体の総語数
            collection_freq_cache_path: コレクション語頻度のキャッシュファイルパス（AvgICTFと同じキャッシュを使用）
            use_pyserini: pyseriniのAnalyzerを使用するかどうか（デフォルト: False）
        """
        self.query_data_list = query_data_list
        self.collection_freq = collection_freq
        self.total_terms = total_terms
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
                            self.total_terms = cache_data.get('total_terms', sum(cache_data.get('collection_freq', {}).values()))
                        else:
                            # clarity.pyの形式（辞書のみ）
                            self.collection_freq = cache_data
                            self.total_terms = sum(cache_data.values()) if cache_data else 0
                    print(f"Loaded collection frequency from cache: {cache_path}")
                except Exception as e:
                    print(f"Warning: Failed to load collection frequency cache: {e}")
                    self.collection_freq = {}
                    self.total_terms = 0
    
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
    
    def _compute_scs(self, query_tokens: List[str]) -> float:
        """
        クエリのSimplified Clarity Score (SCS)を計算
        
        SCS(q) = Σ_{t in q} P(t|q) * log(P(t|q) / P(t|D))
        
        ここで：
        - P(t|q) = tf(t,q) / |q| （クエリ内での語の出現割合）
        - P(t|D) = tf(t,D) / |D| （コレクション全体での語の出現割合）
        
        Args:
            query_tokens: クエリのトークンリスト
        
        Returns:
            SCS値
        """
        if not self.collection_freq or not self.total_terms or len(query_tokens) == 0:
            return 0.0
        
        # クエリ内の語頻度を計算
        query_term_freq = Counter(query_tokens)
        query_length = len(query_tokens)
        
        # SCS = Σ_{t in q} P(t|q) * log(P(t|q) / P(t|D))
        scs = 0.0
        
        for term, tf_query in query_term_freq.items():
            # P(t|q) = tf(t,q) / |q|
            p_t_given_q = tf_query / query_length
            
            # P(t|D) = tf(t,D) / |D|
            tf_collection = self.collection_freq.get(term, 0)
            if tf_collection == 0:
                # コレクションに存在しない語はスキップ（または0として扱う）
                continue
            
            p_t_given_d = tf_collection / self.total_terms
            
            # log(P(t|q) / P(t|D)) を計算
            if p_t_given_d > 0:
                log_ratio = math.log(p_t_given_q / p_t_given_d)
                scs += p_t_given_q * log_ratio
        
        return scs
    
    def compute(self) -> pd.DataFrame:
        """
        各クエリについて、Simplified Clarity Score (SCS)を計算
        
        Returns:
            DataFrame with columns: conv_id, turn_id, simplified_clarity, num_terms
        """
        results = []
        
        for query_data in tqdm(self.query_data_list, desc="Computing Simplified Clarity", unit="query"):
            question = query_data.get('question', query_data.get('query', ''))
            conv_id = query_data.get('conv_id', '')
            turn_id = query_data.get('turn_id', '')
            
            if not question:
                results.append({
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'simplified_clarity': np.nan,
                    'num_terms': 0
                })
                continue
            
            # クエリをトークン化
            tokens = self._tokenize(question)
            
            if len(tokens) == 0:
                results.append({
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'simplified_clarity': np.nan,
                    'num_terms': 0
                })
                continue
            
            # SCSを計算
            scs = self._compute_scs(tokens)
            
            results.append({
                'conv_id': conv_id,
                'turn_id': turn_id,
                'simplified_clarity': scs,
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
        collection_freq_cache_path: Optional[str] = None,
        use_pyserini: bool = False
    ) -> None:
        """
        ベースJSONファイルからSimplified Clarity Score (SCS)を計算し、ベースJSONに追記して出力
        
        Args:
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス
            dataset: データセット名（"INSCIT" または "AmbigNQ"）
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
        
        # コレクション語頻度を計算または読み込む（AvgICTFと同じキャッシュを使用）
        if collection_freq_cache_path is None:
            base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
            collection_freq_cache_path = str(base_dir / "QPP" / "pre_retrieval" / ".collection_freq_cache" / f"{dataset}_collection_freq.pkl")
        
        from .avgictf import AvgICTF
        collection_freq, total_terms = AvgICTF.compute_collection_frequency(
            dataset=dataset,
            collection_freq_cache_path=collection_freq_cache_path
        )
        
        print("Computing Simplified Clarity scores...")
        analyzer = cls(
            query_data_list,
            collection_freq=collection_freq,
            total_terms=total_terms,
            collection_freq_cache_path=collection_freq_cache_path,
            use_pyserini=use_pyserini
        )
        
        # Simplified Clarityを計算
        scs_stats = analyzer.compute()
        
        # CSV出力
        csv_data = []
        for _, row in scs_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'simplified_clarity': float(row['simplified_clarity']) if not np.isnan(row['simplified_clarity']) else None,
                'num_terms': int(row['num_terms'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        scs_dict = {}
        for _, row in scs_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            scs_dict[key] = {
                'simplified_clarity': float(row['simplified_clarity']) if not np.isnan(row['simplified_clarity']) else None,
                'num_terms': int(row['num_terms'])
            }
        
        # 各ターンにSimplified Clarityを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in scs_dict:
                    stats = scs_dict[key]
                    turn['simplified_clarity'] = stats['simplified_clarity']
                    turn['num_terms'] = stats['num_terms']
                else:
                    turn['simplified_clarity'] = None
                    turn['num_terms'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

