"""
AvgICTF (Average ICTF) 手法の実装
クエリに含まれる各語のICTF（逆コレクション頻度）を平均したもの
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
import gzip
import sys
import math

# pyseriniのAnalyzerを使用する場合（オプション）
try:
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False


class AvgICTF:
    """AvgICTF (Average ICTF) 計算クラス"""
    
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
            collection_freq_cache_path: コレクション語頻度のキャッシュファイルパス
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
                        # clarity.pyと同じ形式（辞書）または新しい形式（辞書+total_terms）に対応
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
            # 正規表現ベースのトークナイザー（post_retrieval/clarity.pyと同じ）
            text = text.lower()
            # 英数字とアポストロフィのみを保持
            tokens = re.findall(r"\b[a-z0-9']+\b", text)
            return tokens
    
    def _compute_ictf(self, term: str) -> float:
        """
        語のICTFを計算
        
        ICTF(t) = log(|D| / tf(t, D))
        
        Args:
            term: 語
        
        Returns:
            ICTF値
        """
        if not self.collection_freq or not self.total_terms:
            return 0.0
        
        # 語の出現回数
        tf = self.collection_freq.get(term, 0)
        
        if tf == 0:
            # 語がコレクションに存在しない場合
            return 0.0
        
        # ICTF = log(|D| / tf(t, D))
        ictf = math.log(self.total_terms / tf)
        return ictf
    
    def compute(self) -> pd.DataFrame:
        """
        各クエリについて、AvgICTFを計算
        
        AvgICTF(q) = (1/m) * Σ ICTF(t_i)
        
        Returns:
            DataFrame with columns: conv_id, turn_id, avgictf, num_terms
        """
        results = []
        
        for query_data in tqdm(self.query_data_list, desc="Computing AvgICTF", unit="query"):
            question = query_data.get('question', query_data.get('query', ''))
            conv_id = query_data.get('conv_id', '')
            turn_id = query_data.get('turn_id', '')
            
            if not question:
                results.append({
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'avgictf': np.nan,
                    'num_terms': 0
                })
                continue
            
            # クエリをトークン化
            tokens = self._tokenize(question)
            
            if len(tokens) == 0:
                results.append({
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'avgictf': np.nan,
                    'num_terms': 0
                })
                continue
            
            # 各トークンのICTFを計算
            ictf_values = []
            for token in tokens:
                ictf = self._compute_ictf(token)
                ictf_values.append(ictf)
            
            # AvgICTF = 平均ICTF
            avgictf = np.mean(ictf_values) if ictf_values else np.nan
            
            results.append({
                'conv_id': conv_id,
                'turn_id': turn_id,
                'avgictf': avgictf,
                'num_terms': len(tokens)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compute_collection_frequency(
        dataset: str,
        collection_freq_cache_path: Optional[str] = None,
        max_docs: Optional[int] = None
    ) -> tuple:
        """
        コレクション全体の語頻度（出現回数）を計算
        
        Args:
            dataset: データセット名（"INSCIT" または "AmbigNQ"）
            collection_freq_cache_path: キャッシュファイルのパス
            max_docs: 最大処理文書数（Noneの場合は全件）
        
        Returns:
            (collection_freq, total_terms) のタプル
            - collection_freq: 語の出現回数の辞書（語 -> 出現回数）
            - total_terms: コレクション全体の総語数
        """
        # キャッシュから読み込む
        if collection_freq_cache_path:
            cache_path = Path(collection_freq_cache_path)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        # 新しい形式（辞書+total_terms）または古い形式（辞書のみ）に対応
                        if isinstance(cache_data, dict) and 'collection_freq' in cache_data:
                            collection_freq = cache_data.get('collection_freq', {})
                            total_terms = cache_data.get('total_terms', sum(collection_freq.values()))
                        else:
                            # clarity.pyの形式（辞書のみ）
                            collection_freq = cache_data
                            total_terms = sum(cache_data.values()) if cache_data else 0
                    print(f"Loaded collection frequency from cache: {cache_path}")
                    return collection_freq, total_terms
                except Exception as e:
                    print(f"Warning: Failed to load cache: {e}")
        
        print(f"Computing collection frequency for {dataset}...")
        collection_freq = Counter()
        
        if dataset == "INSCIT":
            # INSCIT: JSON形式のファイルを読み込む
            collection_path = Path("/mnt/nas_syno/daiki/Datasets/INSCIT/data/text_0420_processed")
            if not collection_path.exists():
                # 代替パスを試す
                collection_path = Path("/mnt/nas_syno/daiki/Datasets/INSCIT/data/corpus.zip")
                if collection_path.exists():
                    import zipfile
                    with zipfile.ZipFile(collection_path, 'r') as zip_ref:
                        # zip内のファイルを処理
                        file_list = [f for f in zip_ref.namelist() if f.endswith('.json')]
                        for file_name in tqdm(
                            file_list[:max_docs] if max_docs else file_list,
                            desc="Processing files",
                            file=sys.stderr,
                            mininterval=1.0
                        ):
                            with zip_ref.open(file_name) as f:
                                data = json.load(f)
                                for doc in data:
                                    # INSCITのJSONファイルはpassagesフィールドを持つ
                                    if 'passages' in doc and isinstance(doc['passages'], list):
                                        # passages内の各passageからtextを取得
                                        for passage in doc['passages']:
                                            if 'text' in passage and passage['text']:
                                                text = passage['text']
                                                tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                                collection_freq.update(tokens)  # 出現回数をカウント
                                    elif 'text' in doc:
                                        # 直接textフィールドがある場合（他の形式）
                                        text = doc['text']
                                        tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                        collection_freq.update(tokens)  # 出現回数をカウント
                    # 総語数を計算
                    total_terms = sum(collection_freq.values())
                    if collection_freq_cache_path:
                        cache_path = Path(collection_freq_cache_path)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, 'wb') as f:
                            pickle.dump({
                                'collection_freq': dict(collection_freq),
                                'total_terms': total_terms
                            }, f)
                        print(f"Saved collection frequency to cache: {cache_path}")
                    return dict(collection_freq), total_terms
            
            # JSONファイルを処理
            json_files = list(collection_path.glob("*.json"))
            if max_docs:
                json_files = json_files[:max_docs]
            
            for json_file in tqdm(
                json_files,
                desc="Processing files",
                file=sys.stderr,
                mininterval=1.0
            ):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        for doc in data:
                            # INSCITのJSONファイルはpassagesフィールドを持つ
                            if 'passages' in doc and isinstance(doc['passages'], list):
                                # passages内の各passageからtextを取得
                                for passage in doc['passages']:
                                    if 'text' in passage and passage['text']:
                                        text = passage['text']
                                        tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                        collection_freq.update(tokens)  # 出現回数をカウント
                            elif 'text' in doc:
                                # 直接textフィールドがある場合（他の形式）
                                text = doc['text']
                                tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                collection_freq.update(tokens)  # 出現回数をカウント
                except Exception as e:
                    print(f"Warning: Failed to process {json_file}: {e}")
                    continue
        
        elif dataset == "AmbigNQ":
            # AmbigNQ: TSV形式のファイルを読み込む
            collection_path = Path("/mnt/nas_syno/daiki/Datasets/AmbigQA/codes/data/wikipedia_split/psgs_w100.tsv.gz")
            if not collection_path.exists():
                raise FileNotFoundError(f"Collection file not found: {collection_path}")
            
            with gzip.open(collection_path, 'rt', encoding='utf-8') as f:
                # ヘッダーをスキップ
                next(f)
                for line in tqdm(
                    f,
                    desc="Processing documents",
                    file=sys.stderr,
                    mininterval=1.0
                ):
                    if max_docs and len(collection_freq) > 0 and sum(collection_freq.values()) >= max_docs * 1000:  # 大まかな制限
                        break
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        text = parts[1]  # text列
                        tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                        collection_freq.update(tokens)  # 出現回数をカウント
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # 総語数を計算
        total_terms = sum(collection_freq.values())
        
        # キャッシュに保存
        if collection_freq_cache_path:
            cache_path = Path(collection_freq_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'collection_freq': dict(collection_freq),
                    'total_terms': total_terms
                }, f)
            print(f"Saved collection frequency to cache: {cache_path}")
        
        print(f"Computed collection frequency: {len(collection_freq)} unique terms, {total_terms} total terms")
        return dict(collection_freq), total_terms
    
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
        ベースJSONファイルからAvgICTFスコアを計算し、ベースJSONに追記して出力
        
        Args:
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス
            dataset: データセット名（"INSCIT" または "AmbigNQ"）
            collection_freq_cache_path: コレクション語頻度のキャッシュファイルパス
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
        
        # コレクション語頻度を計算または読み込む
        if collection_freq_cache_path is None:
            base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
            collection_freq_cache_path = str(base_dir / "QPP" / "pre_retrieval" / ".collection_freq_cache" / f"{dataset}_collection_freq.pkl")
        
        collection_freq, total_terms = cls.compute_collection_frequency(
            dataset=dataset,
            collection_freq_cache_path=collection_freq_cache_path
        )
        
        print("Computing AvgICTF scores...")
        analyzer = cls(
            query_data_list,
            collection_freq=collection_freq,
            total_terms=total_terms,
            collection_freq_cache_path=collection_freq_cache_path,
            use_pyserini=use_pyserini
        )
        
        # AvgICTFを計算
        avgictf_stats = analyzer.compute()
        
        # CSV出力
        csv_data = []
        for _, row in avgictf_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'avgictf': float(row['avgictf']) if not np.isnan(row['avgictf']) else None,
                'num_terms': int(row['num_terms'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        avgictf_dict = {}
        for _, row in avgictf_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            avgictf_dict[key] = {
                'avgictf': float(row['avgictf']) if not np.isnan(row['avgictf']) else None,
                'num_terms': int(row['num_terms'])
            }
        
        # 各ターンにAvgICTFを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in avgictf_dict:
                    stats = avgictf_dict[key]
                    turn['avgictf'] = stats['avgictf']
                    turn['num_terms'] = stats['num_terms']
                else:
                    turn['avgictf'] = None
                    turn['num_terms'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

