"""
AvgIDF (Average IDF) 手法の実装
クエリに含まれる各語のIDF（逆文書頻度）を平均したもの
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from tqdm import tqdm
import json
from pathlib import Path
import re
from collections import Counter, defaultdict
import pickle
import gzip
import sys
import math
import tempfile
import shutil

# pyseriniのAnalyzerを使用する場合（オプション）
try:
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    PYSERINI_AVAILABLE = True
except ImportError:
    PYSERINI_AVAILABLE = False


class AvgIDF:
    """AvgIDF (Average IDF) 計算クラス"""
    
    def __init__(
        self,
        query_data_list: List[Dict],
        document_freq: Optional[Dict[str, int]] = None,
        total_documents: Optional[int] = None,
        document_freq_cache_path: Optional[str] = None,
        use_pyserini: bool = False
    ):
        """
        Args:
            query_data_list: クエリデータのリスト（各要素は{'conv_id', 'turn_id', 'question'}を持つ）
            document_freq: 語を含む文書数の辞書（語 -> 文書数）
            total_documents: コレクション全体の文書数
            document_freq_cache_path: 文書頻度のキャッシュファイルパス
            use_pyserini: pyseriniのAnalyzerを使用するかどうか（デフォルト: False）
        """
        self.query_data_list = query_data_list
        self.document_freq = document_freq
        self.total_documents = total_documents
        self.document_freq_cache_path = document_freq_cache_path
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
    
    def _compute_idf(self, term: str) -> float:
        """
        語のIDFを計算
        
        IDF(t) = log(N / Nt)
        
        Args:
            term: 語
        
        Returns:
            IDF値
        """
        if not self.document_freq or not self.total_documents:
            return 0.0
        
        # 語を含む文書数
        Nt = self.document_freq.get(term, 0)
        
        if Nt == 0:
            # 語がコレクションに存在しない場合、大きなIDF値を返す
            # または0を返す（実装によって異なる）
            return 0.0
        
        # IDF = log(N / Nt)
        idf = math.log(self.total_documents / Nt)
        return idf
    
    def compute(self) -> pd.DataFrame:
        """
        各クエリについて、AvgIDFを計算
        
        AvgIDF(q) = (1/m) * Σ IDF(t_i)
        
        Returns:
            DataFrame with columns: conv_id, turn_id, avgidf, num_terms
        """
        results = []
        
        for query_data in tqdm(self.query_data_list, desc="Computing AvgIDF", unit="query"):
            question = query_data.get('question', query_data.get('query', ''))
            conv_id = query_data.get('conv_id', '')
            turn_id = query_data.get('turn_id', '')
            
            if not question:
                results.append({
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'avgidf': np.nan,
                    'num_terms': 0
                })
                continue
            
            # クエリをトークン化
            tokens = self._tokenize(question)
            
            if len(tokens) == 0:
                results.append({
                    'conv_id': conv_id,
                    'turn_id': turn_id,
                    'avgidf': np.nan,
                    'num_terms': 0
                })
                continue
            
            # 各トークンのIDFを計算
            idf_values = []
            for token in tokens:
                idf = self._compute_idf(token)
                idf_values.append(idf)
            
            # AvgIDF = 平均IDF
            avgidf = np.mean(idf_values) if idf_values else np.nan
            
            results.append({
                'conv_id': conv_id,
                'turn_id': turn_id,
                'avgidf': avgidf,
                'num_terms': len(tokens)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compute_document_frequency(
        dataset: str,
        document_freq_cache_path: Optional[str] = None,
        max_docs: Optional[int] = None,
        batch_size: int = 100000
    ) -> tuple:
        """
        コレクション全体の文書頻度（DF）を計算
        
        Args:
            dataset: データセット名（"INSCIT" または "AmbigNQ"）
            document_freq_cache_path: キャッシュファイルのパス
            max_docs: 最大処理文書数（Noneの場合は全件）
            batch_size: バッチ処理のサイズ（AmbigNQの場合、この数の文書を処理したら中間結果をマージ）
        
        Returns:
            (document_freq, total_documents) のタプル
            - document_freq: 語を含む文書数の辞書（語 -> 文書数）
            - total_documents: コレクション全体の文書数
        """
        # キャッシュから読み込む
        if document_freq_cache_path:
            cache_path = Path(document_freq_cache_path)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        document_freq = cache_data.get('document_freq', {})
                        total_documents = cache_data.get('total_documents', 0)
                    print(f"Loaded document frequency from cache: {cache_path}")
                    return document_freq, total_documents
                except Exception as e:
                    print(f"Warning: Failed to load cache: {e}")
        
        print(f"Computing document frequency for {dataset}...")
        # 各語が含まれる文書の集合を保持
        # メモリ効率のため、AmbigNQではバッチ処理を使用
        # バッチごとにディスクにキャッシュし、最後にマージする方式
        if dataset == "AmbigNQ":
            # AmbigNQの場合：バッチごとにディスクにキャッシュ
            # 一時ディレクトリを作成
            temp_dir = Path(tempfile.mkdtemp(prefix="avgidf_batch_"))
            batch_cache_files: List[Path] = []
            print(f"Using temporary directory for batch caches: {temp_dir}", file=sys.stderr)
        else:
            # INSCITの場合：従来通りsetを保持
            term_documents: Dict[str, set] = defaultdict(set)
        total_documents = 0
        doc_id_mapping: Dict[str, int] = {}  # 文書ID文字列 -> 整数のマッピング（AmbigNQ用）
        next_doc_id_int = 0  # 次の整数ID（AmbigNQ用）
        
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
                                    doc_id = f"{file_name}_{total_documents}"
                                    # INSCITのJSONファイルはpassagesフィールドを持つ
                                    if 'passages' in doc and isinstance(doc['passages'], list):
                                        # passages内の各passageからtextを取得
                                        for passage in doc['passages']:
                                            if 'text' in passage and passage['text']:
                                                text = passage['text']
                                                tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                                for token in set(tokens):  # 文書内のユニークな語のみ
                                                    term_documents[token].add(doc_id)
                                    elif 'text' in doc:
                                        # 直接textフィールドがある場合（他の形式）
                                        text = doc['text']
                                        tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                        for token in set(tokens):  # 文書内のユニークな語のみ
                                            term_documents[token].add(doc_id)
                                    total_documents += 1
                    # 文書頻度に変換（集合のサイズ）
                    document_freq = {term: len(docs) for term, docs in term_documents.items()}
                    if document_freq_cache_path:
                        cache_path = Path(document_freq_cache_path)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, 'wb') as f:
                            pickle.dump({
                                'document_freq': document_freq,
                                'total_documents': total_documents
                            }, f)
                        print(f"Saved document frequency to cache: {cache_path}")
                    return document_freq, total_documents
            
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
                            doc_id = f"{json_file.name}_{total_documents}"
                            # INSCITのJSONファイルはpassagesフィールドを持つ
                            if 'passages' in doc and isinstance(doc['passages'], list):
                                # passages内の各passageからtextを取得
                                for passage in doc['passages']:
                                    if 'text' in passage and passage['text']:
                                        text = passage['text']
                                        tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                        for token in set(tokens):  # 文書内のユニークな語のみ
                                            term_documents[token].add(doc_id)
                            elif 'text' in doc:
                                # 直接textフィールドがある場合（他の形式）
                                text = doc['text']
                                tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                for token in set(tokens):  # 文書内のユニークな語のみ
                                    term_documents[token].add(doc_id)
                            total_documents += 1
                except Exception as e:
                    print(f"Warning: Failed to process {json_file}: {e}")
                    continue
        
        elif dataset == "AmbigNQ":
            # AmbigNQ: TSV形式のファイルを読み込む
            collection_path = Path("/mnt/nas_syno/daiki/Datasets/AmbigQA/codes/data/wikipedia_split/psgs_w100.tsv.gz")
            if not collection_path.exists():
                raise FileNotFoundError(f"Collection file not found: {collection_path}")
            
            # ファイルサイズを取得（進捗表示のため）
            file_size = collection_path.stat().st_size
            print(f"File size: {file_size / (1024**3):.2f} GB", file=sys.stderr)
            
            # AmbigNQのコレクションサイズは約2100万〜2500万文書
            # より正確な進捗表示のため、推定総文書数を使用
            estimated_total_docs = 21000000  # デフォルトは2100万
            # ファイルサイズから推定（1文書あたり約200バイトと仮定、gzip圧縮比3倍を考慮）
            size_based_estimate = int((file_size * 3) / 200)
            if 20000000 <= size_based_estimate <= 30000000:
                estimated_total_docs = size_based_estimate
            print(f"Estimated total documents: {estimated_total_docs:,}", file=sys.stderr)
            
            # メモリ効率のため、バッチ処理を実装
            # 一定数の文書を処理したら、中間結果をマージしてメモリを解放
            batch_term_documents: Dict[str, set] = defaultdict(set)
            batch_doc_id_mapping: Dict[str, int] = {}
            batch_next_doc_id_int = 0
            batch_doc_count = 0
            
            # 進捗表示のためのカウンター
            bytes_read = 0
            
            with gzip.open(collection_path, 'rt', encoding='utf-8') as f:
                # ヘッダーをスキップ
                header_line = next(f)
                bytes_read += len(header_line.encode('utf-8'))
                
                # tqdmプログレスバー（推定総文書数を使用）
                pbar = tqdm(
                    desc="Processing documents",
                    total=estimated_total_docs,
                    unit="docs",
                    file=sys.stderr,
                    mininterval=1.0,
                    dynamic_ncols=True,
                    unit_scale=True
                )
                
                try:
                    for line in f:
                        # 進捗更新（読み込んだバイト数）
                        line_bytes = len(line.encode('utf-8'))
                        bytes_read += line_bytes
                        
                        if max_docs and total_documents >= max_docs:
                            break
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            doc_id_str = parts[0]  # 文書ID（文字列）
                            text = parts[1]  # text列
                            
                            # バッチ内で文書IDを整数にマッピング（メモリ節約）
                            if doc_id_str not in batch_doc_id_mapping:
                                batch_doc_id_mapping[doc_id_str] = batch_next_doc_id_int
                                batch_next_doc_id_int += 1
                            doc_id_int = batch_doc_id_mapping[doc_id_str]
                            
                            tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                            for token in set(tokens):  # 文書内のユニークな語のみ
                                batch_term_documents[token].add(doc_id_int)
                            
                            batch_doc_count += 1
                            total_documents += 1
                            
                            # 進捗バーを更新（1文書ごと、tqdmが自動的に更新頻度を調整）
                            pbar.update(1)
                            # パーセンテージを計算（処理済み文書数 / 推定総文書数）
                            progress_pct = min(100.0, (total_documents / estimated_total_docs) * 100)
                            pbar.set_postfix({
                                'progress': f"{progress_pct:.1f}%",
                                'batch_terms': f"{len(batch_term_documents):,}",
                                'batches': f"{len(batch_cache_files)}"
                            })
                            
                            # バッチサイズに達したら、バッチの結果をディスクにキャッシュ
                            if batch_doc_count >= batch_size:
                                # バッチ内での出現文書数をカウント
                                batch_doc_counts: Dict[str, int] = {}
                                for token, doc_set in batch_term_documents.items():
                                    batch_doc_counts[token] = len(doc_set)
                                
                                # バッチの結果をディスクに保存
                                batch_cache_file = temp_dir / f"batch_{len(batch_cache_files):06d}.pkl"
                                with open(batch_cache_file, 'wb') as f:
                                    pickle.dump(batch_doc_counts, f)
                                batch_cache_files.append(batch_cache_file)
                                
                                # バッチのdoc_id_mappingをグローバルにマージ（実際には使わないが、整合性のため）
                                for doc_id_str in batch_doc_id_mapping.keys():
                                    if doc_id_str not in doc_id_mapping:
                                        doc_id_mapping[doc_id_str] = len(doc_id_mapping)
                                
                                # バッチをクリアしてメモリを解放
                                batch_term_documents.clear()
                                batch_doc_id_mapping.clear()
                                batch_doc_counts.clear()
                                batch_next_doc_id_int = 0
                                batch_doc_count = 0
                                
                                # ガベージコレクションを促す
                                import gc
                                gc.collect()
                                
                                # メモリ使用量を監視（オプション）
                                progress_pct = min(100.0, (total_documents / estimated_total_docs) * 100)
                                try:
                                    import psutil
                                    import os
                                    process = psutil.Process(os.getpid())
                                    mem_info = process.memory_info()
                                    mem_gb = mem_info.rss / (1024**3)
                                    pbar.set_postfix({
                                        'progress': f"{progress_pct:.1f}%",
                                        'batches': f"{len(batch_cache_files)}",
                                        'mem': f"{mem_gb:.1f}GB"
                                    })
                                except ImportError:
                                    pbar.set_postfix({
                                        'progress': f"{progress_pct:.1f}%",
                                        'batches': f"{len(batch_cache_files)}",
                                        'cached': "yes"
                                    })
                                
                                print(f"  Processed {total_documents:,} documents ({progress_pct:.1f}%), cached batch {len(batch_cache_files)}. Memory cleared.", file=sys.stderr)
                
                finally:
                    pbar.close()
                
                # 最後のバッチをキャッシュ
                if batch_doc_count > 0:
                    # バッチ内での出現文書数をカウント
                    batch_doc_counts: Dict[str, int] = {}
                    for token, doc_set in batch_term_documents.items():
                        batch_doc_counts[token] = len(doc_set)
                    
                    # バッチの結果をディスクに保存
                    batch_cache_file = temp_dir / f"batch_{len(batch_cache_files):06d}.pkl"
                    with open(batch_cache_file, 'wb') as f:
                        pickle.dump(batch_doc_counts, f)
                    batch_cache_files.append(batch_cache_file)
                    
                    for doc_id_str in batch_doc_id_mapping.keys():
                        if doc_id_str not in doc_id_mapping:
                            doc_id_mapping[doc_id_str] = len(doc_id_mapping)
                    
                    batch_term_documents.clear()
                    batch_doc_id_mapping.clear()
                    batch_doc_counts.clear()
                    import gc
                    gc.collect()
                    print(f"  Processed final batch. Total: {total_documents:,} documents, {len(batch_cache_files)} batches cached.", file=sys.stderr)
            
            # AmbigNQの場合：すべてのバッチキャッシュを読み込んでマージ
            print(f"Merging {len(batch_cache_files)} batch caches...", file=sys.stderr)
            document_freq: Dict[str, int] = defaultdict(int)
            for batch_file in tqdm(batch_cache_files, desc="Merging batches", file=sys.stderr, mininterval=1.0):
                with open(batch_file, 'rb') as f:
                    batch_doc_counts = pickle.load(f)
                    for token, count in batch_doc_counts.items():
                        document_freq[token] += count
            
            # 一時ディレクトリを削除
            import shutil
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}", file=sys.stderr)
            
            # defaultdictから通常のdictに変換
            document_freq = dict(document_freq)
            import gc
            gc.collect()
            
        elif dataset == "INSCIT":
            # INSCITの処理（従来通り）
            # ... (既存のINSCIT処理コード) ...
            pass
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # INSCITの場合のみ、term_documentsからdocument_freqに変換
        if dataset == "INSCIT":
            # 文書頻度に変換（集合のサイズ）
            document_freq = {term: len(docs) for term, docs in term_documents.items()}
            term_documents.clear()
            import gc
            gc.collect()
        
        # キャッシュに保存
        if document_freq_cache_path:
            cache_path = Path(document_freq_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'document_freq': document_freq,
                    'total_documents': total_documents
                }, f)
            print(f"Saved document frequency to cache: {cache_path}")
        
        print(f"Computed document frequency: {len(document_freq)} unique terms, {total_documents} documents")
        return document_freq, total_documents
    
    @classmethod
    def compute_from_files(
        cls,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: str,
        dataset: str = "INSCIT",
        document_freq_cache_path: Optional[str] = None,
        use_pyserini: bool = False
    ) -> None:
        """
        ベースJSONファイルからAvgIDFスコアを計算し、ベースJSONに追記して出力
        
        Args:
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス
            dataset: データセット名（"INSCIT" または "AmbigNQ"）
            document_freq_cache_path: 文書頻度のキャッシュファイルパス
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
        
        # 文書頻度を計算または読み込む
        if document_freq_cache_path is None:
            base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
            document_freq_cache_path = str(base_dir / "QPP" / "pre_retrieval" / ".document_freq_cache" / f"{dataset}_document_freq.pkl")
        
        document_freq, total_documents = cls.compute_document_frequency(
            dataset=dataset,
            document_freq_cache_path=document_freq_cache_path
        )
        
        print("Computing AvgIDF scores...")
        analyzer = cls(
            query_data_list,
            document_freq=document_freq,
            total_documents=total_documents,
            document_freq_cache_path=document_freq_cache_path,
            use_pyserini=use_pyserini
        )
        
        # AvgIDFを計算
        avgidf_stats = analyzer.compute()
        
        # CSV出力
        csv_data = []
        for _, row in avgidf_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'avgidf': float(row['avgidf']) if not np.isnan(row['avgidf']) else None,
                'num_terms': int(row['num_terms'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        avgidf_dict = {}
        for _, row in avgidf_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            avgidf_dict[key] = {
                'avgidf': float(row['avgidf']) if not np.isnan(row['avgidf']) else None,
                'num_terms': int(row['num_terms'])
            }
        
        # 各ターンにAvgIDFを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in avgidf_dict:
                    stats = avgidf_dict[key]
                    turn['avgidf'] = stats['avgidf']
                    turn['num_terms'] = stats['num_terms']
                else:
                    turn['avgidf'] = None
                    turn['num_terms'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

