"""
Post-retrieval QPP用の文書内容分析
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import hashlib
import pickle
from pathlib import Path
from collections import Counter
from math import log
from data_loader import TurnData
from data_loader import DPRResultLoader


class ContentAnalyzer:
    def __init__(
        self, 
        turn_data_list: List[TurnData], 
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        self.turn_data_list = turn_data_list
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        
        # キャッシュ設定
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".embedding_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "document_embeddings.pkl"
        self.embedding_cache: Dict[str, np.ndarray] = {}
        self._load_cache()
    
    def _load_cache(self):
        """キャッシュファイルから埋め込みを読み込む"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings from {self.cache_file}")
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                self.embedding_cache = {}
        else:
            self.embedding_cache = {}
    
    def _save_cache(self):
        """埋め込みキャッシュをファイルに保存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """文書テキストのハッシュを計算"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _extract_title_from_id(self, doc_id: str) -> str:
        """文書IDからタイトルを抽出（コロンの前まで）"""
        if ':' in doc_id:
            return doc_id.split(':')[0]
        return doc_id
    
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
    
    def _load_bert_model(self):
        """BERT-uncasedモデルとトークナイザーをロード"""
        if self.tokenizer is None or self.model is None:
            model_name = 'bert-base-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
    
    def _encode_document(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        単一文書の埋め込みを計算（キャッシュを使用）
        
        Args:
            text: 文書テキスト
            use_cache: キャッシュを使用するかどうか
        
        Returns:
            埋め込みベクトル (768次元)
        """
        # キャッシュから取得を試みる
        if use_cache:
            text_hash = self._get_text_hash(text)
            if text_hash in self.embedding_cache:
                return self.embedding_cache[text_hash]
        
        # キャッシュにない場合は計算
        self._load_bert_model()
        
        # トークナイズ
        encoded = self.tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # デバイスに移動
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 埋め込みを計算
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            # [CLS]トークンの埋め込みを使用（または平均プーリング）
            # [CLS]トークンを使用する場合
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embedding_vector = embedding[0]  # (768,) の形状
        
        # キャッシュに保存
        if use_cache:
            text_hash = self._get_text_hash(text)
            self.embedding_cache[text_hash] = embedding_vector
        
        return embedding_vector
    
    def _compute_turn_similarity_statistics(
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
            # 上位k件を取得（最大100件）
            documents = turn.documents
            if top_k is not None:
                documents = documents[:top_k]
            documents = documents[:100]  # 最大100件に制限
            
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
    
    def save_cache(self):
        """埋め込みキャッシュを保存"""
        self._save_cache()
        print(f"Saved {len(self.embedding_cache)} embeddings to cache")
    
    def _compute_title_concentration_index(
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
            # 上位k件を取得（最大100件）
            documents = turn.documents
            if top_k is not None:
                documents = documents[:top_k]
            documents = documents[:100]  # 最大100件に制限
            
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
    
    def _compute_title_entropy(
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
            # 上位k件を取得（最大100件）
            documents = turn.documents
            if top_k is not None:
                documents = documents[:top_k]
            documents = documents[:100]  # 最大100件に制限
            
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
    
    def _compute_unique_titles(
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
            # 上位k件を取得（最大100件）
            documents = turn.documents
            if top_k is not None:
                documents = documents[:top_k]
            documents = documents[:100]  # 最大100件に制限
            
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
    
    def _compute_nqc(
        self,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        各ターンについて、上位k件のスコアからNQC（Normalized Query Clarity）を計算
        
        NQC = √(Σ (si - μ)^2 / k) / μ
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
        
        Returns:
            DataFrame with columns: conv_id, turn_id, nqc, num_documents, score_mean, score_std
        """
        results = []
        
        for turn in tqdm(self.turn_data_list, desc="Computing NQC", unit="turn"):
            # 上位k件を取得（最大100件）
            documents = turn.documents
            if top_k is not None:
                documents = documents[:top_k]
            documents = documents[:100]  # 最大100件に制限
            
            if len(documents) == 0:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'nqc': np.nan,
                    'num_documents': 0,
                    'score_mean': np.nan,
                    'score_std': np.nan
                })
                continue
            
            # スコアを取得
            scores = np.array([doc.score for doc in documents])
            k = len(scores)
            
            # 平均値を計算
            mu = np.mean(scores)
            
            if mu == 0 or k == 0:
                # 平均が0またはkが0の場合はNQCを計算できない
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'nqc': np.nan,
                    'num_documents': k,
                    'score_mean': mu,
                    'score_std': np.nan
                })
                continue
            
            # 標準偏差を計算（母標準偏差、ddof=0）
            # √(Σ (si - μ)^2 / k)
            std = np.std(scores, ddof=0)
            
            # NQC = std / μ
            nqc = std / mu
            
            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'nqc': nqc,
                'num_documents': k,
                'score_mean': mu,
                'score_std': std
            })
        
        return pd.DataFrame(results)
    
    def _compute_and_merge_similarity_scores(
        self,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> None:
        """
        類似度統計を計算し、dev/train.json形式のファイルに追記して出力
        
        Args:
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス（Noneの場合は自動生成）
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
        """
        import json
        
        # 類似度統計を計算
        similarity_stats = self._compute_turn_similarity_statistics(top_k=top_k)
        
        # LCIを計算（埋め込み計算は不要なので高速）
        lci_stats = self._compute_title_concentration_index(top_k=top_k)
        
        # CSV出力（output_csv_pathが指定されていない場合は自動生成）
        if output_csv_path is None:
            output_csv_path = Path(output_json_path).with_suffix('.csv')
        else:
            output_csv_path = Path(output_csv_path)
        
        # 類似度統計とLCIをマージ
        merged_stats = similarity_stats.merge(
            lci_stats[['conv_id', 'turn_id', 'lci']],
            on=['conv_id', 'turn_id'],
            how='left'
        )
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in merged_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'mean_similarity': row['mean_similarity'] if not np.isnan(row['mean_similarity']) else None,
                'sum_similarity': row['sum_similarity'] if not np.isnan(row['sum_similarity']) else None,
                'num_similarity_pairs': int(row['num_pairs']),
                'num_retrieved_documents': int(row['num_documents']),
                'lci': float(row['lci']) if 'lci' in row and not np.isnan(row['lci']) else 0.0
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        similarity_dict = {}
        for _, row in merged_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            similarity_dict[key] = {
                'mean_similarity': float(row['mean_similarity']) if not np.isnan(row['mean_similarity']) else None,
                'sum_similarity': float(row['sum_similarity']) if not np.isnan(row['sum_similarity']) else None,
                'num_pairs': int(row['num_pairs']),
                'num_documents': int(row['num_documents']),
                'lci': float(row['lci']) if 'lci' in row and not np.isnan(row['lci']) else 0.0
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
                    turn['lci'] = stats['lci']
                else:
                    # 見つからない場合はNoneを設定
                    turn['mean_similarity'] = None
                    turn['sum_similarity'] = None
                    turn['num_similarity_pairs'] = 0
                    turn['num_retrieved_documents'] = 0
                    turn['lci'] = 0.0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def compute_similarity_from_files(
        cls,
        dpr_json_path: str,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: str,
        top_k: Optional[int] = None,
        device: Optional[str] = None
    ) -> None:
        """
        DPR結果ファイルから類似度統計を計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
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
        
        # 類似度統計を計算してマージ
        print("Computing similarity statistics and merging with base JSON...")
        analyzer._compute_and_merge_similarity_scores(
            base_json_path=base_json_path,
            output_json_path=output_json_path,
            output_csv_path=output_csv_path,
            top_k=top_k
        )
        
        # キャッシュを保存
        analyzer.save_cache()
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")
    
    @classmethod
    def compute_lci_from_files(
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
        import json
        
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing Local Concentration Index (LCI)...")
        # LCI計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # LCIを計算
        lci_stats = analyzer._compute_title_concentration_index(top_k=top_k, window=window)
        
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
    
    @classmethod
    def compute_entropy_from_files(
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
        import json
        
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing Title Distribution Entropy...")
        # エントロピー計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # エントロピーを計算
        entropy_stats = analyzer._compute_title_entropy(top_k=top_k)
        
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
    
    @classmethod
    def compute_unique_titles_from_files(
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
        import json
        
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing unique titles count...")
        # ユニークタイトル数計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # ユニークタイトル数を計算
        unique_titles_stats = analyzer._compute_unique_titles(top_k=top_k)
        
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
    
    @classmethod
    def compute_nqc_from_files(
        cls,
        dpr_json_path: str,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: str,
        top_k: Optional[int] = None
    ) -> None:
        """
        DPR結果ファイルから上位k件のスコアからNQC（Normalized Query Clarity）を計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
        """
        import json
        
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        print("Computing NQC (Normalized Query Clarity)...")
        # NQC計算は埋め込み不要なので、キャッシュディレクトリは不要
        analyzer = cls(turn_data_list, device=None, cache_dir=None)
        
        # NQCを計算
        nqc_stats = analyzer._compute_nqc(top_k=top_k)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in nqc_stats.iterrows():
            csv_data.append({
                'conv_id': row['conv_id'],
                'turn_id': row['turn_id'],
                'nqc': float(row['nqc']) if not np.isnan(row['nqc']) else None,
                'num_retrieved_documents': int(row['num_documents']),
                'score_mean': float(row['score_mean']) if not np.isnan(row['score_mean']) else None,
                'score_std': float(row['score_std']) if not np.isnan(row['score_std']) else None
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        nqc_dict = {}
        for _, row in nqc_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            nqc_dict[key] = {
                'nqc': float(row['nqc']) if not np.isnan(row['nqc']) else None,
                'num_documents': int(row['num_documents']),
                'score_mean': float(row['score_mean']) if not np.isnan(row['score_mean']) else None,
                'score_std': float(row['score_std']) if not np.isnan(row['score_std']) else None
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンにNQCを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in nqc_dict:
                    stats = nqc_dict[key]
                    turn['nqc'] = stats['nqc']
                    turn['num_retrieved_documents'] = stats['num_documents']
                    turn['score_mean'] = stats['score_mean']
                    turn['score_std'] = stats['score_std']
                else:
                    turn['nqc'] = None
                    turn['num_retrieved_documents'] = 0
                    turn['score_mean'] = None
                    turn['score_std'] = None
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")
