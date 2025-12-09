"""
Clarity (Query Clarity) 手法の実装
クエリの語彙分布とコレクション全体の語彙分布のKLダイバージェンスを計算
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
import hashlib
import tempfile
import shutil
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class Clarity(BaseQPPAnalyzer):
    """Clarity (Query Clarity) 計算クラス"""
    
    def __init__(
        self, 
        turn_data_list: List[TurnData], 
        collection_freq: Optional[Dict[str, int]] = None,
        collection_freq_cache_path: Optional[str] = None,
        document_lm_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        super().__init__(turn_data_list, device=device, cache_dir=cache_dir)
        self.collection_freq = collection_freq
        self.collection_freq_cache_path = collection_freq_cache_path
        
        # コレクション語頻度が提供されていない場合はキャッシュから読み込む
        if self.collection_freq is None and collection_freq_cache_path:
            cache_path = Path(collection_freq_cache_path)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        self.collection_freq = pickle.load(f)
                    print(f"Loaded collection frequency from cache: {cache_path}")
                except Exception as e:
                    print(f"Warning: Failed to load collection frequency cache: {e}")
                    self.collection_freq = None
        
        # コレクション言語モデルを事前に計算してメモリにキャッシュ
        self.collection_language_model: Optional[Dict[str, float]] = None
        if self.collection_freq is not None and len(self.collection_freq) > 0:
            self.collection_language_model = self._compute_collection_language_model()
            if self.collection_language_model:
                total_cf = sum(self.collection_freq.values())
                print(f"Computed collection language model: {len(self.collection_language_model)} unique words, {total_cf:,} total tokens")
        
        # 文書言語モデルのキャッシュ設定
        if document_lm_cache_dir:
            self.document_lm_cache_dir = Path(document_lm_cache_dir)
            self.document_lm_cache_dir.mkdir(parents=True, exist_ok=True)
            # メモリキャッシュのサイズ制限（最大1000エントリ、メモリ節約のため）
            self.document_lm_cache: Dict[str, Dict[str, float]] = {}
            self.document_lm_cache_max_size = 1000  # メモリキャッシュの最大サイズ
        else:
            self.document_lm_cache_dir = None
            self.document_lm_cache = {}
            self.document_lm_cache_max_size = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        テキストをトークン化（単語に分割）
        
        Args:
            text: トークン化するテキスト
        
        Returns:
            トークンのリスト
        """
        # 小文字に変換し、単語に分割
        text = text.lower()
        # 英数字とアポストロフィのみを保持
        tokens = re.findall(r"\b[a-z0-9']+\b", text)
        return tokens
    
    def _compute_document_word_freq(self, text: str) -> Counter:
        """
        文書の語頻度を計算
        
        Args:
            text: 文書テキスト
        
        Returns:
            語頻度のCounter
        """
        tokens = self._tokenize(text)
        return Counter(tokens)
    
    def _get_document_lm_cache_key(self, doc_text: str, mu: float) -> str:
        """
        文書言語モデルのキャッシュキーを生成
        
        Args:
            doc_text: 文書テキスト
            mu: Dirichlet smoothingパラメータ
        
        Returns:
            キャッシュキー（ハッシュ値）
        """
        # テキストとmuパラメータからハッシュを生成
        text_hash = hashlib.md5(doc_text.encode('utf-8')).hexdigest()
        key = f"{text_hash}_{mu:.1f}"
        return key
    
    def _compute_document_language_model(
        self,
        doc_text: str,
        mu: float = 2000.0,
        use_cache: bool = True
    ) -> Dict[str, float]:
        """
        文書言語モデル P(t|d) を計算（Dirichlet smoothing）
        
        P(t|d) = (tf(t,d) + mu * P(t|C)) / (|d| + mu)
        
        Args:
            doc_text: 文書テキスト
            mu: Dirichlet smoothingパラメータ（デフォルト: 2000.0）
            use_cache: キャッシュを使用するかどうか
        
        Returns:
            P(t|d) の辞書
        """
        # キャッシュから取得を試みる
        if use_cache and self.document_lm_cache_dir:
            cache_key = self._get_document_lm_cache_key(doc_text, mu)
            
            # メモリキャッシュから取得
            if cache_key in self.document_lm_cache:
                return self.document_lm_cache[cache_key]
            
            # ファイルキャッシュから取得
            cache_file = self.document_lm_cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        P_t_d = pickle.load(f)
                    # メモリキャッシュにも保存（サイズ制限あり）
                    if len(self.document_lm_cache) >= self.document_lm_cache_max_size:
                        # 最も古いエントリを削除（FIFO方式）
                        oldest_key = next(iter(self.document_lm_cache))
                        del self.document_lm_cache[oldest_key]
                    self.document_lm_cache[cache_key] = P_t_d
                    return P_t_d
                except Exception as e:
                    # キャッシュファイルが破損している場合は再計算
                    pass
        
        # キャッシュにない場合は計算
        # 文書内の語頻度を計算
        doc_word_freq = self._compute_document_word_freq(doc_text)
        doc_len = sum(doc_word_freq.values())
        
        if doc_len == 0:
            P_t_d = {}
        else:
            # コレクション言語モデルを取得（必要に応じて計算）
            P_t_C = self._compute_collection_language_model(use_cache=True)
            if not P_t_C:
                # コレクション言語モデルがない場合は単純なMLE
                P_t_d = {word: freq / doc_len for word, freq in doc_word_freq.items()}
            else:
                # Dirichlet smoothingでP(t|d)を計算
                P_t_d = {}
                for word, freq in doc_word_freq.items():
                    P_t_C_word = P_t_C.get(word, 0.0)
                    P_t_d[word] = (freq + mu * P_t_C_word) / (doc_len + mu)
        
        # キャッシュに保存
        if use_cache and self.document_lm_cache_dir:
            cache_key = self._get_document_lm_cache_key(doc_text, mu)
            cache_file = self.document_lm_cache_dir / f"{cache_key}.pkl"
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(P_t_d, f)
                # メモリキャッシュにも保存（サイズ制限あり）
                if len(self.document_lm_cache) >= self.document_lm_cache_max_size:
                    # 最も古いエントリを削除（FIFO方式）
                    oldest_key = next(iter(self.document_lm_cache))
                    del self.document_lm_cache[oldest_key]
                self.document_lm_cache[cache_key] = P_t_d
            except Exception as e:
                # キャッシュ保存に失敗しても計算結果は返す
                pass
        
        return P_t_d
    
    def _estimate_relevance_model_em(
        self,
        documents: List,
        query_tokens: List[str],
        doc_language_models: Optional[List[Dict[str, float]]] = None,
        initial_scores: Optional[np.ndarray] = None,
        max_iter: int = 50,
        epsilon: float = 1e-6,
        mu: float = 2000.0,
        verbose: bool = False
    ) -> tuple:
        """
        EMアルゴリズムでRelevance Model（RM1）を推定し、P(d|q)を返す
        
        Args:
            documents: 上位k文書のリスト
            query_tokens: クエリのトークンリスト
            doc_language_models: 各文書の言語モデルP(t|d)のリスト（Noneの場合は計算）
            initial_scores: 初期スコア（DPRスコアなど）。Noneの場合は均等重み
            max_iter: 最大反復回数
            epsilon: 収束判定の閾値
            mu: Dirichlet smoothingパラメータ
            verbose: 進捗を表示するかどうか
        
        Returns:
            (P(d|q)の配列, doc_language_models) のタプル
        """
        k = len(documents)
        if k == 0 or len(query_tokens) == 0:
            return np.array([]), []
        
        # 各文書の言語モデルP(t|d)を計算（まだ計算されていない場合）
        if doc_language_models is None:
            doc_language_models = []
            for doc in documents:
                P_t_d = self._compute_document_language_model(doc.text, mu=mu)
                doc_language_models.append(P_t_d)
        
        # 初期化: P^(0)(d) = softmax(score(d,q))
        if initial_scores is not None and len(initial_scores) == k:
            scores = np.array(initial_scores)
            if np.all(scores == scores[0]):
                P_d_q = np.ones(k) / k
            else:
                exp_scores = np.exp(scores - np.max(scores))
                P_d_q = exp_scores / np.sum(exp_scores)
        else:
            # 均等重みで初期化
            P_d_q = np.ones(k) / k
        
        # EMアルゴリズム
        for iteration in range(max_iter):
            P_d_q_old = P_d_q.copy()
            
            # E-step: P^(i)(d|t) を計算
            # P(d|t) = P(t|d) * P(d) / Σ_d' P(t|d') * P(d')
            P_d_t = np.zeros((len(query_tokens), k))
            
            for t_idx, term in enumerate(query_tokens):
                denominator = 0.0
                numerators = np.zeros(k)
                
                for d_idx in range(k):
                    P_t_d_val = doc_language_models[d_idx].get(term, 0.0)
                    numerators[d_idx] = P_t_d_val * P_d_q[d_idx]
                    denominator += numerators[d_idx]
                
                if denominator > 0:
                    P_d_t[t_idx, :] = numerators / denominator
                else:
                    # すべて0の場合は均等分布
                    P_d_t[t_idx, :] = np.ones(k) / k
            
            # M-step: P^(i+1)(d) = (1/|q|) * Σ_t P^(i)(d|t)
            P_d_q = np.mean(P_d_t, axis=0)
            
            # 正規化（念のため）
            total = np.sum(P_d_q)
            if total > 0:
                P_d_q = P_d_q / total
            else:
                P_d_q = np.ones(k) / k
            
            # 収束判定
            max_diff = np.max(np.abs(P_d_q - P_d_q_old))
            if verbose and iteration % 10 == 0:
                print(f"  EM iteration {iteration}: max_diff = {max_diff:.6f}", file=sys.stderr)
            
            if max_diff < epsilon:
                if verbose:
                    print(f"  EM converged at iteration {iteration}", file=sys.stderr)
                break
        
        return P_d_q, doc_language_models
    
    def _compute_query_language_model(
        self, 
        documents: List,
        query_tokens: List[str],
        use_em: bool = True,
        use_scores: bool = True,
        max_em_iter: int = 50,
        em_epsilon: float = 1e-6,
        mu: float = 2000.0,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        クエリ言語モデル P(w|R_q) を計算（Relevance Model）
        
        P(w|R_q) = Σ_d P(w|d) * P(d|q)
        
        Args:
            documents: 上位k文書のリスト
            query_tokens: クエリのトークンリスト
            use_em: EMアルゴリズムを使用するかどうか（True: 正式なClarity, False: 簡略版）
            use_scores: DPRスコアを使用して初期化するかどうか（use_em=Trueの場合）
            max_em_iter: EMアルゴリズムの最大反復回数
            em_epsilon: EMアルゴリズムの収束判定閾値
            mu: Dirichlet smoothingパラメータ
            verbose: 進捗を表示するかどうか
        
        Returns:
            P(w|R_q) の辞書
        """
        k = len(documents)
        if k == 0:
            return {}
        
        # 各文書の言語モデルP(t|d)を事前に計算（一度だけ）
        doc_language_models = []
        for doc in documents:
            P_t_d = self._compute_document_language_model(doc.text, mu=mu)
            doc_language_models.append(P_t_d)
        
        # P(d|q) を計算
        if use_em:
            # EMアルゴリズムで推定（正式なClarity）
            if use_scores:
                initial_scores = np.array([doc.score for doc in documents])
            else:
                initial_scores = None
            
            P_d_q, doc_language_models = self._estimate_relevance_model_em(
                documents=documents,
                query_tokens=query_tokens,
                doc_language_models=doc_language_models,  # 事前計算したものを渡す
                initial_scores=initial_scores,
                max_iter=max_em_iter,
                epsilon=em_epsilon,
                mu=mu,
                verbose=verbose
            )
        else:
            # 簡略版: DPRスコアをsoftmaxで正規化
            if use_scores:
                scores = np.array([doc.score for doc in documents])
                if np.all(scores == scores[0]):
                    P_d_q = np.ones(k) / k
                else:
                    exp_scores = np.exp(scores - np.max(scores))
                    P_d_q = exp_scores / np.sum(exp_scores)
            else:
                P_d_q = np.ones(k) / k
        
        # P(w|R_q) = Σ_d P(w|d) * P(d|q) を計算（事前計算したdoc_language_modelsを使用）
        P_w_Rq = Counter()
        
        for i, P_w_d in enumerate(doc_language_models):
            for word, prob in P_w_d.items():
                P_w_Rq[word] += prob * P_d_q[i]
        
        # 正規化（確率分布にする）
        total = sum(P_w_Rq.values())
        if total > 0:
            P_w_Rq = {word: prob / total for word, prob in P_w_Rq.items()}
        else:
            P_w_Rq = {}
        
        return P_w_Rq
    
    def _compute_collection_language_model(self, use_cache: bool = True) -> Dict[str, float]:
        """
        コレクション言語モデル P(w|C) を計算
        
        P(w|C) = cf(w) / Σ_w' cf(w')
        
        Args:
            use_cache: メモリキャッシュを使用するかどうか（デフォルト: True）
        
        Returns:
            P(w|C) の辞書
        """
        # メモリキャッシュから取得（既に計算済みの場合）
        if use_cache and self.collection_language_model is not None:
            return self.collection_language_model
        
        if self.collection_freq is None or len(self.collection_freq) == 0:
            return {}
        
        total_cf = sum(self.collection_freq.values())
        if total_cf == 0:
            return {}
        
        # 計算（軽量なので毎回計算しても問題ない）
        P_w_C = {word: freq / total_cf for word, freq in self.collection_freq.items()}
        
        # メモリキャッシュに保存（__init__で既に計算されている場合は不要だが、念のため）
        if use_cache:
            self.collection_language_model = P_w_C
        
        return P_w_C
    
    def _compute_kl_divergence(
        self, 
        P_w_Q: Dict[str, float], 
        P_w_C: Dict[str, float]
    ) -> float:
        """
        KLダイバージェンスを計算
        
        Clarity(Q) = Σ_w P(w|Q) * log(P(w|Q) / P(w|C))
        
        Args:
            P_w_Q: クエリ言語モデル
            P_w_C: コレクション言語モデル
        
        Returns:
            Clarityスコア
        """
        if not P_w_Q or not P_w_C:
            return np.nan
        
        clarity = 0.0
        for word, prob_q in P_w_Q.items():
            if prob_q > 0:
                prob_c = P_w_C.get(word, 1e-10)  # ゼロ除算を避けるため小さな値を設定
                if prob_c > 0:
                    clarity += prob_q * np.log(prob_q / prob_c)
        
        return clarity
    
    def compute(
        self,
        top_k: Optional[int] = None,
        use_scores: bool = True,
        use_em: bool = True,
        max_em_iter: int = 50,
        em_epsilon: float = 1e-6,
        mu: float = 2000.0,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        各ターンについて、Clarityスコアを計算
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            use_scores: DPRスコアを使用して初期化するかどうか（use_em=Trueの場合）
            use_em: EMアルゴリズムを使用するかどうか（True: 正式なClarity, False: 簡略版）
            max_em_iter: EMアルゴリズムの最大反復回数
            em_epsilon: EMアルゴリズムの収束判定閾値
            mu: Dirichlet smoothingパラメータ
            verbose: 進捗を表示するかどうか
        
        Returns:
            DataFrame with columns: conv_id, turn_id, clarity, num_documents
        """
        if self.collection_freq is None:
            raise ValueError("Collection frequency is required. Please provide collection_freq or load from cache.")
        
        # コレクション言語モデルを取得（必要に応じて計算、同じインスタンス内ではキャッシュされる）
        P_w_C = self._compute_collection_language_model(use_cache=True)
        if not P_w_C:
            raise ValueError("Collection language model is empty. Please check collection_freq.")
        
        results = []
        
        # tqdmの設定: nohup環境でも進捗が見えるように設定
        for turn in tqdm(
            self.turn_data_list, 
            desc="Computing Clarity" + (" (EM)" if use_em else " (simplified)"), 
            unit="turn",
            file=sys.stderr,  # 標準エラー出力に進捗を表示（nohupでも見える）
            mininterval=1.0,  # 最低1秒間隔で更新（ログファイルの肥大化を防ぐ）
            disable=False
        ):
            documents = self._get_documents(turn, top_k=top_k)
            
            if len(documents) == 0:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'clarity': np.nan,
                    'num_documents': 0
                })
                continue
            
            # クエリをトークン化
            question = turn.question
            query_tokens = self._tokenize(question)
            
            if len(query_tokens) == 0:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'clarity': np.nan,
                    'num_documents': len(documents)
                })
                continue
            
            # クエリ言語モデルを計算（Relevance Model）
            P_w_Rq = self._compute_query_language_model(
                documents=documents,
                query_tokens=query_tokens,
                use_em=use_em,
                use_scores=use_scores,
                max_em_iter=max_em_iter,
                em_epsilon=em_epsilon,
                mu=mu,
                verbose=verbose
            )
            
            if not P_w_Rq:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'clarity': np.nan,
                    'num_documents': len(documents)
                })
                continue
            
            # KLダイバージェンスを計算
            clarity = self._compute_kl_divergence(P_w_Rq, P_w_C)
            
            results.append({
                'conv_id': turn.conv_id,
                'turn_id': turn.turn_id,
                'clarity': clarity,
                'num_documents': len(documents)
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def compute_collection_frequency(
        dataset: str,
        collection_freq_cache_path: Optional[str] = None,
        max_docs: Optional[int] = None,
        batch_size: int = 50000  # 大規模コーパス用にバッチ処理（AmbigNQ用）
    ) -> Dict[str, int]:
        """
        コレクション全体の語頻度を計算
        
        Args:
            dataset: データセット名（"INSCIT" または "AmbigNQ"）
            collection_freq_cache_path: キャッシュファイルのパス
            max_docs: 最大処理文書数（Noneの場合は全件）
        
        Returns:
            語頻度の辞書
        """
        # キャッシュから読み込む
        if collection_freq_cache_path:
            cache_path = Path(collection_freq_cache_path)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        collection_freq = pickle.load(f)
                    print(f"Loaded collection frequency from cache: {cache_path}")
                    return collection_freq
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
                                                collection_freq.update(tokens)
                                    elif 'text' in doc:
                                        # 直接textフィールドがある場合（他の形式）
                                        text = doc['text']
                                        tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                        collection_freq.update(tokens)
                    if collection_freq_cache_path:
                        cache_path = Path(collection_freq_cache_path)
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(cache_path, 'wb') as f:
                            pickle.dump(dict(collection_freq), f)
                        print(f"Saved collection frequency to cache: {cache_path}")
                    return dict(collection_freq)
            
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
                                        collection_freq.update(tokens)
                            elif 'text' in doc:
                                # 直接textフィールドがある場合（他の形式）
                                text = doc['text']
                                tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                                collection_freq.update(tokens)
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
            # バッチごとにディスクにキャッシュし、最後にマージする方式
            temp_dir = Path(tempfile.mkdtemp(prefix="clarity_collection_freq_batch_"))
            batch_cache_files: List[Path] = []
            print(f"Using temporary directory for batch caches: {temp_dir}", file=sys.stderr)
            
            batch_collection_freq = Counter()
            doc_count = 0
            batch_doc_count = 0
            
            with gzip.open(collection_path, 'rt', encoding='utf-8') as f:
                # ヘッダーをスキップ
                header_line = next(f)
                
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
                        if max_docs and doc_count >= max_docs:
                            break
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            text = parts[1]  # text列
                            tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                            batch_collection_freq.update(tokens)
                            doc_count += 1
                            batch_doc_count += 1
                            
                            # 進捗バーを更新
                            pbar.update(1)
                            progress_pct = min(100.0, (doc_count / estimated_total_docs) * 100)
                            pbar.set_postfix({
                                'progress': f"{progress_pct:.1f}%",
                                'batch_terms': f"{len(batch_collection_freq):,}",
                                'batches': f"{len(batch_cache_files)}"
                            })
                            
                            # バッチサイズに達したら、バッチの結果をディスクにキャッシュ
                            if batch_doc_count >= batch_size:
                                # バッチの結果をディスクに保存
                                batch_cache_file = temp_dir / f"batch_{len(batch_cache_files):06d}.pkl"
                                with open(batch_cache_file, 'wb') as f:
                                    pickle.dump(dict(batch_collection_freq), f)
                                batch_cache_files.append(batch_cache_file)
                                
                                # バッチをクリアしてメモリを解放
                                batch_collection_freq.clear()
                                batch_doc_count = 0
                                
                                # ガベージコレクションを促す
                                import gc
                                gc.collect()
                                
                                # メモリ使用量を監視（オプション）
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
                                
                                print(f"  Processed {doc_count:,} documents ({progress_pct:.1f}%), cached batch {len(batch_cache_files)}. Memory cleared.", file=sys.stderr)
                
                finally:
                    pbar.close()
                
                # 最後のバッチをキャッシュ
                if batch_doc_count > 0:
                    batch_cache_file = temp_dir / f"batch_{len(batch_cache_files):06d}.pkl"
                    with open(batch_cache_file, 'wb') as f:
                        pickle.dump(dict(batch_collection_freq), f)
                    batch_cache_files.append(batch_cache_file)
                    
                    batch_collection_freq.clear()
                    import gc
                    gc.collect()
                    print(f"  Processed final batch. Total: {doc_count:,} documents, {len(batch_cache_files)} batches cached.", file=sys.stderr)
            
            # すべてのバッチキャッシュを読み込んでマージ
            print(f"Merging {len(batch_cache_files)} batch caches...", file=sys.stderr)
            collection_freq = Counter()
            for batch_file in tqdm(batch_cache_files, desc="Merging batches", file=sys.stderr, mininterval=1.0):
                with open(batch_file, 'rb') as f:
                    batch_freq = pickle.load(f)
                    collection_freq.update(batch_freq)
            
            # 一時ディレクトリを削除
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory {temp_dir}: {e}", file=sys.stderr)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        # キャッシュに保存
        if collection_freq_cache_path:
            cache_path = Path(collection_freq_cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(dict(collection_freq), f)
            print(f"Saved collection frequency to cache: {cache_path}")
        
        print(f"Computed collection frequency: {len(collection_freq)} unique words")
        return dict(collection_freq)
    
    @classmethod
    def compute_from_files(
        cls,
        dpr_json_path: str,
        base_json_path: str,
        output_json_path: str,
        output_csv_path: Optional[str] = None,
        dataset: str = "INSCIT",
        top_k: Optional[int] = None,
        use_scores: bool = True,
        use_em: bool = True,
        max_em_iter: int = 50,
        em_epsilon: float = 1e-6,
        mu: float = 2000.0,
        verbose: bool = False,
        collection_freq_cache_path: Optional[str] = None,
        document_lm_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ) -> None:
        """
        DPR結果ファイルからClarityスコアを計算し、ベースJSONに追記して出力
        
        Args:
            dpr_json_path: DPR結果ファイルのパス（dpr_dev.json または dpr_train.json）
            base_json_path: ベースとなるdev.jsonまたはtrain.jsonのパス
            output_json_path: 出力先のJSONファイルパス
            output_csv_path: 出力先のCSVファイルパス（Noneの場合は自動生成）
            dataset: データセット名（"INSCIT" または "AmbigNQ"）
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            use_scores: DPRスコアを使用して初期化するかどうか（use_em=Trueの場合）
            use_em: EMアルゴリズムを使用するかどうか（True: 正式なClarity, False: 簡略版）
            max_em_iter: EMアルゴリズムの最大反復回数
            em_epsilon: EMアルゴリズムの収束判定閾値
            mu: Dirichlet smoothingパラメータ
            verbose: 進捗を表示するかどうか
            collection_freq_cache_path: コレクション語頻度のキャッシュファイルパス
            document_lm_cache_dir: 文書言語モデルのキャッシュディレクトリ（Noneの場合は自動生成）
            device: 使用するデバイス（Noneの場合は自動選択）
            cache_dir: キャッシュディレクトリのパス（Noneの場合はキャッシュを使用しない）
        """
        print(f"Loading DPR results from: {dpr_json_path}")
        turn_data_list = DPRResultLoader(dpr_json_path).load_data()
        print(f"Loaded {len(turn_data_list)} turns")
        
        # コレクション語頻度を計算または読み込む
        if collection_freq_cache_path is None:
            base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
            collection_freq_cache_path = str(base_dir / "QPP" / "post_retrieval" / ".collection_cache" / f"{dataset}_collection_freq.pkl")
        
        collection_freq = cls.compute_collection_frequency(
            dataset=dataset,
            collection_freq_cache_path=collection_freq_cache_path
        )
        
        # 文書言語モデルのキャッシュディレクトリを設定
        if document_lm_cache_dir is None:
            base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
            document_lm_cache_dir = str(base_dir / "QPP" / "post_retrieval" / ".document_lm_cache" / dataset)
        
        print("Computing Clarity scores...")
        analyzer = cls(
            turn_data_list, 
            collection_freq=collection_freq,
            collection_freq_cache_path=collection_freq_cache_path,
            document_lm_cache_dir=document_lm_cache_dir,
            device=device,
            cache_dir=cache_dir
        )
        
        # Clarityを計算
        print(f"Computing Clarity using {'EM algorithm' if use_em else 'simplified method'}...")
        clarity_stats = analyzer.compute(
            top_k=top_k,
            use_scores=use_scores,
            use_em=use_em,
            max_em_iter=max_em_iter,
            em_epsilon=em_epsilon,
            mu=mu,
            verbose=verbose
        )
        
        # CSV出力（output_csv_pathが指定されていない場合は自動生成）
        if output_csv_path is None:
            output_csv_path = Path(output_json_path).with_suffix('.csv')
        else:
            output_csv_path = Path(output_csv_path)
        
        # CSV用のDataFrameを準備
        csv_data = []
        for _, row in clarity_stats.iterrows():
            csv_data.append({
                'conv_id': str(row['conv_id']),
                'turn_id': int(row['turn_id']),
                'clarity': row['clarity'] if not np.isnan(row['clarity']) else None,
                'num_documents': int(row['num_documents'])
            })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df['conv_id'] = csv_df['conv_id'].astype(str)
        csv_df.to_csv(output_csv_path, index=False)
        print(f"CSV saved to: {output_csv_path}")
        
        # キーを(conv_id, turn_id)にして辞書化
        clarity_dict = {}
        for _, row in clarity_stats.iterrows():
            key = (row['conv_id'], str(row['turn_id']))
            clarity_dict[key] = {
                'clarity': float(row['clarity']) if not np.isnan(row['clarity']) else None,
                'num_documents': int(row['num_documents'])
            }
        
        # ベースJSONを読み込む
        with open(base_json_path, 'r', encoding='utf-8') as f:
            base_data = json.load(f)
        
        # 各ターンにClarityを追記
        for conversation in base_data:
            for turn in conversation:
                conv_id = turn['conv_id']
                turn_id = str(turn['turn_id'])
                key = (conv_id, turn_id)
                
                if key in clarity_dict:
                    stats = clarity_dict[key]
                    turn['clarity'] = stats['clarity']
                    turn['num_documents'] = stats['num_documents']
                else:
                    turn['clarity'] = None
                    turn['num_documents'] = 0
        
        # JSON出力
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(base_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nOutput saved to: {output_json_path}")
        print(f"CSV saved to: {output_csv_path}")

