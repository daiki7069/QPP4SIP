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
from .base import BaseQPPAnalyzer
from data_loader import TurnData, DPRResultLoader


class Clarity(BaseQPPAnalyzer):
    """Clarity (Query Clarity) 計算クラス"""
    
    def __init__(
        self, 
        turn_data_list: List[TurnData], 
        collection_freq: Optional[Dict[str, int]] = None,
        collection_freq_cache_path: Optional[str] = None,
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
    
    def _compute_query_language_model(
        self, 
        documents: List, 
        use_scores: bool = True
    ) -> Dict[str, float]:
        """
        クエリ言語モデル P(w|Q) を計算
        
        P(w|Q) = Σ_d P(w|d) * P(d|Q)
        
        Args:
            documents: 上位k文書のリスト
            use_scores: DPRスコアを使用してP(d|Q)を計算するかどうか
        
        Returns:
            P(w|Q) の辞書
        """
        k = len(documents)
        if k == 0:
            return {}
        
        # P(d|Q) を計算（DPRスコアをsoftmaxで正規化）
        if use_scores:
            scores = np.array([doc.score for doc in documents])
            # スコアがすべて同じ場合の処理
            if np.all(scores == scores[0]):
                P_d_Q = np.ones(k) / k
            else:
                # softmaxで正規化
                exp_scores = np.exp(scores - np.max(scores))  # 数値安定性のため
                P_d_Q = exp_scores / np.sum(exp_scores)
        else:
            # 均等重み
            P_d_Q = np.ones(k) / k
        
        # P(w|Q) = Σ_d P(w|d) * P(d|Q) を計算
        P_w_Q = Counter()
        
        for i, doc in enumerate(documents):
            # P(w|d) を計算（文書内の語頻度）
            doc_word_freq = self._compute_document_word_freq(doc.text)
            doc_len = sum(doc_word_freq.values())
            
            if doc_len > 0:
                for word, freq in doc_word_freq.items():
                    P_w_d = freq / doc_len
                    P_w_Q[word] += P_w_d * P_d_Q[i]
        
        # 正規化（確率分布にする）
        total = sum(P_w_Q.values())
        if total > 0:
            P_w_Q = {word: prob / total for word, prob in P_w_Q.items()}
        else:
            P_w_Q = {}
        
        return P_w_Q
    
    def _compute_collection_language_model(self) -> Dict[str, float]:
        """
        コレクション言語モデル P(w|C) を計算
        
        P(w|C) = cf(w) / Σ_w' cf(w')
        
        Returns:
            P(w|C) の辞書
        """
        if self.collection_freq is None or len(self.collection_freq) == 0:
            return {}
        
        total_cf = sum(self.collection_freq.values())
        if total_cf == 0:
            return {}
        
        P_w_C = {word: freq / total_cf for word, freq in self.collection_freq.items()}
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
        use_scores: bool = True
    ) -> pd.DataFrame:
        """
        各ターンについて、Clarityスコアを計算
        
        Args:
            top_k: 上位k件の文書のみを処理（Noneの場合は全件、最大100件）
            use_scores: DPRスコアを使用してP(d|Q)を計算するかどうか
        
        Returns:
            DataFrame with columns: conv_id, turn_id, clarity, num_documents
        """
        if self.collection_freq is None:
            raise ValueError("Collection frequency is required. Please provide collection_freq or load from cache.")
        
        # コレクション言語モデルを計算（一度だけ）
        P_w_C = self._compute_collection_language_model()
        
        if not P_w_C:
            raise ValueError("Collection language model is empty. Please check collection_freq.")
        
        results = []
        
        # tqdmの設定: nohup環境でも進捗が見えるように設定
        for turn in tqdm(
            self.turn_data_list, 
            desc="Computing Clarity", 
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
            
            # クエリ言語モデルを計算
            P_w_Q = self._compute_query_language_model(documents, use_scores=use_scores)
            
            if not P_w_Q:
                results.append({
                    'conv_id': turn.conv_id,
                    'turn_id': turn.turn_id,
                    'clarity': np.nan,
                    'num_documents': len(documents)
                })
                continue
            
            # KLダイバージェンスを計算
            clarity = self._compute_kl_divergence(P_w_Q, P_w_C)
            
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
        max_docs: Optional[int] = None
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
            
            doc_count = 0
            with gzip.open(collection_path, 'rt', encoding='utf-8') as f:
                # ヘッダーをスキップ
                next(f)
                for line in tqdm(
                    f, 
                    desc="Processing documents",
                    file=sys.stderr,
                    mininterval=1.0
                ):
                    if max_docs and doc_count >= max_docs:
                        break
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        text = parts[1]  # text列
                        tokens = re.findall(r"\b[a-z0-9']+\b", text.lower())
                        collection_freq.update(tokens)
                        doc_count += 1
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
        collection_freq_cache_path: Optional[str] = None,
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
            use_scores: DPRスコアを使用してP(d|Q)を計算するかどうか
            collection_freq_cache_path: コレクション語頻度のキャッシュファイルパス
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
        
        print("Computing Clarity scores...")
        analyzer = cls(
            turn_data_list, 
            collection_freq=collection_freq,
            collection_freq_cache_path=collection_freq_cache_path,
            device=device,
            cache_dir=cache_dir
        )
        
        # Clarityを計算
        clarity_stats = analyzer.compute(top_k=top_k, use_scores=use_scores)
        
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

