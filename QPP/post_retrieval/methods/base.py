"""
Post-retrieval QPP用のベースクラス
共通機能（BERT埋め込み、キャッシュ、タイトル抽出など）を提供
"""
import numpy as np
from typing import List, Optional, Dict
import torch
from transformers import BertTokenizer, BertModel
import hashlib
import pickle
from pathlib import Path
from data_loader import TurnData


class BaseQPPAnalyzer:
    """QPP分析のベースクラス"""
    
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
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True)
            self.cache_file = self.cache_dir / "document_embeddings.pkl"
            self.embedding_cache: Dict[str, np.ndarray] = {}
            self._load_cache()
        else:
            self.cache_dir = None
            self.cache_file = None
            self.embedding_cache = {}
    
    def _load_cache(self):
        """キャッシュファイルから埋め込みを読み込む"""
        if self.cache_file and self.cache_file.exists():
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
        if self.cache_file:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.embedding_cache, f)
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")
    
    def save_cache(self):
        """埋め込みキャッシュを保存"""
        self._save_cache()
        if self.cache_file:
            print(f"Saved {len(self.embedding_cache)} embeddings to cache")
    
    def _get_text_hash(self, text: str) -> str:
        """文書テキストのハッシュを計算"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _extract_title_from_id(self, doc_id: str) -> str:
        """文書IDからタイトルを抽出（コロンの前まで）"""
        if ':' in doc_id:
            return doc_id.split(':')[0]
        return doc_id
    
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
        if use_cache and self.cache_file:
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
            # [CLS]トークンの埋め込みを使用
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        embedding_vector = embedding[0]  # (768,) の形状
        
        # キャッシュに保存
        if use_cache and self.cache_file:
            text_hash = self._get_text_hash(text)
            self.embedding_cache[text_hash] = embedding_vector
        
        return embedding_vector
    
    def _get_documents(self, turn: TurnData, top_k: Optional[int] = None, max_docs: int = 100) -> List:
        """
        ターンから文書リストを取得（top_kとmax_docsで制限）
        
        Args:
            turn: TurnDataオブジェクト
            top_k: 上位k件の文書のみを処理（Noneの場合は全件）
            max_docs: 最大文書数（デフォルト: 100）
        
        Returns:
            文書リスト
        """
        documents = turn.documents
        if top_k is not None:
            documents = documents[:top_k]
        documents = documents[:max_docs]  # 最大100件に制限
        return documents

