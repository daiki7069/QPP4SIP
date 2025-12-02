"""
BERT-QPP: Contextualized Pre-trained Transformers for Query Performance Prediction
学習と評価の実装（bi-encoder形式とcross-encoder形式）
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from data_loader import QPPTrainingData, QPPDataLoader

# wandbのインポート（オプション）
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class QPPDataset(Dataset):
    """QPP学習用のデータセット"""
    
    def __init__(self, training_data: List[QPPTrainingData], tokenizer, max_length: int = 512):
        self.training_data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        data = self.training_data[idx]
        return {
            'query': data.query,
            'doc_text': data.doc_text,
            'qpp_score': data.qpp_score,
            'conv_id': data.conv_id,
            'turn_id': data.turn_id
        }


class BERTQPPBiEncoderModel(nn.Module):
    """BERT-QPP Bi-Encoderモデル"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', hidden_size: int = 768):
        super().__init__()
        self.query_encoder = BertModel.from_pretrained(model_name)
        self.doc_encoder = BertModel.from_pretrained(model_name)
        
        # 埋め込みの差からQPPスコアを予測するMLP
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        # クエリとドキュメントをエンコード
        query_outputs = self.query_encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
        doc_outputs = self.doc_encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)
        
        # [CLS]トークンの埋め込みを使用
        query_embedding = query_outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        doc_embedding = doc_outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # 埋め込みの差を使用してQPPスコアを予測
        embedding_diff = query_embedding - doc_embedding
        qpp_score = self.predictor(embedding_diff)  # (batch_size, 1)
        
        return qpp_score.squeeze(-1)  # (batch_size,)


class BERTQPPCrossEncoderModel(nn.Module):
    """BERT-QPP Cross-Encoderモデル"""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1  # 回帰タスク
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits.squeeze(-1)  # (batch_size,)


def collate_fn_bi(batch, tokenizer, max_length: int = 512):
    """Bi-encoder用のcollate関数"""
    queries = [item['query'] for item in batch]
    doc_texts = [item['doc_text'] for item in batch]
    qpp_scores = torch.FloatTensor([item['qpp_score'] for item in batch])
    
    # クエリとドキュメントをトークナイズ
    query_encoded = tokenizer(
        queries,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    doc_encoded = tokenizer(
        doc_texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'query_input_ids': query_encoded['input_ids'],
        'query_attention_mask': query_encoded['attention_mask'],
        'doc_input_ids': doc_encoded['input_ids'],
        'doc_attention_mask': doc_encoded['attention_mask'],
        'qpp_scores': qpp_scores
    }


def collate_fn_cross(batch, tokenizer, max_length: int = 512):
    """Cross-encoder用のcollate関数"""
    queries = [item['query'] for item in batch]
    doc_texts = [item['doc_text'] for item in batch]
    qpp_scores = torch.FloatTensor([item['qpp_score'] for item in batch])
    
    # クエリとドキュメントを結合してトークナイズ
    encoded = tokenizer(
        queries,
        doc_texts,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask'],
        'qpp_scores': qpp_scores
    }


class BERTQPPTrainer:
    """BERT-QPPの学習と評価クラス"""
    
    def __init__(
        self,
        model_type: str = 'bi',  # 'bi' or 'cross'
        model_name: str = 'bert-base-uncased',
        device: Optional[str] = None,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Args:
            model_type: 'bi' (bi-encoder) または 'cross' (cross-encoder)
            model_name: 事前学習済みBERTモデル名
            device: 使用するデバイス
            learning_rate: 学習率
            batch_size: バッチサイズ
            max_length: 最大シーケンス長
        """
        self.model_type = model_type
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        
        # トークナイザー
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # モデル
        if model_type == 'bi':
            self.model = BERTQPPBiEncoderModel(model_name).to(self.device)
            self.collate_fn = lambda batch: collate_fn_bi(batch, self.tokenizer, max_length)
        elif model_type == 'cross':
            self.model = BERTQPPCrossEncoderModel(model_name).to(self.device)
            self.collate_fn = lambda batch: collate_fn_cross(batch, self.tokenizer, max_length)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # 損失関数（MSE Loss）
        self.criterion = nn.MSELoss()
        
        # オプティマイザー
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 学習履歴
        self.train_losses = []
        self.dev_losses = []
        self.dev_correlations = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc="Training", leave=False):
            # データをデバイスに移動
            if self.model_type == 'bi':
                query_input_ids = batch['query_input_ids'].to(self.device)
                query_attention_mask = batch['query_attention_mask'].to(self.device)
                doc_input_ids = batch['doc_input_ids'].to(self.device)
                doc_attention_mask = batch['doc_attention_mask'].to(self.device)
                qpp_scores = batch['qpp_scores'].to(self.device)
                
                # 順伝播
                self.optimizer.zero_grad()
                predictions = self.model(
                    query_input_ids, query_attention_mask,
                    doc_input_ids, doc_attention_mask
                )
            else:  # cross
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                qpp_scores = batch['qpp_scores'].to(self.device)
                
                # 順伝播
                self.optimizer.zero_grad()
                predictions = self.model(input_ids, attention_mask)
            
            # 損失計算
            loss = self.criterion(predictions, qpp_scores)
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, dev_loader: DataLoader) -> Tuple[float, float, float]:
        """
        検証データで評価
        
        Returns:
            (平均損失, Pearson相関係数, Spearman相関係数)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc="Evaluating", leave=False):
                # データをデバイスに移動
                if self.model_type == 'bi':
                    query_input_ids = batch['query_input_ids'].to(self.device)
                    query_attention_mask = batch['query_attention_mask'].to(self.device)
                    doc_input_ids = batch['doc_input_ids'].to(self.device)
                    doc_attention_mask = batch['doc_attention_mask'].to(self.device)
                    qpp_scores = batch['qpp_scores'].to(self.device)
                    
                    predictions = self.model(
                        query_input_ids, query_attention_mask,
                        doc_input_ids, doc_attention_mask
                    )
                else:  # cross
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    qpp_scores = batch['qpp_scores'].to(self.device)
                    
                    predictions = self.model(input_ids, attention_mask)
                
                # 損失計算
                loss = self.criterion(predictions, qpp_scores)
                total_loss += loss.item()
                
                # 予測とラベルを保存
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(qpp_scores.cpu().numpy())
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 相関係数を計算
        pearson_corr, _ = pearsonr(all_predictions, all_labels)
        spearman_corr, _ = spearmanr(all_predictions, all_labels)
        
        return avg_loss, pearson_corr, spearman_corr
    
    def train_and_evaluate(
        self,
        train_data: List[QPPTrainingData],
        dev_data: List[QPPTrainingData],
        num_epochs: int = 10,
        save_dir: Optional[str] = None,
        use_wandb: bool = False
    ) -> Dict:
        """
        学習と検証を実行
        
        Args:
            train_data: 学習データ
            dev_data: 検証データ
            num_epochs: エポック数
            save_dir: モデル保存ディレクトリ
        
        Returns:
            学習履歴の辞書
        """
        # データローダーを作成
        train_dataset = QPPDataset(train_data, self.tokenizer, self.max_length)
        dev_dataset = QPPDataset(dev_data, self.tokenizer, self.max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )
        
        # 最良モデルの追跡
        best_dev_corr = -1.0
        best_epoch = 0
        
        # wandbロギング（使用可能な場合）
        if use_wandb and WANDB_AVAILABLE:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent))
            from wandb_utils import log_training_metrics, log_evaluation_metrics, log_best_model
            log_training_fn = log_training_metrics
            log_eval_fn = log_evaluation_metrics
            log_best_fn = log_best_model
        else:
            log_training_fn = None
            log_eval_fn = None
            log_best_fn = None
        
        # 学習ループ
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 学習
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            # wandbに学習メトリクスを記録
            if log_training_fn:
                log_training_fn(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    learning_rate=self.learning_rate
                )
            
            # 検証
            dev_loss, pearson_corr, spearman_corr = self.evaluate(dev_loader)
            self.dev_losses.append(dev_loss)
            self.dev_correlations.append({'pearson': pearson_corr, 'spearman': spearman_corr})
            
            print(f"Dev Loss: {dev_loss:.4f}")
            print(f"Dev Pearson Correlation: {pearson_corr:.4f}")
            print(f"Dev Spearman Correlation: {spearman_corr:.4f}")
            
            # wandbに検証メトリクスを記録
            if log_eval_fn:
                log_eval_fn(
                    epoch=epoch + 1,
                    dev_loss=dev_loss,
                    pearson_corr=pearson_corr,
                    spearman_corr=spearman_corr
                )
            
            # 最良モデルの保存
            if pearson_corr > best_dev_corr:
                best_dev_corr = pearson_corr
                best_epoch = epoch + 1
                
                if save_dir:
                    self.save_model(save_dir, epoch=epoch)
                    print(f"Best model saved (Pearson: {pearson_corr:.4f})")
                
                # wandbに最良モデルの情報を記録
                if log_best_fn:
                    log_best_fn(
                        epoch=best_epoch,
                        best_dev_corr=best_dev_corr,
                        spearman_corr=spearman_corr
                    )
        
        print(f"\nTraining completed. Best model at epoch {best_epoch} (Pearson: {best_dev_corr:.4f})")
        
        return {
            'train_losses': self.train_losses,
            'dev_losses': self.dev_losses,
            'dev_correlations': self.dev_correlations,
            'best_epoch': best_epoch,
            'best_dev_corr': best_dev_corr
        }
    
    def save_model(self, save_dir: str, epoch: int):
        """モデルを保存"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if self.model_type == 'bi':
            torch.save({
                'epoch': epoch,
                'model_type': 'bi',
                'query_encoder': self.model.query_encoder.state_dict(),
                'doc_encoder': self.model.doc_encoder.state_dict(),
                'predictor': self.model.predictor.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, save_path / 'best_model.pth')
        else:  # cross
            torch.save({
                'epoch': epoch,
                'model_type': 'cross',
                'model': self.model.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, save_path / 'best_model.pth')
    
    def load_model(self, model_path: str):
        """モデルを読み込み"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if self.model_type == 'bi':
            self.model.query_encoder.load_state_dict(checkpoint['query_encoder'])
            self.model.doc_encoder.load_state_dict(checkpoint['doc_encoder'])
            self.model.predictor.load_state_dict(checkpoint['predictor'])
        else:  # cross
            self.model.model.load_state_dict(checkpoint['model'])
        
        print(f"Model loaded from {model_path}")

