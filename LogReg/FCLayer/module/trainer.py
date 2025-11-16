"""
全結合層モデルの訓練クラス
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .wandb_utils import (
    log_training_metrics,
    log_validation_metrics,
    log_evaluation_metrics,
    log_roc_curve,
    log_pr_curve,
    log_confusion_matrix
)


class FCTrainer:
    """全結合層モデルの訓練クラス"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        use_class_weights: bool = True,
        class_counts: Optional[Dict[int, int]] = None,
        early_stop_patience: int = 10,
        use_wandb: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stop_patience = early_stop_patience
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # 損失関数（クラス重み付き）
        if use_class_weights and class_counts is not None:
            total = sum(class_counts.values())
            class_weights = torch.tensor([
                total / (len(class_counts) * class_counts[0]),
                total / (len(class_counts) * class_counts[1])
            ], dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # オプティマイザー
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 学習率スケジューラー
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 学習履歴
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self) -> float:
        """1エポックの訓練"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        global_step = len(self.train_losses) * len(self.train_loader)
        
        for batch_idx, (features, labels) in enumerate(tqdm(self.train_loader, desc='Training')):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 順伝播
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # wandbに記録（バッチごと）
            if self.use_wandb:
                current_lr = self.optimizer.param_groups[0]['lr']
                log_training_metrics(
                    step=global_step + batch_idx,
                    loss=loss.item(),
                    learning_rate=current_lr
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, Dict]:
        """検証"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc='Validating'):
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, dim=1)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # クラス1の確率
        
        avg_loss = total_loss / num_batches
        
        # メトリクス計算
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_predictions),
            'f1': f1_score(all_labels, all_predictions),
            'auc': roc_auc_score(all_labels, all_probabilities),
            'ap': average_precision_score(all_labels, all_probabilities)
        }
        
        return avg_loss, metrics
    
    def train(self) -> Tuple[nn.Module, List[float], List[float]]:
        """モデルを訓練する"""
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # 訓練
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 検証
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            
            # 学習率スケジューラー更新
            self.scheduler.step(val_loss)
            
            # wandbに記録（エポックごと）
            # ステップはバッチ数に基づいて計算（バッチごとのログと重複しないように）
            if self.use_wandb:
                epoch_step = (epoch + 1) * len(self.train_loader)
                log_validation_metrics(
                    epoch=epoch + 1,
                    val_loss=val_loss,
                    metrics=val_metrics,
                    step=epoch_step
                )
            
            print(f'Epoch {epoch+1}/{self.num_epochs} - '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_metrics["accuracy"]:.4f}, '
                  f'Val F1: {val_metrics["f1"]:.4f}, '
                  f'Val AUC: {val_metrics["auc"]:.4f}')
            
            # ベストモデルの保存
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # ベストモデルをロード
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        return self.model, self.train_losses, self.val_losses
    
    def evaluate(
        self,
        data_loader: DataLoader,
        split: str = "test"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        モデルを評価する
        
        Args:
            data_loader: データローダー
            split: データセット分割名（"train" or "test"）
        
        Returns:
            (predictions, probabilities)
        """
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(data_loader, desc=f'Evaluating {split}'):
                features = features.to(self.device)
                
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # クラス1の確率
                all_labels.extend(labels.cpu().numpy())
        
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        labels = np.array(all_labels)
        
        # メトリクス計算
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score, average_precision_score,
            precision_score, recall_score
        )
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions),
            'recall': recall_score(labels, predictions),
            'f1': f1_score(labels, predictions),
            'auc': roc_auc_score(labels, probabilities),
            'ap': average_precision_score(labels, probabilities)
        }
        
        print(f'\n{split.upper()} Metrics:')
        for key, value in metrics.items():
            print(f'  {key}: {value:.4f}')
        
        # wandbに記録
        if self.use_wandb:
            log_evaluation_metrics(epoch=0, eval_results=metrics)
            log_roc_curve(labels, probabilities, split=split, auc=metrics['auc'])
            log_pr_curve(labels, probabilities, split=split, ap=metrics['ap'])
            log_confusion_matrix(labels, predictions, split=split)
        
        return predictions, probabilities
    
    def plot_loss_curves(self, output_dir: Path):
        """学習曲線を描画して保存"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        loss_curve_path = output_dir / 'loss_curves.png'
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        print(f"  - 学習曲線を保存しました: {loss_curve_path}")
        plt.close()
        
        # wandbに記録
        if self.use_wandb:
            try:
                wandb.log({
                    "loss_curves": wandb.Image(str(loss_curve_path))
                })
            except Exception as e:
                print(f"警告: wandbへの学習曲線の記録に失敗しました: {e}")

