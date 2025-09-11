import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import os
import json
from tqdm import tqdm
import argparse

from model.SIPModel import SIPRecognizer
from dataset.dataset import create_data_loaders
from model.util import create_class_weights, get_loss

class SIPTrainer:
    """
    SIP予測モデルの学習クラス
    """
    def __init__(self, model, train_loader, dev_loader, test_loader, 
                 learning_rate=2e-5, device='cuda', use_class_weights=True,
                 class_counts=None, save_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.use_class_weights = use_class_weights
        self.num_classes = model.num_initiatives
        self.save_dir = save_dir
        
        # クラス重みの設定
        if use_class_weights and class_counts is not None:
            self.class_weights = create_class_weights(class_counts, method='balanced')
            self.class_weights = self.class_weights.to(device)
            print(f"Using class weights for {len(class_counts)} classes:")
            for i, weight in enumerate(self.class_weights):
                print(f"  w{i}: {weight:.3f}")
        else:
            self.class_weights = None
            print("Not using class weights")
        
        # オプティマイザー
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 学習履歴
        self.train_losses = []
        self.dev_losses = []
        self.dev_accuracies = []
        
    def train_epoch(self):
        """1エポックの学習"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # 順伝播
            self.optimizer.zero_grad()
            probs, logits = self.model(input_ids, attention_mask)
            
            # utilのget_loss関数を使用
            loss = get_loss(logits, labels, self.class_weights)
            
            # 逆伝播
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, data_loader, split_name="dev"):
        """評価"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating {split_name}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                probs, logits = self.model(input_ids, attention_mask)
                
                # 評価時は重みなし損失を使用（公平な比較のため）
                loss = get_loss(logits, labels, None)
                
                total_loss += loss.item()
                
                # 予測とラベルを保存
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self, num_epochs=10, save_dir=None):
        """学習の実行"""
        if save_dir is None:
            save_dir = self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        best_f1 = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # 学習
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # 開発データで評価
            dev_results = self.evaluate(self.dev_loader, "dev")
            self.dev_losses.append(dev_results['loss'])
            self.dev_accuracies.append(dev_results['accuracy'])
            
            print(f"Dev Loss: {dev_results['loss']:.4f}")
            print(f"Dev Accuracy: {dev_results['accuracy']:.4f}")
            print(f"Dev F1: {dev_results['f1']:.4f}")
            
            # ベストモデルの保存
            if dev_results['f1'] > best_f1:
                best_f1 = dev_results['f1']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'dev_f1': best_f1,
                    'dev_results': dev_results,
                    'class_weights': self.class_weights if self.use_class_weights else None
                }, os.path.join(save_dir, 'best_model.pth'))
                print(f"New best model saved with F1: {best_f1:.4f}")
        
        # 学習履歴の保存
        history = {
            'train_losses': self.train_losses,
            'dev_losses': self.dev_losses,
            'dev_accuracies': self.dev_accuracies,
            'use_class_weights': self.use_class_weights,
            'num_classes': self.num_classes
        }
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    def test(self, model_path=None):
        """テストデータでの評価"""
        # デフォルトパスを設定
        if model_path is None:
            model_path = os.path.join(self.save_dir, 'best_model.pth')
        
        # ベストモデルを読み込み
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # テスト評価
        test_results = self.evaluate(self.test_loader, "test")
        
        print("\n=== Test Results ===")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        print(f"F1: {test_results['f1']:.4f}")
        
        # 詳細な分類レポート
        print("\nClassification Report:")
        print(classification_report(
            test_results['labels'], 
            test_results['predictions'],
            target_names=['No Initiative', 'Initiative']
        ))
        
        return test_results

def main():
    parser = argparse.ArgumentParser(description='SIP Model Training')
    parser.add_argument('--train_csv', default='data/train_sip.csv', help='Training data CSV')
    parser.add_argument('--dev_csv', default='data/dev_sip.csv', help='Development data CSV')
    parser.add_argument('--test_csv', default='data/test_sip.csv', help='Test data CSV')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--save_dir', default='/mnt/disk6/daiki/Models/Mix-Initiative-Dialogue/checkpoints', help='Save directory')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')
    parser.add_argument('--class_counts', nargs='+', type=int, default=[2309, 357], 
                       help='Number of samples for each class (e.g., 2309 357 for 2 classes)')
    parser.add_argument('--num_initiatives', type=int, default=2, help='Number of initiative classes')
    
    args = parser.parse_args()
    
    # クラス数の検証
    if len(args.class_counts) != args.num_initiatives:
        raise ValueError(f"Number of class counts ({len(args.class_counts)}) must match num_initiatives ({args.num_initiatives})")
    
    # データローダーを作成
    train_loader, dev_loader, test_loader = create_data_loaders(
        args.train_csv, args.dev_csv, args.test_csv,
        batch_size=args.batch_size, max_length=args.max_length
    )
    
    # モデルを作成
    model = SIPRecognizer(num_initiatives=args.num_initiatives)
    
    # トレーナーを作成
    trainer = SIPTrainer(
        model, train_loader, dev_loader, test_loader,
        learning_rate=args.learning_rate, device=args.device,
        use_class_weights=args.use_class_weights,
        class_counts=args.class_counts,
        save_dir=args.save_dir
    )
    
    # 学習
    trainer.train(num_epochs=args.num_epochs)
    
    # テスト
    trainer.test()

if __name__ == "__main__":
    main() 