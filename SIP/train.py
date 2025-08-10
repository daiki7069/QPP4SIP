import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import os
import json
from tqdm import tqdm
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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
        self.dev_f1s = []
        
        # 結果保存用ディレクトリ
        self.results_dir = os.path.join(save_dir, 'results')
        os.makedirs(self.results_dir, exist_ok=True)
        
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
        all_probabilities = []
        all_dialogue_histories = []
        all_dialogue_ids = []
        all_turn_indices = []
        
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
                all_probabilities.extend(probs.cpu().numpy())
                
                # 対話履歴とメタデータも保存
                if 'dialogue_history' in batch:
                    all_dialogue_histories.extend(batch['dialogue_history'])
                if 'dialogue_id' in batch:
                    all_dialogue_ids.extend(batch['dialogue_id'])
                if 'turn_index' in batch:
                    all_turn_indices.extend(batch['turn_index'])
        
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
            'labels': all_labels,
            'probabilities': all_probabilities,
            'dialogue_histories': all_dialogue_histories,
            'dialogue_ids': all_dialogue_ids,
            'turn_indices': all_turn_indices
        }
    
    def save_epoch_model(self, epoch, dev_results, is_best=False):
        """エポックごとのモデル保存"""
        # エポックごとのモデル保存
        epoch_model_path = os.path.join(self.save_dir, f'epoch_{epoch+1:03d}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dev_f1': dev_results['f1'],
            'dev_results': dev_results,
            'class_weights': self.class_weights if self.use_class_weights else None,
            'timestamp': datetime.now().isoformat()
        }, epoch_model_path)
        
        if is_best:
            # ベストモデルも別途保存
            best_model_path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'dev_f1': dev_results['f1'],
                'dev_results': dev_results,
                'class_weights': self.class_weights if self.use_class_weights else None,
                'timestamp': datetime.now().isoformat()
            }, best_model_path)
            print(f"New best model saved with F1: {dev_results['f1']:.4f}")
        
        print(f"Epoch {epoch+1} model saved: {epoch_model_path}")
    
    def save_epoch_results(self, epoch, train_loss, dev_results):
        """エポックごとの結果保存"""
        epoch_results = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'train_loss': train_loss,
            'dev_loss': dev_results['loss'],
            'dev_accuracy': dev_results['accuracy'],
            'dev_precision': dev_results['precision'],
            'dev_recall': dev_results['recall'],
            'dev_f1': dev_results['f1']
        }
        
        # JSON形式で保存
        epoch_json_path = os.path.join(self.results_dir, f'epoch_{epoch+1:03d}_results.json')
        with open(epoch_json_path, 'w', encoding='utf-8') as f:
            json.dump(epoch_results, f, indent=2, ensure_ascii=False)
        
        # CSV形式で保存（学習履歴の追記）
        epoch_csv_path = os.path.join(self.results_dir, 'training_history.csv')
        epoch_df = pd.DataFrame([epoch_results])
        
        if os.path.exists(epoch_csv_path):
            # 既存ファイルに追記
            epoch_df.to_csv(epoch_csv_path, mode='a', header=False, index=False)
        else:
            # 新規ファイル作成
            epoch_df.to_csv(epoch_csv_path, index=False)
    
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
            self.dev_f1s.append(dev_results['f1'])
            
            print(f"Dev Loss: {dev_results['loss']:.4f}")
            print(f"Dev Accuracy: {dev_results['accuracy']:.4f}")
            print(f"Dev F1: {dev_results['f1']:.4f}")
            
            # エポックごとの結果保存
            self.save_epoch_results(epoch, train_loss, dev_results)
            
            # エポックごとのモデル保存
            is_best = dev_results['f1'] > best_f1
            if is_best:
                best_f1 = dev_results['f1']
            
            self.save_epoch_model(epoch, dev_results, is_best)
        
        # 学習履歴の保存
        history = {
            'train_losses': self.train_losses,
            'dev_losses': self.dev_losses,
            'dev_accuracies': self.dev_accuracies,
            'dev_f1s': self.dev_f1s,
            'use_class_weights': self.use_class_weights,
            'num_classes': self.num_classes,
            'final_timestamp': datetime.now().isoformat()
        }
        
        # JSON形式で保存
        history_json_path = os.path.join(self.results_dir, 'training_history.json')
        with open(history_json_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        # 学習曲線のプロット
        self.plot_training_curves()
        
        print(f"\n学習完了！結果は {self.results_dir} に保存されました。")
    
    def plot_training_curves(self):
        """学習曲線のプロット"""
        plt.figure(figsize=(15, 5))
        
        # 損失のプロット
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses, label='Train Loss', color='blue')
        plt.plot(self.dev_losses, label='Dev Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 精度のプロット
        plt.subplot(1, 3, 2)
        plt.plot(self.dev_accuracies, label='Dev Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        # F1スコアのプロット
        plt.subplot(1, 3, 3)
        plt.plot(self.dev_f1s, label='Dev F1', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # 画像として保存
        plot_path = os.path.join(self.results_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"学習曲線を保存しました: {plot_path}")
    
    def detailed_error_analysis(self, test_results):
        """詳細なエラー分析"""
        print("\n=== 詳細エラー分析 ===")
        
        # 混同行列
        cm = confusion_matrix(test_results['labels'], test_results['predictions'])
        print(f"混同行列:")
        print(f"          予測")
        print(f"         No_Init  Init")
        print(f"実際 No_Init  {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"     Init     {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # 混同行列の可視化
        self.plot_confusion_matrix(cm)
        
        # 確率分布の統計
        probabilities = np.array(test_results['probabilities'])
        print(f"\n確率分布の統計:")
        print(f"SIP=0 (No Initiative): 平均={probabilities[:, 0].mean():.3f}, 標準偏差={probabilities[:, 0].std():.3f}")
        print(f"SIP=1 (Initiative):    平均={probabilities[:, 1].mean():.3f}, 標準偏差={probabilities[:, 1].std():.3f}")
        
        # 確率分布の可視化
        self.plot_probability_distributions(probabilities)
        
        # 確信度の高い予測の分析
        confidence_threshold = 0.8
        high_conf_predictions = probabilities.max(axis=1) > confidence_threshold
        high_conf_correct = np.array(test_results['predictions'])[high_conf_predictions] == np.array(test_results['labels'])[high_conf_predictions]
        
        print(f"\n高確信度予測 (>0.8): {high_conf_predictions.sum()}/{len(probabilities)}")
        if high_conf_predictions.sum() > 0:
            print(f"高確信度予測の正解率: {high_conf_correct.mean():.3f}")
        
        # 誤分類の詳細分析
        misclassified = np.array(test_results['predictions']) != np.array(test_results['labels'])
        if misclassified.sum() > 0:
            print(f"\n誤分類サンプル数: {misclassified.sum()}")
            
            # 誤分類の確率分布
            misclassified_probs = probabilities[misclassified]
            print(f"誤分類の確率分布:")
            print(f"  平均確信度: {misclassified_probs.max(axis=1).mean():.3f}")
            print(f"  最小確信度: {misclassified_probs.max(axis=1).min():.3f}")
            
            # 誤分類の対話履歴例（最初の5件）
            if 'dialogue_histories' in test_results and len(test_results['dialogue_histories']) > 0:
                print(f"\n誤分類の対話履歴例（最初の5件）:")
                for i in range(min(5, misclassified.sum())):
                    idx = np.where(misclassified)[0][i]
                    true_label = test_results['labels'][idx]
                    pred_label = test_results['predictions'][idx]
                    prob = probabilities[idx]
                    dialogue_history = test_results['dialogue_histories'][idx] if idx < len(test_results['dialogue_histories']) else "N/A"
                    
                    print(f"  サンプル {i+1}:")
                    print(f"    真のラベル: {true_label} ({'No Initiative' if true_label == 0 else 'Initiative'})")
                    print(f"    予測ラベル: {pred_label} ({'No Initiative' if pred_label == 0 else 'Initiative'})")
                    print(f"    予測確率: SIP=0: {prob[0]:.3f}, SIP=1: {prob[1]:.3f}")
                    print(f"    対話履歴: {dialogue_history[:100]}{'...' if len(dialogue_history) > 100 else ''}")
                    print()
    
    def plot_confusion_matrix(self, cm):
        """混同行列の可視化"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Initiative', 'Initiative'],
                    yticklabels=['No Initiative', 'Initiative'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 画像として保存
        cm_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"混同行列を保存しました: {cm_path}")
    
    def plot_probability_distributions(self, probabilities):
        """確率分布の可視化"""
        plt.figure(figsize=(15, 5))
        
        # SIP=0の確率分布
        plt.subplot(1, 3, 1)
        plt.hist(probabilities[:, 0], bins=20, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Probability SIP=0')
        plt.ylabel('Frequency')
        plt.title('Distribution of SIP=0 Probabilities')
        plt.grid(True, alpha=0.3)
        
        # SIP=1の確率分布
        plt.subplot(1, 3, 2)
        plt.hist(probabilities[:, 1], bins=20, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Probability SIP=1')
        plt.ylabel('Frequency')
        plt.title('Distribution of SIP=1 Probabilities')
        plt.grid(True, alpha=0.3)
        
        # 予測確信度の分布
        plt.subplot(1, 3, 3)
        confidence = np.max(probabilities, axis=1)
        plt.hist(confidence, bins=20, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Confidence')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 画像として保存
        prob_path = os.path.join(self.results_dir, 'probability_distributions.png')
        plt.savefig(prob_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"確率分布を保存しました: {prob_path}")
    
    def save_detailed_results(self, test_results, save_path):
        """詳細な結果をCSVファイルに保存"""
        results_df = pd.DataFrame({
            'dialogue_id': test_results.get('dialogue_ids', ['N/A'] * len(test_results['predictions'])),
            'turn_index': test_results.get('turn_indices', ['N/A'] * len(test_results['predictions'])),
            'dialogue_history': test_results.get('dialogue_histories', ['N/A'] * len(test_results['predictions'])),
            'true_label': test_results['labels'],
            'predicted_label': test_results['predictions'],
            'prob_sip_0': [prob[0] for prob in test_results['probabilities']],
            'prob_sip_1': [prob[1] for prob in test_results['probabilities']],
            'prediction_confidence': [max(prob) for prob in test_results['probabilities']],
            'is_correct': [pred == true for pred, true in zip(test_results['predictions'], test_results['labels'])]
        })
        
        results_df.to_csv(save_path, index=False)
        print(f"詳細結果を保存しました: {save_path}")
        
        # JSON形式でも保存
        json_path = save_path.replace('.csv', '.json')
        results_dict = {
            'summary': {
                'total_samples': len(results_df),
                'correct_predictions': results_df['is_correct'].sum(),
                'incorrect_predictions': (~results_df['is_correct']).sum(),
                'accuracy': results_df['is_correct'].mean(),
                'timestamp': datetime.now().isoformat()
            },
            'class_breakdown': {
                'sip_0': {
                    'total': len(results_df[results_df['true_label'] == 0]),
                    'correct': len(results_df[(results_df['true_label'] == 0) & (results_df['is_correct'] == True)]),
                    'accuracy': len(results_df[(results_df['true_label'] == 0) & (results_df['is_correct'] == True)]) / len(results_df[results_df['true_label'] == 0]) if len(results_df[results_df['true_label'] == 0]) > 0 else 0
                },
                'sip_1': {
                    'total': len(results_df[results_df['true_label'] == 1]),
                    'correct': len(results_df[(results_df['true_label'] == 1) & (results_df['is_correct'] == True)]),
                    'accuracy': len(results_df[(results_df['true_label'] == 1) & (results_df['is_correct'] == True)]) / len(results_df[results_df['true_label'] == 1]) if len(results_df[results_df['true_label'] == 1]) > 0 else 0
                }
            },
            'detailed_results': results_df.to_dict('records')
        }
        
        # numpyの型を標準のPython型に変換
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj

        # JSON保存前に変換
        results_dict = convert_numpy_types(results_dict)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"詳細結果（JSON）を保存しました: {json_path}")
    
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
        
        # 詳細なエラー分析
        self.detailed_error_analysis(test_results)
        
        # 詳細結果をCSVファイルに保存
        detailed_results_path = os.path.join(self.results_dir, 'test_detailed_results.csv')
        self.save_detailed_results(test_results, detailed_results_path)
        
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