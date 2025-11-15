#!/usr/bin/env python3
"""
Clarification予測モデルのFine Tuning
queryを入力として、response_typeがclarifyかどうかを予測する
"""
import argparse
import json
import os
from typing import Dict, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import StratifiedKFold

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)

from dataset import load_data, ClarificationDataset
from wandb_utils import init_wandb, log_evaluation_metrics, create_config_dict, get_experiment_name, log_confusion_matrix, log_roc_curve
from trainer import WeightedTrainer


def compute_metrics(eval_pred):
    """
    評価指標の計算
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train(args):
    """
    訓練フェーズ
    """
    print("=" * 50)
    print("訓練フェーズ")
    print("=" * 50)
    
    # データの読み込み
    print("データを読み込んでいます...")
    train_data = load_data(args.train_path)
    dev_data = load_data(args.dev_path)
    
    print(f"訓練データ数: {len(train_data)}")
    print(f"開発データ数: {len(dev_data)}")
    
    # ラベルの分布を確認
    from dataset import is_clarification
    train_clarify_count = sum(1 for item in train_data if is_clarification(item['response_type']))
    dev_clarify_count = sum(1 for item in dev_data if is_clarification(item['response_type']))
    train_not_clarify_count = len(train_data) - train_clarify_count
    print(f"訓練データ - Clarification: {train_clarify_count}, その他: {train_not_clarify_count}")
    print(f"開発データ - Clarification: {dev_clarify_count}, その他: {len(dev_data) - dev_clarify_count}")
    
    # クラス重みの計算
    class_weights = None
    if args.use_class_weights:
        # クラス重みを計算（逆頻度ベース）
        total = len(train_data)
        weight_not_clarify = total / (2.0 * train_not_clarify_count) if train_not_clarify_count > 0 else 1.0
        weight_clarify = total / (2.0 * train_clarify_count) if train_clarify_count > 0 else 1.0
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = torch.tensor([weight_not_clarify, weight_clarify], dtype=torch.float32).to(device)
        
        print(f"\nクラス重みを適用します:")
        print(f"  Not Clarification (0): {weight_not_clarify:.4f}")
        print(f"  Clarification (1): {weight_clarify:.4f}")
    else:
        print("\nクラス重みは使用しません")
    
    # トークナイザーとモデルの読み込み
    print(f"モデルを読み込んでいます: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2
    )
    
    # データセットの作成
    train_dataset = ClarificationDataset(train_data, tokenizer, max_length=args.max_length)
    dev_dataset = ClarificationDataset(dev_data, tokenizer, max_length=args.max_length)
    
    # wandbの初期化
    wandb_run = None
    if args.use_wandb:
        config_dict = create_config_dict(args)
        experiment_name = get_experiment_name(args)
        wandb_mode = os.getenv("WANDB_MODE", "online")
        wandb_run = init_wandb(
            project_name=args.wandb_project,
            experiment_name=experiment_name,
            config_dict=config_dict,
            mode=wandb_mode
        )
        print(f"wandbを初期化しました: プロジェクト={args.wandb_project}, 実験名={experiment_name}")
    
    # 訓練引数の設定
    report_to = ["wandb"] if args.use_wandb else ["tensorboard"]
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, 'logs'),
        logging_steps=args.logging_steps,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        fp16=args.fp16,
        seed=args.seed,
        report_to=report_to,
    )
    
    # Trainerの作成
    trainer_class = WeightedTrainer if args.use_class_weights else Trainer
    trainer = trainer_class(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)] if args.early_stopping else None,
    )
    
    # 訓練の実行
    print("訓練を開始します...")
    trainer.train()
    
    # ベストモデルの保存
    trainer.save_model(os.path.join(args.output_dir, 'best_model'))
    tokenizer.save_pretrained(os.path.join(args.output_dir, 'best_model'))
    print(f"モデルを保存しました: {os.path.join(args.output_dir, 'best_model')}")
    
    # 最終評価
    print("\n最終評価結果:")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # wandbに最終評価結果を記録
    if args.use_wandb and wandb_run:
        log_evaluation_metrics(args.num_epochs, eval_results)
        wandb_run.finish()


def train_kfold(args):
    """
    K-fold交差検証による訓練とOut-of-Fold予測
    訓練データ全体に対してリークなしで予測確率を付与
    """
    print("=" * 50)
    print("K-fold交差検証による訓練")
    print("=" * 50)
    
    # データの読み込み
    print("データを読み込んでいます...")
    # 元のJSON構造を保持するために、元のファイルを直接読み込む
    with open(args.train_path, 'r', encoding='utf-8') as f:
        original_train_data = json.load(f)
    
    # フラット化されたデータで訓練を実行
    train_data = load_data(args.train_path)
    
    print(f"訓練データ数: {len(train_data)}")
    
    # ラベルの抽出
    from dataset import is_clarification
    labels = [1 if is_clarification(item['response_type']) else 0 for item in train_data]
    train_clarify_count = sum(labels)
    train_not_clarify_count = len(train_data) - train_clarify_count
    print(f"訓練データ - Clarification: {train_clarify_count}, その他: {train_not_clarify_count}")
    
    # K-fold分割
    kfold = StratifiedKFold(n_splits=args.k_fold, shuffle=True, random_state=args.seed)
    
    # 全foldの予測確率を保存するリスト
    all_predictions = [None] * len(train_data)
    
    # 各foldで訓練と予測
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(train_data, labels)):
        print(f"\n{'='*50}")
        print(f"Fold {fold_idx + 1}/{args.k_fold}")
        print(f"{'='*50}")
        
        # foldごとのデータ分割
        fold_train_data = [train_data[i] for i in train_indices]
        fold_val_data = [train_data[i] for i in val_indices]
        fold_train_labels = [labels[i] for i in train_indices]
        fold_val_labels = [labels[i] for i in val_indices]
        
        print(f"Fold訓練データ数: {len(fold_train_data)}")
        print(f"Fold検証データ数: {len(fold_val_data)}")
        
        # クラス重みの計算（fold訓練データから）
        class_weights = None
        if args.use_class_weights:
            fold_train_clarify_count = sum(fold_train_labels)
            fold_train_not_clarify_count = len(fold_train_data) - fold_train_clarify_count
            total = len(fold_train_data)
            weight_not_clarify = total / (2.0 * fold_train_not_clarify_count) if fold_train_not_clarify_count > 0 else 1.0
            weight_clarify = total / (2.0 * fold_train_clarify_count) if fold_train_clarify_count > 0 else 1.0
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            class_weights = torch.tensor([weight_not_clarify, weight_clarify], dtype=torch.float32).to(device)
        
        # トークナイザーとモデルの読み込み
        print(f"モデルを読み込んでいます: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2
        )
        
        # データセットの作成
        fold_train_dataset = ClarificationDataset(fold_train_data, tokenizer, max_length=args.max_length)
        fold_val_dataset = ClarificationDataset(fold_val_data, tokenizer, max_length=args.max_length)
        
        # fold専用の出力ディレクトリ
        fold_output_dir = os.path.join(args.output_dir, f'fold_{fold_idx + 1}')
        os.makedirs(fold_output_dir, exist_ok=True)
        
        # 訓練引数の設定
        training_args = TrainingArguments(
            output_dir=fold_output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_dir=os.path.join(fold_output_dir, 'logs'),
            logging_steps=args.logging_steps,
            eval_strategy='epoch',
            save_strategy='epoch',
            save_total_limit=1,  # foldごとは1つだけ保存
            load_best_model_at_end=True,
            metric_for_best_model='f1',
            greater_is_better=True,
            fp16=args.fp16,
            seed=args.seed + fold_idx,  # foldごとに異なるシード
            report_to=[],  # foldごとのwandbログは無効化（オプションで有効化可能）
        )
        
        # Trainerの作成
        trainer_class = WeightedTrainer if args.use_class_weights else Trainer
        trainer = trainer_class(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=fold_train_dataset,
            eval_dataset=fold_val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)] if args.early_stopping else None,
        )
        
        # 訓練の実行
        print(f"Fold {fold_idx + 1}の訓練を開始します...")
        trainer.train()
        
        # ベストモデルの保存
        best_model_path = os.path.join(fold_output_dir, 'best_model')
        trainer.save_model(best_model_path)
        tokenizer.save_pretrained(best_model_path)
        print(f"Fold {fold_idx + 1}のベストモデルを保存しました")
        
        # ベストモデルをロード
        model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
        
        # ベストモデルで検証データに対して予測
        print(f"Fold {fold_idx + 1}の検証データに対して予測を実行...")
        val_loader = DataLoader(fold_val_dataset, batch_size=args.batch_size, shuffle=False)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        fold_predictions = []
        fold_logits_list = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                batch_logits = logits.cpu().numpy()
                
                fold_predictions.extend(batch_probs)
                fold_logits_list.extend(batch_logits)
        
        # 予測結果を保存（元のインデックスに対応）
        for val_idx, (pred, logit) in zip(val_indices, zip(fold_predictions, fold_logits_list)):
            all_predictions[val_idx] = {'prob': pred, 'logit': logit}
        
        print(f"Fold {fold_idx + 1}完了")
    
    # 全foldの予測結果を結合して保存
    print(f"\n{'='*50}")
    print("Out-of-Fold予測結果の保存")
    print(f"{'='*50}")
    
    # 予測結果を元のJSON構造に追加
    flat_idx = 0
    for conversation in original_train_data:
        for turn in conversation:
            if flat_idx < len(train_data) and all_predictions[flat_idx] is not None:
                pred_dict = all_predictions[flat_idx]
                prob = pred_dict['prob']
                logit = pred_dict['logit']
                # 予測結果を追加
                turn['prob_not_clarification'] = float(prob[0])
                turn['prob_clarification'] = float(prob[1])
                turn['logit_not_clarification'] = float(logit[0])
                turn['logit_clarification'] = float(logit[1])
                turn['predicted_label'] = int(np.argmax(prob))
                flat_idx += 1
    
    # JSONファイルとして保存
    output_filename = os.path.basename(args.train_path).replace('.json', '_with_predictions.json')
    output_json_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(original_train_data, f, ensure_ascii=False, indent=2)
    
    print(f"予測結果を追加したJSONファイルを保存しました: {output_json_path}")
    print(f"訓練データ数: {len(train_data)}")
    print(f"予測結果数: {len([p for p in all_predictions if p is not None])}")
    
    # TSV形式も保存（後方互換性のため）
    output_path = os.path.join(args.output_dir, 'oof_predictions.txt')
    with open(output_path, 'w', encoding='utf-8') as f:
        # ヘッダー行
        f.write("query\ttrue_label\tpredicted_label\tprob_not_clarification\tprob_clarification\tlogit_not_clarification\tlogit_clarification\ttrue_response_type\n")
        
        # データ行
        for i, item in enumerate(train_data):
            if all_predictions[i] is not None:
                pred_dict = all_predictions[i]
                prob = pred_dict['prob']
                logit = pred_dict['logit']
                query = item['query'].replace('\t', ' ').replace('\n', ' ')
                true_label = labels[i]
                predicted_label = int(np.argmax(prob))
                prob_not_clarification = float(prob[0])
                prob_clarification = float(prob[1])
                logit_not_clarification = float(logit[0])
                logit_clarification = float(logit[1])
                true_response_type = item['response_type'].replace('\t', ' ').replace('\n', ' ')
                
                f.write(f"{query}\t{true_label}\t{predicted_label}\t{prob_not_clarification:.6f}\t{prob_clarification:.6f}\t{logit_not_clarification:.6f}\t{logit_clarification:.6f}\t{true_response_type}\n")
    
    print(f"TSV形式のOut-of-Fold予測結果も保存しました: {output_path}")


def find_kfold_models(model_dir):
    """
    K-foldディレクトリから全foldのモデルパスを検出
    
    Args:
        model_dir: モデルディレクトリのパス
    
    Returns:
        fold_models: fold番号とモデルパスのリスト [(fold_num, model_path), ...]
    """
    fold_models = []
    
    # ディレクトリ内のfold_*ディレクトリを検索
    if not os.path.exists(model_dir):
        return fold_models
    
    for item in os.listdir(model_dir):
        if item.startswith('fold_'):
            fold_dir = os.path.join(model_dir, item)
            if os.path.isdir(fold_dir):
                # best_modelディレクトリを確認
                best_model_path = os.path.join(fold_dir, 'best_model')
                if os.path.exists(best_model_path):
                    try:
                        fold_num = int(item.split('_')[1])
                        fold_models.append((fold_num, best_model_path))
                    except ValueError:
                        continue
    
    # fold番号でソート
    fold_models.sort(key=lambda x: x[0])
    return fold_models


def evaluate(args):
    """
    評価フェーズ
    """
    print("=" * 50)
    print("評価フェーズ")
    print("=" * 50)
    
    # wandbの初期化
    if args.use_wandb:
        config_dict = create_config_dict(args)
        experiment_name = get_experiment_name(args) + "_eval"
        wandb_mode = os.getenv("WANDB_MODE", "online")
        init_wandb(
            project_name=args.wandb_project,
            experiment_name=experiment_name,
            config_dict=config_dict,
            mode=wandb_mode
        )
        print(f"wandbを初期化しました: プロジェクト={args.wandb_project}, 実験名={experiment_name}")
    
    # データの読み込み
    print("データを読み込んでいます...")
    # 元のJSON構造を保持するために、元のファイルを直接読み込む
    with open(args.dev_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    # フラット化されたデータで予測を実行
    dev_data = load_data(args.dev_path)
    print(f"評価データ数: {len(dev_data)}")
    
    # K-foldモデルの検出
    model_path = args.model_path if args.model_path else args.output_dir
    fold_models = find_kfold_models(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if fold_models:
        # K-foldアンサンブル評価
        print(f"\nK-foldアンサンブル評価を実行します（{len(fold_models)} folds）")
        
        # 全foldのモデルを読み込み
        models = []
        tokenizers = []
        for fold_num, fold_model_path in fold_models:
            print(f"  Fold {fold_num}のモデルを読み込んでいます: {fold_model_path}")
            tokenizer = AutoTokenizer.from_pretrained(fold_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(fold_model_path)
            model.eval()
            model.to(device)
            models.append(model)
            tokenizers.append(tokenizer)
        
        # 最初のfoldのトークナイザーを使用（全foldで同じトークナイザーを使用）
        tokenizer = tokenizers[0]
        
        # データセットの作成
        eval_dataset = ClarificationDataset(dev_data, tokenizer, max_length=args.max_length)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 全foldで予測を実行
        print("全foldで予測を実行しています...")
        all_fold_probs = []
        all_fold_logits = []
        
        for fold_idx, model in enumerate(models):
            print(f"  Fold {fold_models[fold_idx][0]}で予測中...")
            fold_probs = []
            fold_logits = []
            
            with torch.no_grad():
                for batch in eval_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                    batch_logits_np = logits.cpu().numpy()
                    fold_probs.append(batch_probs)
                    fold_logits.append(batch_logits_np)
            
            # バッチを結合
            fold_probs = np.concatenate(fold_probs, axis=0)
            fold_logits = np.concatenate(fold_logits, axis=0)
            all_fold_probs.append(fold_probs)
            all_fold_logits.append(fold_logits)
        
        # 確率とlogitの平均を計算（アンサンブル）
        print("アンサンブル予測を計算しています...")
        ensemble_probs = np.mean(all_fold_probs, axis=0)
        ensemble_logits = np.mean(all_fold_logits, axis=0)
        
        # ラベルの取得
        labels = []
        for item in dev_data:
            from dataset import is_clarification
            labels.append(1 if is_clarification(item['response_type']) else 0)
        labels = np.array(labels)
        
        # 予測ラベルの計算
        predictions = np.argmax(ensemble_probs, axis=1)
        probabilities_list = ensemble_probs.tolist()
        logits_list = ensemble_logits.tolist()
        
    else:
        # 通常の単一モデル評価
        model_path = args.model_path if args.model_path else os.path.join(args.output_dir, 'best_model')
        print(f"モデルを読み込んでいます: {model_path}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        model.to(device)
        
        # データセットの作成
        eval_dataset = ClarificationDataset(dev_data, tokenizer, max_length=args.max_length)
        eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
        
        # 予測
        print("予測を実行しています...")
        predictions = []
        labels = []
        probabilities_list = []
        logits_list = []
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels = batch['labels'].numpy()
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
                batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                batch_logits_np = logits.cpu().numpy()
                
                predictions.extend(batch_predictions)
                labels.extend(batch_labels)
                probabilities_list.extend(batch_probs)
                logits_list.extend(batch_logits_np)
        
        labels = np.array(labels)
        predictions = np.array(predictions)
    
    # 評価指標の計算
    accuracy = accuracy_score(labels, predictions)
    precision_score, recall_score, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    
    # AUCの計算（Clarificationクラスの確率を使用）
    prob_clarification = [prob[1] for prob in probabilities_list]
    auc = roc_auc_score(labels, prob_clarification)
    
    # Average Precisionの計算
    ap = average_precision_score(labels, prob_clarification)
    
    # ROC曲線の計算
    fpr, tpr, thresholds = roc_curve(labels, prob_clarification)
    
    # Precision-Recall曲線の計算
    precision, recall, pr_thresholds = precision_recall_curve(labels, prob_clarification)
    
    print("\n評価結果:")
    if fold_models:
        print(f"  (K-foldアンサンブル評価: {len(fold_models)} folds)")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision_score:.4f}")
    print(f"  Recall: {recall_score:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    
    print("\n混同行列:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    
    print("\n分類レポート:")
    print(classification_report(labels, predictions, target_names=['Not Clarification', 'Clarification']))
    
    # ROC曲線を保存
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    title = f'ROC Curve (AUC = {auc:.4f})'
    if fold_models:
        title += f' - K-fold Ensemble ({len(fold_models)} folds)'
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    roc_curve_path = os.path.join(args.output_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nROC曲線を保存しました: {roc_curve_path}")
    
    # PR曲線を描画
    baseline = np.sum(labels) / len(labels)  # ランダム分類器のベースライン
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Random (AP = {baseline:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    title = f'Precision-Recall Curve (AP = {ap:.4f})'
    if fold_models:
        title += f' - K-fold Ensemble ({len(fold_models)} folds)'
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    pr_curve_path = os.path.join(args.output_dir, 'pr_curve.png')
    plt.savefig(pr_curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR曲線を保存しました: {pr_curve_path}")
    
    # wandbに評価結果を記録
    if args.use_wandb:
        eval_results = {
            'accuracy': accuracy,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1,
            'auc': auc,
            'average_precision': ap
        }
        log_evaluation_metrics(0, eval_results)
        log_confusion_matrix(labels, predictions)
        if roc_curve_path:
            log_roc_curve(fpr, tpr, auc, save_path=roc_curve_path)
        else:
            log_roc_curve(fpr, tpr, auc)
        import wandb
        wandb.finish()
    
    # 予測結果を元のJSON構造に追加して保存
    # フラット化されたデータのインデックスを追跡しながら、元の構造に予測結果をマッピング
    flat_idx = 0
    for conversation in original_data:
        for turn in conversation:
            if flat_idx < len(dev_data):
                # 予測結果を追加
                turn['prob_not_clarification'] = float(probabilities_list[flat_idx][0])
                turn['prob_clarification'] = float(probabilities_list[flat_idx][1])
                turn['logit_not_clarification'] = float(logits_list[flat_idx][0])
                turn['logit_clarification'] = float(logits_list[flat_idx][1])
                turn['predicted_label'] = int(predictions[flat_idx])
                flat_idx += 1
    
    # JSONファイルとして保存
    output_json_path = args.dev_path.replace('.json', '_with_predictions.json')
    # 出力ディレクトリに保存する場合
    output_filename = os.path.basename(args.dev_path).replace('.json', '_with_predictions.json')
    output_json_path = os.path.join(args.output_dir, output_filename)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(original_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n予測結果を追加したJSONファイルを保存しました: {output_json_path}")
    
    # TSV形式も保存（オプション）
    if args.save_predictions:
        output_path = os.path.join(args.output_dir, 'predictions.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            # ヘッダー行
            f.write("query\ttrue_label\tpredicted_label\tprob_not_clarification\tprob_clarification\tlogit_not_clarification\tlogit_clarification\ttrue_response_type\n")
            
            # データ行
            for i, item in enumerate(dev_data):
                query = item['query'].replace('\t', ' ').replace('\n', ' ')  # TSVの区切り文字を置換
                true_label = int(labels[i])
                predicted_label = int(predictions[i])
                prob_not_clarification = float(probabilities_list[i][0])
                prob_clarification = float(probabilities_list[i][1])
                logit_not_clarification = float(logits_list[i][0])
                logit_clarification = float(logits_list[i][1])
                true_response_type = item['response_type'].replace('\t', ' ').replace('\n', ' ')
                
                f.write(f"{query}\t{true_label}\t{predicted_label}\t{prob_not_clarification:.6f}\t{prob_clarification:.6f}\t{logit_not_clarification:.6f}\t{logit_clarification:.6f}\t{true_response_type}\n")
        
        print(f"TSV形式の予測結果を保存しました: {output_path}")


def predict(args):
    """
    推論フェーズ（単一のクエリに対する予測）
    """
    print("=" * 50)
    print("推論フェーズ")
    print("=" * 50)
    
    # モデルとトークナイザーの読み込み
    model_path = args.model_path if args.model_path else os.path.join(args.output_dir, 'best_model')
    print(f"モデルを読み込んでいます: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # クエリのトークナイズ
    encoding = tokenizer(
        args.query,
        truncation=True,
        padding='max_length',
        max_length=args.max_length,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 予測
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(logits, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    print(f"\nクエリ: {args.query}")
    print(f"予測: {'Clarification' if prediction == 1 else 'Not Clarification'}")
    print(f"信頼度: {confidence:.4f}")
    print(f"確率分布: Not Clarification={probabilities[0][0].item():.4f}, Clarification={probabilities[0][1].item():.4f}")


def main():
    parser = argparse.ArgumentParser(description='Clarification予測モデルのFine Tuning')
    
    # データセット名（必須）
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['INSCIT', 'AmbigNQ'],
                       help='データセット名（INSCIT または AmbigNQ）')
    
    # データパス
    parser.add_argument('--train_path', type=str, default=None,
                       help='訓練データのパス（未指定の場合はデータセット名から自動設定）')
    parser.add_argument('--dev_path', type=str, default=None,
                       help='開発データのパス（未指定の場合はデータセット名から自動設定）')
    
    # モデル設定
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='事前学習済みモデル名')
    parser.add_argument('--max_length', type=int, default=512,
                       help='最大シーケンス長')
    
    # 訓練設定
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='訓練エポック数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='バッチサイズ')
    parser.add_argument('--k_fold', type=int, default=None,
                       help='K-fold交差検証のK値（指定するとK-fold交差検証モードで実行）')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学習率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='重み減衰')
    parser.add_argument('--logging_steps', type=int, default=100,
                       help='ログ出力間隔')
    parser.add_argument('--early_stopping', action='store_true',
                       help='Early stoppingを使用')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='クラス重みを使用（不均衡データに対応）')
    parser.add_argument('--fp16', action='store_true',
                       help='FP16を使用')
    parser.add_argument('--seed', type=int, default=42,
                       help='乱数シード')
    
    # 出力設定
    parser.add_argument('--output_dir', type=str, default='./output',
                       help='出力ディレクトリ')
    parser.add_argument('--model_path', type=str, default=None,
                       help='評価/推論時に使用するモデルのパス')
    parser.add_argument('--save_predictions', action='store_true',
                       help='予測結果を保存')
    parser.add_argument('--save_roc_curve', action='store_true',
                       help='ROC曲線を画像ファイルとして保存')
    
    # モード選択
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'predict'],
                       default='train', help='実行モード')
    parser.add_argument('--query', type=str, default=None,
                       help='推論モード時のクエリ')
    
    # wandb設定
    parser.add_argument('--use_wandb', action='store_true',
                       help='wandbを使用してログを記録')
    parser.add_argument('--wandb_project', type=str, default='FT-PLM-Clarification',
                       help='wandbプロジェクト名')
    
    # GPU設定
    parser.add_argument('--gpu_ids', type=str, default='0,1',
                       help='使用するGPU ID（カンマ区切り、例: 0,1,2）。デフォルト: 0,1')
    
    args = parser.parse_args()
    
    # データセット名に応じてデフォルトパスを設定
    base_dataset_dir = f'/home/daiki_shibata/pj/QPP4SIP/dataset/{args.dataset}'
    if args.train_path is None:
        args.train_path = os.path.join(base_dataset_dir, 'train.json')
    if args.dev_path is None:
        args.dev_path = os.path.join(base_dataset_dir, 'dev.json')
    
    print(f"データセット: {args.dataset}")
    print(f"訓練データ: {args.train_path}")
    print(f"開発データ: {args.dev_path}")
    
    # GPUの指定
    if args.gpu_ids:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        print(f"使用するGPU: {args.gpu_ids}")
    
    # データセット名を含む出力ディレクトリを作成
    args.output_dir = os.path.join(args.output_dir, args.dataset)
    
    # 実験固有の出力ディレクトリを作成（訓練時のみ）
    if args.mode == 'train':
        experiment_name = get_experiment_name(args)
        args.output_dir = os.path.join(args.output_dir, experiment_name)
        print(f"実験ディレクトリ: {args.output_dir}")
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # モードに応じて実行
    if args.mode == 'train':
        if args.k_fold is not None:
            train_kfold(args)
        else:
            train(args)
    elif args.mode == 'evaluate':
        # モデルの存在確認（k-foldと通常の両方に対応）
        if args.model_path is None:
            # k-foldモデルの検出を試みる
            model_path = args.output_dir
            fold_models = find_kfold_models(model_path)
            if not fold_models:
                # k-foldモデルがない場合、通常のモデルをチェック
                if not os.path.exists(os.path.join(args.output_dir, 'best_model')):
                    print("エラー: モデルが見つかりません。--model_pathを指定するか、先に訓練を実行してください。")
                    return
        evaluate(args)
    elif args.mode == 'predict':
        if args.query is None:
            print("エラー: --queryを指定してください。")
            return
        # モデルの存在確認（k-foldと通常の両方に対応）
        if args.model_path is None:
            # k-foldモデルの検出を試みる
            model_path = args.output_dir
            fold_models = find_kfold_models(model_path)
            if not fold_models:
                # k-foldモデルがない場合、通常のモデルをチェック
                if not os.path.exists(os.path.join(args.output_dir, 'best_model')):
                    print("エラー: モデルが見つかりません。--model_pathを指定するか、先に訓練を実行してください。")
                    return
        predict(args)


if __name__ == '__main__':
    main()

