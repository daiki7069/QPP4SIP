"""
全結合層（Fully Connected Layer）によるclarification分類
"""
import argparse
import sys
import os
from io import StringIO
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    average_precision_score
)

try:
    import wandb
except ImportError:
    wandb = None

# モジュールのインポート
from module import (
    FullyConnectedClassifier,
    FeatureDataset,
    load_json_data,
    extract_labels,
    extract_base_scores,
    load_qpp_scores,
    merge_features,
    balance_label_distribution,
    normalize_features,
    plot_roc_curves,
    plot_pr_curves,
    FCTrainer
)
from module.wandb_utils import (
    init_wandb,
    create_config_dict,
    get_experiment_name
)

# パス設定
BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


def main():
    parser = argparse.ArgumentParser(description="全結合層によるclarification分類")
    
    # データセット名（必須）
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
    )
    
    parser.add_argument(
        "--use-base-score",
        action="store_true",
        help="ベーススコア（logit_clarification）も特徴量として使用する（デフォルト: False、QPPスコアのみ使用）"
    )
    
    parser.add_argument(
        "--sip-experiment-name",
        type=str,
        default=None,
        help="SIP実験名（デフォルト: データセット名に基づいて自動設定）"
    )
    
    parser.add_argument(
        "--balance-label-distribution",
        action="store_true",
        help="訓練データのラベル分布をテストデータと同じ正例率に調整する（デフォルト: False）"
    )
    
    parser.add_argument(
        "--use-combined-normalization",
        action="store_true",
        help="訓練データとテストデータを結合してから正規化する（リーク前提、デフォルト: False）"
    )
    
    parser.add_argument(
        "--use-separate-normalization",
        action="store_true",
        help="訓練データとテストデータをそれぞれ個別に正規化する（各々が平均0、標準偏差1になる、デフォルト: False）"
    )
    
    # モデルハイパーパラメータ
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs='+',
        default=[64, 32],
        help="隠れ層の次元数（デフォルト: [64, 32]）"
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout率（デフォルト: 0.1）"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="バッチサイズ（デフォルト: 32）"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="エポック数（デフォルト: 50）"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="学習率（デフォルト: 0.001）"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="重み減衰（L2正則化、デフォルト: 0.0001）"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="使用デバイス（デフォルト: cuda if available else cpu）"
    )
    
    # wandb関連
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="wandbを使用して実験を記録する（デフォルト: False）"
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="QPP4SIP-FCLayer",
        help="wandbプロジェクト名（デフォルト: QPP4SIP-FCLayer）"
    )
    
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandbのモード（デフォルト: online）"
    )
    
    parser.add_argument(
        "--hide-train-curves",
        action="store_true",
        help="訓練データの曲線を非表示にする（デフォルト: False）"
    )
    
    parser.add_argument(
        "--show-single-metric-curves",
        action="store_true",
        help="入力に使った指標単体での曲線も表示する（デフォルト: False）"
    )
    
    args = parser.parse_args()
    
    use_base_score = args.use_base_score
    
    # パスの動的設定
    dataset_dir = BASE_DIR / "dataset" / args.dataset
    qpp_output_dir = BASE_DIR / "QPP" / "post_retrieval" / "outputs" / args.dataset
    
    # SIP実験名の設定
    if args.sip_experiment_name:
        sip_experiment_name = args.sip_experiment_name
    else:
        # デフォルトの実験名（データセット名に基づく）
        sip_experiment_name = f"{args.dataset}_bert-base_lr2e-05_bs16_kfold5"
    
    sip_output_dir = BASE_DIR / "SIP" / "FT-PLM" / "output" / args.dataset / sip_experiment_name
    output_dir = BASE_DIR / "LogReg" / "FCLayer" / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 出力ファイルの準備（print内容をファイルにも保存）
    output_file = output_dir / "results.txt"
    output_buffer = StringIO()
    
    def print_and_save(*args, **kwargs):
        """printと同時にファイルにも出力"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=output_buffer)
    
    print_and_save("=== 全結合層によるclarification分類 ===\n")
    print_and_save(f"データセット: {args.dataset}")
    print_and_save(f"使用する特徴量: {'ベーススコア + QPPスコア' if use_base_score else 'QPPスコアのみ'}")
    print_and_save(f"ラベル分布の調整: {'有効' if args.balance_label_distribution else '無効'}")
    print_and_save(f"隠れ層の次元数: {args.hidden_dims}")
    print_and_save(f"Dropout率: {args.dropout}")
    print_and_save(f"バッチサイズ: {args.batch_size}")
    print_and_save(f"エポック数: {args.num_epochs}")
    print_and_save(f"学習率: {args.learning_rate}")
    print_and_save(f"重み減衰: {args.weight_decay}")
    print_and_save(f"デバイス: {args.device}")
    print_and_save(f"wandb使用: {'有効' if args.use_wandb else '無効'}")
    
    # 正規化方法の表示
    if args.use_separate_normalization:
        normalization_method = "訓練データとテストデータをそれぞれ個別に正規化"
    elif args.use_combined_normalization:
        normalization_method = "結合データで正規化（リーク前提）"
    else:
        normalization_method = "訓練データの統計量で正規化（通常）"
    print_and_save(f"正規化方法: {normalization_method}")
    
    if args.use_separate_normalization:
        warning_msg = (
            "\n" + "="*80 + "\n"
            "⚠️  注意: 個別正規化が有効になっています！\n"
            "⚠️  このオプションは訓練データとテストデータをそれぞれ個別に正規化します。\n"
            "⚠️  各データセットが独立に正規化されるため、分布の違いは排除されます。\n"
            "⚠️  実際の予測タスクでは使用できません（テストデータの統計量は未知です）。\n"
            "⚠️  分布の違いによる影響を確認するための実験的なオプションです。\n"
            "="*80 + "\n"
        )
        print_and_save(warning_msg)
        print(warning_msg, file=sys.stderr)
    elif args.use_combined_normalization:
        warning_msg = (
            "\n" + "="*80 + "\n"
            "⚠️  警告: リーク前提の正規化が有効になっています！\n"
            "⚠️  このオプションは訓練データとテストデータを結合してから正規化します。\n"
            "⚠️  実際の予測タスクでは使用できません（データリークが発生します）。\n"
            "⚠️  分布の違いによる影響を確認するための実験的なオプションです。\n"
            "="*80 + "\n"
        )
        print_and_save(warning_msg)
        print(warning_msg, file=sys.stderr)
    
    print_and_save()
    
    # wandbの初期化
    wandb_run = None
    if args.use_wandb:
        config_dict = create_config_dict(args)
        experiment_name = get_experiment_name(args)
        wandb_mode = os.getenv("WANDB_MODE", args.wandb_mode)
        wandb_run = init_wandb(
            project_name=args.wandb_project,
            experiment_name=experiment_name,
            config_dict=config_dict,
            mode=wandb_mode
        )
        print_and_save(f"wandbを初期化しました: プロジェクト={args.wandb_project}, 実験名={experiment_name}")
        print_and_save()
    
    # 1. データ読み込み
    print_and_save("1. データ読み込み中...")
    
    # 訓練データ
    train_json_path = dataset_dir / "train.json"
    train_pred_json_path = sip_output_dir / "train_with_predictions.json"
    
    train_data = load_json_data(train_json_path)
    train_labels = extract_labels(train_data)
    train_qpp_scores = load_qpp_scores('train', qpp_output_dir)
    
    print_and_save(f"  - 訓練データ: {len(train_labels)} サンプル")
    
    # ベーススコアの読み込み（使用する場合のみ）
    if use_base_score:
        train_pred_data = load_json_data(train_pred_json_path)
        train_base_scores = extract_base_scores(train_pred_data)
        print_and_save(f"  - ベーススコア: {len(train_base_scores)} サンプル")
    else:
        train_base_scores = {}
    
    for metric_name, scores in train_qpp_scores.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    
    # テストデータ
    dev_json_path = dataset_dir / "dev.json"
    dev_pred_json_path = sip_output_dir / "dev_with_predictions.json"
    
    dev_data = load_json_data(dev_json_path)
    dev_labels = extract_labels(dev_data)
    
    # ベーススコアの読み込み（使用する場合のみ）
    if use_base_score:
        dev_pred_data = load_json_data(dev_pred_json_path)
        dev_base_scores = extract_base_scores(dev_pred_data)
        print_and_save(f"  - ベーススコア: {len(dev_base_scores)} サンプル")
    else:
        dev_base_scores = {}
    
    dev_qpp_scores = load_qpp_scores('dev', qpp_output_dir)
    
    print_and_save(f"  - テストデータ: {len(dev_labels)} サンプル")
    for metric_name, scores in dev_qpp_scores.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    
    # 2. 特徴量マージ
    print_and_save("\n2. 特徴量マージ中...")
    X_train, y_train = merge_features(train_base_scores, train_qpp_scores, train_labels, use_base_score=use_base_score)
    X_test, y_test = merge_features(dev_base_scores, dev_qpp_scores, dev_labels, use_base_score=use_base_score)
    
    print_and_save(f"  - 訓練データ: {len(X_train)} サンプル, {len(X_train.columns)} 特徴量")
    print_and_save(f"  - テストデータ: {len(X_test)} サンプル, {len(X_test.columns)} 特徴量")
    print_and_save(f"  - 特徴量: {list(X_train.columns)}")
    print_and_save(f"  - 訓練データのラベル分布: {y_train.value_counts().to_dict()}")
    print_and_save(f"  - テストデータのラベル分布: {y_test.value_counts().to_dict()}")
    
    # ラベル分布の調整（オプション）
    if args.balance_label_distribution:
        print_and_save("\n2.5. ラベル分布の調整中...")
        test_positive_rate = y_test.mean()
        train_positive_rate_before = y_train.mean()
        
        print_and_save(f"  - 調整前の訓練データの正例率: {train_positive_rate_before:.4f}")
        print_and_save(f"  - テストデータの正例率（目標）: {test_positive_rate:.4f}")
        
        X_train, y_train = balance_label_distribution(
            X_train, y_train, target_positive_rate=test_positive_rate, random_state=42
        )
        
        train_positive_rate_after = y_train.mean()
        print_and_save(f"  - 調整後の訓練データの正例率: {train_positive_rate_after:.4f}")
        print_and_save(f"  - 調整後の訓練データのサンプル数: {len(X_train)}")
        print_and_save(f"  - 調整後のラベル分布: {y_train.value_counts().to_dict()}")
    else:
        print_and_save("\n2.5. ラベル分布の調整: スキップ（--balance-label-distribution が指定されていません）")
    
    # 3. 前処理: z-score正規化
    print_and_save("\n3. z-score正規化中...")
    if args.use_separate_normalization:
        print_and_save("  - 方法: 訓練データとテストデータをそれぞれ個別に正規化")
    elif args.use_combined_normalization:
        print_and_save("  - 方法: 訓練データとテストデータを結合してから正規化（リーク前提）")
    else:
        print_and_save("  - 方法: 訓練データの統計量でテストデータも正規化（通常）")
    X_train_norm, X_test_norm, scalers = normalize_features(
        X_train, X_test, 
        use_combined_normalization=args.use_combined_normalization,
        use_separate_normalization=args.use_separate_normalization
    )
    print_and_save("  - 正規化完了")
    
    # 4. データローダーの作成
    print_and_save("\n4. データローダーの作成中...")
    
    # 訓練データを訓練と検証に分割（80:20）
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_norm.values, y_train.values, test_size=0.2, random_state=42, stratify=y_train.values
    )
    
    train_dataset = FeatureDataset(X_train_split, y_train_split)
    val_dataset = FeatureDataset(X_val_split, y_val_split)
    test_dataset = FeatureDataset(X_test_norm.values, y_test.values)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print_and_save(f"  - 訓練データ: {len(train_dataset)} サンプル")
    print_and_save(f"  - 検証データ: {len(val_dataset)} サンプル")
    print_and_save(f"  - テストデータ: {len(test_dataset)} サンプル")
    
    # 5. モデルの作成
    print_and_save("\n5. モデルの作成中...")
    input_dim = X_train_norm.shape[1]
    model = FullyConnectedClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    )
    print_and_save(f"  - 入力次元: {input_dim}")
    print_and_save(f"  - モデル構造:")
    print_and_save(str(model))
    
    # 6. モデル学習
    print_and_save("\n6. モデル学習中...")
    device = torch.device(args.device)
    
    # クラス重みの計算
    class_counts = y_train.value_counts().to_dict()
    class_counts = {int(k): int(v) for k, v in class_counts.items()}
    
    # Trainerの作成
    trainer = FCTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        use_class_weights=True,
        class_counts=class_counts,
        early_stop_patience=10,
        use_wandb=args.use_wandb
    )
    
    # 訓練実行
    model, train_losses, val_losses = trainer.train()
    print_and_save("  - 学習完了")
    
    # 学習曲線の描画
    trainer.plot_loss_curves(output_dir)
    
    # 7. 評価
    print_and_save("\n7. 評価中...")
    
    # 訓練データ全体で評価
    train_full_dataset = FeatureDataset(X_train_norm.values, y_train.values)
    train_full_loader = DataLoader(train_full_dataset, batch_size=args.batch_size, shuffle=False)
    
    y_train_pred, y_train_proba = trainer.evaluate(train_full_loader, split="train")
    y_test_pred, y_test_proba = trainer.evaluate(test_loader, split="test")
    
    # 訓練データの評価
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_ap = average_precision_score(y_train, y_train_proba)
    
    # テストデータの評価
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    test_ap = average_precision_score(y_test, y_test_proba)
    
    print_and_save("\n=== 訓練データの評価 ===")
    print_and_save(f"Accuracy: {train_acc:.4f}")
    print_and_save(f"F1 Score: {train_f1:.4f}")
    print_and_save(f"AUC-ROC: {train_auc:.4f}")
    print_and_save(f"Average Precision: {train_ap:.4f}")
    print_and_save("\n分類レポート:")
    print_and_save(classification_report(y_train, y_train_pred, target_names=['not_clarification', 'clarification']))
    
    print_and_save("\n=== テストデータの評価 ===")
    print_and_save(f"Accuracy: {test_acc:.4f}")
    print_and_save(f"F1 Score: {test_f1:.4f}")
    print_and_save(f"AUC-ROC: {test_auc:.4f}")
    print_and_save(f"Average Precision: {test_ap:.4f}")
    print_and_save("\n分類レポート:")
    print_and_save(classification_report(y_test, y_test_pred, target_names=['not_clarification', 'clarification']))
    
    # 訓練データとテストデータの評価指標の比較
    print_and_save("\n【訓練データとテストデータの評価指標の比較】")
    print_and_save(f"Accuracy差: {test_acc - train_acc:+.4f} (テスト {'高' if test_acc > train_acc else '低'})")
    print_and_save(f"F1 Score差: {test_f1 - train_f1:+.4f} (テスト {'高' if test_f1 > train_f1 else '低'})")
    print_and_save(f"AUC-ROC差: {test_auc - train_auc:+.4f} (テスト {'高' if test_auc > train_auc else '低'})")
    print_and_save(f"Average Precision差: {test_ap - train_ap:+.4f} (テスト {'高' if test_ap > train_ap else '低'})")
    
    # 8. ROC曲線の描画
    print_and_save("\n8. ROC曲線を描画中...")
    feature_names = list(X_train_norm.columns)
    plot_roc_curves(
        y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc, output_dir,
        hide_train=args.hide_train_curves,
        X_train=X_train_norm if args.show_single_metric_curves else None,
        X_test=X_test_norm if args.show_single_metric_curves else None,
        show_single_metrics=args.show_single_metric_curves,
        feature_names=feature_names
    )
    print_and_save("  - ROC曲線を保存しました")
    
    # 9. PR曲線の描画
    print_and_save("\n9. Precision-Recall曲線を描画中...")
    plot_pr_curves(
        y_train, y_train_proba, y_test, y_test_proba, train_ap, test_ap, output_dir,
        hide_train=args.hide_train_curves,
        X_train=X_train_norm if args.show_single_metric_curves else None,
        X_test=X_test_norm if args.show_single_metric_curves else None,
        show_single_metrics=args.show_single_metric_curves,
        feature_names=feature_names
    )
    print_and_save("  - PR曲線を保存しました")
    
    # 結果をファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
    print_and_save(f"\n結果をファイルに保存しました: {output_file}")
    
    if args.use_separate_normalization:
        print_and_save("\n" + "="*80)
        print_and_save("⚠️  最終注意: この結果は個別正規化を使用しています！")
        print_and_save("⚠️  実際の予測タスクでは使用できません。")
        print_and_save("="*80)
    elif args.use_combined_normalization:
        print_and_save("\n" + "="*80)
        print_and_save("⚠️  最終警告: この結果はリーク前提の正規化を使用しています！")
        print_and_save("⚠️  実際の予測タスクでは使用できません。")
        print_and_save("="*80)
    
    # wandbを終了
    if wandb_run is not None:
        wandb.finish()
    
    print_and_save("\n=== 完了 ===")


if __name__ == "__main__":
    main()
