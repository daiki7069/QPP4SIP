"""
Neural QPP: 学習が必要なQPP手法のメインスクリプト
BERT-QPPの学習と検証を一度に実行
"""
import argparse
from pathlib import Path
from typing import Optional, List, Dict
import json
import os
import numpy as np
import torch
from sklearn.model_selection import KFold

from data_loader import QPPDataLoader, QPPTrainingData
from methods.bertqpp import BERTQPPTrainer

# wandbのインポート（オプション）
try:
    from wandb_utils import init_wandb, create_config_dict, get_experiment_name
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_and_evaluate_bertqpp(
    dataset: str,
    model_type: str = 'bi',  # 'bi' or 'cross'
    metric: str = 'map@20',
    num_epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    device: Optional[str] = None,
    model_name: str = 'bert-base-uncased',
    use_wandb: bool = False,
    wandb_project: str = 'neural-qpp',
    wandb_mode: str = 'online',
    k_fold: Optional[int] = None,
    random_state: int = 42,
    multi_gpu: bool = False,
    retrieval_method: str = 'dpr'
):
    """
    BERT-QPPの学習と検証を実行
    
    Args:
        dataset: データセット名（INSCIT または AmbigNQ）
        model_type: 'bi' (bi-encoder) または 'cross' (cross-encoder)
        metric: QPPスコアの計算メトリクス（'map@20'など）
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        max_length: 最大シーケンス長
        device: 使用するデバイス
        model_name: 事前学習済みBERTモデル名
    """
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "neural_qpp" / "outputs" / dataset / retrieval_method / f"{model_type}_{metric}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ファイルパス
    train_retrieval_json_path = input_dir / f"{retrieval_method}_train.json"
    train_base_json_path = input_dir / f"train.json"
    dev_retrieval_json_path = input_dir / f"{retrieval_method}_dev.json"
    dev_base_json_path = input_dir / f"dev.json"
    
    print(f"データセット: {dataset}")
    print(f"モデルタイプ: {model_type}")
    print(f"メトリクス: {metric}")
    print(f"エポック数: {num_epochs}")
    print(f"バッチサイズ: {batch_size}")
    print(f"学習率: {learning_rate}")
    print(f"出力ディレクトリ: {output_dir}")
    if k_fold:
        print(f"K-fold交差検証: {k_fold} folds")
    
    # 学習データの読み込み
    print("\n=== 学習データの読み込み ===")
    train_loader = QPPDataLoader(
        dpr_json_path=str(train_retrieval_json_path),
        base_json_path=str(train_base_json_path)
    )
    train_data = train_loader.load_training_data(metric=metric, use_first_doc_only=True)
    print(f"学習データ数: {len(train_data)}")
    
    # 検証データの読み込み
    print("\n=== 検証データの読み込み ===")
    dev_loader = QPPDataLoader(
        dpr_json_path=str(dev_retrieval_json_path),
        base_json_path=str(dev_base_json_path)
    )
    dev_data = dev_loader.load_training_data(metric=metric, use_first_doc_only=True)
    print(f"検証データ数: {len(dev_data)}")
    
    # K-fold交差検証の実行
    if k_fold and k_fold > 1:
        return train_and_evaluate_kfold(
            train_data=train_data,
            dev_data=dev_data,
            dataset=dataset,
            model_type=model_type,
            metric=metric,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            device=device,
            model_name=model_name,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            k_fold=k_fold,
            random_state=random_state,
            output_dir=output_dir,
            multi_gpu=multi_gpu,
            base_device=device,
            retrieval_method=retrieval_method
        )
    
    # wandbの初期化（使用する場合）
    wandb_run = None
    if use_wandb and WANDB_AVAILABLE:
        print("\n=== wandbの初期化 ===")
        config_dict = create_config_dict(
            dataset=dataset,
            model_type=model_type,
            metric=metric,
            model_name=model_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length
        )
        experiment_name = get_experiment_name(
            dataset=dataset,
            model_type=model_type,
            metric=metric,
            model_name=model_name,
            learning_rate=learning_rate,
            batch_size=batch_size
        )
        wandb_mode_env = os.getenv("WANDB_MODE", wandb_mode)
        wandb_run = init_wandb(
            project_name=wandb_project,
            experiment_name=experiment_name,
            config_dict=config_dict,
            mode=wandb_mode_env
        )
        print(f"wandbを初期化しました: プロジェクト={wandb_project}, 実験名={experiment_name}")
    elif use_wandb and not WANDB_AVAILABLE:
        print("警告: wandbがインストールされていません。wandbを使用するには 'pip install wandb' を実行してください。")
    
    # トレーナーの作成
    print("\n=== モデルの初期化 ===")
    trainer = BERTQPPTrainer(
        model_type=model_type,
        model_name=model_name,
        device=device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # 学習と検証の実行
    print("\n=== 学習と検証の開始 ===")
    history = trainer.train_and_evaluate(
        train_data=train_data,
        dev_data=dev_data,
        num_epochs=num_epochs,
        save_dir=str(output_dir),
        use_wandb=use_wandb and WANDB_AVAILABLE
    )
    
    # 学習履歴の保存
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"\n学習履歴を保存しました: {history_path}")
    
    # wandbの終了
    if wandb_run:
        import wandb
        wandb.finish()
        print("wandbを終了しました")
    
    # 最終結果の表示
    print("\n=== 学習完了 ===")
    print(f"最良エポック: {history['best_epoch']}")
    print(f"最良Pearson相関係数: {history['best_dev_corr']:.4f}")
    print(f"最良Spearman相関係数: {history['dev_correlations'][history['best_epoch']-1]['spearman']:.4f}")


def train_and_evaluate_kfold(
    train_data: List[QPPTrainingData],
    dev_data: List[QPPTrainingData],
    dataset: str,
    model_type: str,
    metric: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    device: Optional[str],
    model_name: str,
    use_wandb: bool,
    wandb_project: str,
    wandb_mode: str,
    k_fold: int,
    random_state: int,
    output_dir: Path,
    multi_gpu: bool,
    base_device: Optional[str],
    retrieval_method: str = 'dpr'
) -> Dict:
    """
    K-fold交差検証による学習と検証
    
    Args:
        train_data: 学習データ
        dev_data: 検証データ（最終評価用）
        dataset: データセット名
        model_type: モデルタイプ
        metric: メトリクス名
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        max_length: 最大シーケンス長
        device: 使用するデバイス
        model_name: モデル名
        use_wandb: wandbを使用するか
        wandb_project: wandbプロジェクト名
        wandb_mode: wandbモード
        k_fold: K-foldの数
        random_state: 乱数シード
        output_dir: 出力ディレクトリ
    
    Returns:
        学習履歴の辞書
    """
    print(f"\n=== K-fold交差検証の開始 ({k_fold} folds) ===")
    
    # K-fold分割
    kfold = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    train_indices = np.arange(len(train_data))
    
    # 各foldの結果を保存
    fold_histories = []
    fold_best_corrs = []
    
    # OOF予測を保存するリスト（全サンプルに対して）
    oof_predictions = [None] * len(train_data)
    
    available_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_indices)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{k_fold}")
        print(f"{'='*60}")
        
        # foldごとのデータ分割
        fold_train_data = [train_data[i] for i in train_idx]
        fold_val_data = [train_data[i] for i in val_idx]
        
        print(f"Fold訓練データ数: {len(fold_train_data)}")
        print(f"Fold検証データ数: {len(fold_val_data)}")
        
        # トレーナーの作成（各foldで新しく作成）
        if multi_gpu and available_devices > 0:
            fold_device = f"cuda:{fold_idx % available_devices}"
        else:
            # 明示的に指定があればそれを使用、なければ自動
            fold_device = base_device

        print(f"使用デバイス: {fold_device or 'auto'}")

        trainer = BERTQPPTrainer(
            model_type=model_type,
            model_name=model_name,
            device=fold_device,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_length=max_length
        )
        
        # foldごとの出力ディレクトリ
        fold_output_dir = output_dir / f"fold_{fold_idx + 1}"
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        
        # wandbの初期化（foldごと）
        fold_wandb_run = None
        if use_wandb and WANDB_AVAILABLE:
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent))
            from wandb_utils import init_wandb, create_config_dict, get_experiment_name
            
            config_dict = create_config_dict(
                dataset=dataset,
                model_type=model_type,
                metric=metric,
                model_name=model_name,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_length=max_length
            )
            config_dict['k_fold'] = k_fold
            config_dict['fold'] = fold_idx + 1
            
            experiment_name = get_experiment_name(
                dataset=dataset,
                model_type=model_type,
                metric=metric,
                model_name=model_name,
                learning_rate=learning_rate,
                batch_size=batch_size
            )
            experiment_name = f"{experiment_name}_fold{fold_idx + 1}"
            
            wandb_mode_env = os.getenv("WANDB_MODE", wandb_mode)
            fold_wandb_run = init_wandb(
                project_name=wandb_project,
                experiment_name=experiment_name,
                config_dict=config_dict,
                mode=wandb_mode_env
            )
        
        # 学習と検証の実行
        fold_history = trainer.train_and_evaluate(
            train_data=fold_train_data,
            dev_data=fold_val_data,
            num_epochs=num_epochs,
            save_dir=str(fold_output_dir),
            use_wandb=use_wandb and WANDB_AVAILABLE
        )
        
        fold_histories.append(fold_history)
        fold_best_corrs.append(fold_history['best_dev_corr'])
        
        print(f"Fold {fold_idx + 1} 最良Pearson相関係数: {fold_history['best_dev_corr']:.4f}")
        
        # OOF予測の生成（このfoldの検証セットに対して）
        print(f"Fold {fold_idx + 1} のOOF予測を生成中...")
        best_fold_model_path = fold_output_dir / "best_model.pth"
        trainer.load_model(str(best_fold_model_path))
        trainer.model.eval()
        
        from torch.utils.data import DataLoader
        from methods.bertqpp import QPPDataset
        fold_val_dataset = QPPDataset(fold_val_data, trainer.tokenizer, max_length)
        fold_val_loader = DataLoader(
            fold_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=trainer.collate_fn
        )
        
        fold_val_predictions = []
        with torch.no_grad():
            for batch in fold_val_loader:
                if model_type == 'bi':
                    query_input_ids = batch['query_input_ids'].to(trainer.device)
                    query_attention_mask = batch['query_attention_mask'].to(trainer.device)
                    doc_input_ids = batch['doc_input_ids'].to(trainer.device)
                    doc_attention_mask = batch['doc_attention_mask'].to(trainer.device)
                    
                    predictions = trainer.model(
                        query_input_ids, query_attention_mask,
                        doc_input_ids, doc_attention_mask
                    )
                else:  # cross
                    input_ids = batch['input_ids'].to(trainer.device)
                    attention_mask = batch['attention_mask'].to(trainer.device)
                    
                    predictions = trainer.model(input_ids, attention_mask)
                
                fold_val_predictions.extend(predictions.cpu().numpy().flatten())
        
        # OOF予測を元のインデックスに対応させて保存
        for val_idx, pred in zip(val_idx, fold_val_predictions):
            oof_predictions[val_idx] = float(pred)
        
        print(f"Fold {fold_idx + 1} のOOF予測完了（{len(fold_val_predictions)}サンプル）")
        
        # wandbの終了
        if fold_wandb_run:
            import wandb
            wandb.finish()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 全foldの結果を集約
    print(f"\n{'='*60}")
    print("K-fold交差検証の結果")
    print(f"{'='*60}")
    
    mean_corr = np.mean(fold_best_corrs)
    std_corr = np.std(fold_best_corrs)
    
    print(f"平均Pearson相関係数: {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"各foldの結果:")
    for i, corr in enumerate(fold_best_corrs):
        print(f"  Fold {i+1}: {corr:.4f}")
    
    # OOF予測結果をCSVファイルに保存（ロジスティック回帰との統合用）
    print(f"\n=== Out-of-Fold予測結果の保存 ===")
    import pandas as pd
    
    # conv_idとturn_idを取得
    oof_conv_ids = [train_data[i].conv_id for i in range(len(train_data)) if oof_predictions[i] is not None]
    oof_turn_ids = [train_data[i].turn_id for i in range(len(train_data)) if oof_predictions[i] is not None]
    oof_preds = [oof_predictions[i] for i in range(len(train_data)) if oof_predictions[i] is not None]
    
    oof_df = pd.DataFrame({
        'conv_id': oof_conv_ids,
        'turn_id': oof_turn_ids,
        f'bertqpp_{model_type}_oof': oof_preds
    })
    
    oof_output_file = output_dir / f"train_bertqpp_{model_type}_{metric}_oof.csv"
    oof_df.to_csv(oof_output_file, index=False)
    print(f"OOF予測結果を保存しました: {oof_output_file}")
    print(f"OOF予測数: {len(oof_df)}")
    print(f"OOF予測値の統計:")
    print(f"  平均: {oof_df[f'bertqpp_{model_type}_oof'].mean():.4f}")
    print(f"  標準偏差: {oof_df[f'bertqpp_{model_type}_oof'].std():.4f}")
    print(f"  最小値: {oof_df[f'bertqpp_{model_type}_oof'].min():.4f}")
    print(f"  最大値: {oof_df[f'bertqpp_{model_type}_oof'].max():.4f}")
    
    # 最良のfoldを選択
    best_fold_idx = np.argmax(fold_best_corrs)
    print(f"\n最良のfold: Fold {best_fold_idx + 1} (Pearson: {fold_best_corrs[best_fold_idx]:.4f})")
    
    # 最良のfoldのモデルを最終モデルとしてコピー
    best_fold_dir = output_dir / f"fold_{best_fold_idx + 1}"
    best_model_path = best_fold_dir / "best_model.pth"
    final_model_path = output_dir / "best_model.pth"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"最良モデルをコピーしました: {final_model_path}")
    
    # 最終評価（devデータで）
    print(f"\n=== 最終評価（devデータ） ===")
    final_device = base_device
    if multi_gpu and available_devices > 0 and final_device is None:
        final_device = "cuda:0"

    final_trainer = BERTQPPTrainer(
        model_type=model_type,
        model_name=model_name,
        device=final_device,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length
    )
    final_trainer.load_model(str(final_model_path))
    
    from torch.utils.data import DataLoader
    from methods.bertqpp import QPPDataset
    dev_dataset = QPPDataset(dev_data, final_trainer.tokenizer, max_length)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=final_trainer.collate_fn
    )
    
    dev_loss, dev_pearson, dev_spearman = final_trainer.evaluate(dev_loader)
    print(f"Dev Loss: {dev_loss:.4f}")
    print(f"Dev Pearson Correlation: {dev_pearson:.4f}")
    print(f"Dev Spearman Correlation: {dev_spearman:.4f}")
    
    # 学習履歴の保存
    kfold_history = {
        'k_fold': k_fold,
        'fold_histories': fold_histories,
        'fold_best_corrs': fold_best_corrs,
        'mean_corr': float(mean_corr),
        'std_corr': float(std_corr),
        'best_fold': int(best_fold_idx + 1),
        'final_dev_corr': float(dev_pearson),
        'final_dev_spearman': float(dev_spearman)
    }
    
    history_path = output_dir / "kfold_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(kfold_history, f, indent=2, ensure_ascii=False)
    print(f"\nK-fold学習履歴を保存しました: {history_path}")
    
    return kfold_history


def main():
    parser = argparse.ArgumentParser(description="Neural QPP: BERT-QPPの学習と検証")
    
    # データセット名（必須）
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
    )
    
    # モデルタイプ
    parser.add_argument(
        "--model_type",
        type=str,
        default="bi",
        choices=["bi", "cross"],
        help="モデルタイプ: 'bi' (bi-encoder) または 'cross' (cross-encoder, デフォルト: bi)"
    )
    
    # メトリクス
    parser.add_argument(
        "--metric",
        type=str,
        default="map@20",
        choices=["map@20", "map"],
        help="QPPスコアの計算メトリクス（デフォルト: map@20）"
    )
    
    # 学習パラメータ
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="エポック数（デフォルト: 10）"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="バッチサイズ（デフォルト: 16）"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="学習率（デフォルト: 2e-5）"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="最大シーケンス長（デフォルト: 512）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用するデバイス (cuda または cpu, デフォルト: 自動選択)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="事前学習済みBERTモデル名（デフォルト: bert-base-uncased）"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="wandbを使用して学習を記録する（デフォルト: False）"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-qpp",
        help="wandbプロジェクト名（デフォルト: neural-qpp）"
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandbのモード（デフォルト: online）。環境変数WANDB_MODEで上書き可能"
    )
    parser.add_argument(
        "--k_fold",
        type=int,
        default=None,
        help="K-fold交差検証のfold数（指定しない場合は通常のtrain/dev分割を使用）"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）"
    )
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="利用可能な複数GPUをfoldごとに使い分けてOOMを緩和する"
    )
    parser.add_argument(
        "--retrieval_method",
        type=str,
        default="dpr",
        choices=["dpr", "bm25"],
        help="検索手法 (dpr または bm25, デフォルト: dpr)"
    )
    
    args = parser.parse_args()
    
    # 学習と検証の実行
    train_and_evaluate_bertqpp(
        dataset=args.dataset,
        model_type=args.model_type,
        metric=args.metric,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device,
        model_name=args.model_name,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_mode=args.wandb_mode,
        k_fold=args.k_fold,
        random_state=args.random_state,
        multi_gpu=args.multi_gpu,
        retrieval_method=args.retrieval_method
    )


if __name__ == "__main__":
    main()

