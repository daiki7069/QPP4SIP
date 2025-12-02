#!/usr/bin/env python3
"""
Neural QPP: 学習済みモデルを使用した推論スクリプト
予測結果をCSVファイルに保存
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import QPPDataLoader
from methods.bertqpp import BERTQPPTrainer, QPPDataset


def predict_and_save(
    dataset: str,
    split: str,
    model_type: str = 'bi',
    metric: str = 'map@20',
    model_path: Optional[str] = None,
    batch_size: int = 16,
    max_length: int = 512,
    device: Optional[str] = None,
    model_name: str = 'bert-base-uncased'
):
    """
    学習済みモデルで予測を行い、結果をCSVに保存
    
    Args:
        dataset: データセット名（INSCIT または AmbigNQ）
        split: データセットのスプリット（train または dev）
        model_type: モデルタイプ（'bi' または 'cross'）
        metric: メトリクス名（'map@20'など）
        model_path: モデルのパス（Noneの場合は自動検出）
        batch_size: バッチサイズ
        max_length: 最大シーケンス長
        device: 使用するデバイス
        model_name: 事前学習済みBERTモデル名
    """
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    model_dir = base_dir / "QPP" / "neural_qpp" / "outputs" / dataset / f"{model_type}_{metric}"
    
    # モデルパスの決定
    if model_path is None:
        model_path = model_dir / "best_model.pth"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    
    # 出力ディレクトリはモデルパスと同じディレクトリ
    output_dir = model_path.parent
    
    # データの読み込み
    print(f"=== データの読み込み ===")
    dpr_json_path = input_dir / f"dpr_{split}.json"
    data_loader = QPPDataLoader(dpr_json_path=str(dpr_json_path))
    test_data = data_loader.load_training_data(metric=metric, use_first_doc_only=True)
    print(f"データ数: {len(test_data)}")
    
    # トレーナーの初期化
    print(f"=== モデルの読み込み ===")
    trainer = BERTQPPTrainer(
        model_type=model_type,
        model_name=model_name,
        device=device,
        learning_rate=1e-5,  # 推論時は使用しない
        batch_size=batch_size,
        max_length=max_length
    )
    
    # モデルの読み込み
    trainer.load_model(str(model_path))
    trainer.model.eval()
    
    # データセットとデータローダーの作成
    test_dataset = QPPDataset(test_data, trainer.tokenizer, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=trainer.collate_fn
    )
    
    # 予測の実行
    print(f"=== 予測の実行 ===")
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            # データをデバイスに移動
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
            
            # 予測値を保存
            all_predictions.extend(predictions.cpu().numpy().flatten())
    
    # conv_idとturn_idを取得（データセットから）
    all_conv_ids = [data.conv_id for data in test_data]
    all_turn_ids = [data.turn_id for data in test_data]
    
    # DataFrameの作成
    df = pd.DataFrame({
        'conv_id': all_conv_ids,
        'turn_id': all_turn_ids,
        f'bertqpp_{model_type}': all_predictions
    })
    
    # CSVファイルに保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{split}_bertqpp_{model_type}_{metric}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n予測結果を保存しました: {output_file}")
    print(f"予測数: {len(df)}")
    print(f"予測値の統計:")
    print(f"  平均: {df[f'bertqpp_{model_type}'].mean():.4f}")
    print(f"  標準偏差: {df[f'bertqpp_{model_type}'].std():.4f}")
    print(f"  最小値: {df[f'bertqpp_{model_type}'].min():.4f}")
    print(f"  最大値: {df[f'bertqpp_{model_type}'].max():.4f}")


def main():
    parser = argparse.ArgumentParser(description="Neural QPP: 推論スクリプト")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "dev"],
        help="データセットのスプリット（train または dev）"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="bi",
        choices=["bi", "cross"],
        help="モデルタイプ: 'bi' (bi-encoder) または 'cross' (cross-encoder, デフォルト: bi)"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="map@20",
        choices=["map@20", "map"],
        help="メトリクス名（デフォルト: map@20）"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="モデルのパス（デフォルト: outputs/{dataset}/{model_type}_{metric}/best_model.pth）"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="バッチサイズ（デフォルト: 16）"
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
    
    args = parser.parse_args()
    
    predict_and_save(
        dataset=args.dataset,
        split=args.split,
        model_type=args.model_type,
        metric=args.metric,
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()

