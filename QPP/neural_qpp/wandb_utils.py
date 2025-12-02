"""
wandb設定とユーティリティ関数
Neural QPP用のwandbロギング機能
"""
import wandb
import os
from typing import Dict, Optional


def init_wandb(project_name: str, experiment_name: str, config_dict: Dict, mode: str = "online"):
    """
    wandbを初期化
    
    Args:
        project_name: wandbプロジェクト名
        experiment_name: 実験名
        config_dict: 設定辞書
        mode: wandbのモード ("online", "offline", "disabled")
    
    Returns:
        wandb run object
    """
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        config=config_dict,
        mode=mode,
        reinit=True
    )
    
    return run


def log_training_metrics(epoch: int, train_loss: float, learning_rate: Optional[float] = None):
    """
    学習中のメトリクスをwandbに記録
    
    Args:
        epoch: 現在のエポック
        train_loss: 学習損失
        learning_rate: 学習率（オプション）
    """
    log_dict = {
        "train/loss": train_loss,
        "epoch": epoch
    }
    
    if learning_rate is not None:
        log_dict["train/learning_rate"] = learning_rate
    
    wandb.log(log_dict, step=epoch)


def log_evaluation_metrics(
    epoch: int,
    dev_loss: float,
    pearson_corr: float,
    spearman_corr: float
):
    """
    検証結果をwandbに記録
    
    Args:
        epoch: 現在のエポック
        dev_loss: 検証損失
        pearson_corr: Pearson相関係数
        spearman_corr: Spearman相関係数
    """
    log_dict = {
        "dev/loss": dev_loss,
        "dev/pearson_correlation": pearson_corr,
        "dev/spearman_correlation": spearman_corr,
        "epoch": epoch
    }
    
    wandb.log(log_dict, step=epoch)


def log_best_model(epoch: int, best_dev_corr: float, spearman_corr: float):
    """
    最良モデルの情報をwandbに記録
    
    Args:
        epoch: 最良エポック
        best_dev_corr: 最良Pearson相関係数
        spearman_corr: 最良Spearman相関係数
    """
    wandb.log({
        "best/epoch": epoch,
        "best/pearson_correlation": best_dev_corr,
        "best/spearman_correlation": spearman_corr
    })


def create_config_dict(
    dataset: str,
    model_type: str,
    metric: str,
    model_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int
) -> Dict:
    """
    wandbに記録する設定辞書を作成
    
    Args:
        dataset: データセット名
        model_type: モデルタイプ（'bi' または 'cross'）
        metric: QPPスコアの計算メトリクス
        model_name: 事前学習済みBERTモデル名
        num_epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        max_length: 最大シーケンス長
    
    Returns:
        wandb設定辞書
    """
    config = {
        "task": "query_performance_prediction",
        "dataset": dataset,
        "model_type": model_type,
        "metric": metric,
        "model_name": model_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
    }
    
    return config


def get_experiment_name(
    dataset: str,
    model_type: str,
    metric: str,
    model_name: str,
    learning_rate: float,
    batch_size: int
) -> str:
    """
    実験名を生成
    
    Args:
        dataset: データセット名
        model_type: モデルタイプ（'bi' または 'cross'）
        metric: QPPスコアの計算メトリクス
        model_name: 事前学習済みBERTモデル名
        learning_rate: 学習率
        batch_size: バッチサイズ
    
    Returns:
        実験名
    """
    # モデル名からベース部分を抽出
    if "/" in model_name:
        model_base = model_name.split("/")[-1]
    else:
        model_base = model_name
    
    # モデル名を短縮
    if model_base.startswith("bert-"):
        parts = model_base.split("-")
        if len(parts) >= 2:
            model_base = "-".join(parts[:2])
    
    # 学習率を文字列に変換（例: 2e-5 -> "2e-5"）
    lr_str = f"{learning_rate:.0e}".replace("+", "").replace(".0", "")
    
    # 実験名を構築
    experiment_name = f"{dataset}_{model_type}_{model_base}_lr{lr_str}_bs{batch_size}_{metric}"
    
    return experiment_name

