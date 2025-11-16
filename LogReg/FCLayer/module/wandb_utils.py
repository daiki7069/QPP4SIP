"""
wandb設定とユーティリティ関数
"""
import wandb
import os
import tempfile
import matplotlib
matplotlib.use('Agg')  # バックエンドを設定
import matplotlib.pyplot as plt
from typing import Dict, Optional
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


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


def log_training_metrics(step: int, loss: float, learning_rate: Optional[float] = None):
    """
    学習中のメトリクスをwandbに記録
    
    Args:
        step: 現在のステップ
        loss: 損失値
        learning_rate: 学習率（オプション）
    """
    log_dict = {
        "train/loss": loss
    }
    
    if learning_rate is not None:
        log_dict["train/learning_rate"] = learning_rate
    
    wandb.log(log_dict, step=step)


def log_validation_metrics(epoch: int, val_loss: float, metrics: Optional[Dict] = None, step: Optional[int] = None):
    """
    検証時のメトリクスをwandbに記録
    
    Args:
        epoch: エポック番号
        val_loss: 検証損失
        metrics: その他のメトリクス（オプション）
        step: ステップ番号（オプション、指定しない場合はepochを使用）
    """
    log_dict = {
        "val/loss": val_loss,
        "epoch": epoch
    }
    
    if metrics:
        for key, value in metrics.items():
            log_dict[f"val/{key}"] = value
    
    # ステップが指定されていない場合はepochを使用
    if step is None:
        step = epoch
    
    wandb.log(log_dict, step=step)


def log_evaluation_metrics(epoch: int, eval_results: Dict):
    """
    evaluation結果をwandbに記録
    
    Args:
        epoch: エポック
        eval_results: evaluation結果の辞書
    """
    metrics_to_log = {
        "eval/accuracy": eval_results.get("accuracy", 0),
        "eval/precision": eval_results.get("precision", 0),
        "eval/recall": eval_results.get("recall", 0),
        "eval/f1": eval_results.get("f1", 0),
        "eval/auc": eval_results.get("auc", 0),
        "eval/ap": eval_results.get("ap", 0),
        "epoch": epoch
    }
    
    # 損失も記録（存在する場合）
    if "eval_loss" in eval_results:
        metrics_to_log["eval/loss"] = eval_results["eval_loss"]
    
    wandb.log(metrics_to_log, step=epoch)


def log_roc_curve(y_true, y_proba, split: str = "test", auc: Optional[float] = None):
    """
    ROC曲線をwandbに記録
    
    Args:
        y_true: 正解ラベル
        y_proba: 予測確率
        split: データセット分割名（"train" or "test"）
        auc: AUCスコア（オプション、計算されない場合）
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    
    if auc is None:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_proba)
    
    # ROC曲線をプロット
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {split.upper()} (AUC = {auc:.4f})', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 一時ファイルに保存
    temp_dir = tempfile.gettempdir()
    save_path = os.path.join(temp_dir, f'roc_curve_{split}_{wandb.run.id if wandb.run else "temp"}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # wandbに画像として記録
    try:
        wandb.log({
            f"{split}/roc_curve": wandb.Image(save_path)
        })
    except Exception as e:
        print(f"警告: wandbへのROC曲線の記録に失敗しました: {e}")
    
    # AUCスコアも記録
    wandb.log({
        f"{split}/auc": auc
    })
    
    # 一時ファイルを削除
    try:
        if save_path.startswith(temp_dir):
            os.remove(save_path)
    except:
        pass


def log_pr_curve(y_true, y_proba, split: str = "test", ap: Optional[float] = None):
    """
    Precision-Recall曲線をwandbに記録
    
    Args:
        y_true: 正解ラベル
        y_proba: 予測確率
        split: データセット分割名（"train" or "test"）
        ap: Average Precisionスコア（オプション、計算されない場合）
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    
    if ap is None:
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_true, y_proba)
    
    # PR曲線をプロット
    baseline = len(y_true[y_true == 1]) / len(y_true) if len(y_true) > 0 else 0.0
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.4f})')
    plt.axhline(y=baseline, color='navy', lw=2, linestyle='--', label=f'Random (AP = {baseline:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {split.upper()} (AP = {ap:.4f})', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 一時ファイルに保存
    temp_dir = tempfile.gettempdir()
    save_path = os.path.join(temp_dir, f'pr_curve_{split}_{wandb.run.id if wandb.run else "temp"}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # wandbに画像として記録
    try:
        wandb.log({
            f"{split}/pr_curve": wandb.Image(save_path)
        })
    except Exception as e:
        print(f"警告: wandbへのPR曲線の記録に失敗しました: {e}")
    
    # APスコアも記録
    wandb.log({
        f"{split}/ap": ap
    })
    
    # 一時ファイルを削除
    try:
        if save_path.startswith(temp_dir):
            os.remove(save_path)
    except:
        pass


def log_confusion_matrix(y_true, y_pred, class_names=None, split: str = "test"):
    """
    混同行列をwandbに記録
    
    Args:
        y_true: 正解ラベル
        y_pred: 予測ラベル
        class_names: クラス名のリスト（オプション）
        split: データセット分割名（"train" or "test"）
    """
    if class_names is None:
        class_names = ["Not Clarification", "Clarification"]
    
    # 混同行列を計算
    cm = confusion_matrix(y_true, y_pred)
    
    # wandbに記録（可視化用）
    try:
        wandb.log({
            f"{split}/confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
    except Exception as e:
        # wandb.plot.confusion_matrixが使えない場合は、数値のみ記録
        print(f"警告: 混同行列の可視化に失敗しました: {e}")
    
    # 数値も記録
    if cm.shape == (2, 2):
        wandb.log({
            f"{split}/confusion_matrix/tn": int(cm[0][0]),
            f"{split}/confusion_matrix/fp": int(cm[0][1]),
            f"{split}/confusion_matrix/fn": int(cm[1][0]),
            f"{split}/confusion_matrix/tp": int(cm[1][1]),
        })


def create_config_dict(args) -> Dict:
    """
    wandbに記録する設定辞書を作成
    
    Args:
        args: コマンドライン引数
    
    Returns:
        wandb設定辞書
    """
    config = {
        "task": "clarification_prediction_fc",
        "dataset": getattr(args, "dataset", ""),
        "use_base_score": getattr(args, "use_base_score", False),
        "hidden_dims": getattr(args, "hidden_dims", [64, 32]),
        "dropout": getattr(args, "dropout", 0.1),
        "batch_size": getattr(args, "batch_size", 32),
        "num_epochs": getattr(args, "num_epochs", 50),
        "learning_rate": getattr(args, "learning_rate", 0.001),
        "weight_decay": getattr(args, "weight_decay", 0.0001),
        "balance_label_distribution": getattr(args, "balance_label_distribution", False),
        "use_combined_normalization": getattr(args, "use_combined_normalization", False),
        "use_separate_normalization": getattr(args, "use_separate_normalization", False),
    }
    
    return config


def get_experiment_name(args) -> str:
    """
    実験名を生成
    
    Args:
        args: コマンドライン引数
    
    Returns:
        実験名
    """
    dataset = getattr(args, "dataset", "")
    hidden_dims = getattr(args, "hidden_dims", [64, 32])
    lr = getattr(args, "learning_rate", 0.001)
    batch_size = getattr(args, "batch_size", 32)
    
    # 隠れ層の次元数を文字列に変換
    hidden_str = "_".join(map(str, hidden_dims))
    
    experiment_name = f"{dataset}_fc_{hidden_str}_lr{lr}_bs{batch_size}"
    
    if getattr(args, "use_base_score", False):
        experiment_name += "_withbase"
    
    if getattr(args, "balance_label_distribution", False):
        experiment_name += "_balanced"
    
    return experiment_name

