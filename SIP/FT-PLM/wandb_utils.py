"""
wandb設定とユーティリティ関数
学習曲線の可視化とevaluation結果の表示を行うための設定
"""
import wandb
import os
import tempfile
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
        "eval/epoch": epoch
    }
    
    # 損失も記録（存在する場合）
    if "eval_loss" in eval_results:
        metrics_to_log["eval/loss"] = eval_results["eval_loss"]
    
    wandb.log(metrics_to_log, step=epoch)


def log_roc_curve(fpr, tpr, auc, save_path=None):
    """
    ROC曲線をwandbに記録（matplotlibでプロット）
    
    Args:
        fpr: False Positive Rate
        tpr: True Positive Rate
        auc: AUCスコア
        save_path: 画像の保存パス（オプション）
    """
    import matplotlib
    matplotlib.use('Agg')  # バックエンドを設定
    import matplotlib.pyplot as plt
    
    # ROC曲線をプロット
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve (AUC = {auc:.4f})', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 画像を保存
    if save_path is None:
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, f'roc_curve_{wandb.run.id if wandb.run else "temp"}.png')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # wandbに画像として記録
    try:
        wandb.log({
            "roc_curve": wandb.Image(save_path)
        })
    except Exception as e:
        print(f"警告: wandbへのROC曲線の記録に失敗しました: {e}")
    
    # AUCスコアも記録
    wandb.log({
        "eval/auc": auc
    })
    
    # 画像ファイルを削除（一時ファイルの場合）
    if save_path.startswith(tempfile.gettempdir()):
        try:
            os.remove(save_path)
        except:
            pass


def log_confusion_matrix(y_true, y_pred, class_names=None):
    """
    混同行列をwandbに記録
    
    Args:
        y_true: 正解ラベル
        y_pred: 予測ラベル
        class_names: クラス名のリスト（オプション）
    """
    if class_names is None:
        class_names = ["Not Clarification", "Clarification"]
    
    # 混同行列を計算
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # wandbに記録（可視化用）
    try:
        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
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
            "confusion_matrix/tn": int(cm[0][0]),
            "confusion_matrix/fp": int(cm[0][1]),
            "confusion_matrix/fn": int(cm[1][0]),
            "confusion_matrix/tp": int(cm[1][1]),
        })
    
    # 混同行列のテーブルとしても記録
    wandb.log({
        "confusion_matrix/table": wandb.Table(
            columns=class_names + ["Total"],
            data=[
                [int(cm[i][j]) for j in range(len(class_names))] + [int(cm[i].sum())]
                for i in range(len(class_names))
            ] + [[int(cm[:, j].sum()) for j in range(len(class_names))] + [int(cm.sum())]]
        )
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
        "task": "clarification_prediction",
        "dataset": getattr(args, "dataset", ""),
        "model_name": getattr(args, "model_name", "bert-base-uncased"),
        "max_length": getattr(args, "max_length", 512),
        "num_epochs": getattr(args, "num_epochs", 5),
        "batch_size": getattr(args, "batch_size", 16),
        "learning_rate": getattr(args, "learning_rate", 2e-5),
        "weight_decay": getattr(args, "weight_decay", 0.01),
        "seed": getattr(args, "seed", 42),
        "fp16": getattr(args, "fp16", False),
        "early_stopping": getattr(args, "early_stopping", False),
        "use_class_weights": getattr(args, "use_class_weights", False),
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
    model_name = getattr(args, "model_name", "bert-base-uncased")
    
    # モデル名からベース部分を抽出
    # 例: "bert-base-uncased" -> "bert-base", "roberta-base" -> "roberta-base"
    if "/" in model_name:
        # HuggingFace Hubの形式（例: "facebook/roberta-base"）
        model_base = model_name.split("/")[-1]
    else:
        model_base = model_name
    
    # モデル名を短縮（例: "bert-base-uncased" -> "bert-base", "roberta-base" -> "roberta-base"）
    if model_base.startswith("bert-"):
        # "bert-base-uncased" -> "bert-base"
        parts = model_base.split("-")
        if len(parts) >= 2:
            model_base = "-".join(parts[:2])
    elif model_base.startswith("roberta-"):
        # "roberta-base" -> "roberta-base" (そのまま)
        pass
    else:
        # その他の場合は最初の部分のみ
        model_base = model_base.split("-")[0] if "-" in model_base else model_base
    
    lr = getattr(args, "learning_rate", 2e-5)
    batch_size = getattr(args, "batch_size", 16)
    
    # データセット名を実験名に含める
    dataset = getattr(args, "dataset", "")
    
    experiment_name = f"{dataset}_{model_base}_lr{lr}_bs{batch_size}"
    
    if getattr(args, "early_stopping", False):
        experiment_name += "_earlystop"
    
    # K-fold交差検証の場合は実験名に含める
    k_fold = getattr(args, "k_fold", None)
    if k_fold is not None:
        experiment_name += f"_kfold{k_fold}"
    
    return experiment_name

