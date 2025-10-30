"""
wandb設定とユーティリティ関数
学習曲線の可視化とevaluation結果の表示を行うための設定
"""
import wandb
import os


def init_wandb(project_name, experiment_name, config_dict, mode="online"):
    """
    wandbを初期化
    
    Args:
        project_name (str): wandbプロジェクト名
        experiment_name (str): 実験名
        config_dict (dict): 設定辞書
        mode (str): wandbのモード ("online", "offline", "disabled")
    
    Returns:
        wandb run object
    """
    # wandbを初期化
    run = wandb.init(
        project=project_name,
        name=experiment_name,
        config=config_dict,
        mode=mode,
        reinit=True  # 同じプロセス内で複数回初期化する場合に必要
    )
    
    return run


def log_training_metrics(epoch, step, loss_dict, learning_rate=None):
    """
    学習中のメトリクスをwandbに記録
    
    Args:
        epoch (int): 現在のエポック
        step (int): 現在のステップ
        loss_dict (dict): 損失の辞書
        learning_rate (float, optional): 学習率
    """
    # エポック情報を記録
    wandb.log({"epoch": epoch}, step=step)
    
    # 学習率を記録
    if learning_rate is not None:
        wandb.log({"learning_rate": learning_rate}, step=step)
    
    # 各損失を記録
    for loss_name, loss_value in loss_dict.items():
        # タスク名に応じてメトリクス名を調整
        metric_name = f"train/{loss_name}"
        wandb.log({metric_name: loss_value}, step=step)
    
    # 全体の損失も記録（SIPタスクの場合）
    if "loss_distance_crf" in loss_dict and "loss_mle_e" in loss_dict:
        overall_loss = (
            loss_dict["loss_distance_crf"] + 
            loss_dict["loss_mle_e"]
        )
        wandb.log({"train/loss_overall": overall_loss}, step=step)


def log_evaluation_metrics(epoch, eval_results):
    """
    evaluation結果をwandbに記録
    
    Args:
        epoch (int): エポック
        eval_results (dict): evaluation結果の辞書
    """
    # 基本的な評価指標
    metrics_to_log = {
        "eval/f1": eval_results.get("f1", 0),
        "eval/precision": eval_results.get("p", 0),
        "eval/recall": eval_results.get("r", 0),
        "eval/accuracy": eval_results.get("acc", 0),
    }
    
    # ラベル別の精度
    acc_per_label = eval_results.get("acc_per_label", [])
    if len(acc_per_label) >= 2:
        metrics_to_log["eval/accuracy_non_initiative"] = acc_per_label[0]
        metrics_to_log["eval/accuracy_initiative"] = acc_per_label[1]
    
    # ラベル別のprecision
    p_per_label = eval_results.get("p_per_label", [])
    if len(p_per_label) >= 2:
        metrics_to_log["eval/precision_non_initiative"] = p_per_label[0]
        metrics_to_log["eval/precision_initiative"] = p_per_label[1]
    
    # ラベル別のrecall
    r_per_label = eval_results.get("r_per_label", [])
    if len(r_per_label) >= 2:
        metrics_to_log["eval/recall_non_initiative"] = r_per_label[0]
        metrics_to_log["eval/recall_initiative"] = r_per_label[1]
    
    # 対話単位の統計
    if "conversation_stats" in eval_results:
        conv_stats = eval_results["conversation_stats"]
        metrics_to_log.update({
            "eval/avg_conversation_accuracy": conv_stats.get("avg_conversation_accuracy", 0),
            "eval/total_conversations": conv_stats.get("total_conversations", 0),
            "eval/total_turns": conv_stats.get("total_turns", 0),
            "eval/avg_turns_per_conversation": conv_stats.get("avg_turns_per_conversation", 0),
            "eval/conversations_with_initiative": conv_stats.get("conversations_with_initiative", 0),
        })
        
        # Initiativeラベル別の詳細指標
        metrics_to_log.update({
            "eval/initiative_acc": conv_stats.get("initiative_acc", 0),
            "eval/initiative_precision": conv_stats.get("initiative_precision", 0),
            "eval/initiative_recall": conv_stats.get("initiative_recall", 0),
            "eval/initiative_f1": conv_stats.get("initiative_f1", 0),
            "eval/non_initiative_acc": conv_stats.get("non_initiative_acc", 0),
            "eval/non_initiative_precision": conv_stats.get("non_initiative_precision", 0),
            "eval/non_initiative_recall": conv_stats.get("non_initiative_recall", 0),
            "eval/non_initiative_f1": conv_stats.get("non_initiative_f1", 0),
        })
        
        # Hit数と総数
        metrics_to_log.update({
            "eval/initiative_hit_num": conv_stats.get("initiative_hit_num", 0),
            "eval/initiative_total_num": conv_stats.get("initiative_total_num", 0),
            "eval/non_initiative_hit_num": conv_stats.get("non_initiative_hit_num", 0),
            "eval/non_initiative_total_num": conv_stats.get("non_initiative_total_num", 0),
        })
    
    # wandbに記録
    wandb.log(metrics_to_log, step=epoch)


def create_config_dict(args):
    """
    wandbに記録する設定辞書を作成
    
    Args:
        args: コマンドライン引数
    
    Returns:
        dict: wandb設定辞書
    """
    config = {
        "task": getattr(args, "task", "SIP"),
        "model": getattr(args, "model", "music"),
        "dataset": getattr(args, "dataset", "INSCIT"),
        "qpp4sip_pattern": getattr(args, "qpp4sip_pattern", None),
        "epoch_num": getattr(args, "epoch_num", 20),
        "learning_rate": getattr(args, "learning_rate", 2e-5),
        "hidden_size": getattr(args, "hidden_size", 768),
        "dropout": getattr(args, "dropout", 0.1),
        "BiLSTM_layers": getattr(args, "BiLSTM_layers", 1),
        "accumulation_steps": getattr(args, "accumulation_steps", 1),
        "clip": getattr(args, "clip", 1.0),
        "lr_distance_crf": getattr(args, "lr_distance_crf", 1e-3),
        "max_utterance_len": getattr(args, "max_utterance_len", 128),
        "max_context_len": getattr(args, "max_context_len", 384),
        "random_seed": getattr(args, "random_seed", 42),
        "class_imbalance_ratio": getattr(args, "class_imbalance_ratio", 6.5),
        "focal_gamma": getattr(args, "focal_gamma", 2.0),
    }
    
    # QPP関連の設定
    if hasattr(args, "qpp_feature_indices") and args.qpp_feature_indices:
        config["qpp_feature_indices"] = args.qpp_feature_indices
    if hasattr(args, "qpp_target_feature_index"):
        config["qpp_target_feature_index"] = args.qpp_target_feature_index
    if hasattr(args, "qpp_gate_feature_index"):
        config["qpp_gate_feature_index"] = args.qpp_gate_feature_index
    
    return config


def get_experiment_name(args):
    """
    実験名を生成
    
    Args:
        args: コマンドライン引数
    
    Returns:
        str: 実験名
    """
    model = getattr(args, "model", "music")
    pattern = getattr(args, "qpp4sip_pattern", None)
    
    if model == "music":
        return "music"
    elif model == "qpp4sip":
        feature_id = ""
        if hasattr(args, "qpp_feature_indices") and args.qpp_feature_indices:
            feature_id = "".join(map(str, sorted(args.qpp_feature_indices)))
        return f"qpp4sip_{pattern}_{feature_id}" if feature_id else f"qpp4sip_{pattern}"
    elif model == "qpp_gating":
        return f"qpp_gating_{getattr(args, 'qpp_gate_feature_index', 1)}"
    else:
        return model

