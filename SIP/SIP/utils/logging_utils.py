"""
実験ログ管理関連のユーティリティ関数
"""
import os
import logging
from datetime import datetime


def setup_experiment_logging(experiment_dir, experiment_name=None):
    """
    実験用のログ設定を行う
    
    Args:
        experiment_dir (str): 実験ディレクトリ
        experiment_name (str): 実験名
        
    Returns:
        logging.Logger: 設定されたロガー
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    log_dir = os.path.join(experiment_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}.log")
    
    # ロガーの設定
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # 既存のハンドラーをクリア
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # ファイルハンドラー
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # フォーマッター
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # ハンドラーを追加
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_experiment_info(logger, config):
    """
    実験情報をログに記録
    
    Args:
        logger: ロガー
        config (dict): 実験設定
    """
    logger.info("=" * 50)
    logger.info("実験開始")
    logger.info("=" * 50)
    
    for key, value in config.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 50)


def log_training_progress(logger, epoch, total_epochs, train_loss, val_loss=None):
    """
    学習進捗をログに記録
    
    Args:
        logger: ロガー
        epoch (int): 現在のエポック
        total_epochs (int): 総エポック数
        train_loss (float): 訓練損失
        val_loss (float, optional): 検証損失
    """
    if val_loss is not None:
        logger.info(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    else:
        logger.info(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}")


def log_experiment_results(logger, results):
    """
    実験結果をログに記録
    
    Args:
        logger: ロガー
        results (dict): 実験結果
    """
    logger.info("=" * 50)
    logger.info("実験結果")
    logger.info("=" * 50)
    
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")
    
    logger.info("=" * 50)
