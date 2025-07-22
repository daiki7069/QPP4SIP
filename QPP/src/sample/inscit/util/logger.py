import logging
import sys
from typing import Optional
from datetime import datetime


class Logger:
    """ログ表示のためのクラス"""
    
    def __init__(self, name: str = "QPP", level: int = logging.INFO, 
                 log_to_file: bool = False, log_file: Optional[str] = None):
        """
        初期化
        
        Args:
            name: ロガー名
            level: ログレベル
            log_to_file: ファイルにログを出力するかどうか
            log_file: ログファイルのパス（Noneの場合は自動生成）
        """
        self.name = name
        self.level = level
        self.log_to_file = log_to_file
        
        # ロガーを作成
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 既存のハンドラーをクリア
        self.logger.handlers.clear()
        
        # フォーマッターを作成
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # コンソールハンドラーを追加
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # ファイルハンドラーを追加（必要に応じて）
        if log_to_file:
            if log_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"logs/{name}_{timestamp}.log"
            
            # ログディレクトリを作成
            import os
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str):
        """デバッグレベルのログを出力"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """情報レベルのログを出力"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """警告レベルのログを出力"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """エラーレベルのログを出力"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """致命的エラーレベルのログを出力"""
        self.logger.critical(message)
    
    def success(self, message: str):
        """成功メッセージを出力（カスタムレベル）"""
        # 成功メッセージは情報レベルで出力
        self.logger.info(f"✅ {message}")
    
    def progress(self, message: str):
        """進捗メッセージを出力（カスタムレベル）"""
        # 進捗メッセージは情報レベルで出力
        self.logger.info(f"🔄 {message}")
    
    def section(self, title: str):
        """セクションタイトルを出力"""
        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"📋 {title}")
        self.logger.info(f"{'='*50}")
    
    def subsection(self, title: str):
        """サブセクションタイトルを出力"""
        self.logger.info(f"\n{'-'*30}")
        self.logger.info(f"📌 {title}")
        self.logger.info(f"{'-'*30}")


# グローバルロガーインスタンス
_default_logger = None


def get_logger(name: str = "QPP", level: int = logging.INFO, 
               log_to_file: bool = False, log_file: Optional[str] = None) -> Logger:
    """
    ロガーインスタンスを取得
    
    Args:
        name: ロガー名
        level: ログレベル
        log_to_file: ファイルにログを出力するかどうか
        log_file: ログファイルのパス
        
    Returns:
        Loggerインスタンス
    """
    global _default_logger
    
    if _default_logger is None:
        _default_logger = Logger(name, level, log_to_file, log_file)
    
    return _default_logger


def main():
    """メイン関数 - 使用例"""
    # ロガーを作成
    logger = get_logger("TestLogger", log_to_file=True)
    
    # 各種ログレベルのテスト
    logger.section("ログテスト")
    
    logger.debug("これはデバッグメッセージです")
    logger.info("これは情報メッセージです")
    logger.warning("これは警告メッセージです")
    logger.error("これはエラーメッセージです")
    logger.critical("これは致命的エラーメッセージです")
    
    logger.subsection("カスタムメッセージ")
    logger.success("処理が成功しました！")
    logger.progress("処理中... 50%完了")
    
    logger.info("テストが完了しました")


if __name__ == '__main__':
    main() 