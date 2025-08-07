import os
from pathlib import Path

# プロジェクトルートディレクトリ
PROJECT_ROOT = Path(__file__).parent.parent.parent

# サンプルデータディレクトリ設定
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# サンプル設定
SAMPLE_CONFIG = {
    "name": "sample",
    "description": "サンプル設定",
    "version": "1.0.0"
}

# 開発環境設定
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development") 