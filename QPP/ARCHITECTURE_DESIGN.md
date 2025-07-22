# QPP4SIP システムアーキテクチャ設計

## 概要
Query Performance Prediction (QPP) システムの根本的なアーキテクチャ見直し提案

## 現在の状況
- **QPP**: メインプロジェクト（エントリーポイント: `QPP/main.py`）
- **SIP**: 別プロジェクト
- データセット: `/mnt/disk6/daiki/Datasets/` 内にiKAT、INSCITなど
- モデル: 外部から取得、今後は `/mnt/disk6/daiki/Models` からも取得予定

## 提案する新しいアーキテクチャ

### 1. レイヤー分離アーキテクチャ

```
QPP/
├── infrastructure/          # インフラ層
│   ├── data/               # データアクセス層
│   │   ├── dataset_manager.py
│   │   ├── model_manager.py
│   │   └── storage_manager.py
│   ├── retrieval/          # 検索エンジン層
│   │   ├── lucene_retriever.py
│   │   ├── bm25_retriever.py
│   │   └── dpr_retriever.py
│   └── models/             # モデル管理層
│       ├── model_loader.py
│       └── model_registry.py
├── application/            # アプリケーション層
│   ├── services/           # ビジネスロジック
│   │   ├── query_service.py
│   │   ├── retrieval_service.py
│   │   └── evaluation_service.py
│   └── processors/         # データ処理
│       ├── data_preprocessor.py
│       └── query_processor.py
├── interfaces/             # インターフェース層
│   ├── adapters/           # 外部システム適応
│   │   ├── ikat_adapter.py
│   │   ├── inscit_adapter.py
│   │   └── dataset_adapter.py
│   └── api/                # API層
│       ├── rest_api.py
│       └── cli.py
└── config/                 # 設定管理
    ├── settings.py
    └── dataset_config.py
```

### 2. データベース指向設計

`/mnt/disk6/daiki/` をデータベースとして扱う設計：

```python
# infrastructure/data/storage_manager.py
class StorageManager:
    """データベースとしての/mnt/disk6/daiki/を管理"""
    
    def __init__(self):
        self.base_path = "/mnt/disk6/daiki"
        self.datasets_path = f"{self.base_path}/Datasets"
        self.models_path = f"{self.base_path}/Models"
    
    def list_datasets(self) -> List[str]:
        """利用可能なデータセット一覧を取得"""
        
    def get_dataset_info(self, dataset_name: str) -> Dict:
        """データセットのメタデータを取得"""
        
    def list_models(self) -> List[str]:
        """利用可能なモデル一覧を取得"""
```

### 3. プラグイン型アーキテクチャ

```python
# interfaces/adapters/dataset_adapter.py
from abc import ABC, abstractmethod

class DatasetAdapter(ABC):
    """データセット適応器の基底クラス"""
    
    @abstractmethod
    def load_data(self, path: str) -> Any:
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict:
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        pass

class IKATAdapter(DatasetAdapter):
    """iKATデータセット専用適応器"""
    
class INSCITAdapter(DatasetAdapter):
    """INSCITデータセット専用適応器"""
```

### 4. 設定駆動設計

```python
# config/dataset_config.py
DATASET_CONFIGS = {
    "iKAT": {
        "type": "conversation",
        "adapter": "IKATAdapter",
        "data_path": "/mnt/disk6/daiki/Datasets/iKAT",
        "index_path": "/mnt/disk6/daiki/Datasets/iKAT/ikat_demo/index",
        "schema": {
            "turns": "list",
            "utterance": "str",
            "response": "str"
        }
    },
    "INSCIT": {
        "type": "conversation",
        "adapter": "INSCITAdapter", 
        "data_path": "/mnt/disk6/daiki/Datasets/INSCIT",
        "index_path": "/mnt/disk6/daiki/Datasets/INSCIT/models/DPR/retrieval_data/wikipedia/index",
        "schema": {
            "dialogue": "dict",
            "turns": "list",
            "labels": "list"
        }
    }
}
```

### 5. 新しいエントリーポイント

```python
# QPP/main.py
from application.services.query_service import QueryService
from infrastructure.data.storage_manager import StorageManager
from config.settings import load_config

def main():
    # 設定読み込み
    config = load_config()
    
    # ストレージマネージャー初期化
    storage = StorageManager()
    
    # サービス初期化
    query_service = QueryService(storage, config)
    
    # コマンドライン引数処理
    # またはAPIサーバー起動
    
if __name__ == "__main__":
    main()
```

## メリット

1. **拡張性**: 新しいデータセットやモデルの追加が容易
2. **保守性**: レイヤー分離により修正範囲を限定
3. **再利用性**: 共通コンポーネントの再利用
4. **テスト容易性**: 各レイヤーを独立してテスト可能
5. **設定駆動**: コード変更なしで新しいデータセットに対応

## 実装方針

1. **段階的移行**: 既存コードを壊さずに段階的に移行
2. **後方互換性**: 既存のAPIやインターフェースを維持
3. **設定ファイル**: データセットやモデルの設定を外部化
4. **プラグイン開発**: 新しいデータセット用のアダプターを簡単に追加可能

## 次のステップ

1. 現在のコードベースの詳細分析
2. 移行計画の策定
3. プロトタイプの作成
4. 段階的な実装とテスト
 