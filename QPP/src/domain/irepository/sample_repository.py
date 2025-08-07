from abc import ABC, abstractmethod

class ISampleRepository(ABC):
    """サンプルリポジトリインターフェース"""
    
    @abstractmethod
    def get_dataset_length(self, dataset_name: str) -> int:
        """指定されたデータセットの長さを取得"""
        pass 