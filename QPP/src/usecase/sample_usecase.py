from abc import ABC, abstractmethod
from domain.irepository.sample_repository import SampleRepository

class SampleUseCase(ABC):
    """サンプルユースケース抽象クラス"""
    
    def __init__(self, repository: SampleRepository):
        self.repository = repository
    
    @abstractmethod
    def get_dataset_length(self, dataset_name: str) -> int:
        """データセットの長さを取得"""
        pass 