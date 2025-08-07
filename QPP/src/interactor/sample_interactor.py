from typing import Dict, Any
from usecase.sample_usecase import SampleUseCase

class SampleInteractor(SampleUseCase):
    """サンプルインタラクター"""
    
    def __init__(self, repository):
        super().__init__(repository)
    
    def get_dataset_length(self, dataset_name: str) -> int:
        """データセットの長さを取得"""
        print(f"SampleInteractor: データセットの長さを取得中 - dataset_name: {dataset_name}")
        
        length = self.repository.get_dataset_length(dataset_name)
        
        print(f"SampleInteractor: データセット長さ取得完了 - {dataset_name}: {length}")
        return length 