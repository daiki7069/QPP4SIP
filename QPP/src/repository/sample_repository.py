import json
import os
from domain.irepository.sample_repository import ISampleRepository

class SampleRepository(ISampleRepository):
    """サンプルリポジトリ実装"""
    
    def __init__(self, data_dir: str = "data/sample"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def get_dataset_length(self, dataset_name: str) -> int:
        """指定されたデータセットの長さを取得"""
        print(f"SampleRepositoryImpl: データセットの長さを取得中 - dataset_name: {dataset_name}")
        
        # サンプルデータセットの長さを返す（実際の実装ではファイルから読み込む）
        sample_lengths = {
            "sample_dataset_1": 100,
            "sample_dataset_2": 200,
            "sample_dataset_3": 150
        }
        
        length = sample_lengths.get(dataset_name, 0)
        print(f"SampleRepositoryImpl: データセット長さ取得完了 - {dataset_name}: {length}")
        
        return length 