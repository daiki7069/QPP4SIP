#!/usr/bin/env python3
"""
サンプルクリーンアーキテクチャの使用例
"""

import logging
from injector import Injector

from src.interactor.sample_interactor import SampleInteractor
from src.registory.sample_registory import SampleRegistory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def main():
    """メイン実行例"""
    print("=== サンプルクリーンアーキテクチャの使用例 ===")

    injector = Injector(SampleRegistory)
    sample_interactor = injector.get(SampleInteractor)
    sample_interactor.get_dataset_length("sample_dataset_1")
    
    
if __name__ == "__main__":
    main()