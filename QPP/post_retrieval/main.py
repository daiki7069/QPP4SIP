"""
Post-retrieval QPP用のサンプルスクリプト
"""
import os
import pandas as pd
from datetime import datetime
from data_loader import DPRResultLoader
from score_analysis import ScoreAnalyzer
from content_analysis import ContentAnalyzer


def main():
    # データの読み込み
    loader = DPRResultLoader("/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/retrieval_outputs_own/results")
    dev_data = loader.load_data("train")
    print(f"Loaded {len(dev_data)} turns")
    
    # スコア分析
    score_analyzer = ScoreAnalyzer(dev_data)
    score_stats = score_analyzer.get_score_statistics()
    print(f"Score statistics: {score_stats}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    score_output_path = os.path.join("outputs", f"score_statistics_{timestamp}.csv")
    score_stats.to_csv(score_output_path, index=False)
    print(f"Score statistics saved to: {score_output_path}")
     
    # # 内容分析
    # content_analyzer = ContentAnalyzer(dev_data)
    # question_doc_pairs = content_analyzer.get_question_document_pairs()
    # print(f"Question-document pairs: {question_doc_pairs.shape}")


if __name__ == "__main__":
    main()
