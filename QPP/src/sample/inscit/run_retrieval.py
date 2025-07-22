#!/usr/bin/env python3
"""
会話データに対する検索処理のエントリポイント
"""
import json
from module.lucene_retriever import LuceneRetriever
from module.conversation_retrieval_processor import ConversationRetrievalProcessor


def main():
    """メイン処理"""
    # 設定
    index_path = "/mnt/disk6/daiki/Datasets/INSCIT/models/DPR/retrieval_data/wikipedia/index"
    input_file_path = "/mnt/disk6/daiki/Datasets/INSCIT/data/dev_resolved.json"
    
    # 検索エンジンを初期化
    retriever = LuceneRetriever(index_path)
    processor = ConversationRetrievalProcessor(retriever)
    
    # 処理を実行（nDCG@1, @3, @5を計算）
    output_path = processor.process_json_file(
        input_file_path, 
        calculate_ndcg=True, 
        k_values=[1, 3, 5]
    )
    
    # 全体の評価指標を算出
    with open(output_path, 'r', encoding='utf-8') as f:
        processed_data = json.load(f)
    
    overall_metrics = processor.calculate_overall_metrics(processed_data, k_values=[1, 3, 5])
    
    # responseType別の評価指標を算出
    response_type_metrics = processor.calculate_metrics_by_response_type(processed_data, k_values=[1, 3, 5])
    
    print(f"処理完了: {output_path}")
    print(f"nDCG@1: 平均={overall_metrics['avg_ndcg@1']:.4f}, 最大={overall_metrics['max_ndcg@1']:.4f}")
    print(f"nDCG@3: 平均={overall_metrics['avg_ndcg@3']:.4f}, 最大={overall_metrics['max_ndcg@3']:.4f}")
    print(f"nDCG@5: 平均={overall_metrics['avg_ndcg@5']:.4f}, 最大={overall_metrics['max_ndcg@5']:.4f}")
    print(f"Precision@1: 平均={overall_metrics['avg_precision@1']:.4f}, 最大={overall_metrics['max_precision@1']:.4f}")
    print(f"Precision@3: 平均={overall_metrics['avg_precision@3']:.4f}, 最大={overall_metrics['max_precision@3']:.4f}")
    print(f"Precision@5: 平均={overall_metrics['avg_precision@5']:.4f}, 最大={overall_metrics['max_precision@5']:.4f}")
    print(f"Recall@1: 平均={overall_metrics['avg_recall@1']:.4f}, 最大={overall_metrics['max_recall@1']:.4f}")
    print(f"Recall@3: 平均={overall_metrics['avg_recall@3']:.4f}, 最大={overall_metrics['max_recall@3']:.4f}")
    print(f"Recall@5: 平均={overall_metrics['avg_recall@5']:.4f}, 最大={overall_metrics['max_recall@5']:.4f}")
    print(f"評価クエリ数: {overall_metrics['num_queries']}")
    
    print("\n=== ResponseType別の評価指標 ===")
    for response_type, metrics in response_type_metrics.items():
        print(f"\n{response_type}:")
        count = metrics.get('count_ndcg@1', 0)
        print(f"  サンプル数: {count}")
        print(f"  nDCG@1: 平均={metrics['avg_ndcg@1']:.4f}, 最大={metrics['max_ndcg@1']:.4f}, 最小={metrics['min_ndcg@1']:.4f}")
        print(f"  nDCG@3: 平均={metrics['avg_ndcg@3']:.4f}, 最大={metrics['max_ndcg@3']:.4f}, 最小={metrics['min_ndcg@3']:.4f}")
        print(f"  nDCG@5: 平均={metrics['avg_ndcg@5']:.4f}, 最大={metrics['max_ndcg@5']:.4f}, 最小={metrics['min_ndcg@5']:.4f}")
        print(f"  Precision@1: 平均={metrics['avg_precision@1']:.4f}, 最大={metrics['max_precision@1']:.4f}")
        print(f"  Precision@3: 平均={metrics['avg_precision@3']:.4f}, 最大={metrics['max_precision@3']:.4f}")
        print(f"  Precision@5: 平均={metrics['avg_precision@5']:.4f}, 最大={metrics['max_precision@5']:.4f}")
        print(f"  Recall@1: 平均={metrics['avg_recall@1']:.4f}, 最大={metrics['max_recall@1']:.4f}")
        print(f"  Recall@3: 平均={metrics['avg_recall@3']:.4f}, 最大={metrics['max_recall@3']:.4f}")
        print(f"  Recall@5: 平均={metrics['avg_recall@5']:.4f}, 最大={metrics['max_recall@5']:.4f}")


if __name__ == "__main__":
    main() 