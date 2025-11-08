"""
Retrieval結果と対話データを合体するサンプルスクリプト
"""
import os
import pandas as pd
from datetime import datetime
from data_loader import RetrievalDataLoader


def main():
    # データローダーの初期化
    loader = RetrievalDataLoader(
        dialogue_data_dir="/mnt/nas_syno/daiki/Datasets/INSCIT/data",
        retrieval_results_dir="/mnt/nas_syno/daiki/Datasets/INSCIT/models/DPR/retrieval_outputs_own/results"
    )
    
    # データの処理
    print("Processing dev data...")
    evidence_only_df, evidence_prev_evidence_df = loader.process_data("dev")
    
    print(f"Evidence only results: {evidence_only_df.shape}")
    print(f"Evidence + PrevEvidence results: {evidence_prev_evidence_df.shape}")
    
    # CSV出力
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # evidenceのみの場合
    if not evidence_only_df.empty:
        evidence_only_path = os.path.join("outputs", f"retrieval_evidence_only_{timestamp}.csv")
        evidence_only_df.to_csv(evidence_only_path, index=False)
        print(f"Evidence only results saved to: {evidence_only_path}")
        
        # サンプルデータの表示
        print("\nEvidence only sample:")
        print(evidence_only_df[['conv_id', 'turn_id', 'found_ratio', 'mrr', 'ndcg@1', 'ndcg@5', 'ndcg@10']].head())
    
    # evidence + prevEvidenceの場合
    if not evidence_prev_evidence_df.empty:
        evidence_prev_evidence_path = os.path.join("outputs", f"retrieval_evidence_prev_evidence_{timestamp}.csv")
        evidence_prev_evidence_df.to_csv(evidence_prev_evidence_path, index=False)
        print(f"Evidence + PrevEvidence results saved to: {evidence_prev_evidence_path}")
        
        # サンプルデータの表示
        print("\nEvidence + PrevEvidence sample:")
        print(evidence_prev_evidence_df[['conv_id', 'turn_id', 'found_ratio', 'mrr', 'ndcg@1', 'ndcg@5', 'ndcg@10']].head())
    
    # 統計情報の表示
    if not evidence_only_df.empty:
        print(f"\nEvidence only statistics:")
        print(f"Found ratio: {evidence_only_df['found_ratio'].mean():.3f}")
        print(f"MRR: {evidence_only_df['mrr'].mean():.3f}")
        print(f"NDCG@1: {evidence_only_df['ndcg@1'].mean():.3f}")
        print(f"NDCG@5: {evidence_only_df['ndcg@5'].mean():.3f}")
        print(f"NDCG@10: {evidence_only_df['ndcg@10'].mean():.3f}")
    
    if not evidence_prev_evidence_df.empty:
        print(f"\nEvidence + PrevEvidence statistics:")
        print(f"Found ratio: {evidence_prev_evidence_df['found_ratio'].mean():.3f}")
        print(f"MRR: {evidence_prev_evidence_df['mrr'].mean():.3f}")
        print(f"NDCG@1: {evidence_prev_evidence_df['ndcg@1'].mean():.3f}")
        print(f"NDCG@5: {evidence_prev_evidence_df['ndcg@5'].mean():.3f}")
        print(f"NDCG@10: {evidence_prev_evidence_df['ndcg@10'].mean():.3f}")


if __name__ == "__main__":
    main()
