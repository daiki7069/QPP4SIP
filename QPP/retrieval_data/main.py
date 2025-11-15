"""
Retrieval結果と対話データを合体するスクリプト
"""
import argparse
import pandas as pd
from pathlib import Path
from data_loader import RetrievalDataLoader


def main():
    parser = argparse.ArgumentParser(description="Retrieval結果と対話データを合体するスクリプト")
    
    # データセット名（必須）
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="データセットのスプリット (train, dev, test, デフォルト: dev)"
    )
    
    args = parser.parse_args()
    
    # ベースパスの設定
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    dialogue_data_dir = base_dir / "dataset" / args.dataset
    retrieval_results_dir = base_dir / "dataset" / args.dataset  # dpr_{split}.jsonはdatasetディレクトリ内にある
    output_dir = base_dir / "QPP" / "retrieval_data" / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"データセット: {args.dataset}")
    print(f"スプリット: {args.split}")
    print(f"対話データディレクトリ: {dialogue_data_dir}")
    print(f"Retrieval結果ディレクトリ: {retrieval_results_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    
    # データローダーの初期化
    loader = RetrievalDataLoader(
        dialogue_data_dir=str(dialogue_data_dir),
        retrieval_results_dir=str(retrieval_results_dir)
    )
    
    # データの処理
    print(f"\nProcessing {args.split} data...")
    evidence_only_df, evidence_prev_evidence_df = loader.process_data(args.split)
    
    print(f"Evidence only results: {evidence_only_df.shape}")
    print(f"Evidence + PrevEvidence results: {evidence_prev_evidence_df.shape}")
    
    # CSV出力
    # evidenceのみの場合
    if not evidence_only_df.empty:
        evidence_only_path = output_dir / f"dpr_{args.split}_only_evidence.csv"
        evidence_only_df.to_csv(evidence_only_path, index=False)
        print(f"Evidence only results saved to: {evidence_only_path}")
        
        # サンプルデータの表示
        print("\nEvidence only sample:")
        print(evidence_only_df[['conv_id', 'turn_id', 'found_ratio', 'mrr', 'ndcg@1', 'ndcg@5', 'ndcg@10']].head())
    
    # evidence + prevEvidenceの場合
    if not evidence_prev_evidence_df.empty:
        evidence_prev_evidence_path = output_dir / f"dpr_{args.split}_evidence_prev_evidence.csv"
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
