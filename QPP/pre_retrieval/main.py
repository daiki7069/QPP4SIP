"""
Pre-retrieval QPP用のスクリプト
"""
import argparse
from pathlib import Path
from typing import Optional
from methods.avgidf import AvgIDF
from methods.avgictf import AvgICTF
from methods.maxidf import MaxIDF
from methods.maxscq import MaxSCQ
from methods.simplifiedclarity import SimplifiedClarity


def compute_avgidf(split: str, dataset: str, use_pyserini: bool = False):
    """クエリに含まれる各語のIDF（逆文書頻度）の平均（AvgIDF）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "pre_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文書頻度のキャッシュディレクトリ
    document_freq_cache_dir = base_dir / "QPP" / "pre_retrieval" / ".document_freq_cache"
    document_freq_cache_dir.mkdir(parents=True, exist_ok=True)
    document_freq_cache_path = str(document_freq_cache_dir / f"{dataset}_document_freq.pkl")
    
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_avgidf.json"
    output_csv_path = output_dir / f"{split}_avgidf.csv"
    
    AvgIDF.compute_from_files(
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=str(output_csv_path),
        dataset=dataset,
        document_freq_cache_path=document_freq_cache_path,
        use_pyserini=use_pyserini
    )


def compute_avgictf(split: str, dataset: str, use_pyserini: bool = False):
    """クエリに含まれる各語のICTF（逆コレクション頻度）の平均（AvgICTF）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "pre_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # コレクション語頻度のキャッシュディレクトリ
    collection_freq_cache_dir = base_dir / "QPP" / "pre_retrieval" / ".collection_freq_cache"
    collection_freq_cache_dir.mkdir(parents=True, exist_ok=True)
    collection_freq_cache_path = str(collection_freq_cache_dir / f"{dataset}_collection_freq.pkl")
    
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_avgictf.json"
    output_csv_path = output_dir / f"{split}_avgictf.csv"
    
    AvgICTF.compute_from_files(
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=str(output_csv_path),
        dataset=dataset,
        collection_freq_cache_path=collection_freq_cache_path,
        use_pyserini=use_pyserini
    )


def compute_maxidf(split: str, dataset: str, use_pyserini: bool = False):
    """クエリに含まれる各語のIDF（逆文書頻度）の最大値（MaxIDF）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "pre_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文書頻度のキャッシュディレクトリ（AvgIDFと同じキャッシュを使用）
    document_freq_cache_dir = base_dir / "QPP" / "pre_retrieval" / ".document_freq_cache"
    document_freq_cache_dir.mkdir(parents=True, exist_ok=True)
    document_freq_cache_path = str(document_freq_cache_dir / f"{dataset}_document_freq.pkl")
    
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_maxidf.json"
    output_csv_path = output_dir / f"{split}_maxidf.csv"
    
    MaxIDF.compute_from_files(
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=str(output_csv_path),
        dataset=dataset,
        document_freq_cache_path=document_freq_cache_path,
        use_pyserini=use_pyserini
    )


def compute_maxscq(split: str, dataset: str, use_pyserini: bool = False):
    """クエリに含まれる各語のSCQ（Similarity of Collection and Query）の最大値（MaxSCQ）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "pre_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 文書頻度のキャッシュディレクトリ（AvgIDFと同じキャッシュを使用）
    document_freq_cache_dir = base_dir / "QPP" / "pre_retrieval" / ".document_freq_cache"
    document_freq_cache_dir.mkdir(parents=True, exist_ok=True)
    document_freq_cache_path = str(document_freq_cache_dir / f"{dataset}_document_freq.pkl")
    
    # コレクション語頻度のキャッシュディレクトリ（AvgICTFと同じキャッシュを使用）
    collection_freq_cache_dir = base_dir / "QPP" / "pre_retrieval" / ".collection_freq_cache"
    collection_freq_cache_dir.mkdir(parents=True, exist_ok=True)
    collection_freq_cache_path = str(collection_freq_cache_dir / f"{dataset}_collection_freq.pkl")
    
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_maxscq.json"
    output_csv_path = output_dir / f"{split}_maxscq.csv"
    
    MaxSCQ.compute_from_files(
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=str(output_csv_path),
        dataset=dataset,
        document_freq_cache_path=document_freq_cache_path,
        collection_freq_cache_path=collection_freq_cache_path,
        use_pyserini=use_pyserini
    )


def compute_simplified_clarity(split: str, dataset: str, use_pyserini: bool = False):
    """クエリ言語モデルとコレクション言語モデルのKLダイバージェンスの簡略版（Simplified Clarity）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "pre_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # コレクション語頻度のキャッシュディレクトリ（AvgICTFと同じキャッシュを使用）
    collection_freq_cache_dir = base_dir / "QPP" / "pre_retrieval" / ".collection_freq_cache"
    collection_freq_cache_dir.mkdir(parents=True, exist_ok=True)
    collection_freq_cache_path = str(collection_freq_cache_dir / f"{dataset}_collection_freq.pkl")
    
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_simplified_clarity.json"
    output_csv_path = output_dir / f"{split}_simplified_clarity.csv"
    
    SimplifiedClarity.compute_from_files(
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=str(output_csv_path),
        dataset=dataset,
        collection_freq_cache_path=collection_freq_cache_path,
        use_pyserini=use_pyserini
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-retrieval QPP分析スクリプト")
    
    # データセット名（必須）
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
    )
    
    parser.add_argument(
        "--metric",
        choices=[
            "avgidf",
            "avgictf",
            "maxidf",
            "maxscq",
            "simplified_clarity",
            "all"
        ],
        help="実行モード: 'avgidf', 'avgictf', 'maxidf', 'maxscq', 'simplified_clarity', または 'all'"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["dev", "train", "all"],
        help="データセットの種類 (dev, train, または all, デフォルト: all)"
    )
    parser.add_argument(
        "--use_pyserini",
        action="store_true",
        default=False,
        help="pyseriniのAnalyzerを使用してトークン化する（BM25インデックス作成時と同じ）"
    )
    
    args = parser.parse_args()
    
    print(f"データセット: {args.dataset}")
    print(f"スプリット: {args.split}")
    print(f"メトリクス: {args.metric}")
    
    # 処理するsplitを決定
    if args.split == "all":
        splits_to_process = ["dev", "train"]
    else:
        splits_to_process = [args.split]
    
    if args.metric == "avgidf" or args.metric == "all":
        print("=== AvgIDF (Average IDF) を計算 ===")
        for split in splits_to_process:
            print(f"\n--- {split.upper()} データを処理中 ---")
            compute_avgidf(
                split=split,
                dataset=args.dataset,
                use_pyserini=args.use_pyserini
            )
    
    if args.metric == "avgictf" or args.metric == "all":
        print("=== AvgICTF (Average ICTF) を計算 ===")
        for split in splits_to_process:
            print(f"\n--- {split.upper()} データを処理中 ---")
            compute_avgictf(
                split=split,
                dataset=args.dataset,
                use_pyserini=args.use_pyserini
            )
    
    if args.metric == "maxidf" or args.metric == "all":
        print("=== MaxIDF (Maximum IDF) を計算 ===")
        for split in splits_to_process:
            print(f"\n--- {split.upper()} データを処理中 ---")
            compute_maxidf(
                split=split,
                dataset=args.dataset,
                use_pyserini=args.use_pyserini
            )
    
    if args.metric == "maxscq" or args.metric == "all":
        print("=== MaxSCQ (Maximum SCQ) を計算 ===")
        for split in splits_to_process:
            print(f"\n--- {split.upper()} データを処理中 ---")
            compute_maxscq(
                split=split,
                dataset=args.dataset,
                use_pyserini=args.use_pyserini
            )
    
    if args.metric == "simplified_clarity" or args.metric == "all":
        print("=== Simplified Clarity (SCS) を計算 ===")
        for split in splits_to_process:
            print(f"\n--- {split.upper()} データを処理中 ---")
            compute_simplified_clarity(
                split=split,
                dataset=args.dataset,
                use_pyserini=args.use_pyserini
            )


if __name__ == "__main__":
    main()

