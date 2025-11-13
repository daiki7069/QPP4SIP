"""
Post-retrieval QPP用のスクリプト
"""
import argparse
from pathlib import Path
from typing import Optional
from methods.similarity import Similarity
from methods.lci import LCI
from methods.entropy import Entropy
from methods.unique_titles import UniqueTitles
from methods.nqc import NQC
import torch

# 入力ディレクトリを固定
INPUT_DIR = Path("/home/daiki_shibata/pj/QPP4SIP/dataset/INSCIT")
OUTPUT_DIR = Path("/home/daiki_shibata/pj/QPP4SIP/QPP/post_retrieval/outputs")


def compute_similarity(split: str, top_k: int, device: Optional[str] = None):
    """類似度統計を計算"""
    dpr_json_path = INPUT_DIR / f"dpr_{split}.json"
    base_json_path = INPUT_DIR / f"{split}.json"
    output_json_path = INPUT_DIR / f"{split}_similarity.json"
    output_csv_path = OUTPUT_DIR / f"{split}_similarity.csv"
    Similarity.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k,
        device=device
    )


def compute_lci(split: str, top_k: int, window: int = 3):
    """タイトル列の局所的集中度（LCI）を計算"""
    dpr_json_path = INPUT_DIR / f"dpr_{split}.json"
    base_json_path = INPUT_DIR / f"{split}.json"
    output_json_path = INPUT_DIR / f"{split}_lci.json"
    output_csv_path = OUTPUT_DIR / f"{split}_lci.csv"
    LCI.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k,
        window=window
    )


def compute_entropy(split: str, top_k: int):
    """タイトル分布の正規化エントロピーを計算"""
    dpr_json_path = INPUT_DIR / f"dpr_{split}.json"
    base_json_path = INPUT_DIR / f"{split}.json"
    output_json_path = INPUT_DIR / f"{split}_entropy.json"
    output_csv_path = OUTPUT_DIR / f"{split}_entropy.csv"

    Entropy.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k
    )


def compute_unique_titles(split: str, top_k: int):
    """top_kに含まれるユニークなタイトルの種類数を計算"""
    dpr_json_path = INPUT_DIR / f"dpr_{split}.json"
    base_json_path = INPUT_DIR / f"{split}.json"
    output_json_path = INPUT_DIR / f"{split}_unique_titles.json"
    output_csv_path = OUTPUT_DIR / f"{split}_unique_titles.csv"

    UniqueTitles.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k
    )


def compute_nqc(split: str, top_k: int):
    """上位k件のスコアからNQC（Normalized Query Clarity）を計算"""
    dpr_json_path = INPUT_DIR / f"dpr_{split}.json"
    base_json_path = INPUT_DIR / f"{split}.json"
    output_json_path = INPUT_DIR / f"{split}_nqc.json"
    output_csv_path = OUTPUT_DIR / f"{split}_nqc.csv"

    NQC.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k
    )


def main():
    parser = argparse.ArgumentParser(description="Post-retrieval QPP分析スクリプト")
    parser.add_argument(
        "--metric",
        choices=["similarity", "lci", "entropy", "unique_titles", "nqc", "all"],
        help="実行モード: 'similarity' (類似度統計), 'lci' (局所的集中度), 'entropy' (エントロピー), 'unique_titles' (ユニークタイトル数), 'nqc' (Normalized Query Clarity), または 'all' (全て)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="データセットの種類 (dev または train, デフォルト: dev)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="上位k件の文書のみを処理 (デフォルト: 100)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用するデバイス (cuda または cpu, デフォルト: 自動選択)"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="LCI計算の半径 (lciモードのみ, デフォルト: 3)"
    )
    
    args = parser.parse_args()
    
    # デバイスの設定
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if args.metric == "similarity" or args.metric == "all":
        print("=== 類似度統計を計算 ===")
        compute_similarity(
            split=args.split,
            top_k=args.top_k,
            device=device
        )
    if args.metric == "lci" or args.metric == "all":
        print("=== 局所的集中度（LCI）を計算 ===")
        compute_lci(
            split=args.split,
            top_k=args.top_k,
            window=args.window
        )
    if args.metric == "entropy" or args.metric == "all":
        print("=== タイトル分布エントロピーを計算 ===")
        compute_entropy(
            split=args.split,
            top_k=args.top_k
        )
    if args.metric == "unique_titles" or args.metric == "all":
        print("=== ユニークタイトル数を計算 ===")
        compute_unique_titles(
            split=args.split,
            top_k=args.top_k
        )
    if args.metric == "nqc" or args.metric == "all":
        print("=== NQC (Normalized Query Clarity) を計算 ===")
        compute_nqc(
            split=args.split,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()
