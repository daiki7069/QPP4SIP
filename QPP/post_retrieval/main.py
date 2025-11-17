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
from methods.wig import WIG
from methods.coherency import Coherency
import torch


def compute_similarity(split: str, dataset: str, top_k: int, device: Optional[str] = None):
    """類似度統計を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_similarity.json"
    output_csv_path = output_dir / f"{split}_similarity.csv"
    Similarity.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k,
        device=device
    )


def compute_lci(split: str, dataset: str, top_k: int, window: int = 3):
    """タイトル列の局所的集中度（LCI）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_lci.json"
    output_csv_path = output_dir / f"{split}_lci.csv"
    LCI.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k,
        window=window
    )


def compute_entropy(split: str, dataset: str, top_k: int):
    """タイトル分布の正規化エントロピーを計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_entropy.json"
    output_csv_path = output_dir / f"{split}_entropy.csv"

    Entropy.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k
    )


def compute_unique_titles(split: str, dataset: str, top_k: int):
    """top_kに含まれるユニークなタイトルの種類数を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_unique_titles.json"
    output_csv_path = output_dir / f"{split}_unique_titles.csv"

    UniqueTitles.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k
    )


def compute_nqc(split: str, dataset: str, top_k: int):
    """上位k件のスコアからNQC（Normalized Query Clarity）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_nqc.json"
    output_csv_path = output_dir / f"{split}_nqc.csv"

    NQC.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k
    )


def compute_wig(split: str, dataset: str, top_k: int, k: int = 20, bg_ratio: float = 0.5):
    """DPRスコアからWIG（Weighted Information Gain）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_wig.json"
    output_csv_path = output_dir / f"{split}_wig.csv"

    WIG.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k,
        k=k,
        bg_ratio=bg_ratio
    )


def compute_coherency(split: str, dataset: str, top_k: int, top_t: Optional[int] = None, use_weighted: bool = True, device: Optional[str] = None):
    """文書間ネットワークを構築し、ACC/WACCを計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_coherency.json"
    output_csv_path = output_dir / f"{split}_coherency.csv"

    Coherency.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k,
        top_t=top_t,
        use_weighted=use_weighted,
        device=device
    )


def main():
    parser = argparse.ArgumentParser(description="Post-retrieval QPP分析スクリプト")
    
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
        choices=["similarity", "lci", "entropy", "unique_titles", "nqc", "wig", "coherency", "all"],
        help="実行モード: 'similarity' (類似度統計), 'lci' (局所的集中度), 'entropy' (エントロピー), 'unique_titles' (ユニークタイトル数), 'nqc' (Normalized Query Clarity), 'wig' (Weighted Information Gain), 'coherency' (ACC/WACC), または 'all' (全て)"
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
    parser.add_argument(
        "--wig_k",
        type=int,
        default=50,
        help="WIG計算に使うtop-k文書数 (wigモードのみ, デフォルト: 50)"
    )
    parser.add_argument(
        "--wig_bg_ratio",
        type=float,
        default=0.5,
        help="WIG計算の背景モデルとして使う割合 (wigモードのみ, デフォルト: 0.5)"
    )
    parser.add_argument(
        "--top_t",
        type=int,
        default=None,
        help="ネットワーク構築の対象とする上位t文書 (coherencyモードのみ, デフォルト: top_kと同じ)"
    )
    parser.add_argument(
        "--use_weighted",
        action="store_true",
        default=True,
        help="重み付きエッジを使用するかどうか (coherencyモードのみ, デフォルト: True)"
    )
    parser.add_argument(
        "--no_weighted",
        action="store_false",
        dest="use_weighted",
        help="重み付きエッジを使用しない (coherencyモードのみ)"
    )
    
    args = parser.parse_args()
    
    print(f"データセット: {args.dataset}")
    print(f"スプリット: {args.split}")
    print(f"メトリクス: {args.metric}")
    
    # デバイスの設定
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    if args.metric == "similarity" or args.metric == "all":
        print("=== 類似度統計を計算 ===")
        compute_similarity(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k,
            device=device
        )
    if args.metric == "lci" or args.metric == "all":
        print("=== 局所的集中度（LCI）を計算 ===")
        compute_lci(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k,
            window=args.window
        )
    if args.metric == "entropy" or args.metric == "all":
        print("=== タイトル分布エントロピーを計算 ===")
        compute_entropy(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k
        )
    if args.metric == "unique_titles" or args.metric == "all":
        print("=== ユニークタイトル数を計算 ===")
        compute_unique_titles(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k
        )
    if args.metric == "nqc" or args.metric == "all":
        print("=== NQC (Normalized Query Clarity) を計算 ===")
        compute_nqc(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k
        )
    if args.metric == "wig" or args.metric == "all":
        print("=== WIG (Weighted Information Gain) を計算 ===")
        compute_wig(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k,
            k=args.wig_k,
            bg_ratio=args.wig_bg_ratio
        )
    if args.metric == "coherency" or args.metric == "all":
        print("=== Coherency (ACC/WACC) を計算 ===")
        compute_coherency(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k,
            top_t=args.top_t,
            use_weighted=args.use_weighted,
            device=device
        )


if __name__ == "__main__":
    main()
