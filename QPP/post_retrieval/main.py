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
from methods.smv import SMV
from methods.nsv import NSV
from methods.wig import WIG
from methods.coherency import Coherency
from methods.clarity import Clarity
import torch


def compute_similarity(split: str, dataset: str, top_k: int, device: Optional[str] = None):
    """類似度統計を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # キャッシュディレクトリを構築（dataset/splitで分離）
    cache_dir = base_dir / "QPP" / "post_retrieval" / ".embedding_cache" / dataset / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    
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
        device=device,
        cache_dir=str(cache_dir)
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


def compute_smv(split: str, dataset: str, top_k: int):
    """上位k件のスコアからSMV（Score Magnitude Variance）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_smv.json"
    output_csv_path = output_dir / f"{split}_smv.csv"

    SMV.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        top_k=top_k
    )


def compute_nsv_metric(split: str, dataset: str, top_k: int):
    """上位k件のスコアからNSV（Normalized Score Variance / N(σ)）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_nsv.json"
    output_csv_path = output_dir / f"{split}_nsv.csv"

    NSV.compute_from_files(
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
    
    # キャッシュディレクトリを構築（dataset/splitで分離）
    cache_dir = base_dir / "QPP" / "post_retrieval" / ".embedding_cache" / dataset / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    
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
        device=device,
        cache_dir=str(cache_dir)
    )


def compute_clarity(split: str, dataset: str, top_k: int, use_scores: bool = True, device: Optional[str] = None):
    """クエリの語彙分布とコレクション全体の語彙分布のKLダイバージェンス（Clarity）を計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # コレクション語頻度のキャッシュディレクトリ
    collection_cache_dir = base_dir / "QPP" / "post_retrieval" / ".collection_cache"
    collection_cache_dir.mkdir(parents=True, exist_ok=True)
    collection_freq_cache_path = str(collection_cache_dir / f"{dataset}_collection_freq.pkl")
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_clarity.json"
    output_csv_path = output_dir / f"{split}_clarity.csv"

    Clarity.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        dataset=dataset,
        top_k=top_k,
        use_scores=use_scores,
        collection_freq_cache_path=collection_freq_cache_path,
        device=device,
        cache_dir=None
    )


def compute_bertqpp_bi(split: str, dataset: str, top_k: int, model_path: Optional[str] = None, device: Optional[str] = None):
    """BERT-QPP bi-encoder形式でQPPスコアを計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # キャッシュディレクトリを構築（dataset/splitで分離）
    cache_dir = base_dir / "QPP" / "post_retrieval" / ".embedding_cache" / dataset / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_bertqpp_bi.json"
    output_csv_path = output_dir / f"{split}_bertqpp_bi.csv"
    
    BERTQPPBiEncoder.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        model_path=model_path,
        top_k=1,  # BERT-QPPは通常最初のドキュメントのみを使用
        use_first_doc_only=True,
        device=device,
        cache_dir=str(cache_dir)
    )


def compute_bertqpp_cross(split: str, dataset: str, top_k: int, model_path: Optional[str] = None, device: Optional[str] = None):
    """BERT-QPP cross-encoder形式でQPPスコアを計算"""
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / dataset
    output_dir = base_dir / "QPP" / "post_retrieval" / "outputs" / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # キャッシュディレクトリを構築（dataset/splitで分離）
    cache_dir = base_dir / "QPP" / "post_retrieval" / ".embedding_cache" / dataset / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    dpr_json_path = input_dir / f"dpr_{split}.json"
    base_json_path = input_dir / f"{split}.json"
    output_json_path = input_dir / f"{split}_bertqpp_cross.json"
    output_csv_path = output_dir / f"{split}_bertqpp_cross.csv"
    
    BERTQPPCrossEncoder.compute_from_files(
        dpr_json_path=str(dpr_json_path),
        base_json_path=str(base_json_path),
        output_json_path=str(output_json_path),
        output_csv_path=output_csv_path,
        model_path=model_path,
        top_k=1,  # BERT-QPPは通常最初のドキュメントのみを使用
        use_first_doc_only=True,
        device=device,
        cache_dir=str(cache_dir)
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
        choices=[
            "similarity",
            "lci",
            "entropy",
            "unique_titles",
            "nqc",
            "smv",
            "nsv",
            "wig",
            "coherency",
            "clarity",
            "bertqpp_bi",
            "bertqpp_cross",
            "all"
        ],
        help="実行モード: 'similarity', 'lci', 'entropy', 'unique_titles', 'nqc', 'smv', 'nsv', 'wig', 'coherency', 'clarity', 'bertqpp_bi', 'bertqpp_cross', または 'all'"
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
    parser.add_argument(
        "--use_scores",
        action="store_true",
        default=True,
        help="DPRスコアを使用してP(d|Q)を計算するかどうか (clarityモードのみ, デフォルト: True)"
    )
    parser.add_argument(
        "--no_scores",
        action="store_false",
        dest="use_scores",
        help="DPRスコアを使用しない（均等重み） (clarityモードのみ)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="BERT-QPP用のファインチューニング済みモデルのパス (bertqpp_bi/bertqpp_crossモードのみ, デフォルト: bert-base-uncasedを使用)"
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
    if args.metric == "smv" or args.metric == "all":
        print("=== SMV (Score Magnitude Variance) を計算 ===")
        compute_smv(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k
        )
    if args.metric == "nsv" or args.metric == "all":
        print("=== NSV (Normalized Score Variance / N(σ)) を計算 ===")
        compute_nsv_metric(
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
    if args.metric == "clarity" or args.metric == "all":
        print("=== Clarity (Query Clarity) を計算 ===")
        compute_clarity(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k,
            use_scores=args.use_scores,
            device=device
        )
    if args.metric == "bertqpp_bi" or args.metric == "all":
        print("=== BERT-QPP (bi-encoder) を計算 ===")
        compute_bertqpp_bi(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k,
            model_path=args.model_path,
            device=device
        )
    if args.metric == "bertqpp_cross" or args.metric == "all":
        print("=== BERT-QPP (cross-encoder) を計算 ===")
        compute_bertqpp_cross(
            split=args.split,
            dataset=args.dataset,
            top_k=args.top_k,
            model_path=args.model_path,
            device=device
        )


if __name__ == "__main__":
    main()
