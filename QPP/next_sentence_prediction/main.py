"""
Next Sentence Prediction (NSP) 実験スクリプト

モード1: デモ用にBERTのNSP動作を確認
モード2: 検索済みtop-k文書からcoherencyグラフを構築し、NC/ANCを計算
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import BertForNextSentencePrediction, BertTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
QPP_DIR = CURRENT_DIR.parent
PROJECT_ROOT = QPP_DIR.parent

if str(QPP_DIR) not in sys.path:
    sys.path.insert(0, str(QPP_DIR))

from methods.nsp_graph import NSPGraphAnalyzer  # noqa: E402
from post_retrieval.data_loader import DPRResultLoader  # noqa: E402


def check_nsp(sent_a: str, sent_b: str, model, tokenizer, device: str = "cpu"):
    """
    NSP で文のペアが続きかどうかを判定
    """
    encoded = tokenizer(
        sent_a,
        sent_b,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits  # [batch, 2] → [IsNext, NotNext]

    probs = F.softmax(logits, dim=-1)[0]
    is_next_prob = probs[0].item()
    not_next_prob = probs[1].item()

    return {
        "sent_a": sent_a,
        "sent_b": sent_b,
        "is_next_prob": is_next_prob,
        "not_next_prob": not_next_prob,
        "prediction": "IsNext" if is_next_prob > not_next_prob else "NotNext"
    }


def print_nsp_result(result: dict):
    """NSP結果を表示"""
    print(f"A: {result['sent_a']}")
    print(f"B: {result['sent_b']}")
    print(f"IsNext prob     : {result['is_next_prob']:.4f}")
    print(f"NotNext prob    : {result['not_next_prob']:.4f}")
    print(f"Prediction      : {result['prediction']}")
    print("-" * 40)


def run_demo_mode(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用モデル: {args.model_name}")
    print(f"使用デバイス: {device}")
    print("=" * 40)

    print("モデルをロード中...")
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertForNextSentencePrediction.from_pretrained(args.model_name)
    model.to(device)
    model.eval()
    print("モデルロード完了")
    print("=" * 40)

    print("\n【テストケース1】自然な続きのペア")
    result1 = check_nsp(
        "I went to the store to buy some groceries.",
        "Then I went home and cooked dinner.",
        model, tokenizer, device
    )
    print_nsp_result(result1)

    print("\n【テストケース2】関係のないペア")
    result2 = check_nsp(
        "I went to the store to buy some groceries.",
        "The quantum Hall effect occurs in two-dimensional electron systems.",
        model, tokenizer, device
    )
    print_nsp_result(result2)

    print("\n【テストケース3】微妙に続きそうなペア")
    result3 = check_nsp(
        "I went to the store to buy some groceries.",
        "He was sick at that time.",
        model, tokenizer, device
    )
    print_nsp_result(result3)

    print("\n【結果の要約】")
    print(f"テストケース1 (自然な続き): IsNext={result1['is_next_prob']:.4f}")
    print(f"テストケース2 (関係ない):   IsNext={result2['is_next_prob']:.4f}")
    print(f"テストケース3 (微妙):       IsNext={result3['is_next_prob']:.4f}")

    if result1['is_next_prob'] > result2['is_next_prob']:
        print("\n✅ NSPは正常に動作しています！")
        print("   自然な続きのペアの方が、関係ないペアより高いIsNext確率を示しています。")
    else:
        print("\n⚠️  注意: 期待される動作と異なる結果が得られました。")


def run_graph_mode(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset_dir = PROJECT_ROOT / "dataset" / args.dataset
    dpr_json_path = dataset_dir / f"dpr_{args.split}.json"

    if not dpr_json_path.exists():
        raise FileNotFoundError(f"{dpr_json_path} が見つかりません")

    print(f"データセット: {args.dataset} / スプリット: {args.split}")
    print(f"top-k: {args.top_k}, NSPしきい値: {args.threshold}")
    print(f"読み込み中: {dpr_json_path}")

    loader = DPRResultLoader(str(dpr_json_path))
    turn_data_list = loader.load_data()
    if args.limit_turns:
        turn_data_list = turn_data_list[:args.limit_turns]
    print(f"読み込み済みターン数: {len(turn_data_list)}")

    analyzer = NSPGraphAnalyzer(
        turn_data_list=turn_data_list,
        model_name=args.model_name,
        device=device,
        threshold=args.threshold,
        top_k=args.top_k,
        batch_size=args.batch_size,
        use_multi_gpu=not args.no_multi_gpu,
    )

    df = analyzer.analyze(show_progress=not args.no_progress)

    output_dir = Path(args.output_dir) if args.output_dir else CURRENT_DIR / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{args.split}_nsp_graph_topk{args.top_k}.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n出力: {csv_path}")
    print(df[["node_connectivity", "average_node_connectivity"]].describe())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NSP demo / coherency graph ツール")
    parser.add_argument(
        "--mode",
        choices=["demo", "graph"],
        default="graph",
        help="demo: NSP動作確認, graph: top-kグラフ解析"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-uncased",
        help="使用するBERTモデル名（デフォルト: bert-base-uncased）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="使用するデバイス (cuda または cpu, デフォルト: 自動選択)"
    )
    parser.add_argument(
        "--dataset",
        choices=["INSCIT", "AmbigNQ"],
        default="INSCIT",
        help="グラフモードで使用するデータセット"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        help="dpr_[split].json を指定 (例: dev, train)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="グラフ構築に使うtop-k文書数"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="NSPでエッジを張る際のIsNext確率しきい値"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="結果CSVの出力先ディレクトリ"
    )
    parser.add_argument(
        "--limit_turns",
        type=int,
        default=None,
        help="データセット先頭から指定数だけ処理（デバッグ用）"
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="tqdmの進捗バーを表示しない"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="NSP推論のバッチサイズ（デフォルト: 64、GPUメモリに応じて調整可能）"
    )
    parser.add_argument(
        "--no_multi_gpu",
        action="store_true",
        help="マルチGPUを使用しない（単一GPUまたはCPU）"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo_mode(args)
    else:
        run_graph_mode(args)


if __name__ == "__main__":
    main()

