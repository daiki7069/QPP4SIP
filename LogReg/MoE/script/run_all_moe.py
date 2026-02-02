#!/usr/bin/env python3
"""
全ゲーティング方式を一括実行し、結果を比較する。
使い方:
  script/run_all_moe.py --dataset AmbigNQ --no-cv
  script/run_all_moe.sh   # 上記を呼ぶラッパー
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# MoE/script/run_all_moe.py -> parent.parent = MoE, parent.parent.parent = LogReg, parent^4 = QPP4SIP
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MOE_DIR = BASE_DIR / "LogReg" / "MoE"


def run_one(dataset: str, gating_type: str, out_dir: Path, no_cv: bool, retrieval: str, **kwargs) -> bool:
    cmd = [
        sys.executable,
        str(MOE_DIR / "main.py"),
        "--dataset", dataset,
        "--gating-type", gating_type,
        "--output-dir", str(out_dir),
        "--retrieval-method", retrieval,
    ]
    if no_cv:
        cmd.append("--no-cv")
    if gating_type == "mlp":
        cmd.extend(["--gate-hidden-size", str(kwargs.get("gate_hidden_size", 32))])
    elif gating_type == "temperature":
        cmd.extend(["--gate-temperature", str(kwargs.get("gate_temperature", 0.5))])
    print(f"\n>>> Running: {' '.join(cmd)}")
    ret = subprocess.run(cmd, cwd=str(MOE_DIR))
    return ret.returncode == 0


def parse_results_txt(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    # ゲーティング方式
    m = re.search(r"ゲーティング方式:\s*(\S+)", text)
    gating = m.group(1) if m else None
    # Test 行 (--no-cv): "Test:  AUC=0.xxxx AP=0.xxxx Acc=0.xxxx F1=0.xxxx"
    m = re.search(r"Test:\s*AUC=([\d.]+)\s+AP=([\d.]+)\s+Acc=([\d.]+)\s+F1=([\d.]+)", text)
    if m:
        return {
            "gating": gating or path.parent.name,
            "AUC": float(m.group(1)),
            "AP": float(m.group(2)),
            "Acc": float(m.group(3)),
            "F1": float(m.group(4)),
            "mode": "test",
        }
    # CV 統合行（旧形式: --- CV 統合 ---）
    m = re.search(r"--- CV 統合 ---\s*\n\s*AUC=([\d.]+)\s+AP=([\d.]+)\s+Acc=([\d.]+)\s+F1=([\d.]+)", text)
    if m:
        return {
            "gating": gating or path.parent.name,
            "AUC": float(m.group(1)),
            "AP": float(m.group(2)),
            "Acc": float(m.group(3)),
            "F1": float(m.group(4)),
            "mode": "cv",
        }
    # CV 統合行（新形式: === テストデータの評価（CV統合） === ブロック内の値）
    block = re.search(r"=== テストデータの評価（CV統合） ===\s*\n(.*?)(?=\n\n|\nFold別:|\Z)", text, re.DOTALL)
    if block:
        block_text = block.group(1)
        m_auc = re.search(r"AUC-ROC:\s*([\d.]+)", block_text)
        m_ap = re.search(r"Average Precision:\s*([\d.]+)", block_text)
        m_acc = re.search(r"Accuracy:\s*([\d.]+)", block_text)
        m_f1 = re.search(r"F1 Score:\s*([\d.]+)", block_text)
        if m_auc and m_ap and m_acc and m_f1:
            return {
                "gating": gating or path.parent.name,
                "AUC": float(m_auc.group(1)),
                "AP": float(m_ap.group(1)),
                "Acc": float(m_acc.group(1)),
                "F1": float(m_f1.group(1)),
                "mode": "cv",
            }
    return None


def main():
    ap = argparse.ArgumentParser(description="全MoEゲーティング方式を一括実行して比較")
    ap.add_argument("--dataset", default="AmbigNQ", choices=["AmbigNQ", "INSCIT"])
    ap.add_argument("--retrieval-method", default="dpr", choices=["dpr", "bm25"])
    ap.add_argument("--no-cv", action="store_true", help="CVなし train/dev 分離評価")
    ap.add_argument("--gate-hidden-size", type=int, default=32)
    ap.add_argument("--gate-temperature", type=float, default=0.5)
    ap.add_argument("--output-dir", type=str, default=None, help="省略時は outputs/<dataset>/run_all_<timestamp>")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else (BASE_DIR / "LogReg" / "MoE" / "outputs" / args.dataset / f"run_all_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    gating_configs = [
        ("linear", {}),
        ("mlp", {"gate_hidden_size": args.gate_hidden_size}),
        ("temperature", {"gate_temperature": args.gate_temperature}),
        ("learned_fixed", {}),
    ]
    ok = True
    for gating_type, kw in gating_configs:
        if not run_one(args.dataset, gating_type, out_dir, args.no_cv, args.retrieval_method, **kw):
            ok = False
            print(f"Warning: {gating_type} failed.")

    # 結果を集約
    rows = []
    for d in sorted(out_dir.iterdir()):
        if not d.is_dir():
            continue
        rf = d / "results.txt"
        if not rf.exists():
            continue
        row = parse_results_txt(rf)
        if row:
            row["run_dir"] = d.name
            rows.append(row)
    if not rows:
        print("No results.txt found.")
        return 1 if not ok else 0

    df = pd.DataFrame(rows)
    # gating でソート（表示用の固定順）
    order = ["linear", "mlp", "temperature", "learned_fixed"]
    df["_order"] = df["gating"].map(lambda x: order.index(x) if x in order else 99)
    df = df.sort_values("_order").drop(columns=["_order"])
    csv_path = out_dir / "comparison.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print("\n" + "=" * 60)
    print("比較結果 (Test / CV)")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\n比較CSV: {csv_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
