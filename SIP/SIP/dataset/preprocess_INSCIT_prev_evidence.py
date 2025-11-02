#!/usr/bin/env python3

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Tuple


CSV_COLUMNS = [
    "num_retrieved_docs",
    "num_evidence_docs",
    "num_prev_evidence_docs",
    "found_ratio",
    "ndcg@1",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "ndcg@50",
]


def load_internal_json(json_path: str) -> List[List[Dict[str, Any]]]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    # 期待構造: [[{turn...}, ...], ...] ただし単一会話のフラットな可能性もある
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return [data]
    return data


def normalize_metrics(
    key_to_metrics: Dict[Tuple[str, int], Dict[str, Any]]
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """
    全メトリクスを0~1の範囲に正規化する
    """
    # 各メトリクスの最大値を計算
    max_values: Dict[str, float] = {}
    for col in CSV_COLUMNS:
        max_values[col] = 0.0
        for metrics in key_to_metrics.values():
            val = metrics.get(col, 0)
            if isinstance(val, (int, float)):
                max_values[col] = max(max_values[col], float(val))
    
    # 正規化を適用（最大値が0の場合は1に正規化、つまり0のまま）
    normalized: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, metrics in key_to_metrics.items():
        normalized_metrics: Dict[str, Any] = {}
        for col in CSV_COLUMNS:
            val = metrics.get(col, 0)
            if isinstance(val, (int, float)):
                max_val = max_values[col]
                if max_val > 0:
                    normalized_metrics[col] = float(val) / max_val
                else:
                    normalized_metrics[col] = 0.0
            else:
                normalized_metrics[col] = 0.0
        normalized[key] = normalized_metrics
    
    return normalized


def load_csv_metrics(csv_path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    key_to_metrics: Dict[Tuple[str, int], Dict[str, Any]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conv_id = row["conv_id"].strip()
            turn_id = int(row["turn_id"])  # CSVのturn_idは整数想定
            metrics: Dict[str, Any] = {}
            for col in CSV_COLUMNS:
                val = row[col]
                if col in ("num_retrieved_docs", "num_evidence_docs", "num_prev_evidence_docs"):
                    try:
                        metrics[col] = int(val)
                    except ValueError:
                        metrics[col] = 0
                else:
                    try:
                        metrics[col] = float(val)
                    except ValueError:
                        metrics[col] = 0.0
            key_to_metrics[(conv_id, turn_id)] = metrics
    
    # 全メトリクスを0~1に正規化
    key_to_metrics = normalize_metrics(key_to_metrics)
    return key_to_metrics


def attach_metrics_to_internal(
    conversations: List[List[Dict[str, Any]]],
    key_to_metrics: Dict[Tuple[str, int], Dict[str, Any]],
) -> Tuple[List[List[Dict[str, Any]]], int]:
    matched = 0
    for conv in conversations:
        for turn in conv:
            conv_id = str(turn.get("conv_id", "")).strip()
            turn_id = int(turn.get("turn_id", 0))
            m = key_to_metrics.get((conv_id, turn_id))
            if m is None:
                continue
            for k, v in m.items():
                turn[k] = v
            matched += 1
    return conversations, matched


def split_response_types(response_type_joined: str, answers: List[str]) -> List[str]:
    if not response_type_joined:
        return [""] * len(answers)
    parts = [p.strip() for p in response_type_joined.split("[SEP]")]
    # join元は " x [SEP] y" のように前後に空白がある可能性があるためトリム
    parts = [p.replace("  ", " ").replace(" [SEP]", "").strip() for p in parts]
    # 個数不一致の時はanswersに合わせて切詰め/パディング
    if len(parts) < len(answers):
        parts = parts + ([""] * (len(answers) - len(parts)))
    elif len(parts) > len(answers):
        parts = parts[: len(answers)]
    return parts


def build_internal_with_metrics(
    conversations: List[List[Dict[str, Any]]]
) -> List[List[Dict[str, Any]]]:
    """
    内部形式の骨格（dev.jsonと同じ: [[turn, ...], ...]）を維持しつつ、
    各ターンにCSVメトリクス（CSV_COLUMNS）を追記した配列を返す。
    """
    out: List[List[Dict[str, Any]]] = []
    for conv in conversations:
        new_conv: List[Dict[str, Any]] = []
        for turn in conv:
            new_turn = dict(turn)
            # 既知のCSV列のみを明示的に保持（他の内部キーはそのまま）
            for k in CSV_COLUMNS:
                if k in turn:
                    new_turn[k] = turn[k]
            new_conv.append(new_turn)
        out.append(new_conv)
    return out


def compute_internal_stats(conversations: List[List[Dict[str, Any]]]) -> Tuple[int, int, Dict[str, int]]:
    num_dialogues = len(conversations)
    num_turns = sum(len(c) for c in conversations)
    label_dist: Dict[str, int] = {}
    for conv in conversations:
        for turn in conv:
            answers: List[str] = turn.get("answer", []) or []
            rtypes = split_response_types(str(turn.get("response_type", "")), answers)
            for rt in rtypes:
                key = rt or ""
                label_dist[key] = label_dist.get(key, 0) + 1
    return num_dialogues, num_turns, label_dist


# 原形式向けの統計は不要


# ルート推測やsplit依存の経路推定は不要のため削除


def main() -> None:
    parser = argparse.ArgumentParser(
        description="INSCIT内部形式JSONにCSVメトリクスを結合し、元のJSON形式へ復元（シンプル版）"
    )
    # 入力JSON/CSVと出力のディレクトリ＋ファイル名（必須・シンプル）
    parser.add_argument("--input_json", required=True, help="内部形式JSONのパス")
    parser.add_argument("--input_csv", required=True, help="CSVのパス")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    parser.add_argument("--output_filename", required=True, help="出力ファイル名（例: dev_dpr_with_prev.json）")
    args = parser.parse_args()

    internal_json = args.input_json
    csv_path = args.input_csv
    out_json = os.path.join(args.output_dir, args.output_filename)

    conversations = load_internal_json(internal_json)
    key_to_metrics = load_csv_metrics(csv_path)

    # 前統計
    before_num_dialogues, before_num_turns, before_label_dist = compute_internal_stats(conversations)

    conversations_with_metrics, matched_pairs = attach_metrics_to_internal(conversations, key_to_metrics)
    output_conversations = build_internal_with_metrics(conversations_with_metrics)

    # 後統計（内部形式と同じ骨格）
    after_num_dialogues, after_num_turns, after_label_dist = compute_internal_stats(output_conversations)

    # マッチ状況
    csv_rows = len(key_to_metrics)
    unmatched_in_json = max(before_num_turns - matched_pairs, 0)
    unmatched_in_csv = max(csv_rows - matched_pairs, 0)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output_conversations, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {out_json}")
    print("Paths:")
    print(f"  input_json: {internal_json}")
    print(f"  input_csv:  {csv_path}")
    print(f"  output:     {out_json}")

    # 統計出力
    print("=== Stats: Before (internal) ===")
    print(f"dialogues: {before_num_dialogues}")
    print(f"turns: {before_num_turns}")
    print(f"label_dist: {before_label_dist}")

    print("=== Stats: After (original) ===")
    print(f"dialogues: {after_num_dialogues}")
    print(f"turns: {after_num_turns}")
    print(f"label_dist: {after_label_dist}")

    print("=== Matching (CSV vs JSON) ===")
    print(f"csv_rows: {csv_rows}")
    print(f"matched_pairs: {matched_pairs}")
    print(f"unmatched_in_json (turns without CSV): {unmatched_in_json}")
    print(f"unmatched_in_csv (CSV rows without JSON): {unmatched_in_csv}")


if __name__ == "__main__":
    main()


