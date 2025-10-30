#!/usr/bin/env python3

import argparse
import json
import os
from typing import List, Dict, Any


def build_internal_INSCIT_from_original(original_data: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    """
    original INSCIT JSON (NASの data/{split}.json) を内部形式に変換する。
    内部形式INSCITの各ターンは以下のキーを持つ:
      - conv_id
      - turn_id
      - query
      - answer(list[str])
      - response_type(str)
      - dialogue_history(list[str])
    """
    conversations: List[List[Dict[str, Any]]] = []

    for conv_id, dialogue in original_data.items():
        turns = dialogue.get("turns", [])
        new_conv: List[Dict[str, Any]] = []

        for idx, turn in enumerate(turns):
            turn_id = idx + 1

            ctx_list = turn.get("context", None)
            query_text = ctx_list[-1]
            
            labels = turn.get("labels", []) or []
            answers: List[str] = []
            response_types: List[str] = []
            for label in labels:
                resp = label.get("response", "")
                answers.append(resp)
                rtype = label.get("responseType", "")
                response_types.append(rtype)

            response_type_joined = " [SEP] ".join(response_types) if response_types else ""

            new_turn = {
                "conv_id": conv_id,
                "turn_id": turn_id,
                "query": query_text,
                "answer": answers,
                "response_type": response_type_joined,
                "dialogue_history": ctx_list,
            }
            new_conv.append(new_turn)

        conversations.append(new_conv)

    return conversations


def main():
    parser = argparse.ArgumentParser(description="Create base INSCIT json from original NAS data")
    parser.add_argument("--original_data_file", required=True, help="INSCITの元データのフルファイルパスを直接指定")
    parser.add_argument("--output_dir", required=True, help="出力ディレクトリ")
    args = parser.parse_args()

    src_path = args.original_data_file
    out_dir = args.output_dir
    out_filename = os.path.basename(src_path)
    out_path = os.path.join(out_dir, out_filename)

    with open(src_path, "r", encoding="utf-8") as f:
        original = json.load(f)

    conversations = build_internal_INSCIT_from_original(original)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, ensure_ascii=False, indent=2)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
