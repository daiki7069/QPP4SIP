#!/usr/bin/env python3
"""
TSVファイルをPKLファイルに変換するスクリプト
既存のSIP処理パイプラインと互換性を保つため、TSVデータを既存のPKL形式に変換する
"""

import pandas as pd
import pickle
import argparse
import os
from typing import Dict, List, Any
import json

def convert_tsv_to_pkl(tsv_file: str, output_pkl: str) -> None:
    """
    TSVファイルを既存のPKL形式に変換する
    
    Args:
        tsv_file: 入力TSVファイルのパス
        output_pkl: 出力PKLファイルのパス
    """
    print(f"TSVファイルを読み込み中: {tsv_file}")
    
    # TSVファイルを読み込み
    df = pd.read_csv(tsv_file, sep='\t')
    print(f"読み込んだ行数: {len(df)}")
    
    # データを会話単位でグループ化
    conversations = {}
    
    for _, row in df.iterrows():
        dialogue_id = row['dialogue_id']
        turn_id = row['turn_id']
        
        if dialogue_id not in conversations:
            conversations[dialogue_id] = {
                'turns': [],
                'dialogue_id': dialogue_id
            }
        
        # SIPラベルの決定（clarification -> 1, その他 -> 0）
        sip_label = 1 if 'clarification' in str(row['response_type']) else 0
        
        # QPP特徴量を抽出
        qpp_features = extract_qpp_features(row)
        
        # ターンデータを作成（既存の形式に合わせる）
        turn_data = {
            'turn_id': turn_id,
            'user_utterance': row['query'],
            'system_utterance': row['response'],
            'user_I_label': 'no_initiative',  # ユーザーは常に質問のみ
            'system_I_label': 'clarification' if sip_label == 1 else 'no_initiative',
            'qpp_features': qpp_features,
            'resolved_query': row['query'],
            'response_type': row['response_type'],
            'dialogue_history': row['dialogue_history']
        }
        
        conversations[dialogue_id]['turns'].append(turn_data)
    
    # 会話をターン順でソート
    for dialogue_id in conversations:
        conversations[dialogue_id]['turns'].sort(key=lambda x: x['turn_id'])
    
    print(f"変換された会話数: {len(conversations)}")
    
    # PKLファイルとして保存
    with open(output_pkl, 'wb') as f:
        pickle.dump(conversations, f)
    
    print(f"PKLファイルを保存しました: {output_pkl}")

def extract_qpp_features(row) -> Dict[str, float]:
    """QPP特徴量を抽出"""
    qpp_feature_names = [
        'mrr', 'found_ratio', 'mean_rank',
        'hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50',
        'precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50',
        'recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50',
        'f1@1', 'f1@5', 'f1@10', 'f1@20', 'f1@50',
        'ndcg@1', 'ndcg@3', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50'
    ]
    
    qpp_features = {}
    for name in qpp_feature_names:
        if name in row:
            qpp_features[name] = float(row[name]) if pd.notna(row[name]) else 0.0
        else:
            qpp_features[name] = 0.0
    
    return qpp_features

def convert_multiple_tsv_files(tsv_files: List[str], output_dir: str) -> None:
    """
    複数のTSVファイルを一括でPKLに変換する
    
    Args:
        tsv_files: TSVファイルのパスのリスト
        output_dir: 出力ディレクトリ
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for tsv_file in tsv_files:
        # ファイル名から出力ファイル名を生成
        base_name = os.path.splitext(os.path.basename(tsv_file))[0]
        output_pkl = os.path.join(output_dir, f"{base_name}.pkl")
        
        print(f"\n=== {tsv_file} を変換中 ===")
        convert_tsv_to_pkl(tsv_file, output_pkl)

def main():
    parser = argparse.ArgumentParser(description='TSVファイルをPKLファイルに変換')
    parser.add_argument('--input_tsv', type=str, 
                       help='入力TSVファイルのパス')
    parser.add_argument('--output_pkl', type=str,
                       help='出力PKLファイルのパス')
    parser.add_argument('--batch_convert', action='store_true',
                       help='複数ファイルの一括変換モード')
    parser.add_argument('--tsv_files', nargs='+', 
                       help='一括変換するTSVファイルのリスト')
    parser.add_argument('--output_dir', type=str,
                       help='一括変換時の出力ディレクトリ')
    
    args = parser.parse_args()
    
    if args.batch_convert:
        if not args.tsv_files or not args.output_dir:
            print("一括変換モードでは --tsv_files と --output_dir が必要です")
            return
        convert_multiple_tsv_files(args.tsv_files, args.output_dir)
    else:
        if not args.input_tsv or not args.output_pkl:
            print("単一ファイル変換モードでは --input_tsv と --output_pkl が必要です")
            return
        convert_tsv_to_pkl(args.input_tsv, args.output_pkl)

if __name__ == "__main__":
    main()
