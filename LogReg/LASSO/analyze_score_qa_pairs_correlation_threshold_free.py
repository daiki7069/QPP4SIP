#!/usr/bin/env python3
"""
閾値に依存しない方法で、各QAペア数カテゴリでの平均スコアを計算
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# 既存の分析結果を読み込む
analysis_dir = Path('/home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO/outputs/AmbigNQ/score_qa_pairs_analysis')

# データを読み込む（元のスクリプトから生成されたデータを使用）
# 実際には、元のスクリプトを実行してから、その結果を読み込む必要がある

def calculate_mean_scores_by_category():
    """各カテゴリでの平均スコアを計算（閾値に依存しない）"""
    
    # 元のデータを読み込む（dev.jsonから）
    data_path = Path('/home/daiki_shibata/pj/QPP4SIP/dataset/AmbigNQ/dev.json')
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # スコアを読み込む
    import sys
    sys.path.append('/home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO')
    from analyze_score_qa_pairs_correlation import (
        load_qa_pairs_count, load_bert_scores, load_qpp_scores,
        load_integrated_score, create_qa_pairs_category
    )
    
    qa_pairs_dict = load_qa_pairs_count(data_path)
    bert_scores = load_bert_scores(
        Path('/home/daiki_shibata/pj/QPP4SIP/SIP/FT-PLM/output/AmbigNQ/AmbigNQ_bert-base_lr2e-05_bs16_kfold5/dev_with_predictions.json')
    )
    qpp_scores_dict = load_qpp_scores(Path('/home/daiki_shibata/pj/QPP4SIP/dataset/AmbigNQ'), split='dev')
    
    # データを結合
    data_list = []
    for key in set(qa_pairs_dict.keys()) | set(bert_scores.keys()):
        conv_id, turn_id = key
        row = {
            'conv_id': conv_id,
            'turn_id': turn_id,
            'qa_pairs_count': qa_pairs_dict.get(key, 0),
            'BERT': bert_scores.get(key, np.nan),
        }
        
        # QPPスコア（NQCのみ、符号反転）
        if 'nqc' in qpp_scores_dict and key in qpp_scores_dict['nqc']:
            row['QPP'] = -qpp_scores_dict['nqc'][key]
        else:
            row['QPP'] = np.nan
        
        row['qa_pairs_category'] = create_qa_pairs_category(row['qa_pairs_count'])
        data_list.append(row)
    
    df = pd.DataFrame(data_list)
    
    # 各カテゴリでの平均スコアを計算
    results = []
    for method in ['QPP', 'BERT']:
        for category in ['1 QA pair', '2 QA pairs', '3 QA pairs', '4+ QA pairs']:
            mask = (df['qa_pairs_category'] == category) & ~pd.isna(df[method])
            if mask.sum() > 0:
                mean_score = df.loc[mask, method].mean()
                std_score = df.loc[mask, method].std()
                median_score = df.loc[mask, method].median()
                results.append({
                    'method': method,
                    'qa_pairs_category': category,
                    'mean_score': mean_score,
                    'std_score': std_score,
                    'median_score': median_score,
                    'count': mask.sum()
                })
    
    results_df = pd.DataFrame(results)
    print("\n=== 閾値に依存しない方法：各カテゴリでの平均スコア ===")
    print(results_df.pivot(index='method', columns='qa_pairs_category', values='mean_score').round(4))
    
    print("\n=== 各カテゴリでの中央値スコア ===")
    print(results_df.pivot(index='method', columns='qa_pairs_category', values='median_score').round(4))
    
    # CSVに保存
    results_df.to_csv(analysis_dir / 'mean_scores_by_category.csv', index=False)
    print(f"\n結果を保存しました: {analysis_dir / 'mean_scores_by_category.csv'}")

if __name__ == '__main__':
    calculate_mean_scores_by_category()

