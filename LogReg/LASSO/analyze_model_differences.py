"""
LogRegが正解し、BERTが外したクエリを分析するスクリプト
"""
import argparse
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


def load_json_data(json_path: Path) -> List[List[Dict]]:
    """JSONデータを読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_query_info(data: List[List[Dict]]) -> Dict[Tuple[str, int], Dict]:
    """
    クエリ情報を抽出
    戻り値: {(conv_id, turn_id): {query, label, ...}}
    """
    query_info = {}
    for conversation in data:
        for turn in conversation:
            conv_id = str(turn.get('conv_id', ''))
            turn_id = int(turn.get('turn_id', 0))
            key = (conv_id, turn_id)
            
            query = turn.get('query', turn.get('question', ''))
            label = turn.get('response_type', '')
            # response_typeを0/1に変換
            if label == 'clarification':
                label = 1
            elif label == 'answer':
                label = 0
            else:
                label = None
            
            query_info[key] = {
                'query': query,
                'label': label,
                'conv_id': conv_id,
                'turn_id': turn_id,
                'answer': turn.get('answer', turn.get('answers', '')),
                'ctxs': turn.get('ctxs', [])[:5] if 'ctxs' in turn else []  # 上位5件の検索結果
            }
    
    return query_info


def analyze_model_differences(
    results_csv_path: Path,
    dataset_json_path: Path,
    output_dir: Path,
    threshold: float = 0.5,
    base_model: str = None
):
    """
    LogRegが正解し、ベースモデル（BERT/RoBERTa）が外したクエリを分析
    
    Args:
        results_csv_path: results.csvのパス
        dataset_json_path: データセットのJSONファイル（dev.jsonなど）のパス
        output_dir: 出力ディレクトリ
        threshold: 予測の閾値（デフォルト: 0.5）
        base_model: ベースモデル名（'BERT'または'RoBERTa'、Noneの場合は自動検出）
    """
    print(f"=== LogRegが正解し、ベースモデルが外したクエリを分析 ===")
    print(f"results.csv: {results_csv_path}")
    print(f"dataset JSON: {dataset_json_path}")
    print(f"threshold: {threshold}")
    print()
    
    # results.csvを読み込む
    results_df = pd.read_csv(results_csv_path)
    print(f"results.csvを読み込み: {len(results_df)} 行")
    
    # 必要なカラムが存在するか確認
    if 'label' not in results_df.columns or 'LogReg' not in results_df.columns:
        print(f"エラー: 必要なカラムが見つかりません: label, LogReg")
        print(f"利用可能なカラム: {results_df.columns.tolist()}")
        return
    
    # ベースモデルを自動検出
    if base_model is None:
        if 'BERT' in results_df.columns:
            base_model = 'BERT'
        elif 'RoBERTa' in results_df.columns:
            base_model = 'RoBERTa'
        elif 'Transfer' in results_df.columns:
            base_model = 'Transfer'
        else:
            print(f"エラー: ベースモデルのカラム（BERT/RoBERTa/Transfer）が見つかりません")
            print(f"利用可能なカラム: {results_df.columns.tolist()}")
            return
    
    if base_model not in results_df.columns:
        print(f"エラー: 指定されたベースモデル '{base_model}' が見つかりません")
        print(f"利用可能なカラム: {results_df.columns.tolist()}")
        return
    
    print(f"ベースモデル: {base_model}")
    print()
    
    # データセットJSONを読み込む
    data = load_json_data(dataset_json_path)
    query_info = extract_query_info(data)
    print(f"データセットJSONを読み込み: {len(query_info)} クエリ")
    print()
    
    # 予測を計算
    # LogRegは確率値（0-1）なので閾値0.5を使用
    results_df['LogReg_pred'] = (results_df['LogReg'] >= 0.5).astype(int)
    
    # ベースモデル（BERT/RoBERTa/Transfer）はlogit値なので閾値0.0を使用
    # logit > 0 が正例、logit <= 0 が負例
    results_df[f'{base_model}_pred'] = (results_df[base_model] > 0.0).astype(int)
    
    # LogRegが正解し、ベースモデルが外したクエリを特定
    # 条件: LogRegの予測 == 正解ラベル かつ ベースモデルの予測 != 正解ラベル
    correct_logreg = results_df['LogReg_pred'] == results_df['label']
    wrong_base = results_df[f'{base_model}_pred'] != results_df['label']
    target_indices = results_df[correct_logreg & wrong_base].index
    
    print(f"LogRegが正解し、{base_model}が外したクエリ: {len(target_indices)} 件")
    print()
    
    if len(target_indices) == 0:
        print("該当するクエリが見つかりませんでした。")
        return
    
    # 詳細情報を収集
    detailed_results = []
    for idx in target_indices:
        row = results_df.iloc[idx]
        
        # クエリ情報を取得（results.csvの行番号とデータセットの対応を確認）
        # 注意: results.csvの行順序がデータセットの順序と一致していることを前提とする
        # より確実にするため、conv_idとturn_idでマッチングする必要がある場合がある
        
        result_dict = {
            'index': int(idx),
            'label': int(row['label']),
            'LogReg_score': float(row['LogReg']),
            'LogReg_pred': int(row['LogReg_pred']),
            f'{base_model}_score': float(row[base_model]),
            f'{base_model}_pred': int(row[f'{base_model}_pred']),
        }
        
        # 特徴量スコアを追加
        feature_columns = [col for col in results_df.columns 
                          if col not in ['label', 'LogReg', 'L1', 'L2', 'ENet', 'BERT', 'RoBERTa', 'Transfer']]
        for col in feature_columns:
            result_dict[col] = float(row[col]) if pd.notna(row[col]) else None
        
        # クエリテキストを取得（可能な場合）
        # 注意: results.csvの行順序がデータセットの順序と一致していることを前提とする
        if idx < len(data):
            # データセットからクエリ情報を取得
            # データセットは会話のリストのリスト形式
            flat_idx = 0
            found = False
            for conversation in data:
                for turn in conversation:
                    if flat_idx == idx:
                        result_dict['query'] = turn.get('query', turn.get('question', ''))
                        result_dict['answer'] = turn.get('answer', turn.get('answers', ''))
                        result_dict['conv_id'] = str(turn.get('conv_id', ''))
                        result_dict['turn_id'] = int(turn.get('turn_id', 0))
                        # 検索結果の上位3件のタイトルを取得
                        ctxs = turn.get('ctxs', [])[:3]
                        result_dict['retrieval_titles'] = [ctx.get('title', '') for ctx in ctxs]
                        found = True
                        break
                    flat_idx += 1
                if found:
                    break
        
        detailed_results.append(result_dict)
    
    # DataFrameに変換
    detailed_df = pd.DataFrame(detailed_results)
    
    # 統計情報を表示
    print("=== 統計情報 ===")
    print(f"総数: {len(detailed_df)} 件")
    print(f"正解ラベル分布:")
    print(detailed_df['label'].value_counts().to_dict())
    print()
    
    # 特徴量の統計
    print("=== 特徴量の統計（LogReg正解・ベースモデル外し vs 全体） ===")
    # 統計に使用する特徴量カラム（モデルスコアや予測結果、メタ情報を除外）
    exclude_cols = ['index', 'label', 'LogReg_score', 'LogReg_pred', 
                    'BERT_score', 'BERT_pred', 'RoBERTa_score', 'RoBERTa_pred',
                    'Transfer_score', 'Transfer_pred', 'query', 'answer', 
                    'conv_id', 'turn_id', 'retrieval_titles', 'query_length']
    feature_columns = [col for col in detailed_df.columns if col not in exclude_cols]
    
    for col in feature_columns:
        # results_dfにも存在する特徴量のみ統計を計算
        if col in results_df.columns and detailed_df[col].dtype in [np.float64, np.int64]:
            target_mean = detailed_df[col].mean()
            all_mean = results_df[col].mean()
            diff = target_mean - all_mean
            print(f"{col:20s}: 対象={target_mean:8.4f}, 全体={all_mean:8.4f}, 差={diff:8.4f}")
    print()
    
    # クエリ長の統計
    if 'query' in detailed_df.columns:
        detailed_df['query_length'] = detailed_df['query'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)
        all_query_lengths = []
        for conversation in data:
            for turn in conversation:
                query = turn.get('query', turn.get('question', ''))
                all_query_lengths.append(len(str(query).split()))
        
        print("=== クエリ長の統計 ===")
        print(f"対象クエリの平均単語数: {detailed_df['query_length'].mean():.2f}")
        print(f"全体の平均単語数: {np.mean(all_query_lengths):.2f}")
        print(f"対象クエリの最小単語数: {detailed_df['query_length'].min()}")
        print(f"対象クエリの最大単語数: {detailed_df['query_length'].max()}")
        print()
    
    # CSVファイルに保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / f"logreg_correct_{base_model.lower()}_wrong.csv"
    detailed_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"詳細結果を保存: {output_csv_path}")
    
    # クエリテキストのみのリストも保存
    if 'query' in detailed_df.columns:
        query_list_path = output_dir / f"logreg_correct_{base_model.lower()}_wrong_queries.txt"
        base_score_col = f'{base_model}_score'
        with open(query_list_path, 'w', encoding='utf-8') as f:
            for idx, row in detailed_df.iterrows():
                query = row.get('query', '') if 'query' in row else ''
                label = row.get('label', '') if 'label' in row else ''
                logreg_score = row.get('LogReg_score', '') if 'LogReg_score' in row else ''
                base_score = row.get(base_score_col, '') if base_score_col in row else ''
                f.write(f"Label: {label}, LogReg: {logreg_score:.4f}, {base_model}: {base_score:.4f}\n")
                f.write(f"Query: {query}\n")
                f.write("-" * 80 + "\n")
        print(f"クエリリストを保存: {query_list_path}")
    
    print()
    print("=== 分析完了 ===")


def main():
    parser = argparse.ArgumentParser(
        description="LogRegが正解し、BERTが外したクエリを分析"
    )
    parser.add_argument(
        "--results-csv",
        type=str,
        required=True,
        help="results.csvのパス"
    )
    parser.add_argument(
        "--dataset-json",
        type=str,
        required=True,
        help="データセットのJSONファイル（dev.jsonなど）のパス"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリ"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="予測の閾値（デフォルト: 0.5）"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        choices=['BERT', 'RoBERTa', 'Transfer'],
        help="ベースモデル名（BERT/RoBERTa/Transfer、Noneの場合は自動検出）"
    )
    
    args = parser.parse_args()
    
    results_csv_path = Path(args.results_csv)
    dataset_json_path = Path(args.dataset_json)
    output_dir = Path(args.output_dir)
    
    if not results_csv_path.exists():
        print(f"エラー: results.csvが見つかりません: {results_csv_path}")
        return
    
    if not dataset_json_path.exists():
        print(f"エラー: データセットJSONが見つかりません: {dataset_json_path}")
        return
    
    analyze_model_differences(
        results_csv_path=results_csv_path,
        dataset_json_path=dataset_json_path,
        output_dir=output_dir,
        threshold=args.threshold,
        base_model=args.base_model
    )


if __name__ == "__main__":
    main()

