#!/usr/bin/env python3
"""
複数シード推論の結果を集計するスクリプト
"""
import os
import sys
import argparse
import numpy as np
from collections import defaultdict

def parse_inference_file(file_path):
    """
    推論結果ファイルを解析する関数
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # ヘッダーをスキップ
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    turn_id = parts[0]
                    predicted = parts[1]
                    true_label = parts[2]
                    qpp_value = float(parts[3])
                    initiative_prob = float(parts[4])
                    resolved_query = parts[5]
                    
                    results.append({
                        'turn_id': turn_id,
                        'predicted': predicted,
                        'true_label': true_label,
                        'qpp_value': qpp_value,
                        'initiative_prob': initiative_prob,
                        'resolved_query': resolved_query
                    })
    return results

def calculate_metrics(predictions, true_labels):
    """
    評価指標を計算する関数
    """
    # 精度
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    # F1スコア、Precision、Recall
    tp = fp = fn = 0
    for pred, true in zip(predictions, true_labels):
        if pred == "Initiative" and true == "Initiative":
            tp += 1
        elif pred == "Initiative" and true == "Non-initiative":
            fp += 1
        elif pred == "Non-initiative" and true == "Initiative":
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, f1, precision, recall

def aggregate_results(output_dir, num_runs=10, target_epoch=1):
    """
    複数シード推論の結果を集計する関数
    """
    print(f"複数シード推論結果を集計中... (実行回数: {num_runs}, 対象エポック: {target_epoch})")
    
    # 各エポックの結果を集計
    epoch_results = {}
    
    # ファイルを検索
    for filename in os.listdir(output_dir):
        if filename.startswith("dev_") and filename.endswith(".txt"):
            # ファイル名からエポックと実行番号を抽出
            # dev_1_1.1.txt -> ["1", "1.1"] -> epoch_id=1, run_id=1
            parts = filename.replace("dev_", "").replace(".txt", "").split("_")
            if len(parts) >= 2:
                epoch_id = int(parts[0])
                # run_idは最初の数字部分のみを取得
                run_id_str = parts[1].split(".")[0]
                run_id = int(run_id_str)
                
                # 特定のエポックのみを処理
                if epoch_id == target_epoch:
                    if epoch_id not in epoch_results:
                        epoch_results[epoch_id] = []
                    
                    file_path = os.path.join(output_dir, filename)
                    results = parse_inference_file(file_path)
                    epoch_results[epoch_id].append((run_id, results))
    
    # 各エポックの結果を集計
    for epoch_id in sorted(epoch_results.keys()):
        print(f"\n=== エポック {epoch_id} ===")
        
        runs = epoch_results[epoch_id]
        if len(runs) != num_runs:
            print(f"警告: エポック {epoch_id} の実行回数が {len(runs)} です（期待値: {num_runs}）")
        
        # 各サンプルについて、全実行での予測を集計
        if not runs:
            continue
            
        # 最初の実行の結果をベースにする
        base_results = runs[0][1]
        num_samples = len(base_results)
        
        # 各サンプルについて、全実行での予測を集計
        averaged_predictions = []
        averaged_probs = []
        
        for i in range(num_samples):
            # 各サンプルについて、全実行での予測を集計
            predictions = []
            probs = []
            
            for run_id, results in runs:
                if i < len(results):
                    predictions.append(results[i]['predicted'])
                    probs.append(results[i]['initiative_prob'])
            
            # 予測の多数決
            initiative_count = predictions.count("Initiative")
            if initiative_count > len(predictions) / 2:
                averaged_predictions.append("Initiative")
            else:
                averaged_predictions.append("Non-initiative")
            
            # 確率の平均
            avg_prob = sum(probs) / len(probs) if probs else 0.0
            averaged_probs.append(avg_prob)
        
        # 評価指標を計算
        true_labels = [result['true_label'] for result in base_results]
        accuracy, f1, precision, recall = calculate_metrics(averaged_predictions, true_labels)
        
        print(f"精度 (Accuracy): {accuracy:.4f}")
        print(f"F1スコア: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # 結果をファイルに保存
        result_file = os.path.join(output_dir, f"multi_seed_aggregated_epoch_{epoch_id}.txt")
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"複数シード推論集計結果 (エポック {epoch_id})\n")
            f.write(f"実行回数: {len(runs)}\n\n")
            
            f.write("=== 評価指標 ===\n")
            f.write(f"精度 (Accuracy): {accuracy:.4f}\n")
            f.write(f"F1スコア: {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n\n")
            
            f.write("=== 詳細結果 ===\n")
            f.write("Turn ID\t予測\t正解\tQPP値\t予測確率\tResolved Query\n")
            for i, (turn_id, pred, true, qpp, prob, query) in enumerate(zip(
                [result['turn_id'] for result in base_results],
                averaged_predictions,
                true_labels,
                [result['qpp_value'] for result in base_results],
                averaged_probs,
                [result['resolved_query'] for result in base_results]
            )):
                f.write(f"{turn_id}\t{pred}\t{true}\t{qpp:.4f}\t{prob:.4f}\t{query}\n")
        
        print(f"集計結果を保存しました: {result_file}")

def main():
    parser = argparse.ArgumentParser(description="複数シード推論の結果を集計する")
    parser.add_argument("--output_dir", type=str, required=True, help="出力ディレクトリ")
    parser.add_argument("--num_runs", type=int, default=10, help="実行回数")
    parser.add_argument("--target_epoch", type=int, default=1, help="対象エポック")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"エラー: 出力ディレクトリが見つかりません: {args.output_dir}")
        sys.exit(1)
    
    aggregate_results(args.output_dir, args.num_runs, args.target_epoch)
    print("\n=== 集計完了 ===")

if __name__ == "__main__":
    main()
