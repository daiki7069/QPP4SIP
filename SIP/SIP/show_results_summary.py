#!/usr/bin/env python3
"""
推論結果のサマリーを表示するスクリプト
"""

import argparse
import os
import re
import numpy as np

def parse_result_file(result_file):
    """結果ファイルを解析して辞書のリストを返す"""
    results = []
    
    if not os.path.exists(result_file):
        print(f"結果ファイルが見つかりません: {result_file}")
        return results
    
    with open(result_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # エポック番号と結果辞書を抽出
            match = re.match(r'(\d+):\s*(.+)', line)
            if match:
                epoch = int(match.group(1))
                result_str = match.group(2)
                
                # 辞書文字列を評価（安全のため制限）
                try:
                    result_dict = eval(result_str)
                    results.append((epoch, result_dict))
                except:
                    print(f"エポック {epoch} の結果を解析できませんでした")
                    continue
    
    return results

def print_summary(results, dataset_type):
    """結果のサマリーを表示"""
    if not results:
        print("結果が見つかりませんでした。")
        return
    
    print(f"\n=== {dataset_type.upper()} データセット 評価結果サマリー ===")
    print(f"評価エポック数: {len(results)}")
    print()
    
    # ヘッダー
    print(f"{'エポック':<8} {'精度':<8} {'再現率':<8} {'F1スコア':<8} {'正解率':<8} {'Initiative':<12} {'Non-initiative':<15}")
    print("-" * 80)
    
    # 各エポックの結果
    for epoch, result in results:
        f1 = result.get('f1', 0)
        precision = result.get('p', 0)
        recall = result.get('r', 0)
        accuracy = result.get('acc', 0)
        
        # ラベル別の結果
        acc_per_label = result.get('acc_per_label', [0, 0])
        total_num = result.get('total_num', [0, 0])
        hit_num = result.get('hit_num', [0, 0])
        
        initiative_acc = acc_per_label[1] if len(acc_per_label) > 1 else 0
        non_initiative_acc = acc_per_label[0] if len(acc_per_label) > 0 else 0
        
        print(f"{epoch:<8} {precision:<8.2f} {recall:<8.2f} {f1:<8.2f} {accuracy:<8.2f} "
              f"{initiative_acc:<12.2f} {non_initiative_acc:<15.2f}")
    
    # 最良の結果を表示
    if results:
        best_f1 = max(results, key=lambda x: x[1].get('f1', 0))
        best_acc = max(results, key=lambda x: x[1].get('acc', 0))
        
        print("\n=== 最良の結果 ===")
        print(f"最高F1スコア: {best_f1[1].get('f1', 0):.2f} (エポック {best_f1[0]})")
        print(f"最高正解率: {best_acc[1].get('acc', 0):.2f} (エポック {best_acc[0]})")
        
        # 詳細情報
        print(f"\n=== エポック {best_f1[0]} の詳細結果 ===")
        result = best_f1[1]
        print(f"精度 (Precision): {result.get('p', 0):.2f}")
        print(f"再現率 (Recall): {result.get('r', 0):.2f}")
        print(f"F1スコア: {result.get('f1', 0):.2f}")
        print(f"正解率 (Accuracy): {result.get('acc', 0):.2f}")
        
        total_num = result.get('total_num', [0, 0])
        hit_num = result.get('hit_num', [0, 0])
        print(f"\nラベル別結果:")
        print(f"  Non-initiative: {hit_num[0]}/{total_num[0]} ({acc_per_label[0]:.2f}%)")
        print(f"  Initiative: {hit_num[1]}/{total_num[1]} ({acc_per_label[1]:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='推論結果のサマリーを表示・保存')
    parser.add_argument('--result_file', type=str, required=True, help='結果ファイルのパス')
    parser.add_argument('--dataset_type', type=str, default='dev', help='データセットタイプ')
    parser.add_argument('--output_file', type=str, help='サマリーを保存するファイルパス（指定しない場合は表示のみ）')
    
    args = parser.parse_args()
    
    results = parse_result_file(args.result_file)
    
    if args.output_file:
        # ファイルに保存
        with open(args.output_file, 'w', encoding='utf-8') as f:
            import sys
            from io import StringIO
            
            # 標準出力をキャプチャ
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            print_summary(results, args.dataset_type)
            
            # キャプチャした内容を取得
            summary_text = sys.stdout.getvalue()
            sys.stdout = old_stdout
            
            # ファイルに書き込み
            f.write(summary_text)
            print(f"サマリーを保存しました: {args.output_file}")
    else:
        # 表示のみ
        print_summary(results, args.dataset_type)

if __name__ == "__main__":
    main()
