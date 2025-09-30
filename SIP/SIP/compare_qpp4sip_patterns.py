#!/usr/bin/env python3
"""
QPP4SIPの3つのパターンを比較するスクリプト
"""

import os
import subprocess
import time
from pathlib import Path

def run_experiment(pattern, epochs=20):
    """指定されたパターンで実験を実行"""
    print(f"\n🚀 {pattern} パターンの学習を開始...")
    
    cmd = [
        "uv", "run", "python", "model/run.py",
        "--mode", "train",
        "--task", "SIP",
        "--name", "QPP4SIP",
        "--dataset", "QPP4SIP",
        "--model", "qpp4sip",
        "--qpp4sip_pattern", pattern,
        "--input_path", "./dataset/pkl/bm25_train.pkl",
        "--output_path", f"./output/qpp4sip_{pattern}",
        "--saved_model_path", f"./checkpoints/qpp4sip_{pattern}",
        "--log_path", f"./logs/qpp4sip_{pattern}",
        "--hidden_size", "768",
        "--dropout", "0.1",
        "--BiLSTM_layers", "1",
        "--epoch_num", str(epochs),
        "--batch_size", "1",
        "--learning_rate", "2e-5",
        "--lr_crf", "1e-3",
        "--accumulation_steps", "1",
        "--clip", "1.0",
        "--max_utterance_len", "128",
        "--max_context_len", "384",
        "--random_seed", "42",
        "--class_imbalance_ratio", "6.5",
        "--focal_gamma", "2.0"
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        print(f"✅ {pattern} パターンの学習が完了しました (所要時間: {end_time - start_time:.1f}秒)")
        return True
    else:
        print(f"❌ {pattern} パターンの学習に失敗しました")
        print(f"エラー: {result.stderr}")
        return False

def run_inference(pattern, epochs=20):
    """指定されたパターンで推論を実行"""
    print(f"\n🔍 {pattern} パターンの推論を開始...")
    
    cmd = [
        "uv", "run", "python", "model/run.py",
        "--mode", "inference",
        "--task", "SIP",
        "--name", "QPP4SIP",
        "--dataset", "QPP4SIP",
        "--model", "qpp4sip",
        "--qpp4sip_pattern", pattern,
        "--input_path", "./dataset/pkl/bm25_dev.pkl",
        "--output_path", f"./output/qpp4sip_{pattern}",
        "--saved_model_path", f"./checkpoints/qpp4sip_{pattern}",
        "--log_path", f"./logs/qpp4sip_{pattern}",
        "--hidden_size", "768",
        "--dropout", "0.1",
        "--BiLSTM_layers", "1",
        "--epoch_num", str(epochs),
        "--batch_size", "1",
        "--learning_rate", "2e-5",
        "--lr_crf", "1e-3",
        "--accumulation_steps", "1",
        "--clip", "1.0",
        "--max_utterance_len", "128",
        "--max_context_len", "384",
        "--random_seed", "42",
        "--class_imbalance_ratio", "6.5",
        "--focal_gamma", "2.0"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {pattern} パターンの推論が完了しました")
        return True
    else:
        print(f"❌ {pattern} パターンの推論に失敗しました")
        print(f"エラー: {result.stderr}")
        return False

def run_evaluation(pattern, epochs=20):
    """指定されたパターンで評価を実行"""
    print(f"\n📊 {pattern} パターンの評価を開始...")
    
    cmd = [
        "uv", "run", "python", "evaluation/evaluation.py",
        "--prediction_path", f"./output/qpp4sip_{pattern}",
        "--label_path", "./dataset/pkl/bm25_dev.pkl",
        "--task", "SIP",
        "--dataset_type", "dev",
        "--epoch_num", str(epochs),
        "--dataset", "QPP4SIP"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {pattern} パターンの評価が完了しました")
        return True
    else:
        print(f"❌ {pattern} パターンの評価に失敗しました")
        print(f"エラー: {result.stderr}")
        return False

def generate_summary(pattern):
    """指定されたパターンのサマリーを生成"""
    print(f"\n📋 {pattern} パターンのサマリーを生成...")
    
    cmd = [
        "uv", "run", "python", "show_results_summary.py",
        "--result_file", f"./output/qpp4sip_{pattern}/result.dev.txt",
        "--dataset_type", "dev",
        "--output_file", f"./output/qpp4sip_{pattern}/summary_dev.txt"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {pattern} パターンのサマリーが生成されました")
        return True
    else:
        print(f"❌ {pattern} パターンのサマリー生成に失敗しました")
        return False

def main():
    """メイン実行関数"""
    patterns = ["feature_fusion", "auxiliary_head", "policy_gating"]
    epochs = 20
    
    print("🎯 QPP4SIP 3パターン比較実験を開始します")
    print(f"対象パターン: {patterns}")
    print(f"エポック数: {epochs}")
    
    results = {}
    
    for pattern in patterns:
        print(f"\n{'='*50}")
        print(f"🔄 {pattern.upper()} パターンの実験")
        print(f"{'='*50}")
        
        # 学習
        if run_experiment(pattern, epochs):
            # 推論
            if run_inference(pattern, epochs):
                # 評価
                if run_evaluation(pattern, epochs):
                    # サマリー生成
                    if generate_summary(pattern):
                        results[pattern] = "成功"
                    else:
                        results[pattern] = "サマリー生成失敗"
                else:
                    results[pattern] = "評価失敗"
            else:
                results[pattern] = "推論失敗"
        else:
            results[pattern] = "学習失敗"
    
    # 結果サマリー
    print(f"\n{'='*50}")
    print("📊 実験結果サマリー")
    print(f"{'='*50}")
    for pattern, status in results.items():
        print(f"{pattern}: {status}")
    
    # 比較レポートの生成
    print(f"\n📋 比較レポートを生成中...")
    generate_comparison_report(patterns)

def generate_comparison_report(patterns):
    """比較レポートを生成"""
    report = []
    report.append("# QPP4SIP パターン比較レポート")
    report.append("")
    report.append("## 実験概要")
    report.append("- データセット: INSCIT (TSV変換済み)")
    report.append("- エポック数: 20")
    report.append("- 評価指標: F1スコア, 正解率, Initiative予測率")
    report.append("")
    report.append("## パターン説明")
    report.append("1. **Feature Fusion**: QPP特徴量をBERT表現に直接結合")
    report.append("2. **Auxiliary Head**: QPP回帰タスクを追加したマルチタスク学習")
    report.append("3. **Policy Gating**: QPPスコアに基づいて閾値を動的調整")
    report.append("")
    
    for pattern in patterns:
        summary_file = f"./output/qpp4sip_{pattern}/summary_dev.txt"
        if os.path.exists(summary_file):
            report.append(f"## {pattern.upper()} パターンの結果")
            with open(summary_file, 'r') as f:
                report.append(f"```")
                report.append(f.read())
                report.append(f"```")
            report.append("")
    
    # レポートを保存
    with open("./output/qpp4sip_comparison_report.md", "w") as f:
        f.write("\n".join(report))
    
    print("✅ 比較レポートが生成されました: ./output/qpp4sip_comparison_report.md")

if __name__ == "__main__":
    main()
