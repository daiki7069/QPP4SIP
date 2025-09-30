#!/usr/bin/env python3
"""
学習の進行状況を監視するスクリプト
"""

import os
import time
import subprocess
from datetime import datetime

def check_training_progress():
    """学習の進行状況をチェック"""
    patterns = ["feature_fusion", "auxiliary_head", "policy_gating"]
    
    print(f"\n🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 学習状況チェック")
    print("="*60)
    
    for pattern in patterns:
        checkpoint_dir = f"./checkpoints/qpp4sip_{pattern}"
        output_dir = f"./output/qpp4sip_{pattern}"
        
        # チェックポイント数
        if os.path.exists(checkpoint_dir):
            checkpoint_count = len([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        else:
            checkpoint_count = 0
        
        # ログファイルの確認
        log_file = f"./logs/qpp4sip_{pattern}/train.log"
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                else:
                    last_line = "ログファイルが空です"
        else:
            last_line = "ログファイルが存在しません"
        
        # プロセス確認
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        is_running = pattern in result.stdout
        
        status = "🟢 実行中" if is_running else "🔴 停止"
        
        print(f"{pattern:15} | {status} | チェックポイント: {checkpoint_count:2d} | 最新ログ: {last_line[:50]}...")
    
    print("="*60)

def main():
    """メイン実行関数"""
    print("🔍 QPP4SIP学習監視を開始します (Ctrl+Cで終了)")
    
    try:
        while True:
            check_training_progress()
            time.sleep(30)  # 30秒ごとにチェック
    except KeyboardInterrupt:
        print("\n👋 監視を終了します")

if __name__ == "__main__":
    main()
