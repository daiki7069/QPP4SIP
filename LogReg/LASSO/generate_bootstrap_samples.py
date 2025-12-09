"""
ブートストラップサンプル（インデックス）を生成するスクリプト
対応のあるブートストラップのために、同じサンプルセットを複数の特徴量組み合わせで使用する
"""
import argparse
import numpy as np
from pathlib import Path
from module.bootstrap import generate_bootstrap_samples, save_bootstrap_samples


def main():
    parser = argparse.ArgumentParser(description="ブートストラップサンプル（インデックス）を生成")
    
    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="元のサンプル数"
    )
    
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=1000,
        help="ブートストラップの反復回数（デフォルト: 1000）"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="保存先パス（.pkl形式）"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）"
    )
    
    args = parser.parse_args()
    
    print(f"ブートストラップサンプルを生成中...")
    print(f"  サンプル数: {args.n_samples}")
    print(f"  反復回数: {args.n_iterations}")
    print(f"  乱数シード: {args.random_state}")
    
    # ブートストラップサンプルを生成
    bootstrap_samples = generate_bootstrap_samples(
        n_samples=args.n_samples,
        n_iterations=args.n_iterations,
        random_state=args.random_state
    )
    
    # 保存
    output_path = Path(args.output_path)
    save_bootstrap_samples(bootstrap_samples, output_path)
    
    print(f"ブートストラップサンプルを保存しました: {output_path}")
    print(f"  生成されたサンプル数: {len(bootstrap_samples)}")


if __name__ == "__main__":
    main()

