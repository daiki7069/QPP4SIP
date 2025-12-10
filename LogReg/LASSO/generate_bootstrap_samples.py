"""
ブートストラップサンプル（インデックス）を生成するスクリプト
対応のあるブートストラップのために、同じサンプルセットを複数の特徴量組み合わせで使用する
"""
import argparse
import numpy as np
from pathlib import Path
from module.bootstrap import generate_bootstrap_samples, save_bootstrap_samples
from module import load_json_data, extract_labels


def main():
    parser = argparse.ArgumentParser(description="ブートストラップサンプル（インデックス）を生成")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（サンプル数を自動取得）"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="元のサンプル数（指定しない場合はデータセットから自動取得）"
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
        default=None,
        help="保存先パス（.pkl形式、デフォルト: outputs/{dataset}/bootstrap_samples.pkl）"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）"
    )
    
    args = parser.parse_args()
    
    # サンプル数の取得
    if args.n_samples is None:
        # データセットからサンプル数を取得（devデータを使用）
        base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
        dataset_dir = base_dir / "dataset" / args.dataset
        dev_json_path = dataset_dir / "dev.json"
        
        if not dev_json_path.exists():
            print(f"エラー: データセットファイルが見つかりません: {dev_json_path}")
            return
        
        dev_data = load_json_data(dev_json_path)
        dev_labels = extract_labels(dev_data)
        n_samples = len(dev_labels)
        print(f"データセットからサンプル数を取得: {n_samples}")
    else:
        n_samples = args.n_samples
    
    # 出力パスの設定
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        base_dir = Path("/home/daiki_shibata/pj/QPP4SIP/LogReg/LASSO")
        output_path = base_dir / "outputs" / args.dataset / "bootstrap_samples.pkl"
    
    print(f"ブートストラップサンプルを生成中...")
    print(f"  データセット: {args.dataset}")
    print(f"  サンプル数: {n_samples}")
    print(f"  反復回数: {args.n_iterations}")
    print(f"  乱数シード: {args.random_state}")
    
    # ブートストラップサンプルを生成
    bootstrap_samples = generate_bootstrap_samples(
        n_samples=n_samples,
        n_iterations=args.n_iterations,
        random_state=args.random_state
    )
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_bootstrap_samples(bootstrap_samples, output_path)
    
    print(f"ブートストラップサンプルを保存しました: {output_path}")
    print(f"  生成されたサンプル数: {len(bootstrap_samples)}")


if __name__ == "__main__":
    main()

