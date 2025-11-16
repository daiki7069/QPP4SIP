"""
データをCSVに変換するスクリプト
"""
import argparse
import pandas as pd
from pathlib import Path
from data_loader import INSCITDataLoader


def main():
    parser = argparse.ArgumentParser(description="データをCSVに変換するスクリプト")
    
    # データセット名（必須）
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="dev",
        choices=["train", "dev", "test"],
        help="データセットのスプリット (train, dev, test, デフォルト: dev)"
    )
    
    args = parser.parse_args()
    
    # ベースパスの設定
    base_dir = Path("/home/daiki_shibata/pj/QPP4SIP")
    input_dir = base_dir / "dataset" / args.dataset
    output_dir = base_dir / "QPP" / "raw_data" / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"データセット: {args.dataset}")
    print(f"スプリット: {args.split}")
    print(f"入力ディレクトリ: {input_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    
    # データの読み込み
    loader = INSCITDataLoader(str(input_dir), dataset=args.dataset)
    data = loader.load_data(args.split)
    print(f"Loaded {len(data)} dialogue turns")
    
    # DataFrameに変換
    df = loader.convert_to_dataframe(data)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # CSV出力
    output_path = output_dir / f"{args.split}.csv"
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    
    # サンプルデータの表示
    print("\nSample data:")
    print(df.head())


if __name__ == "__main__":
    main()
