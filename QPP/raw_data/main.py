"""
INSCITデータをCSVに変換するサンプルスクリプト
"""
import os
import pandas as pd
from datetime import datetime
from data_loader import INSCITDataLoader


def main():
    # データの読み込み
    loader = INSCITDataLoader("/mnt/nas_syno/daiki/Datasets/INSCIT/data")
    dev_data = loader.load_data("test")
    print(f"Loaded {len(dev_data)} dialogue turns")
    
    # DataFrameに変換
    df = loader.convert_to_dataframe(dev_data)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # CSV出力
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("outputs", f"inscit_dev_{timestamp}.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    
    # サンプルデータの表示
    print("\nSample data:")
    print(df.head())


if __name__ == "__main__":
    main()
