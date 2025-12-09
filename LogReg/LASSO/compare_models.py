"""
回帰モデル同士の比較検定（DeLongの検定）を実行するエントリポイント
複数の特徴量組み合わせの回帰モデル間でAUCを比較する
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from module.bootstrap import delong_test


def main():
    parser = argparse.ArgumentParser(
        description="回帰モデル同士の比較検定（DeLongの検定）を実行"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名"
    )
    
    parser.add_argument(
        "--model-a-path",
        type=str,
        required=True,
        help="モデルAの予測結果ファイル（CSV形式: y_true, y_pred_proba列が必要）"
    )
    
    parser.add_argument(
        "--model-b-path",
        type=str,
        required=True,
        help="モデルBの予測結果ファイル（CSV形式: y_true, y_pred_proba列が必要）"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="検定結果の保存先（CSV形式、デフォルト: モデルAのディレクトリに保存）"
    )
    
    parser.add_argument(
        "--model-a-name",
        type=str,
        default="Model A",
        help="モデルAの表示名（デフォルト: 'Model A'）"
    )
    
    parser.add_argument(
        "--model-b-name",
        type=str,
        default="Model B",
        help="モデルBの表示名（デフォルト: 'Model B'）"
    )
    
    args = parser.parse_args()
    
    # ファイルの読み込み
    model_a_path = Path(args.model_a_path)
    model_b_path = Path(args.model_b_path)
    
    if not model_a_path.exists():
        print(f"エラー: モデルAのファイルが見つかりません: {model_a_path}")
        return
    
    if not model_b_path.exists():
        print(f"エラー: モデルBのファイルが見つかりません: {model_b_path}")
        return
    
    # データの読み込み
    try:
        df_a = pd.read_csv(model_a_path)
        df_b = pd.read_csv(model_b_path)
    except Exception as e:
        print(f"エラー: ファイルの読み込みに失敗しました: {e}")
        return
    
    # 必要なカラムの確認
    required_columns = ['y_true', 'y_pred_proba']
    for col in required_columns:
        if col not in df_a.columns:
            print(f"エラー: モデルAのファイルに必要なカラム '{col}' がありません")
            return
        if col not in df_b.columns:
            print(f"エラー: モデルBのファイルに必要なカラム '{col}' がありません")
            return
    
    # データの整合性確認（同じy_trueかどうか）
    y_true_a = df_a['y_true'].values
    y_true_b = df_b['y_true'].values
    
    if len(y_true_a) != len(y_true_b):
        print(f"エラー: モデルAとモデルBのサンプル数が異なります (A: {len(y_true_a)}, B: {len(y_true_b)})")
        return
    
    if not np.array_equal(y_true_a, y_true_b):
        print("警告: モデルAとモデルBのy_trueが異なります。最初のモデルのy_trueを使用します。")
        y_true = y_true_a
    else:
        y_true = y_true_a
    
    y_pred_proba_a = df_a['y_pred_proba'].values
    y_pred_proba_b = df_b['y_pred_proba'].values
    
    # DeLongの検定を実行
    print(f"\n{'='*80}")
    print(f"DeLongの検定: {args.model_a_name} vs {args.model_b_name}")
    print(f"{'='*80}")
    
    delong_result = delong_test(
        y_true=y_true,
        y_pred_proba_a=y_pred_proba_a,
        y_pred_proba_b=y_pred_proba_b
    )
    
    # 結果の表示
    print(f"\n検定結果:")
    print(f"  {args.model_a_name}: AUC = {delong_result['auc_a']:.4f}")
    print(f"  {args.model_b_name}: AUC = {delong_result['auc_b']:.4f}")
    print(f"  AUC差: {delong_result['auc_diff']:.4f}")
    print(f"  Z統計量: {delong_result['z_stat']:.4f}")
    print(f"  p値: {delong_result['p_value']:.4f}")
    print(f"  有意差: {'あり' if delong_result['significant'] else 'なし'} (α=0.05)")
    
    # 結果の保存
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = model_a_path.parent / "delong_test_comparison.csv"
    
    result_df = pd.DataFrame([{
        'model_a_name': args.model_a_name,
        'model_a_path': str(model_a_path),
        'model_b_name': args.model_b_name,
        'model_b_path': str(model_b_path),
        **delong_result
    }])
    result_df.to_csv(output_path, index=False)
    print(f"\n検定結果を保存: {output_path}")


if __name__ == "__main__":
    main()

