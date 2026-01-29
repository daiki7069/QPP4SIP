"""
AmbigNQで学習したモデルをClariQで評価するスクリプト
"""
import argparse
import pickle
import warnings
import sys
from io import StringIO
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, average_precision_score
import warnings

from module import (
    load_json_data,
    extract_labels,
    load_qpp_scores,
    load_base_scores,
    find_common_nsp_top_k,
    merge_features,
    normalize_features,
    get_feature_dir_name,
    create_model
)

# パス設定
BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


def save_model_and_scaler(model, scalers, feature_names, output_dir, model_type):
    """モデルとscalerを保存"""
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # モデルを保存
    model_path = model_dir / f"{model_type}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"モデルを保存しました: {model_path}")
    
    # scalerを保存
    scaler_path = model_dir / f"{model_type}_scalers.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scalers, f)
    print(f"Scalerを保存しました: {scaler_path}")
    
    # 特徴量名を保存
    feature_names_path = model_dir / f"{model_type}_feature_names.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"特徴量名を保存しました: {feature_names_path}")


def load_model_and_scaler(model_dir, model_type):
    """モデルとscalerを読み込み"""
    model_path = model_dir / f"{model_type}_model.pkl"
    scaler_path = model_dir / f"{model_type}_scalers.pkl"
    feature_names_path = model_dir / f"{model_type}_feature_names.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scalerファイルが見つかりません: {scaler_path}")
    if not feature_names_path.exists():
        raise FileNotFoundError(f"特徴量名ファイルが見つかりません: {feature_names_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    print(f"モデルを読み込みました: {model_path}")
    print(f"Scalerを読み込みました: {scaler_path}")
    print(f"特徴量名を読み込みました: {feature_names_path}")
    
    return model, scalers, feature_names


def apply_scalers(X, scalers):
    """保存されたscalerを使ってデータを正規化"""
    X_norm = X.copy()
    for column in X.columns:
        if column in scalers:
            X_norm[column] = scalers[column].transform(X[[column]]).flatten()
        else:
            print(f"警告: 特徴量 '{column}' のscalerが見つかりません。スキップします。")
    return X_norm


def main():
    parser = argparse.ArgumentParser(description="AmbigNQで学習したモデルをClariQで評価")
    
    # モデル保存先（AmbigNQの出力ディレクトリ）
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="AmbigNQで学習したモデルの保存ディレクトリ（例: outputs/AmbigNQ/post/clarity_ns50_nqc_smv_wig/models）"
    )
    
    # モデルタイプ
    parser.add_argument(
        "--model-type",
        type=str,
        default="l1",
        choices=["l1", "l2", "elasticnet", "none", "bolasso", "lars_traps", "lars_cv"],
        help="評価するモデルタイプ（デフォルト: l1）"
    )
    
    # 特徴量タイプ
    parser.add_argument(
        "--feature-types",
        type=str,
        nargs='+',
        default=["post"],
        choices=["pre", "post", "nsp", "base"],
        help="使用する特徴量タイプ（デフォルト: post）"
    )
    
    # 検索手法
    parser.add_argument(
        "--retrieval-method",
        type=str,
        default="dpr",
        choices=["dpr", "bm25"],
        help="検索手法 (dpr または bm25, デフォルト: dpr)"
    )
    
    # 出力ディレクトリ
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="評価結果の出力ディレクトリ（デフォルト: outputs/ClariQ/transfer_from_AmbigNQ/）"
    )
    
    args = parser.parse_args()
    
    # パスの設定
    model_dir = BASE_DIR / "LogReg" / "LASSO" / args.model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_dir}")
    
    clariq_dataset_dir = BASE_DIR / "dataset" / "ClariQ"
    clariq_qpp_output_dir = BASE_DIR / "QPP" / "post_retrieval" / "outputs" / "ClariQ" / args.retrieval_method
    clariq_pre_retrieval_output_dir = BASE_DIR / "QPP" / "pre_retrieval" / "outputs" / "ClariQ"
    clariq_nsp_output_dir = BASE_DIR / "QPP" / "next_sentence_prediction" / "outputs" / "ClariQ"
    
    if args.output_dir:
        output_dir = BASE_DIR / "LogReg" / "LASSO" / args.output_dir
    else:
        output_dir = BASE_DIR / "LogReg" / "LASSO" / "outputs" / "ClariQ" / "transfer_from_AmbigNQ"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AmbigNQで学習したモデルをClariQで評価")
    print("=" * 80)
    print(f"モデルディレクトリ: {model_dir}")
    print(f"モデルタイプ: {args.model_type}")
    print(f"特徴量タイプ: {args.feature_types}")
    print(f"検索手法: {args.retrieval_method}")
    print(f"出力ディレクトリ: {output_dir}")
    print()
    
    # モデルとscalerを読み込み
    print("モデルとscalerを読み込み中...")
    model, scalers, saved_feature_names = load_model_and_scaler(model_dir, args.model_type)
    print(f"保存された特徴量: {saved_feature_names}")
    print()
    
    # ClariQデータの読み込み
    print("ClariQデータを読み込み中...")
    clariq_dev_json_path = clariq_dataset_dir / "dev.json"
    if not clariq_dev_json_path.exists():
        raise FileNotFoundError(f"ClariQ dev.jsonが見つかりません: {clariq_dev_json_path}")
    
    clariq_dev_data = load_json_data(clariq_dev_json_path)
    clariq_dev_labels = extract_labels(clariq_dev_data)
    print(f"  - ClariQ devデータ: {len(clariq_dev_labels)} サンプル")
    
    # 特徴量の読み込み
    use_post = 'post' in args.feature_types
    use_pre = 'pre' in args.feature_types
    use_nsp = 'nsp' in args.feature_types
    use_bert = 'base' in args.feature_types
    
    # NSP top_kの検出
    nsp_top_k = None
    if use_nsp and clariq_nsp_output_dir.exists():
        nsp_top_k = find_common_nsp_top_k(clariq_nsp_output_dir, splits=['dev'])
        if nsp_top_k:
            print(f"  - NSP top_k: {nsp_top_k}")
    
    # ClariQの特徴量を読み込み
    clariq_dev_scores = {}
    clariq_dev_qpp_scores = load_qpp_scores(
        'dev',
        clariq_qpp_output_dir,
        nsp_output_dir=clariq_nsp_output_dir,
        nsp_top_k=nsp_top_k,
        pre_retrieval_output_dir=clariq_pre_retrieval_output_dir,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=use_nsp
    )
    clariq_dev_scores.update(clariq_dev_qpp_scores)
    
    if use_bert:
        clariq_dev_base_scores = load_base_scores(
            'dev', 'ClariQ', BASE_DIR,
            base_experiment_names=None,
            use_bert=use_bert,
            use_roberta=False,
            use_transfer=False
        )
        if clariq_dev_base_scores:
            clariq_dev_scores.update(clariq_dev_base_scores)
    
    # 特徴量をマージ
    clariq_dev_features, clariq_feature_names = merge_features(
        clariq_dev_scores,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=use_nsp,
        use_bert=use_bert
    )
    
    print(f"  - 特徴量数: {len(clariq_feature_names)}")
    print(f"  - 特徴量: {clariq_feature_names}")
    
    # 保存された特徴量と一致するか確認
    missing_features = set(saved_feature_names) - set(clariq_feature_names)
    if missing_features:
        print(f"警告: 以下の特徴量がClariQデータに存在しません: {missing_features}")
        print("存在する特徴量のみを使用します。")
    
    # 共通の特徴量のみを使用
    common_features = list(set(saved_feature_names) & set(clariq_feature_names))
    if not common_features:
        raise ValueError("共通の特徴量がありません。")
    
    print(f"  - 共通特徴量数: {len(common_features)}")
    print(f"  - 共通特徴量: {common_features}")
    
    # 共通特徴量のみを抽出
    clariq_dev_features = clariq_dev_features[common_features]
    
    # データの整合性確認
    if len(clariq_dev_features) != len(clariq_dev_labels):
        raise ValueError(f"特徴量数 ({len(clariq_dev_features)}) とラベル数 ({len(clariq_dev_labels)}) が一致しません。")
    
    print()
    
    # 正規化（保存されたscalerを使用）
    print("特徴量を正規化中...")
    clariq_dev_features_norm = apply_scalers(clariq_dev_features, scalers)
    print("正規化完了")
    print()
    
    # 予測
    print("予測を実行中...")
    clariq_dev_proba = model.predict_proba(clariq_dev_features_norm)[:, 1]
    clariq_dev_pred = model.predict(clariq_dev_features_norm)
    print("予測完了")
    print()
    
    # 評価
    print("=" * 80)
    print("評価結果")
    print("=" * 80)
    
    clariq_auc = roc_auc_score(clariq_dev_labels, clariq_dev_proba)
    clariq_ap = average_precision_score(clariq_dev_labels, clariq_dev_proba)
    clariq_acc = accuracy_score(clariq_dev_labels, clariq_dev_pred)
    clariq_f1 = f1_score(clariq_dev_labels, clariq_dev_pred)
    
    print(f"Accuracy: {clariq_acc:.4f}")
    print(f"F1 Score: {clariq_f1:.4f}")
    print(f"AUC-ROC: {clariq_auc:.4f}")
    print(f"Average Precision: {clariq_ap:.4f}")
    print()
    
    # 分類レポート
    print("分類レポート:")
    print(classification_report(clariq_dev_labels, clariq_dev_pred, target_names=['Non-Clarification', 'Clarification']))
    
    # 結果を保存
    results_dir = output_dir / f"{args.model_type}_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 結果をCSVに保存
    results_df = pd.DataFrame({
        'label': clariq_dev_labels,
        'pred_proba': clariq_dev_proba,
        'pred': clariq_dev_pred
    })
    results_csv_path = results_dir / "predictions.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"予測結果を保存しました: {results_csv_path}")
    
    # 評価指標を保存
    metrics_df = pd.DataFrame([{
        'model_type': args.model_type,
        'accuracy': clariq_acc,
        'f1_score': clariq_f1,
        'auc_roc': clariq_auc,
        'average_precision': clariq_ap
    }])
    metrics_csv_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"評価指標を保存しました: {metrics_csv_path}")
    
    # 結果をテキストファイルに保存
    results_txt_path = results_dir / "results.txt"
    with open(results_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("AmbigNQで学習したモデルをClariQで評価\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"モデルタイプ: {args.model_type}\n")
        f.write(f"特徴量: {common_features}\n")
        f.write(f"検索手法: {args.retrieval_method}\n\n")
        f.write("評価結果:\n")
        f.write(f"  Accuracy: {clariq_acc:.4f}\n")
        f.write(f"  F1 Score: {clariq_f1:.4f}\n")
        f.write(f"  AUC-ROC: {clariq_auc:.4f}\n")
        f.write(f"  Average Precision: {clariq_ap:.4f}\n\n")
        f.write("分類レポート:\n")
        f.write(classification_report(clariq_dev_labels, clariq_dev_pred, target_names=['Non-Clarification', 'Clarification']))
    print(f"結果を保存しました: {results_txt_path}")
    
    print()
    print("=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()

