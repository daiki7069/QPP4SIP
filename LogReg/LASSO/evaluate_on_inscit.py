"""
AmbigNQで学習したモデルをINSCITで評価するスクリプト
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
    create_model,
    get_coefficients,
    plot_roc_curves,
    plot_pr_curves,
    plot_threshold_f1_curves,
    plot_feature_distributions,
    plot_correlation_heatmaps
)

# パス設定
BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


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
    parser = argparse.ArgumentParser(description="AmbigNQで学習したモデルをINSCITで評価")
    
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
    
    # 特徴量タイプ（必須、pre, post, pre post, pre post bert など）
    parser.add_argument(
        "--feature-types",
        type=str,
        nargs='+',  # 1個以上（必須）
        required=True,
        choices=["pre", "post", "nsp", "bert", "roberta", "transfer"],
        help="使用する特徴量タイプ（必須、例: post, 'pre post', 'pre post bert'）"
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
        help="評価結果の出力ディレクトリ（デフォルト: outputs/INSCIT/transfer_from_AmbigNQ/）"
    )
    
    args = parser.parse_args()
    
    # パスの設定
    model_dir = BASE_DIR / "LogReg" / "LASSO" / args.model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"モデルディレクトリが見つかりません: {model_dir}")
    
    inscit_dataset_dir = BASE_DIR / "dataset" / "INSCIT"
    inscit_qpp_output_dir = BASE_DIR / "QPP" / "post_retrieval" / "outputs" / "INSCIT" / args.retrieval_method
    inscit_pre_retrieval_output_dir = BASE_DIR / "QPP" / "pre_retrieval" / "outputs" / "INSCIT"
    inscit_nsp_output_dir = BASE_DIR / "QPP" / "next_sentence_prediction" / "outputs" / "INSCIT"
    
    # 出力ディレクトリの設定（通常のINSCIT評価と同じ構造を使用）
    if args.output_dir:
        base_output_dir = BASE_DIR / "LogReg" / "LASSO" / args.output_dir
    else:
        base_output_dir = BASE_DIR / "LogReg" / "LASSO" / "outputs" / "INSCIT"
    
    # まず、共通特徴量名から通常のディレクトリ構造を生成するために、
    # モデルから読み込んだ特徴量名を使用してget_feature_dir_nameを呼び出す
    # ただし、この時点ではまだ特徴量名が確定していないので、
    # 後で確定したら再計算する必要がある
    # とりあえず、通常のINSCIT評価と同じ構造にするため、output_dirを設定
    output_dir = base_output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("AmbigNQで学習したモデルをINSCITで評価")
    print("=" * 80)
    print(f"モデルディレクトリ: {model_dir}")
    print(f"モデルタイプ: {args.model_type}")
    print(f"特徴量タイプ: {args.feature_types}")
    print(f"検索手法: {args.retrieval_method}")
    # モデルとscalerを読み込み
    print("モデルとscalerを読み込み中...")
    model, scalers, saved_feature_names = load_model_and_scaler(model_dir, args.model_type)
    print(f"保存された特徴量: {saved_feature_names}")
    print()
    
    # INSCITデータの読み込み
    print("INSCITデータを読み込み中...")
    inscit_dev_json_path = inscit_dataset_dir / "dev.json"
    if not inscit_dev_json_path.exists():
        raise FileNotFoundError(f"INSCIT dev.jsonが見つかりません: {inscit_dev_json_path}")
    
    inscit_dev_data = load_json_data(inscit_dev_json_path)
    inscit_dev_labels = extract_labels(inscit_dev_data)
    print(f"  - INSCIT devデータ: {len(inscit_dev_labels)} サンプル")
    
    # 指定された特徴量タイプを使用
    specified_types = set(args.feature_types)
    print(f"  - 指定された特徴量タイプ: {sorted(specified_types)}")
    
    # 保存された特徴量名から実際に使用されている特徴量タイプを確認
    detected_feature_types = set()
    for feature_name in saved_feature_names:
        if feature_name.startswith('pre_'):
            detected_feature_types.add('pre')
        elif feature_name.startswith('nsp_'):
            detected_feature_types.add('nsp')
        elif 'logit_clarification' in feature_name or 'prob_clarification' in feature_name:
            if 'roberta' in feature_name:
                detected_feature_types.add('roberta')
            elif 'bert' in feature_name:
                detected_feature_types.add('bert')
            else:
                detected_feature_types.add('base')
        else:
            # post-retrieval QPPスコア（nqc, clarity, wig, smv, n_sigma_50など）
            detected_feature_types.add('post')
    
    print(f"  - 保存されたモデルの特徴量タイプ: {sorted(detected_feature_types)}")
    
    # 指定されたタイプと保存されたタイプが一致するか確認
    if not specified_types.issubset(detected_feature_types):
        missing = specified_types - detected_feature_types
        print(f"  ⚠️  警告: 指定された特徴量タイプ {sorted(missing)} が保存されたモデルに含まれていません。")
        print(f"     保存されたモデルには {sorted(detected_feature_types)} が含まれています。")
    
    use_post = 'post' in specified_types
    use_pre = 'pre' in specified_types
    use_nsp = 'nsp' in specified_types
    use_bert = 'bert' in specified_types
    use_roberta = 'roberta' in specified_types
    use_transfer = 'transfer' in specified_types
    
    print(f"  - 使用する特徴量タイプ: post={use_post}, pre={use_pre}, nsp={use_nsp}, bert={use_bert}, roberta={use_roberta}")
    
    # NSP top_kの検出
    nsp_top_k = None
    if use_nsp and inscit_nsp_output_dir.exists():
        nsp_top_k = find_common_nsp_top_k(inscit_nsp_output_dir, splits=['dev'])
        if nsp_top_k:
            print(f"  - NSP top_k: {nsp_top_k}")
    
    # INSCITの特徴量を読み込み
    inscit_dev_scores = {}
    inscit_dev_qpp_scores = load_qpp_scores(
        'dev',
        inscit_qpp_output_dir,
        nsp_output_dir=inscit_nsp_output_dir,
        nsp_top_k=nsp_top_k,
        pre_retrieval_output_dir=inscit_pre_retrieval_output_dir,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=use_nsp
    )
    inscit_dev_scores.update(inscit_dev_qpp_scores)
    
    if use_bert or use_roberta:
        inscit_dev_base_scores = load_base_scores(
            'dev', 'INSCIT', BASE_DIR,
            base_experiment_names=None,
            use_bert=use_bert,
            use_roberta=use_roberta,
            use_transfer=use_transfer
        )
        if inscit_dev_base_scores:
            inscit_dev_scores.update(inscit_dev_base_scores)
            for feature_name, scores in inscit_dev_base_scores.items():
                print(f"  - {feature_name}: {len(scores)} サンプル")
    
    # 特徴量をマージ
    inscit_dev_features, inscit_dev_labels_series = merge_features(
        inscit_dev_scores,
        inscit_dev_labels
    )
    
    inscit_feature_names = list(inscit_dev_features.columns)
    print(f"  - 特徴量数: {len(inscit_feature_names)}")
    print(f"  - 特徴量: {inscit_feature_names}")
    
    # 保存された特徴量と一致するか確認
    missing_features = set(saved_feature_names) - set(inscit_feature_names)
    if missing_features:
        print(f"警告: 以下の特徴量がINSCITデータに存在しません: {missing_features}")
        print("存在する特徴量のみを使用します。")
    
    # 共通の特徴量のみを使用（保存された順序に合わせる）
    common_features = [f for f in saved_feature_names if f in inscit_feature_names]
    if not common_features:
        raise ValueError("共通の特徴量がありません。")
    
    print(f"  - 共通特徴量数: {len(common_features)}")
    print(f"  - 共通特徴量: {common_features}")
    
    # 保存された順序で共通特徴量のみを抽出
    inscit_dev_features = inscit_dev_features[common_features]
    
    # データの整合性確認
    if len(inscit_dev_features) != len(inscit_dev_labels_series):
        raise ValueError(f"特徴量数 ({len(inscit_dev_features)}) とラベル数 ({len(inscit_dev_labels_series)}) が一致しません。")
    
    # ラベルをSeriesから配列に変換
    inscit_dev_labels_array = inscit_dev_labels_series.values
    
    # 通常のINSCIT評価と同じディレクトリ構造を生成（get_feature_dir_nameを使用）
    feature_dir_name = get_feature_dir_name(common_features, retrieval_method=args.retrieval_method)
    feature_output_dir = output_dir / feature_dir_name
    # 転移学習の結果は、通常の評価結果とは別のサブディレクトリに保存
    transfer_output_dir = feature_output_dir / "transfer_from_AmbigNQ"
    transfer_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"出力ディレクトリ: {transfer_output_dir}")
    print()
    
    # 正規化（保存されたscalerを使用）
    print("特徴量を正規化中...")
    inscit_dev_features_norm = apply_scalers(inscit_dev_features, scalers)
    print("正規化完了")
    print()
    
    # 予測
    print("予測を実行中...")
    inscit_dev_proba = model.predict_proba(inscit_dev_features_norm)[:, 1]
    inscit_dev_pred = model.predict(inscit_dev_features_norm)
    print("予測完了")
    print()
    
    # 評価
    print("=" * 80)
    print("評価結果")
    print("=" * 80)
    
    inscit_auc = roc_auc_score(inscit_dev_labels_array, inscit_dev_proba)
    inscit_ap = average_precision_score(inscit_dev_labels_array, inscit_dev_proba)
    inscit_acc = accuracy_score(inscit_dev_labels_array, inscit_dev_pred)
    inscit_f1 = f1_score(inscit_dev_labels_array, inscit_dev_pred)
    
    print(f"Accuracy: {inscit_acc:.4f}")
    print(f"F1 Score: {inscit_f1:.4f}")
    print(f"AUC-ROC: {inscit_auc:.4f}")
    print(f"Average Precision: {inscit_ap:.4f}")
    print()
    
    # 分類レポート
    print("分類レポート:")
    print(classification_report(inscit_dev_labels_array, inscit_dev_pred, target_names=['Non-Clarification', 'Clarification']))
    
    # 結果を保存（転移学習用のディレクトリに保存）
    results_dir = transfer_output_dir / f"{args.model_type}_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 結果をCSVに保存
    results_df = pd.DataFrame({
        'label': inscit_dev_labels_array,
        'pred_proba': inscit_dev_proba,
        'pred': inscit_dev_pred
    })
    results_csv_path = results_dir / "predictions.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"予測結果を保存しました: {results_csv_path}")
    
    # 評価指標を保存
    metrics_df = pd.DataFrame([{
        'model_type': args.model_type,
        'accuracy': inscit_acc,
        'f1_score': inscit_f1,
        'auc_roc': inscit_auc,
        'average_precision': inscit_ap
    }])
    metrics_csv_path = results_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"評価指標を保存しました: {metrics_csv_path}")
    
    # 結果をテキストファイルに保存（AmbigNQの形式に合わせて詳細化）
    results_txt_path = results_dir / "results.txt"
    
    # モデルタイプ名のマッピング
    model_type_name = {
        'l1': 'L1正則化ロジスティック回帰',
        'l2': 'L2正則化ロジスティック回帰',
        'elasticnet': 'ElasticNet正則化ロジスティック回帰',
        'none': 'ペナルティなしロジスティック回帰',
        'randomforest': 'RandomForest分類器'
    }.get(args.model_type, args.model_type)
    
    # 特徴量タイプ名の生成
    feature_type_names = []
    if use_pre:
        feature_type_names.append("QPPスコア（pre）")
    if use_post:
        feature_type_names.append("QPPスコア（post）")
    if use_nsp:
        feature_type_names.append("NSPスコア")
    if use_bert:
        feature_type_names.append("BERTスコア")
    if use_roberta:
        feature_type_names.append("RoBERTaスコア")
    if use_transfer:
        feature_type_names.append("Transferスコア")
    feature_type_str = " + ".join(feature_type_names) if feature_type_names else "QPPスコア"
    
    # 正規化方法の判定（scalerのタイプから推測）
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    normalization_type = "Min-Max正規化" if any(isinstance(s, MinMaxScaler) for s in scalers.values() if s is not None) else "z-score正規化"
    
    with open(results_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"使用する特徴量タイプ: {', '.join(args.feature_types)}\n")
        f.write(f"=== {model_type_name}によるclarification分類 ===\n\n")
        f.write(f"データセット: INSCIT（転移学習評価）\n")
        f.write(f"転移元データセット: AmbigNQ\n")
        f.write(f"検索手法: {args.retrieval_method}\n")
        f.write(f"使用する特徴量: {feature_type_str}\n")
        f.write(f"  設定: --feature-typesで指定\n")
        f.write(f"モデルタイプ: {model_type_name}\n")
        f.write(f"正規化方法: AmbigNQの訓練データの統計量で{normalization_type}（転移学習）\n\n")
        
        f.write("1. データ読み込み中...\n")
        f.write(f"  - INSCIT devデータ: {len(inscit_dev_labels_array)} サンプル\n")
        for feature_name in common_features:
            f.write(f"  - {feature_name}: {len(inscit_dev_features)} サンプル\n")
        f.write("\n")
        
        f.write("2. 特徴量マージ中...\n")
        f.write(f"  - INSCIT devデータ: {len(inscit_dev_features)} サンプル, {len(common_features)} 特徴量\n")
        f.write(f"  - 特徴量: {common_features}\n")
        f.write(f"  - INSCIT devデータのラベル分布: {dict(zip(*np.unique(inscit_dev_labels_array, return_counts=True)))} (正例率: {inscit_dev_labels_array.mean():.4f})\n\n")
        
        f.write("【データ分布の確認（正規化前）】\n")
        f.write("INSCIT devデータの特徴量統計（正規化前）:\n")
        f.write(inscit_dev_features.describe().to_string())
        f.write(f"\n\nINSCIT devデータのラベル分布: {dict(zip(*np.unique(inscit_dev_labels_array, return_counts=True)))} (正例率: {inscit_dev_labels_array.mean():.4f})\n\n")
        
        f.write(f"3. {normalization_type}中...\n")
        f.write("  - 方法: AmbigNQの訓練データの統計量でINSCIT devデータも正規化（転移学習）\n")
        f.write("  - 正規化完了\n\n")
        
        f.write("【データ分布の確認（正規化後）】\n")
        f.write("INSCIT devデータの特徴量統計（正規化後）:\n")
        f.write(inscit_dev_features_norm.describe().to_string())
        f.write("\n\n")
        
        # モデルの係数を取得
        coefficients = get_coefficients(model, common_features, args.model_type)
        coefficient_label = "重要度" if args.model_type == 'randomforest' else "係数"
        f.write(f"4. モデル情報（AmbigNQで学習）...\n")
        f.write(f"  - モデルタイプ: {model_type_name}\n")
        f.write(f"\n特徴量の{coefficient_label}:\n")
        feature_importance = pd.DataFrame({
            'feature': common_features,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        f.write(feature_importance.to_string(index=False))
        f.write("\n\n")
        
        f.write("5. 評価中...\n\n")
        f.write("=== INSCIT devデータの評価 ===\n")
        f.write(f"Accuracy: {inscit_acc:.4f}\n")
        f.write(f"F1 Score: {inscit_f1:.4f}\n")
        f.write(f"AUC-ROC: {inscit_auc:.4f}\n")
        f.write(f"Average Precision: {inscit_ap:.4f}\n\n")
        f.write("分類レポート:\n")
        f.write(classification_report(inscit_dev_labels_array, inscit_dev_pred, target_names=['Non-Clarification', 'Clarification']))
        f.write("\n\n")
        
        f.write("6. ROC曲線を描画中...\n")
        f.write("  - ROC曲線を保存しました\n\n")
        f.write("7. Precision-Recall曲線を描画中...\n")
        f.write("  - PR曲線を保存しました\n\n")
        f.write("8. 閾値とF1スコアの関係を描画中...\n")
        f.write("  - 閾値とF1スコアの関係を保存しました\n\n")
        f.write("9. 特徴量のスコア分布を描画中...\n")
        f.write("  - 特徴量のスコア分布を保存しました\n\n")
        f.write("10. 相関係数ヒートマップを描画中...\n")
        f.write("  - 相関係数ヒートマップを保存しました\n")
    
    print(f"結果を保存しました: {results_txt_path}")
    
    # グラフを描画
    print()
    print("=" * 80)
    print("グラフを描画中...")
    print("=" * 80)
    
    graphs_dir = results_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROC曲線
    print("1. ROC曲線を描画中...")
    plot_roc_curves(
        y_train=None,  # 訓練データはないのでNone
        y_train_proba=None,
        y_test=inscit_dev_labels_array,
        y_test_proba=inscit_dev_proba,
        train_auc=None,
        test_auc=inscit_auc,
        output_dir=graphs_dir,
        hide_train=True,
        X_train=None,
        X_test=inscit_dev_features_norm,
        show_single_metrics=True
    )
    print(f"  - ROC曲線を保存しました: {graphs_dir / 'roc_curves.png'}")
    
    # 2. Precision-Recall曲線
    print("2. Precision-Recall曲線を描画中...")
    plot_pr_curves(
        y_train=None,
        y_train_proba=None,
        y_test=inscit_dev_labels_array,
        y_test_proba=inscit_dev_proba,
        train_ap=None,
        test_ap=inscit_ap,
        output_dir=graphs_dir,
        hide_train=True,
        X_train=None,
        X_test=inscit_dev_features_norm,
        show_single_metrics=True
    )
    print(f"  - PR曲線を保存しました: {graphs_dir / 'pr_curves.png'}")
    
    # 3. 閾値とF1スコアの関係
    print("3. 閾値とF1スコアの関係を描画中...")
    plot_threshold_f1_curves(
        y_train=None,
        y_train_proba=None,
        y_test=inscit_dev_labels_array,
        y_test_proba=inscit_dev_proba,
        output_dir=graphs_dir,
        hide_train=True,
        X_train=None,
        X_test=inscit_dev_features_norm,
        show_single_metrics=True
    )
    print(f"  - 閾値とF1スコアの関係を保存しました: {graphs_dir / 'threshold_f1_curves.png'}")
    
    # 4. 特徴量のスコア分布
    print("4. 特徴量のスコア分布を描画中...")
    # X_trainがNoneの場合は、X_testをX_trainとしても使用（hide_train=Trueなので表示されない）
    plot_feature_distributions(
        X_train=inscit_dev_features_norm,  # 訓練データがないので、テストデータを代用（hide_train=Trueで非表示）
        X_test=inscit_dev_features_norm,
        output_dir=graphs_dir,
        hide_train=True,
        feature_names=common_features,
        y_train=inscit_dev_labels_array,  # 訓練データがないので、テストデータを代用
        y_test=inscit_dev_labels_array
    )
    print(f"  - 特徴量のスコア分布を保存しました: {graphs_dir / 'feature_distributions.png'}")
    
    # 5. 相関係数ヒートマップ
    print("5. 相関係数ヒートマップを描画中...")
    # X_trainがNoneの場合は、X_testのみを使用
    # y_testをSeriesに変換（plot_correlation_heatmapsがSeriesを期待しているため）
    inscit_dev_labels_series = pd.Series(inscit_dev_labels_array)
    plot_correlation_heatmaps(
        X_train=inscit_dev_features_norm,  # 訓練データがないので、テストデータを代用
        X_test=inscit_dev_features_norm,
        y_train=inscit_dev_labels_series,  # 訓練データがないので、テストデータを代用
        y_test=inscit_dev_labels_series,
        output_dir=graphs_dir
    )
    print(f"  - 相関係数ヒートマップを保存しました: {graphs_dir / 'correlation_heatmaps.png'}")
    
    print()
    print("=" * 80)
    print("完了")
    print("=" * 80)


if __name__ == "__main__":
    main()

