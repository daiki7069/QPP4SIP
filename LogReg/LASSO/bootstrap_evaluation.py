"""
ブートストラップ評価を実行するエントリーポイント
単体指標の最高ROCと回帰モデルのROCを比較する
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from module import (
    load_json_data,
    extract_labels,
    load_qpp_scores,
    load_base_scores,
    find_common_nsp_top_k,
    merge_features,
    normalize_features,
    get_feature_dir_name
)
from module.bootstrap import (
    load_bootstrap_samples,
    evaluate_bootstrap,
    save_bootstrap_results,
    delong_test
)


BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


def main():
    parser = argparse.ArgumentParser(
        description="ブートストラップ評価を実行（単体指標の最高ROCと回帰モデルのROCを比較）"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名"
    )
    
    parser.add_argument(
        "--retrieval-method",
        type=str,
        default="dpr",
        choices=["dpr", "bm25"],
        help="検索手法 (dpr または bm25, デフォルト: dpr)"
    )
    
    parser.add_argument(
        "--bootstrap-samples-path",
        type=str,
        required=True,
        help="ブートストラップサンプル（インデックス）のパス（generate_bootstrap_samples.pyで生成）"
    )
    
    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="回帰モデルの予測結果ファイル（CSV形式: y_true, y_pred_proba列が必要、main.pyで生成）"
    )
    
    parser.add_argument(
        "--use-minmax-normalization",
        action="store_true",
        help="Min-Max正規化を使用（デフォルト: z-score正規化）"
    )
    
    args = parser.parse_args()
    
    # パスの設定
    dataset_dir = BASE_DIR / "dataset" / args.dataset
    qpp_output_dir = BASE_DIR / "QPP" / "post_retrieval" / "outputs" / args.dataset / args.retrieval_method
    pre_retrieval_output_dir = BASE_DIR / "QPP" / "pre_retrieval" / "outputs" / args.dataset
    nsp_output_dir = BASE_DIR / "QPP" / "next_sentence_prediction" / "outputs" / args.dataset
    output_dir = BASE_DIR / "LogReg" / "LASSO" / "outputs" / args.dataset
    
    # ブートストラップサンプルの読み込み
    bootstrap_samples_path = Path(args.bootstrap_samples_path)
    if not bootstrap_samples_path.exists():
        print(f"エラー: ブートストラップサンプルファイルが見つかりません: {bootstrap_samples_path}")
        return
    
    bootstrap_samples = load_bootstrap_samples(bootstrap_samples_path)
    print(f"ブートストラップサンプルを読み込み: {len(bootstrap_samples)} 反復")
    
    # 回帰モデルの予測結果を読み込み
    predictions_path = Path(args.predictions_path)
    if not predictions_path.exists():
        print(f"エラー: 予測結果ファイルが見つかりません: {predictions_path}")
        return
    
    predictions_df = pd.read_csv(predictions_path)
    if 'y_true' not in predictions_df.columns or 'y_pred_proba' not in predictions_df.columns:
        print(f"エラー: 予測結果ファイルに必要なカラム（y_true, y_pred_proba）がありません")
        return
    
    y_true_regression = predictions_df['y_true'].values
    y_pred_proba_regression = predictions_df['y_pred_proba'].values
    
    # データの読み込み（単体指標の評価用）
    train_json_path = dataset_dir / "train.json"
    train_data = load_json_data(train_json_path)
    train_labels = extract_labels(train_data)
    
    dev_json_path = dataset_dir / "dev.json"
    dev_data = load_json_data(dev_json_path)
    dev_labels = extract_labels(dev_data)
    
    # NSP top_kの検出
    nsp_top_k = None
    if nsp_output_dir.exists():
        nsp_top_k = find_common_nsp_top_k(nsp_output_dir, splits=['train', 'dev'])
    
    # スコアの読み込み
    train_all_scores = {}
    train_qpp_scores = load_qpp_scores(
        'train',
        qpp_output_dir,
        nsp_output_dir=nsp_output_dir,
        nsp_top_k=nsp_top_k,
        pre_retrieval_output_dir=pre_retrieval_output_dir
    )
    train_all_scores.update(train_qpp_scores)
    
    train_base_scores = load_base_scores('train', args.dataset, BASE_DIR, base_experiment_names=None)
    if train_base_scores:
        train_all_scores.update(train_base_scores)
    
    dev_all_scores = {}
    dev_qpp_scores = load_qpp_scores(
        'dev',
        qpp_output_dir,
        nsp_output_dir=nsp_output_dir,
        nsp_top_k=nsp_top_k,
        pre_retrieval_output_dir=pre_retrieval_output_dir
    )
    dev_all_scores.update(dev_qpp_scores)
    
    dev_base_scores = load_base_scores('dev', args.dataset, BASE_DIR, base_experiment_names=None)
    if dev_base_scores:
        dev_all_scores.update(dev_base_scores)
    
    # 特徴量マージ
    X_train, y_train = merge_features(train_all_scores, train_labels)
    X_test, y_test = merge_features(dev_all_scores, dev_labels)
    
    # 各単体指標についてロジスティック回帰を実行し、ROCを計算
    print("\n各単体指標についてロジスティック回帰を実行中...")
    single_metric_aucs = {}
    single_metric_probas = {}
    
    for metric_name in X_train.columns:
        # 正規化
        X_train_metric = X_train[[metric_name]]
        X_test_metric = X_test[[metric_name]]
        X_train_metric_norm, X_test_metric_norm = normalize_features(
            X_train_metric, X_test_metric,
            use_combined_normalization=False,
            use_separate_normalization=False,
            use_minmax_normalization=args.use_minmax_normalization
        )
        
        # 単一指標でロジスティック回帰
        single_metric_model = LogisticRegression(
            penalty='l2',
            solver='liblinear',
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        single_metric_model.fit(X_train_metric_norm, y_train)
        y_test_proba_single = single_metric_model.predict_proba(X_test_metric_norm)[:, 1]
        
        try:
            test_auc_single = roc_auc_score(y_test, y_test_proba_single)
            single_metric_aucs[metric_name] = test_auc_single
            single_metric_probas[metric_name] = y_test_proba_single
        except ValueError:
            pass
    
    if len(single_metric_aucs) == 0:
        print("エラー: 単体指標のROCを計算できませんでした")
        return
    
    # 最高ROCを出した指標を見つける
    best_metric = max(single_metric_aucs, key=single_metric_aucs.get)
    best_metric_auc = single_metric_aucs[best_metric]
    best_metric_proba = single_metric_probas[best_metric]
    
    print(f"\n単体指標の最高ROC: {best_metric} (AUC = {best_metric_auc:.4f})")
    
    # 回帰モデルのROC
    regression_auc = roc_auc_score(y_true_regression, y_pred_proba_regression)
    print(f"回帰モデルのROC: AUC = {regression_auc:.4f}")
    
    # ブートストラップ評価を実行
    print("\nブートストラップ評価を実行中...")
    
    # 回帰モデルのブートストラップ評価
    regression_bootstrap_results = evaluate_bootstrap(
        y_true=y_true_regression,
        y_pred_proba=y_pred_proba_regression,
        bootstrap_samples=bootstrap_samples,
        feature_combination_name="regression"
    )
    
    # 単体指標のブートストラップ評価
    single_metric_bootstrap_results = evaluate_bootstrap(
        y_true=y_test.values,
        y_pred_proba=best_metric_proba,
        bootstrap_samples=bootstrap_samples,
        feature_combination_name=f"single_{best_metric}"
    )
    
    # 結果を結合
    all_bootstrap_results = pd.concat([
        regression_bootstrap_results,
        single_metric_bootstrap_results
    ], ignore_index=True)
    
    # 結果を保存
    feature_dir_name = get_feature_dir_name(list(X_train.columns))
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    bootstrap_results_path = feature_output_dir / "bootstrap_results_single_vs_regression.csv"
    save_bootstrap_results(all_bootstrap_results, bootstrap_results_path)
    print(f"ブートストラップ結果を保存: {bootstrap_results_path}")
    
    # サマリーを表示
    print("\nブートストラップ結果のサマリー:")
    for combination in ["regression", f"single_{best_metric}"]:
        print(f"\n{combination}:")
        for metric in ['auc', 'ap', 'f1', 'accuracy']:
            metric_values = all_bootstrap_results[
                (all_bootstrap_results['feature_combination'] == combination) &
                (all_bootstrap_results['metric_name'] == metric)
            ]['value'].values
            mean_val = np.mean(metric_values)
            std_val = np.std(metric_values)
            ci_lower = np.percentile(metric_values, 2.5)
            ci_upper = np.percentile(metric_values, 97.5)
            print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    
    # DeLongの検定
    print("\nDeLongの検定を実行中...")
    delong_result = delong_test(
        y_true=y_true_regression,
        y_pred_proba_a=y_pred_proba_regression,
        y_pred_proba_b=best_metric_proba
    )
    
    print(f"\nDeLongの検定結果:")
    print(f"  モデルA (回帰): AUC = {delong_result['auc_a']:.4f}")
    print(f"  モデルB (単体指標 {best_metric}): AUC = {delong_result['auc_b']:.4f}")
    print(f"  AUC差: {delong_result['auc_diff']:.4f}")
    print(f"  Z統計量: {delong_result['z_stat']:.4f}")
    print(f"  p値: {delong_result['p_value']:.4f}")
    print(f"  有意差: {'あり' if delong_result['significant'] else 'なし'} (α=0.05)")
    
    # 結果を保存
    delong_result_path = feature_output_dir / "delong_test_single_metric_bootstrap.csv"
    result_df = pd.DataFrame([{
        'best_single_metric': best_metric,
        'best_single_metric_auc': best_metric_auc,
        'regression_auc': regression_auc,
        **delong_result
    }])
    result_df.to_csv(delong_result_path, index=False)
    print(f"DeLongの検定結果を保存: {delong_result_path}")


if __name__ == "__main__":
    main()

