"""
統計的検定（ANOVA/Tukey HSD）用の10-fold CV実験スクリプト
各FoldのAUCを記録し、R解析用のCSVを出力する
"""
import argparse
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random

# main.pyのモジュールをインポート
from module import (
    load_json_data,
    extract_labels,
    load_qpp_scores,
    load_base_scores,
    merge_features,
    normalize_features,
    create_model,
    find_common_nsp_top_k
)

# パス設定
BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


def format_metric_name(metric_name: str) -> str:
    """
    メトリクス名を表示用に整形
    
    Args:
        metric_name: メトリクス名（例: 'pre_avgidf', 'nqc', 'wig'）
    
    Returns:
        整形されたメトリクス名（例: 'AvgIDF', 'NQC', 'WIG'）
    """
    # メトリクス名のマッピング
    metric_mapping = {
        # pre-retrieval
        'pre_avgidf': 'AvgIDF',
        'pre_avgictf': 'AvgICTF',
        'pre_maxidf': 'MaxIDF',
        'pre_maxscq': 'MaxSCQ',
        'pre_simplified_clarity': 'SClarity',
        # post-retrieval
        'nqc': 'NQC',
        'wig': 'WIG',
        'smv': 'SMV',
        'nsv': 'NSV',
        'clarity': 'Clarity',
        'similarity': 'Similarity',
        'entropy': 'Entropy',
        'lci': 'LCI',
        'unique_titles': 'UniqueTitles',
        'acc': 'ACC',
        'wacc': 'WACC',
        'n_sigma_50': 'nSigma'
    }
    
    # マッピングに存在する場合はそれを使用
    if metric_name in metric_mapping:
        return metric_mapping[metric_name]
    
    # マッピングにない場合は、pre_を削除して整形
    if metric_name.startswith('pre_'):
        metric_name = metric_name.replace('pre_', '')
    
    # アンダースコアを削除してタイトルケースに変換
    return metric_name.replace('_', ' ').title()


def run_cv_for_model(
    X_combined: pd.DataFrame,
    y_combined: pd.Series,
    model_type: str,
    feature_setting: str,
    qpp_metric_names: list = None,
    n_folds: int = 10,
    use_minmax_normalization: bool = True,
    random_state: int = 42
) -> list:
    """
    指定されたモデルタイプで10-fold CVを実行し、各FoldのAUCを返す
    
    Args:
        X_combined: 特徴量データ
        y_combined: ラベルデータ
        model_type: モデルタイプ（'none', 'l1', 'l2', 'elasticnet'）
        feature_setting: 特徴量設定名（ログ用）
        qpp_metric_names: QPP指標名のリスト（単一指標のAUC計算用）
        n_folds: CVのfold数（デフォルト: 10）
        use_minmax_normalization: Min-Max正規化を使用するか
        random_state: 乱数シード
    
    Returns:
        fold_results: 各Foldの結果リスト [{'fold_id': 1, 'system_name': '...', 'auc_score': 0.xxx}, ...]
    """
    from sklearn.linear_model import LogisticRegression
    
    fold_results = []
    
    # StratifiedKFoldでk分割
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    print(f"  [{feature_setting}] {model_type} モデルで10-fold CVを実行中...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_combined, y_combined)):
        # foldごとのデータ分割
        X_fold_train = X_combined.iloc[train_idx].reset_index(drop=True)
        y_fold_train = y_combined.iloc[train_idx].reset_index(drop=True)
        X_fold_test = X_combined.iloc[test_idx].reset_index(drop=True)
        y_fold_test = y_combined.iloc[test_idx].reset_index(drop=True)
        
        # 正規化（訓練データの統計量でテストデータも正規化）
        X_fold_train_norm, X_fold_test_norm = normalize_features(
            X_fold_train, X_fold_test,
            use_combined_normalization=False,
            use_separate_normalization=False,
            use_minmax_normalization=use_minmax_normalization
        )
        
        # モデル学習
        model = create_model(
            model_type=model_type,
            max_iter=1000,
            random_state=random_state,
            class_weight='balanced',
            tol=1e-8,
            non_negative=False
        )
        model.fit(X_fold_train_norm, y_fold_train)
        
        # 予測
        y_fold_test_proba = model.predict_proba(X_fold_test_norm)[:, 1]
        
        # AUCを計算
        fold_test_auc = roc_auc_score(y_fold_test, y_fold_test_proba)
        
        # システム名を生成
        system_name_map = {
            'none': 'LogReg',
            'l1': 'L1',
            'l2': 'L2',
            'elasticnet': 'ENet'
        }
        system_name = system_name_map.get(model_type, model_type)
        
        # 結果を保存（統合指標）
        fold_results.append({
            'fold_id': fold_idx + 1,  # 整数で保存
            'system_name': system_name,
            'auc_score': fold_test_auc
        })
        
        # QPP指標の単体AUCを計算
        if qpp_metric_names:
            for metric_name in qpp_metric_names:
                if metric_name not in X_fold_train.columns:
                    continue
                
                try:
                    # 単一指標で正規化
                    X_train_metric = X_fold_train[[metric_name]]
                    X_test_metric = X_fold_test[[metric_name]]
                    X_train_metric_norm, X_test_metric_norm = normalize_features(
                        X_train_metric, X_test_metric,
                        use_combined_normalization=False,
                        use_separate_normalization=False,
                        use_minmax_normalization=use_minmax_normalization
                    )
                    
                    # 単一指標でロジスティック回帰
                    single_metric_model = LogisticRegression(
                        penalty='l2',
                        solver='liblinear',
                        C=1.0,
                        max_iter=1000,
                        random_state=random_state,
                        class_weight='balanced'
                    )
                    single_metric_model.fit(X_train_metric_norm, y_fold_train)
                    y_test_proba_single = single_metric_model.predict_proba(X_test_metric_norm)[:, 1]
                    
                    # AUCを計算
                    single_metric_auc = roc_auc_score(y_fold_test, y_test_proba_single)
                    
                    # 結果を保存（QPP指標）
                    # メトリクス名を整形（表示用）
                    metric_display_name = format_metric_name(metric_name)
                    fold_results.append({
                        'fold_id': fold_idx + 1,  # 整数で保存
                        'system_name': metric_display_name,
                        'auc_score': single_metric_auc
                    })
                except (ValueError, KeyError) as e:
                    # エラーが発生した場合はスキップ
                    continue
        
        print(f"    Fold {fold_idx + 1}/{n_folds}: AUC = {fold_test_auc:.4f}")
    
    return fold_results


def main():
    # ランダムシードを固定
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    parser = argparse.ArgumentParser(description="統計的検定用の10-fold CV実験")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力ディレクトリ（デフォルト: outputs/{dataset}/statistical_evaluation/）"
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=10,
        help="CVのfold数（デフォルト: 10）"
    )
    parser.add_argument(
        "--retrieval-method",
        type=str,
        default="dpr",
        choices=["dpr", "bm25"],
        help="検索手法（デフォルト: dpr）"
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    if args.output_dir is None:
        output_dir = BASE_DIR / "LogReg" / "LASSO" / "outputs" / args.dataset / "statistical_evaluation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 実験設定の定義
    feature_settings = {
        'pre': ['pre'],
        'post': ['post'],
        'pre_post': ['pre', 'post'],
        'pre_post_bert': ['pre', 'post', 'bert'],
        'pre_post_roberta': ['pre', 'post', 'roberta']
    }
    
    # モデルタイプの定義
    model_types = ['none', 'l1', 'l2', 'elasticnet']
    
    # パスの設定
    dataset_dir = BASE_DIR / "dataset" / args.dataset
    qpp_output_dir = BASE_DIR / "QPP" / "post_retrieval" / "outputs" / args.dataset / args.retrieval_method
    pre_retrieval_output_dir = BASE_DIR / "QPP" / "pre_retrieval" / "outputs" / args.dataset
    nsp_output_dir = BASE_DIR / "QPP" / "next_sentence_prediction" / "outputs" / args.dataset
    
    # NSPのtop_k値を検出
    nsp_top_k = None
    if nsp_output_dir.exists():
        nsp_top_k = find_common_nsp_top_k(nsp_output_dir, splits=['train', 'dev'])
        if nsp_top_k is not None:
            print(f"NSP top_k値: {nsp_top_k} (trainとdevで共通)")
    
    # 全結果を保存するリスト
    all_results = []
    
    # 各実験設定を実行
    for setting_name, feature_types in feature_settings.items():
        print(f"\n{'='*80}")
        print(f"実験設定: {setting_name} (特徴量: {', '.join(feature_types)})")
        print(f"{'='*80}")
        
        # 特徴量タイプの設定
        use_post = 'post' in feature_types
        use_pre = 'pre' in feature_types
        use_bert = 'bert' in feature_types
        use_roberta = 'roberta' in feature_types
        use_nsp = 'nsp' in feature_types
        
        # データの読み込み
        print("データを読み込み中...")
        
        # trainとdevのデータを読み込み
        train_json_path = dataset_dir / "train.json"
        dev_json_path = dataset_dir / "dev.json"
        train_data = load_json_data(train_json_path)
        dev_data = load_json_data(dev_json_path)
        
        # ラベルの抽出
        train_labels = extract_labels(train_data)
        dev_labels = extract_labels(dev_data)
        
        # 特徴量の読み込み
        all_scores = {}
        
        # Post-retrieval QPP指標
        if use_post:
            train_qpp_scores = load_qpp_scores(
                'train', 
                qpp_output_dir,
                nsp_output_dir=None,
                nsp_top_k=None,
                pre_retrieval_output_dir=None,
                use_post=True,
                use_pre=False,
                use_nsp=False
            )
            dev_qpp_scores = load_qpp_scores(
                'dev',
                qpp_output_dir,
                nsp_output_dir=None,
                nsp_top_k=None,
                pre_retrieval_output_dir=None,
                use_post=True,
                use_pre=False,
                use_nsp=False
            )
            # trainとdevを結合
            for metric_name, scores in train_qpp_scores.items():
                all_scores[metric_name] = {**scores, **dev_qpp_scores.get(metric_name, {})}
        
        # Pre-retrieval QPP指標
        if use_pre:
            train_pre_scores = load_qpp_scores(
                'train',
                qpp_output_dir,
                nsp_output_dir=None,
                nsp_top_k=None,
                pre_retrieval_output_dir=pre_retrieval_output_dir,
                use_post=False,
                use_pre=True,
                use_nsp=False
            )
            dev_pre_scores = load_qpp_scores(
                'dev',
                qpp_output_dir,
                nsp_output_dir=None,
                nsp_top_k=None,
                pre_retrieval_output_dir=pre_retrieval_output_dir,
                use_post=False,
                use_pre=True,
                use_nsp=False
            )
            # trainとdevを結合
            for metric_name, scores in train_pre_scores.items():
                all_scores[metric_name] = {**scores, **dev_pre_scores.get(metric_name, {})}
        
        # BERT/RoBERTaスコア
        if use_bert or use_roberta:
            base_experiment_names = []
            if use_bert:
                base_experiment_names.append('bert')
            if use_roberta:
                base_experiment_names.append('roberta')
            
            train_base_scores = load_base_scores(
                'train', args.dataset, BASE_DIR,
                base_experiment_names=base_experiment_names,
                use_bert=use_bert,
                use_roberta=use_roberta,
                use_transfer=False,
                use_full_train_model_for_dev=False
            )
            dev_base_scores = load_base_scores(
                'dev', args.dataset, BASE_DIR,
                base_experiment_names=base_experiment_names,
                use_bert=use_bert,
                use_roberta=use_roberta,
                use_transfer=False,
                use_full_train_model_for_dev=False
            )
            # trainとdevを結合
            for metric_name, scores in train_base_scores.items():
                all_scores[metric_name] = {**scores, **dev_base_scores.get(metric_name, {})}
        
        # ラベルを結合
        all_labels = {**train_labels, **dev_labels}
        
        # 特徴量をマージ
        X_combined, y_combined = merge_features(all_scores, all_labels)
        
        print(f"  結合データ: {len(X_combined)} サンプル, {len(X_combined.columns)} 特徴量")
        print(f"  ラベル分布: {y_combined.value_counts().to_dict()} (正例率: {y_combined.mean():.4f})")
        
        # QPP指標名のリストを取得（単一指標のAUC計算用）
        qpp_metric_names = []
        if use_pre:
            # pre-retrieval QPP指標
            pre_metrics = [col for col in X_combined.columns if col.startswith('pre_')]
            qpp_metric_names.extend(pre_metrics)
        if use_post:
            # post-retrieval QPP指標（BERT/RoBERTaのlogitは除く）
            post_metrics = [col for col in X_combined.columns 
                          if col in ['nqc', 'wig', 'smv', 'nsv', 'clarity', 'similarity', 'entropy', 'lci', 'unique_titles', 'acc', 'wacc', 'n_sigma_50']]
            qpp_metric_names.extend(post_metrics)
        
        # 各モデルタイプでCVを実行
        for model_type in model_types:
            fold_results = run_cv_for_model(
                X_combined=X_combined,
                y_combined=y_combined,
                model_type=model_type,
                feature_setting=setting_name,
                qpp_metric_names=qpp_metric_names,
                n_folds=args.cv_folds,
                use_minmax_normalization=True,
                random_state=RANDOM_SEED
            )
            
            # 結果に実験設定名と検索手法を追加
            for result in fold_results:
                result['feature_setting'] = setting_name
                result['retrieval_method'] = args.retrieval_method
                all_results.append(result)
    
    # 結果をDataFrameに変換
    results_df = pd.DataFrame(all_results)
    
    # CSVを保存（検索手法を含むファイル名）
    output_csv = output_dir / f"eval_results_{args.retrieval_method}.csv"
    results_df.to_csv(output_csv, index=False)
    
    # 後方互換性のため、dprの場合はeval_results.csvにも保存
    if args.retrieval_method == "dpr":
        output_csv_legacy = output_dir / "eval_results.csv"
        results_df.to_csv(output_csv_legacy, index=False)
    print(f"\n{'='*80}")
    print(f"結果を保存しました: {output_csv}")
    print(f"{'='*80}")
    print(f"\nデータ形状: {results_df.shape}")
    print(f"\n最初の10行:")
    print(results_df.head(10))
    print(f"\n統計情報:")
    print(results_df.groupby(['retrieval_method', 'feature_setting', 'system_name'])['auc_score'].agg(['mean', 'std', 'count']))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

