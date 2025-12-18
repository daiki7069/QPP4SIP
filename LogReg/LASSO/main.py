"""
ロジスティック回帰・RandomForestによるclarification分類
L1/L2/ElasticNet正則化ロジスティック回帰とRandomForest分類器をサポート
"""
import argparse
import warnings
import sys
from io import StringIO
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, average_precision_score
from sklearn.model_selection import StratifiedKFold
import warnings

from module import (
    load_json_data,
    extract_labels,
    load_qpp_scores,
    load_base_scores,
    load_base_probabilities,
    find_common_nsp_top_k,
    merge_features,
    balance_label_distribution,
    normalize_features,
    get_feature_dir_name,
    plot_roc_curves,
    plot_pr_curves,
    plot_single_metric_roc_curves,
    plot_single_metric_pr_curves,
    plot_threshold_f1_curves,
    plot_feature_distributions,
    plot_correlation_heatmaps,
    plot_confidence_analysis,
    plot_overconfidence_analysis,
    plot_multiple_models_comparison,
    NonNegativeLogisticRegression,
    BOLASSOModel,
    LARSTrapsModel,
    LARSCVModel,
    create_model,
    get_coefficients
)
from module.visualization import format_feature_name
from module.bootstrap import (
    delong_test,
    load_bootstrap_samples,
    evaluate_bootstrap,
    save_bootstrap_results,
    compare_feature_combinations
)


# パス設定（main関数内で動的に設定）
BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


# 以下のクラスと関数は module/models.py と module/feature_selection.py に移動しました
# - NonNegativeLogisticRegression
# - BOLASSOModel
# - LARSTrapsModel
# - LARSCVModel
# - create_model
# - get_coefficients
# - bootstrap_sample
# - bolasso_feature_selection
# - lars_traps_feature_selection


def main():
    # ランダムシードを固定して実験の再現性を確保
    import random
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    parser = argparse.ArgumentParser(description="ロジスティック回帰・RandomForestによるclarification分類")
    
    # データセット名（必須）
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
    )
    
    # モデルタイプ
    parser.add_argument(
        "--model-type",
        type=str,
        default="all",
        choices=["l1", "l2", "elasticnet", "randomforest", "none", "bolasso", "lars_traps", "lars_cv", "all"],
        help="モデルタイプ（l1: L1正則化, l2: L2正則化, elasticnet: ElasticNet, randomforest: RandomForest, none: ペナルティなし, bolasso: BOLASSO, lars_traps: LARS-Traps, lars_cv: LARS-CV, all: 全てのモデルを実行して比較, デフォルト: all）"
    )
    
    # BOLASSO用のパラメータ
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100,
        help="BOLASSOのブートストラップサンプル数（デフォルト: 100、--model-type bolassoの場合のみ有効）"
    )
    
    parser.add_argument(
        "--selection-threshold",
        type=float,
        default=0.5,
        help="BOLASSOの特徴量選択閾値（0.0-1.0、デフォルト: 0.5、--model-type bolassoの場合のみ有効）"
    )
    
    # LARS-Traps用のパラメータ
    parser.add_argument(
        "--n-random-traps",
        type=int,
        default=10,
        help="LARS-Trapsのランダムトラップ数（デフォルト: 10、--model-type lars_trapsの場合のみ有効）"
    )
    
    # LARS-CV用のパラメータ
    parser.add_argument(
        "--lars-cv-folds",
        type=int,
        default=5,
        help="LARS-CVのクロスバリデーションフォールド数（デフォルト: 5、--model-type lars_cvの場合のみ有効）"
    )
    
    
    parser.add_argument(
        "--balance-label-distribution",
        action="store_true",
        help="訓練データのラベル分布をテストデータと同じ正例率に調整する（デフォルト: False）"
    )
    
    parser.add_argument(
        "--use-combined-normalization",
        action="store_true",
        help="訓練データとテストデータを結合してから正規化する（リーク前提、デフォルト: False）"
    )
    
    parser.add_argument(
        "--use-separate-normalization",
        action="store_true",
        help="訓練データとテストデータをそれぞれ個別に正規化する（各々が平均0、標準偏差1になる、デフォルト: False）"
    )
    
    parser.add_argument(
        "--use-minmax-normalization",
        action="store_true",
        help="[0,1]正規化（Min-Max正規化）を使用する（デフォルト: False、z-score正規化を使用）"
    )
    
    parser.add_argument(
        "--hide-train-curves",
        action="store_true",
        default=True,
        help="訓練データの曲線を非表示にする（デフォルト: True）"
    )
    
    parser.add_argument(
        "--show-single-metric-curves",
        action="store_true",
        default=True,
        help="入力に使った指標単体での曲線も表示する（デフォルト: True）"
    )
    
    parser.add_argument(
        "--non-negative-coefficients",
        action="store_true",
        help="係数を非負に制約する（デフォルト: False）"
    )
    
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="最大反復回数（デフォルト: 1000）"
    )
    
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-8,
        help="収束判定の閾値（デフォルト: 1e-8、より厳しくする場合は1e-9など）"
    )
    
    parser.add_argument(
        "--no-cv",
        action="store_false",
        dest="use_cv",
        default=True,
        help="クロスバリデーションを無効化（train/devを分離して評価）。デフォルトではCVが有効（trainとdevの分布が異なるため）"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="クロスバリデーションのfold数（デフォルト: 5、--use-cvが指定されている場合のみ有効）"
    )
    
    parser.add_argument(
        "--retrieval-method",
        type=str,
        default="dpr",
        choices=["dpr", "bm25"],
        help="検索手法 (dpr または bm25, デフォルト: dpr)"
    )
    
    parser.add_argument(
        "--delong-test",
        action="store_true",
        help="DeLongの検定を実行する（デフォルト: False、単体指標の最高ROCと回帰モデルのROCを比較）"
    )
    
    parser.add_argument(
        "--bootstrap-test",
        action="store_true",
        help="ブートストラップ評価を実行する（デフォルト: False、単体指標の最高ROCと回帰モデルのROCを比較、--bootstrap-samples-pathが必要）"
    )
    
    parser.add_argument(
        "--bootstrap-samples-path",
        type=str,
        default=None,
        help="既存のブートストラップサンプル（インデックス）のパス（--bootstrap-test使用時は必須）"
    )
    
    parser.add_argument(
        "--feature-types",
        type=str,
        nargs='+',
        choices=['post', 'pre', 'bert', 'roberta', 'nsp', 'transfer'],
        default=['post', 'pre', 'nsp'],
        help="使用する特徴量タイプを指定（複数指定可能: post, pre, bert, roberta, nsp, transfer）。デフォルト: post, pre, nsp"
    )
    
    args = parser.parse_args()
    
    # 全てのモデルタイプを定義
    ALL_MODEL_TYPES = ['l1', 'l2', 'elasticnet', 'none', 'bolasso', 'lars_traps', 'lars_cv', 'randomforest']
    
    # model_typeが"all"の場合、全てのモデルタイプを実行
    if args.model_type == 'all':
        model_types_to_run = ALL_MODEL_TYPES
        run_all_models = True
    else:
        model_types_to_run = [args.model_type]
        run_all_models = False
    
    # パスの動的設定
    dataset_dir = BASE_DIR / "dataset" / args.dataset
    qpp_output_dir = BASE_DIR / "QPP" / "post_retrieval" / "outputs" / args.dataset / args.retrieval_method
    pre_retrieval_output_dir = BASE_DIR / "QPP" / "pre_retrieval" / "outputs" / args.dataset
    nsp_output_dir = BASE_DIR / "QPP" / "next_sentence_prediction" / "outputs" / args.dataset
    output_dir = BASE_DIR / "LogReg" / "LASSO" / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 出力ファイルの準備（print内容をファイルにも保存）
    output_file = output_dir / "results.txt"
    output_buffer = StringIO()
    
    def print_and_save(*args, **kwargs):
        """printと同時にファイルにも出力"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=output_buffer)
    
    # 訓練データとテストデータで共通して存在するNSPのtop_k値を検出
    nsp_top_k = None
    if nsp_output_dir.exists():
        nsp_top_k = find_common_nsp_top_k(nsp_output_dir, splits=['train', 'dev'])
        if nsp_top_k is not None:
            print_and_save(f"NSP top_k値: {nsp_top_k} (trainとdevで共通)")
        else:
            print_and_save("Warning: trainとdevで共通するNSP top_k値が見つかりませんでした")
    
    # 特徴量タイプの設定（argparseのデフォルト機能を使用）
    use_post = 'post' in args.feature_types
    use_pre = 'pre' in args.feature_types
    use_bert = 'bert' in args.feature_types
    use_roberta = 'roberta' in args.feature_types
    use_nsp = 'nsp' in args.feature_types
    use_transfer = 'transfer' in args.feature_types
    print_and_save(f"使用する特徴量タイプ: {', '.join(args.feature_types)}")
    
    # 後方互換性のため
    use_pre_retrieval = use_pre
    use_base = use_bert or use_roberta or use_transfer
    
    model_type_name = {
        'l1': 'L1正則化付きロジスティック回帰',
        'l2': 'L2正則化付きロジスティック回帰',
        'elasticnet': 'ElasticNet正則化付きロジスティック回帰',
        'randomforest': 'RandomForest分類器',
        'none': 'ペナルティなしロジスティック回帰',
        'bolasso': 'BOLASSO（Bootstrap-enhanced LASSO）',
        'lars_traps': 'LARS-Traps',
        'lars_cv': 'LARS-CV'
    }.get(args.model_type, args.model_type)
    print_and_save(f"=== {model_type_name}によるclarification分類 ===\n")
    print_and_save(f"データセット: {args.dataset}")
    print_and_save(f"検索手法: {args.retrieval_method}")
    # 使用する特徴量タイプに基づいて説明文を生成
    feature_parts = []
    if use_post:
        feature_parts.append("post")
    if use_pre_retrieval:
        feature_parts.append("pre")
    if use_nsp:
        feature_parts.append("nsp")
    if use_base:
        if use_transfer:
            feature_parts.append("transfer")
        if use_bert or use_roberta:
            feature_parts.append("base")
    feature_desc = "QPPスコア（" + " + ".join(feature_parts) + "）"
    print_and_save(f"使用する特徴量: {feature_desc}")
    print_and_save(f"  設定: --feature-typesで指定")
    print_and_save(f"モデルタイプ: {model_type_name}")
    print_and_save(f"ラベル分布の調整: {'有効' if args.balance_label_distribution else '無効'}")
    print_and_save(f"非負制約: {'有効' if args.non_negative_coefficients and args.model_type == 'l1' else '無効'}")
    print_and_save(f"クロスバリデーション: {'有効' if args.use_cv else '無効'}")
    if args.use_cv:
        print_and_save(f"  Fold数: {args.cv_folds}")
    
    # 正規化方法の表示
    normalization_type = "[0,1]正規化（Min-Max）" if args.use_minmax_normalization else "z-score正規化"
    if args.use_separate_normalization:
        normalization_method = f"訓練データとテストデータをそれぞれ個別に{normalization_type}"
    elif args.use_combined_normalization:
        normalization_method = f"結合データで{normalization_type}（リーク前提）"
    else:
        normalization_method = f"訓練データの統計量で{normalization_type}（通常）"
    print_and_save(f"正規化方法: {normalization_method}")
    
    if args.use_separate_normalization:
        warning_msg = (
            "\n" + "="*80 + "\n"
            "⚠️  注意: 個別正規化が有効になっています！\n"
            "⚠️  このオプションは訓練データとテストデータをそれぞれ個別に正規化します。\n"
            "⚠️  各データセットが独立に正規化されるため、分布の違いは排除されます。\n"
            "⚠️  実際の予測タスクでは使用できません（テストデータの統計量は未知です）。\n"
            "⚠️  分布の違いによる影響を確認するための実験的なオプションです。\n"
            "="*80 + "\n"
        )
        print_and_save(warning_msg)
        # 標準エラー出力にも警告を出力（目立つように）
        print(warning_msg, file=sys.stderr)
    elif args.use_combined_normalization:
        warning_msg = (
            "\n" + "="*80 + "\n"
            "⚠️  警告: リーク前提の正規化が有効になっています！\n"
            "⚠️  このオプションは訓練データとテストデータを結合してから正規化します。\n"
            "⚠️  実際の予測タスクでは使用できません（データリークが発生します）。\n"
            "⚠️  分布の違いによる影響を確認するための実験的なオプションです。\n"
            "="*80 + "\n"
        )
        print_and_save(warning_msg)
        # 標準エラー出力にも警告を出力（目立つように）
        print(warning_msg, file=sys.stderr)
    
    print_and_save()
    
    # 1. データ読み込み
    print_and_save("1. データ読み込み中...")
    
    # 訓練データ
    train_json_path = dataset_dir / "train.json"
    train_data = load_json_data(train_json_path)
    train_labels = extract_labels(train_data)
    
    print_and_save(f"  - 訓練データ: {len(train_labels)} サンプル")
    
    # 全てのスコアを読み込む（post、nsp、base）
    train_all_scores = {}
    
    # Post-retrieval、Pre-retrieval、NSPスコアを読み込む
    # pre_retrieval_output_dirはconfig.pyにPRE_RETRIEVAL_CONFIGSが定義されていれば自動的に使用
    train_qpp_scores = load_qpp_scores(
        'train', 
        qpp_output_dir, 
        nsp_output_dir=nsp_output_dir, 
        nsp_top_k=nsp_top_k,
        pre_retrieval_output_dir=pre_retrieval_output_dir,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=use_nsp
    )
    train_all_scores.update(train_qpp_scores)
    for metric_name, scores in train_qpp_scores.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    
    # ベーススコアを読み込む（data_loader内のコメントアウトで選択）
    train_base_scores = load_base_scores('train', args.dataset, BASE_DIR, base_experiment_names=None, use_bert=use_bert, use_roberta=use_roberta, use_transfer=use_transfer)
    if train_base_scores:
        train_all_scores.update(train_base_scores)
        for feature_name, scores in train_base_scores.items():
            print_and_save(f"  - {feature_name}: {len(scores)} サンプル")
    
    # テストデータ
    dev_json_path = dataset_dir / "dev.json"
    dev_data = load_json_data(dev_json_path)
    dev_labels = extract_labels(dev_data)
    
    print_and_save(f"  - テストデータ: {len(dev_labels)} サンプル")
    
    # 全てのスコアを読み込む（post、nsp、base）
    dev_all_scores = {}
    
    # Post-retrieval、Pre-retrieval、NSPスコアを読み込む
    # pre_retrieval_output_dirはconfig.pyにPRE_RETRIEVAL_CONFIGSが定義されていれば自動的に使用
    dev_qpp_scores = load_qpp_scores(
        'dev', 
        qpp_output_dir, 
        nsp_output_dir=nsp_output_dir, 
        nsp_top_k=nsp_top_k,
        pre_retrieval_output_dir=pre_retrieval_output_dir,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=use_nsp
    )
    dev_all_scores.update(dev_qpp_scores)
    for metric_name, scores in dev_qpp_scores.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    
    # ベーススコアを読み込む（data_loader内のコメントアウトで選択）
    dev_base_scores = load_base_scores('dev', args.dataset, BASE_DIR, base_experiment_names=None, use_bert=use_bert, use_roberta=use_roberta, use_transfer=use_transfer)
    if dev_base_scores:
        dev_all_scores.update(dev_base_scores)
        for feature_name, scores in dev_base_scores.items():
            print_and_save(f"  - {feature_name}: {len(scores)} サンプル")
    
    # 2. 特徴量マージ
    print_and_save("\n2. 特徴量マージ中...")
    X_train, y_train = merge_features(train_all_scores, train_labels)
    X_test, y_test = merge_features(dev_all_scores, dev_labels)
    
    print_and_save(f"  - 訓練データ: {len(X_train)} サンプル, {len(X_train.columns)} 特徴量")
    print_and_save(f"  - テストデータ: {len(X_test)} サンプル, {len(X_test.columns)} 特徴量")
    print_and_save(f"  - 特徴量: {list(X_train.columns)}")
    print_and_save(f"  - 訓練データのラベル分布: {y_train.value_counts().to_dict()}")
    print_and_save(f"  - テストデータのラベル分布: {y_test.value_counts().to_dict()}")
    
    # クロスバリデーションが有効な場合、trainとdevを統合
    if args.use_cv:
        print_and_save("\n2.1. trainとdevを統合中...")
        X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
        print_and_save(f"  - 統合データ: {len(X_combined)} サンプル, {len(X_combined.columns)} 特徴量")
        print_and_save(f"  - 統合データのラベル分布: {y_combined.value_counts().to_dict()} (正例率: {y_combined.mean():.4f})")
    
    # ラベル分布の調整（オプション、クロスバリデーションの場合は統合データに対しては適用しない）
    if args.balance_label_distribution and not args.use_cv:
        print_and_save("\n2.5. ラベル分布の調整中...")
        test_positive_rate = y_test.mean()
        train_positive_rate_before = y_train.mean()
        
        print_and_save(f"  - 調整前の訓練データの正例率: {train_positive_rate_before:.4f}")
        print_and_save(f"  - テストデータの正例率（目標）: {test_positive_rate:.4f}")
        
        X_train, y_train = balance_label_distribution(
            X_train, y_train, target_positive_rate=test_positive_rate, random_state=42
        )
        
        train_positive_rate_after = y_train.mean()
        print_and_save(f"  - 調整後の訓練データの正例率: {train_positive_rate_after:.4f}")
        print_and_save(f"  - 調整後の訓練データのサンプル数: {len(X_train)}")
        print_and_save(f"  - 調整後のラベル分布: {y_train.value_counts().to_dict()}")
    elif args.balance_label_distribution and args.use_cv:
        print_and_save("\n2.5. ラベル分布の調整: スキップ（クロスバリデーション使用時は統合データに対しては適用しません）")
    else:
        print_and_save("\n2.5. ラベル分布の調整: スキップ（--balance-label-distribution が指定されていません）")
    
    # クロスバリデーションが有効な場合は、ここで分岐
    if args.use_cv:
        # クロスバリデーション処理
        print_and_save("\n" + "="*80)
        print_and_save("クロスバリデーション評価を開始します")
        print_and_save("="*80)
        
        # StratifiedKFoldでk分割
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        
        # 各foldの結果を保存
        fold_results = []
        all_test_proba = []
        all_test_labels = []
        fold_models = []  # 各foldのモデルを保存
        all_cv_test_proba = []  # 各foldのテストデータの予測確率（単一指標用）
        all_cv_test_labels = []  # 各foldのテストデータのラベル（単一指標用）
        all_cv_test_features = []  # 各foldのテストデータの特徴量（単一指標用）
        
        # Confidence Analysis用の変数を初期化
        single_metric_probas = {}
        best_metric = None
        best_metric_auc = None
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_combined, y_combined)):
            print_and_save(f"\n{'='*80}")
            print_and_save(f"Fold {fold_idx + 1}/{args.cv_folds}")
            print_and_save(f"{'='*80}")
            
            # foldごとのデータ分割
            X_fold_train = X_combined.iloc[train_idx].reset_index(drop=True)
            y_fold_train = y_combined.iloc[train_idx].reset_index(drop=True)
            X_fold_test = X_combined.iloc[test_idx].reset_index(drop=True)
            y_fold_test = y_combined.iloc[test_idx].reset_index(drop=True)
            
            print_and_save(f"  - Fold訓練データ: {len(X_fold_train)} サンプル")
            print_and_save(f"  - Foldテストデータ: {len(X_fold_test)} サンプル")
            print_and_save(f"  - Fold訓練データのラベル分布: {y_fold_train.value_counts().to_dict()} (正例率: {y_fold_train.mean():.4f})")
            print_and_save(f"  - Foldテストデータのラベル分布: {y_fold_test.value_counts().to_dict()} (正例率: {y_fold_test.mean():.4f})")
            
            # 正規化（訓練データの統計量でテストデータも正規化）
            normalization_type_name = "[0,1]正規化（Min-Max）" if args.use_minmax_normalization else "z-score正規化"
            X_fold_train_norm, X_fold_test_norm = normalize_features(
                X_fold_train, X_fold_test,
                use_combined_normalization=False,
                use_separate_normalization=False,
                use_minmax_normalization=args.use_minmax_normalization
            )
            
            # モデル学習（全てのモデルを実行する場合は、最初のモデルタイプのみ実行）
            current_model_type = model_types_to_run[0] if run_all_models else args.model_type
            fold_model = create_model(
                model_type=current_model_type,
                    max_iter=args.max_iter,
                    random_state=42,
                    class_weight='balanced',
                tol=args.tol,
                non_negative=args.non_negative_coefficients,
                n_bootstrap=args.n_bootstrap,
                selection_threshold=args.selection_threshold,
                n_random_traps=args.n_random_traps,
                cv=args.lars_cv_folds
            )
            # 特徴量選択モデルの場合はprint_and_save_funcを渡す
            if current_model_type in ['bolasso', 'lars_traps', 'lars_cv']:
                fold_model.fit(X_fold_train_norm, y_fold_train, print_and_save_func=print_and_save)
            else:
                fold_model.fit(X_fold_train_norm, y_fold_train)
            
            # モデルを保存
            fold_models.append(fold_model)
            
            # 予測
            y_fold_train_pred = fold_model.predict(X_fold_train_norm)
            y_fold_test_pred = fold_model.predict(X_fold_test_norm)
            y_fold_train_proba = fold_model.predict_proba(X_fold_train_norm)[:, 1]
            y_fold_test_proba = fold_model.predict_proba(X_fold_test_norm)[:, 1]
            
            # 評価
            fold_train_acc = accuracy_score(y_fold_train, y_fold_train_pred)
            fold_train_f1 = f1_score(y_fold_train, y_fold_train_pred)
            fold_train_auc = roc_auc_score(y_fold_train, y_fold_train_proba)
            fold_train_ap = average_precision_score(y_fold_train, y_fold_train_proba)
            
            fold_test_acc = accuracy_score(y_fold_test, y_fold_test_pred)
            fold_test_f1 = f1_score(y_fold_test, y_fold_test_pred)
            fold_test_auc = roc_auc_score(y_fold_test, y_fold_test_proba)
            fold_test_ap = average_precision_score(y_fold_test, y_fold_test_proba)
            
            fold_results.append({
                'fold': fold_idx + 1,
                'train_acc': fold_train_acc,
                'train_f1': fold_train_f1,
                'train_auc': fold_train_auc,
                'train_ap': fold_train_ap,
                'test_acc': fold_test_acc,
                'test_f1': fold_test_f1,
                'test_auc': fold_test_auc,
                'test_ap': fold_test_ap
            })
            
            all_test_proba.extend(y_fold_test_proba)
            all_test_labels.extend(y_fold_test.tolist())
            
            # 単一指標の曲線用にデータを保存
            all_cv_test_proba.extend(y_fold_test_proba)
            all_cv_test_labels.extend(y_fold_test.tolist())
            all_cv_test_features.append(X_fold_test_norm)
            
            print_and_save(f"\n  Fold {fold_idx + 1} 結果:")
            print_and_save(f"    訓練 - Accuracy: {fold_train_acc:.4f}, F1: {fold_train_f1:.4f}, AUC: {fold_train_auc:.4f}, AP: {fold_train_ap:.4f}")
            print_and_save(f"    テスト - Accuracy: {fold_test_acc:.4f}, F1: {fold_test_f1:.4f}, AUC: {fold_test_auc:.4f}, AP: {fold_test_ap:.4f}")
        
        # 結果の集約
        print_and_save("\n" + "="*80)
        print_and_save("クロスバリデーション結果の集約")
        print_and_save("="*80)
        
        # 各foldの結果をDataFrameにまとめる
        fold_df = pd.DataFrame(fold_results)
        
        print_and_save("\n各Foldの結果:")
        print_and_save(fold_df.to_string(index=False))
        
        print_and_save("\n平均値（標準偏差）:")
        for metric in ['train_acc', 'train_f1', 'train_auc', 'train_ap', 'test_acc', 'test_f1', 'test_auc', 'test_ap']:
            mean_val = fold_df[metric].mean()
            std_val = fold_df[metric].std()
            print_and_save(f"  {metric}: {mean_val:.4f} (±{std_val:.4f})")
        
        # 全体のAUCとAPを計算（全foldのテストデータを統合）
        all_test_auc = roc_auc_score(all_test_labels, all_test_proba)
        all_test_ap = average_precision_score(all_test_labels, all_test_proba)
        print_and_save(f"\n全Fold統合テストデータ:")
        print_and_save(f"  AUC-ROC: {all_test_auc:.4f}")
        print_and_save(f"  Average Precision: {all_test_ap:.4f}")
        
        # 特徴量名に基づいて出力ディレクトリとファイルパスを設定
        feature_names = list(X_combined.columns)
        feature_dir_name = get_feature_dir_name(feature_names)
        feature_output_dir = output_dir / feature_dir_name
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        # サブフォルダを作成
        graphs_dir = feature_output_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = feature_output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        output_file = feature_output_dir / "results.txt"
        
        print_and_save("\n=== クロスバリデーション完了 ===")
        print_and_save("\n" + "="*80)
        print_and_save("クロスバリデーションで得られたモデルを使って、train/devデータに対して評価を実行します")
        print_and_save("="*80)
        
        # クロスバリデーションの結果を使って評価
        # 各foldのテストデータに対する予測結果を既に保存しているので、それを使用
        # all_test_probaとall_test_labelsが既に全foldのテストデータの予測結果とラベルを含んでいる
        
        print_and_save("\n=== クロスバリデーション結果による評価 ===")
        print_and_save("（各foldのテストデータに対する予測結果を使用）")
        
        # CV結果の評価（全foldのテストデータを統合）
        y_cv_test_proba = np.array(all_test_proba)
        y_cv_test_labels = np.array(all_test_labels)
        y_cv_test_pred = (y_cv_test_proba >= 0.5).astype(int)
        
        cv_test_acc = accuracy_score(y_cv_test_labels, y_cv_test_pred)
        cv_test_f1 = f1_score(y_cv_test_labels, y_cv_test_pred)
        cv_test_auc = roc_auc_score(y_cv_test_labels, y_cv_test_proba)
        cv_test_ap = average_precision_score(y_cv_test_labels, y_cv_test_proba)
        
        print_and_save(f"CVテストデータ - Accuracy: {cv_test_acc:.4f}, F1: {cv_test_f1:.4f}, AUC: {cv_test_auc:.4f}, AP: {cv_test_ap:.4f}")
        
        # 単体指標の最高ROCと回帰モデルのDeLong検定（オプション）
        if args.delong_test:
            print_and_save("\n5.6. 単体指標の最高ROCと回帰モデルのDeLong検定を実行中...")
            
            # 各単体指標についてロジスティック回帰を実行し、ROCを計算
            single_metric_aucs = {}
            single_metric_aps = {}
            single_metric_probas = {}
            
            # CV統合データに対して各foldで単体指標の評価を行う
            # 非学習指標（QPPスコア）は直接使用、学習指標（BERT等）はロジスティック回帰を使用
            from config import POST_RETRIEVAL_CONFIGS, PRE_RETRIEVAL_CONFIGS
            from module.data_loader import NSP_METRICS, get_nsp_metric_name
            
            # 非学習指標のリスト（QPPスコア）
            non_learning_metrics = set(POST_RETRIEVAL_CONFIGS.keys()) | set(PRE_RETRIEVAL_CONFIGS.keys())
            # NSPメトリクスも非学習指標として扱う（動的に生成されるため、名前で判定）
            nsp_metric_prefixes = [f"nsp_{metric}_topk" for metric in NSP_METRICS.keys()]
            
            for metric_name in X_combined.columns:
                # 非学習指標かどうかを判定
                is_non_learning = (
                    metric_name in non_learning_metrics or
                    any(metric_name.startswith(prefix) for prefix in nsp_metric_prefixes)
                )
                
                if is_non_learning:
                    # 非学習指標：QPPスコアを直接使用（CV統合テストデータに対応するスコアを使用）
                    # 全foldのテストデータに対応するスコアを収集
                    metric_scores_list = []
                    metric_labels_list = []
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_combined, y_combined)):
                        X_fold_test = X_combined.iloc[test_idx].reset_index(drop=True)
                        y_fold_test = y_combined.iloc[test_idx].reset_index(drop=True)
                        metric_scores_list.append(X_fold_test[metric_name].values)
                        metric_labels_list.append(y_fold_test.values)
                    
                    # 全foldのスコアとラベルを統合
                    metric_scores_all = np.concatenate(metric_scores_list)
                    metric_labels_all = np.concatenate(metric_labels_list)
                    
                    # Min-Max正規化で[0,1]に変換（予測確率として使用）
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    metric_scores_normalized = scaler.fit_transform(metric_scores_all.reshape(-1, 1)).flatten()
                    
                    try:
                        single_metric_aucs[metric_name] = roc_auc_score(metric_labels_all, metric_scores_normalized)
                        single_metric_aps[metric_name] = average_precision_score(metric_labels_all, metric_scores_normalized)
                        single_metric_probas[metric_name] = metric_scores_normalized
                    except ValueError:
                        pass
                else:
                    # 学習指標（BERT等）：ロジスティック回帰を使用
                    metric_aucs = []
                    metric_probas = []
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_combined, y_combined)):
                        X_fold_train = X_combined.iloc[train_idx].reset_index(drop=True)
                        y_fold_train = y_combined.iloc[train_idx].reset_index(drop=True)
                        X_fold_test = X_combined.iloc[test_idx].reset_index(drop=True)
                        y_fold_test = y_combined.iloc[test_idx].reset_index(drop=True)
                        
                        # 正規化
                        X_fold_train_norm, X_fold_test_norm = normalize_features(
                            X_fold_train, X_fold_test,
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
                        single_metric_model.fit(X_fold_train_norm[[metric_name]], y_fold_train)
                        y_fold_test_proba = single_metric_model.predict_proba(X_fold_test_norm[[metric_name]])[:, 1]
                        
                        try:
                            fold_auc = roc_auc_score(y_fold_test, y_fold_test_proba)
                            fold_ap = average_precision_score(y_fold_test, y_fold_test_proba)
                            metric_aucs.append(fold_auc)
                            metric_probas.append(y_fold_test_proba)
                        except ValueError:
                            pass
                    
                    if len(metric_aucs) > 0:
                        # 全foldの予測確率を統合
                        single_metric_probas[metric_name] = np.concatenate(metric_probas)
                        # 統合された予測確率からAUC-ROCとAUC-PRを計算（out-of-fold統合）
                        single_metric_aucs[metric_name] = roc_auc_score(y_cv_test_labels, single_metric_probas[metric_name])
                        single_metric_aps[metric_name] = average_precision_score(y_cv_test_labels, single_metric_probas[metric_name])
            
            if len(single_metric_aucs) > 0:
                # 最高ROCを出した指標を見つける
                best_metric = max(single_metric_aucs, key=single_metric_aucs.get)
                best_metric_auc = single_metric_aucs[best_metric]
                best_metric_proba = single_metric_probas[best_metric]
                
                print_and_save(f"  単体指標の最高ROC: {format_feature_name(best_metric)} (AUC = {best_metric_auc:.4f})")
                print_and_save(f"  回帰モデルのROC: AUC = {cv_test_auc:.4f}")
                
                # DeLongの検定
                delong_result = delong_test(
                    y_true=y_cv_test_labels,
                    y_pred_proba_a=np.array(all_test_proba),
                    y_pred_proba_b=best_metric_proba
                )
                
                print_and_save(f"\n  DeLongの検定結果:")
                print_and_save(f"    モデルA (回帰): AUC = {delong_result['auc_a']:.4f}")
                print_and_save(f"    モデルB (単体指標 {format_feature_name(best_metric)}): AUC = {delong_result['auc_b']:.4f}")
                print_and_save(f"    AUC差: {delong_result['auc_diff']:.4f}")
                print_and_save(f"    Z統計量: {delong_result['z_stat']:.4f}")
                print_and_save(f"    p値: {delong_result['p_value']:.4f}")
                print_and_save(f"    有意差: {'あり' if delong_result['significant'] else 'なし'} (α=0.05)")
                
                # 各指標と回帰モデルとの比較表を作成
                print_and_save("\n  === 各指標とモデルの比較表 ===")
                comparison_data = []
                
                # 各指標のAUC-ROCとAUC-PRを計算
                for metric_name in single_metric_aucs.keys():
                    metric_proba = single_metric_probas[metric_name]
                    metric_auc = single_metric_aucs[metric_name]
                    metric_ap = single_metric_aps[metric_name]
                    
                    comparison_data.append({
                        'Metric': format_feature_name(metric_name),
                        'AUC-ROC': f"{metric_auc:.4f}",
                        'AUC-PR': f"{metric_ap:.4f}",
                        'p-value': '-'
                    })
                
                # 回帰モデルの情報を追加（最高性能の単体指標とのDeLong検定のp-value）
                regression_ap = average_precision_score(y_cv_test_labels, np.array(all_test_proba))
                model_name = {
                    'l1': 'L1 Logistic Regression',
                    'l2': 'L2 Logistic Regression',
                    'elasticnet': 'E-Net',
                    'none': 'No Penalty Logistic Regression',
                    'bolasso': 'BOLASSO',
                    'lars_traps': 'LARS-Traps',
                    'lars_cv': 'LARS-CV',
                    'randomforest': 'Random Forest'
                }.get(args.model_type, f'{args.model_type.upper()} Logistic Regression')
                
                # 最高性能の単体指標とのDeLong検定
                # p値の表示：0.0000になる場合は科学記法で表示
                if np.isnan(delong_result['p_value']):
                    p_value_str = '-'
                elif delong_result['p_value'] < 0.0001:
                    p_value_str = f"{delong_result['p_value']:.2e}"
                else:
                    p_value_str = f"{delong_result['p_value']:.4f}"
                
                comparison_data.append({
                    'Metric': model_name,
                    'AUC-ROC': f"{cv_test_auc:.4f}",
                    'AUC-PR': f"{regression_ap:.4f}",
                    'p-value': p_value_str
                })
                
                # 表を表示
                comparison_df = pd.DataFrame(comparison_data)
                # タブ区切り形式で出力（スプレッドシートにコピペしやすい）
                print_and_save("\n" + comparison_df.to_csv(sep='\t', index=False))
                
                # 結果を保存（delongディレクトリに）
                delong_dir = feature_output_dir / "delong"
                delong_dir.mkdir(parents=True, exist_ok=True)
                delong_result_path = delong_dir / "delong_test_single_metric.csv"
                result_df = pd.DataFrame([{
                    'best_single_metric': best_metric,
                    'best_single_metric_auc': best_metric_auc,
                    'regression_auc': cv_test_auc,
                    **delong_result
                }])
                result_df.to_csv(delong_result_path, index=False)
                print_and_save(f"  - DeLongの検定結果を保存: {delong_result_path}")
                
                # 比較表も保存
                comparison_csv_path = delong_dir / "comparison_table.csv"
                comparison_df.to_csv(comparison_csv_path, index=False)
                print_and_save(f"  - 比較表を保存: {comparison_csv_path}")
            else:
                print_and_save("  ⚠️  警告: 単体指標のROCを計算できませんでした")
        
            # ブートストラップ評価（オプション）
            if args.bootstrap_test:
                if not args.bootstrap_samples_path:
                    print_and_save("  ⚠️  警告: --bootstrap-testを使用する場合は--bootstrap-samples-pathを指定してください")
                else:
                    bootstrap_samples_path = Path(args.bootstrap_samples_path)
                    if not bootstrap_samples_path.exists():
                        print_and_save(f"  ⚠️  警告: ブートストラップサンプルファイルが見つかりません: {bootstrap_samples_path}")
                    else:
                        print_and_save("\n5.7. ブートストラップ評価を実行中（単体指標vs回帰モデル）...")
                        bootstrap_samples = load_bootstrap_samples(bootstrap_samples_path)
                        print_and_save(f"  - ブートストラップサンプルを読み込み: {len(bootstrap_samples)} 反復")
                        
                        # 回帰モデルのブートストラップ評価
                        regression_bootstrap_results = evaluate_bootstrap(
                            y_true=np.array(all_test_labels),
                            y_pred_proba=np.array(all_test_proba),
                            bootstrap_samples=bootstrap_samples,
                            feature_combination_name="regression"
                        )
                        
                        # 単体指標のブートストラップ評価
                        single_metric_bootstrap_results = evaluate_bootstrap(
                            y_true=y_cv_test_labels,
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
                        bootstrap_results_path = csv_dir / "bootstrap_results_single_vs_regression.csv"
                        save_bootstrap_results(all_bootstrap_results, bootstrap_results_path)
                        print_and_save(f"  - ブートストラップ結果を保存: {bootstrap_results_path}")
                        
                        # サマリーを表示
                        print_and_save(f"\n  ブートストラップ結果のサマリー:")
                        for combination in ["regression", f"single_{best_metric}"]:
                            print_and_save(f"\n  {combination}:")
                            for metric in ['auc', 'ap', 'f1', 'accuracy']:
                                metric_values = all_bootstrap_results[
                                    (all_bootstrap_results['feature_combination'] == combination) &
                                    (all_bootstrap_results['metric_name'] == metric)
                                ]['value'].values
                                mean_val = np.mean(metric_values)
                                std_val = np.std(metric_values)
                                ci_lower = np.percentile(metric_values, 2.5)
                                ci_upper = np.percentile(metric_values, 97.5)
                                print_and_save(f"    {metric.upper()}: {mean_val:.4f} ± {std_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        
            # 予測結果を保存（回帰モデル同士の比較用）
            predictions_path = csv_dir / "predictions_for_delong.csv"
            pd.DataFrame({
                'y_true': np.array(all_test_labels),
                'y_pred_proba': np.array(all_test_proba)
            }).to_csv(predictions_path, index=False)
            print_and_save(f"  - 回帰モデル同士の比較用に予測結果を保存: {predictions_path}")
        
        # 可視化用に、train/devデータに対する予測も計算（参考用、評価には使用しない）
        # まず、統合データ全体で正規化パラメータを計算（訓練データの統計量を使用）
        normalization_type_name = "[0,1]正規化（Min-Max）" if args.use_minmax_normalization else "z-score正規化"
        X_train_norm, X_test_norm = normalize_features(
            X_train, X_test,
            use_combined_normalization=False,
            use_separate_normalization=False,
            use_minmax_normalization=args.use_minmax_normalization
        )
        
        # 各foldのモデルで予測してアンサンブル（参考用）
        print_and_save("\n参考: train/devデータに対する予測（評価には使用しません）")
        all_train_proba = []
        all_test_proba_ref = []
        
        for fold_idx, fold_model in enumerate(fold_models):
            train_proba_fold = fold_model.predict_proba(X_train_norm)[:, 1]
            test_proba_fold = fold_model.predict_proba(X_test_norm)[:, 1]
            all_train_proba.append(train_proba_fold)
            all_test_proba_ref.append(test_proba_fold)
        
        # アンサンブル（平均）
        y_train_proba = np.mean(all_train_proba, axis=0)
        y_test_proba = np.mean(all_test_proba_ref, axis=0)
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        y_test_pred = (y_test_proba >= 0.5).astype(int)
        
        # 評価（参考用）
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_proba)
        train_ap = average_precision_score(y_train, y_train_proba)
        
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        test_ap = average_precision_score(y_test, y_test_proba)
        
        print_and_save(f"参考 - 訓練データ: Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, AP: {train_ap:.4f}")
        print_and_save(f"参考 - テストデータ: Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
        
        # 可視化にはCV結果を使用
        y_train_proba_cv = None  # trainは表示しない
        y_test_proba_cv = y_cv_test_proba  # CV結果を使用
        y_test_cv = pd.Series(y_cv_test_labels)  # CV結果のラベルを使用
        train_auc_cv = None
        test_auc_cv = cv_test_auc
        train_ap_cv = None
        test_ap_cv = cv_test_ap
        
        # 特徴量の係数の平均を計算（各foldのモデルの係数の平均）
        all_coefs = np.array([get_coefficients(model, X_train_norm.columns, args.model_type) for model in fold_models])
        mean_coefs = np.mean(all_coefs, axis=0)
        
        coefficient_label = "重要度" if args.model_type == 'randomforest' else "係数"
        print_and_save(f"\n特徴量の{coefficient_label}（各foldの平均）:")
        feature_importance = pd.DataFrame({
            'feature': X_train_norm.columns,
            'coefficient': mean_coefs
        }).sort_values('coefficient', key=abs, ascending=False)
        print_and_save(feature_importance.to_string(index=False))
        
        # 係数をCSVファイルとして保存
        coefficients_csv_path = feature_output_dir / "coefficients.csv"
        feature_importance.to_csv(coefficients_csv_path, index=False)
        print_and_save(f"  - {coefficient_label}をCSVファイルに保存しました: {coefficients_csv_path}")
        
        # 係数が0の特徴量を表示（L1正則化の場合のみ）
        if args.model_type == 'l1':
            zero_coef_features = feature_importance[feature_importance['coefficient'] == 0.0]
            if len(zero_coef_features) > 0:
                print_and_save(f"\n係数が0の特徴量（L1正則化により除外）: {len(zero_coef_features)}個")
                print_and_save(zero_coef_features[['feature']].to_string(index=False))
            else:
                print_and_save("\n係数が0の特徴量はありません（全ての特徴量が使用されています）")
        
        # 可視化のために、モデルオブジェクトを作成（係数の平均を使用）
        # ただし、実際の予測は既にアンサンブルで行っているので、可視化用にダミーモデルを作成
        class EnsembleModel:
            def __init__(self, mean_coefs, intercept_mean):
                self.coef_ = mean_coefs.reshape(1, -1)
                self.intercept_ = intercept_mean
            
            def predict_proba(self, X):
                if isinstance(X, pd.DataFrame):
                    X_array = X.values
                else:
                    X_array = X
                z = X_array @ self.coef_[0] + self.intercept_
                proba_positive = expit(z)
                proba_negative = 1 - proba_positive
                return np.column_stack([proba_negative, proba_positive])
        
        # 切片の平均も計算
        if args.non_negative_coefficients:
            intercept_mean = np.mean([model.intercept_ for model in fold_models])
        else:
            intercept_mean = np.mean([model.intercept_ for model in fold_models])
        
        model = EnsembleModel(mean_coefs, intercept_mean)
        
        # 単一指標の曲線用にCV結果の特徴量データを統合
        X_cv_test_norm = pd.concat(all_cv_test_features, axis=0, ignore_index=True)
        y_cv_test = pd.Series(all_cv_test_labels)
        
        # 予測結果を保存（DeLongの検定用、CV統合データ）
        if args.delong_test:
            feature_names = list(X_combined.columns)
            feature_dir_name = get_feature_dir_name(feature_names)
            feature_output_dir = output_dir / feature_dir_name
            feature_output_dir.mkdir(parents=True, exist_ok=True)
            # サブフォルダを作成
            csv_dir = feature_output_dir / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = csv_dir / "predictions_for_delong.csv"
            pd.DataFrame({
                'y_true': all_test_labels,
                'y_pred_proba': all_test_proba
            }).to_csv(predictions_path, index=False)
            print_and_save(f"  - DeLongの検定用に予測結果を保存（CV統合）: {predictions_path}")
        
        # 特徴量名に基づいて出力ディレクトリとファイルパスを設定
        feature_names = list(X_train_norm.columns)
        feature_dir_name = get_feature_dir_name(feature_names)
        feature_output_dir = output_dir / feature_dir_name
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        # サブフォルダを作成
        graphs_dir = feature_output_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = feature_output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        output_file = feature_output_dir / "results.txt"
    
    # データ分布の確認（正規化前）
    if not args.use_cv:  # クロスバリデーションの場合は既に処理済み
        print_and_save("\n【データ分布の確認（正規化前）】")
        print_and_save("訓練データの特徴量統計（正規化前）:")
        print_and_save(X_train.describe().to_string())
        print_and_save("\nテストデータの特徴量統計（正規化前）:")
        print_and_save(X_test.describe().to_string())
        print_and_save(f"\n訓練データのラベル分布: {y_train.value_counts().to_dict()} (正例率: {y_train.mean():.4f})")
        print_and_save(f"テストデータのラベル分布: {y_test.value_counts().to_dict()} (正例率: {y_test.mean():.4f})")
    
    # 3. 前処理: 正規化
    normalization_type_name = "[0,1]正規化（Min-Max）" if args.use_minmax_normalization else "z-score正規化"
    print_and_save(f"\n3. {normalization_type_name}中...")
    if args.use_separate_normalization:
        print_and_save("  - 方法: 訓練データとテストデータをそれぞれ個別に正規化")
        print_and_save("  ⚠️  注意: 各データセットが独立に正規化されるため、分布の違いは排除されますが、実際の予測タスクでは使用できません。")
    elif args.use_combined_normalization:
        print_and_save("  - 方法: 訓練データとテストデータを結合してから正規化（リーク前提）")
        print_and_save("  ⚠️  警告: データリークが発生しています！実際の予測タスクでは使用しないでください。")
    else:
        print_and_save("  - 方法: 訓練データの統計量でテストデータも正規化（通常）")
    X_train_norm, X_test_norm = normalize_features(
        X_train, X_test, 
        use_combined_normalization=args.use_combined_normalization,
        use_separate_normalization=args.use_separate_normalization,
        use_minmax_normalization=args.use_minmax_normalization
    )
    print_and_save("  - 正規化完了")
    
    # データ分布の確認（正規化後）
    if not args.use_cv:  # クロスバリデーションの場合は既に表示済み
        print_and_save("\n【データ分布の確認（正規化後）】")
        print_and_save("訓練データの特徴量統計（正規化後）:")
        print_and_save(X_train_norm.describe().to_string())
        print_and_save("\nテストデータの特徴量統計（正規化後）:")
        print_and_save(X_test_norm.describe().to_string())
        
        if args.use_combined_normalization:
            # 結合正規化の場合、結合データ全体の統計も表示
            X_combined_norm = pd.concat([X_train_norm, X_test_norm], axis=0, ignore_index=True)
            print_and_save("\n結合データ全体の特徴量統計（正規化後）:")
            print_and_save(X_combined_norm.describe().to_string())
            print_and_save("\n→ 結合正規化により、結合データ全体の平均≈0, 標準偏差≈1になるはず")
        
        # 分布の違いを数値で確認（正規化前）
        print_and_save("\n【分布の違いの分析（正規化前）】")
        for col in X_train.columns:
            train_mean = X_train[col].mean()
            test_mean = X_test[col].mean()
            train_std = X_train[col].std()
            test_std = X_test[col].std()
            mean_diff = abs(train_mean - test_mean) / (abs(train_mean) + 1e-10)
            std_diff = abs(train_std - test_std) / (abs(train_std) + 1e-10)
            print_and_save(f"{col}:")
            print_and_save(f"  平均の差: 訓練={train_mean:.4f}, テスト={test_mean:.4f}, 相対差={mean_diff:.4f}")
            print_and_save(f"  標準偏差の差: 訓練={train_std:.4f}, テスト={test_std:.4f}, 相対差={std_diff:.4f}")
        
        # 分布の違いを数値で確認（正規化後）
        print_and_save("\n【分布の違いの分析（正規化後）】")
        for col in X_train_norm.columns:
            train_norm_mean = X_train_norm[col].mean()
            test_norm_mean = X_test_norm[col].mean()
            train_norm_std = X_train_norm[col].std()
            test_norm_std = X_test_norm[col].std()
            mean_diff_norm = abs(train_norm_mean - test_norm_mean)
            std_diff_norm = abs(train_norm_std - test_norm_std)
            print_and_save(f"{col}:")
            print_and_save(f"  平均: 訓練={train_norm_mean:.6f}, テスト={test_norm_mean:.6f}, 差={mean_diff_norm:.6f}")
            print_and_save(f"  標準偏差: 訓練={train_norm_std:.6f}, テスト={test_norm_std:.6f}, 差={std_diff_norm:.6f}")
            
            if args.use_combined_normalization:
                # 結合正規化の場合、結合データ全体の統計も表示
                X_combined_norm = pd.concat([X_train_norm, X_test_norm], axis=0, ignore_index=True)
                combined_mean = X_combined_norm[col].mean()
                combined_std = X_combined_norm[col].std()
                print_and_save(f"  結合データ全体: 平均={combined_mean:.6f}, 標準偏差={combined_std:.6f}")
                print_and_save(f"  → 結合正規化により、結合データ全体の平均≈0, 標準偏差≈1になるはず")
    
    # Confidence Analysis用の変数を初期化（通常評価の場合）
    if not args.use_cv:
        single_metric_probas = {}
        best_metric = None
        best_metric_auc = None
    
    # 4. モデル学習（クロスバリデーションの場合は既に完了）
    if not args.use_cv:
        print_and_save("\n4. モデル学習中...")
        model_type_name = {
            'l1': 'L1正則化ロジスティック回帰',
            'l2': 'L2正則化ロジスティック回帰',
            'elasticnet': 'ElasticNet正則化ロジスティック回帰',
            'randomforest': 'RandomForest分類器'
        }.get(args.model_type, args.model_type)
        print_and_save(f"  - モデルタイプ: {model_type_name}")
        if args.non_negative_coefficients and args.model_type == 'l1':
            print_and_save("  - 非負制約付きモデルを使用します")
            print_and_save(f"  - 最大反復回数: {args.max_iter}")
        model = create_model(
            model_type=args.model_type,
                max_iter=args.max_iter,
                random_state=42,
                class_weight='balanced',
            tol=args.tol,
            non_negative=args.non_negative_coefficients,
            n_bootstrap=args.n_bootstrap,
            selection_threshold=args.selection_threshold,
            n_random_traps=args.n_random_traps,
            cv=args.lars_cv_folds
        )
        # 特徴量選択モデルの場合はprint_and_save_funcを渡す
        if args.model_type in ['bolasso', 'lars_traps', 'lars_cv']:
            model.fit(X_train_norm, y_train, print_and_save_func=print_and_save)
        else:
            model.fit(X_train_norm, y_train)
        
        # 収束状況を確認（線形モデルの場合のみ）
        if args.model_type in ['l1', 'l2', 'elasticnet', 'none']:
            actual_iter = model.n_iter_[0] if hasattr(model, 'n_iter_') and len(model.n_iter_) > 0 else 'unknown'
            print_and_save(f"  - 実際の反復回数: {actual_iter}")
            
            # 警告をキャッチして収束状況を確認
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # 警告があるかチェック（max_iterに達した場合）
                if w:
                    for warning in w:
                        if "max_iter" in str(warning.message).lower() or "convergence" in str(warning.message).lower():
                            print_and_save(f"  ⚠️  警告: {warning.message}")
                            print_and_save(f"  ⚠️  max_iterを増やすことを検討してください（現在: {args.max_iter}）")
                else:
                    print_and_save("  - 正常に収束しました")
        else:
            print_and_save("  - 学習完了")
        
        # 特徴量の重要度（係数）を表示
        coefficient_label = "重要度" if args.model_type == 'randomforest' else "係数"
        print_and_save(f"\n特徴量の{coefficient_label}:")
        coefficients = get_coefficients(model, X_train_norm.columns, args.model_type)
        feature_importance = pd.DataFrame({
            'feature': X_train_norm.columns,
            'coefficient': coefficients
        }).sort_values('coefficient', key=abs, ascending=False)
        print_and_save(feature_importance.to_string(index=False))
        
        # 係数をCSVファイルとして保存
        feature_names = list(X_train_norm.columns)
        feature_dir_name = get_feature_dir_name(feature_names)
        feature_output_dir = output_dir / feature_dir_name
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        # サブフォルダを作成
        graphs_dir = feature_output_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        csv_dir = feature_output_dir / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        coefficients_csv_path = csv_dir / "coefficients.csv"
        feature_importance.to_csv(coefficients_csv_path, index=False)
        print_and_save(f"  - {coefficient_label}をCSVファイルに保存しました: {coefficients_csv_path}")
        
        # 係数が0の特徴量を表示（L1正則化の場合のみ）
        if args.model_type == 'l1':
            zero_coef_features = feature_importance[feature_importance['coefficient'] == 0.0]
            if len(zero_coef_features) > 0:
                print_and_save(f"\n係数が0の特徴量（L1正則化により除外）: {len(zero_coef_features)}個")
                print_and_save(zero_coef_features[['feature']].to_string(index=False))
            else:
                print_and_save("\n係数が0の特徴量はありません（全ての特徴量が使用されています）")
        
        # 5. 評価
        print_and_save("\n5. 評価中...")
        y_train_pred = model.predict(X_train_norm)
        y_test_pred = model.predict(X_test_norm)
        
        y_train_proba = model.predict_proba(X_train_norm)[:, 1]
        y_test_proba = model.predict_proba(X_test_norm)[:, 1]
        
        # 訓練データの評価
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, y_train_proba)
        train_ap = average_precision_score(y_train, y_train_proba)
        
        # テストデータの評価
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        test_ap = average_precision_score(y_test, y_test_proba)
        
        # 単体指標の最高ROCと回帰モデルのDeLong検定（オプション）
        if args.delong_test:
            print_and_save("\n5.6. 単体指標の最高ROCと回帰モデルのDeLong検定を実行中...")
            
            # 各単体指標についてロジスティック回帰を実行し、ROCを計算
            single_metric_aucs = {}
            single_metric_aps = {}
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
                    test_ap_single = average_precision_score(y_test, y_test_proba_single)
                    single_metric_aucs[metric_name] = test_auc_single
                    single_metric_aps[metric_name] = test_ap_single
                    single_metric_probas[metric_name] = y_test_proba_single
                except ValueError:
                    pass
            
            if len(single_metric_aucs) > 0:
                # 最高ROCを出した指標を見つける
                best_metric = max(single_metric_aucs, key=single_metric_aucs.get)
                best_metric_auc = single_metric_aucs[best_metric]
                best_metric_proba = single_metric_probas[best_metric]
                
                print_and_save(f"  単体指標の最高ROC: {format_feature_name(best_metric)} (AUC = {best_metric_auc:.4f})")
                print_and_save(f"  回帰モデルのROC: AUC = {test_auc:.4f}")
                
                # DeLongの検定
                delong_result = delong_test(
                    y_true=y_test.values,
                    y_pred_proba_a=y_test_proba,
                    y_pred_proba_b=best_metric_proba
                )
                
                print_and_save(f"\n  DeLongの検定結果:")
                print_and_save(f"    モデルA (回帰): AUC = {delong_result['auc_a']:.4f}")
                print_and_save(f"    モデルB (単体指標 {format_feature_name(best_metric)}): AUC = {delong_result['auc_b']:.4f}")
                print_and_save(f"    AUC差: {delong_result['auc_diff']:.4f}")
                print_and_save(f"    Z統計量: {delong_result['z_stat']:.4f}")
                print_and_save(f"    p値: {delong_result['p_value']:.4f}")
                print_and_save(f"    有意差: {'あり' if delong_result['significant'] else 'なし'} (α=0.05)")
                
                # 各指標と回帰モデルとの比較表を作成
                print_and_save("\n  === 各指標とモデルの比較表 ===")
                comparison_data = []
                
                # 各指標のAUC-ROCとAUC-PRを計算
                for metric_name in single_metric_aucs.keys():
                    metric_proba = single_metric_probas[metric_name]
                    metric_auc = single_metric_aucs[metric_name]
                    metric_ap = single_metric_aps[metric_name]
                    
                    comparison_data.append({
                        'Metric': format_feature_name(metric_name),
                        'AUC-ROC': f"{metric_auc:.4f}",
                        'AUC-PR': f"{metric_ap:.4f}",
                        'p-value': '-'
                    })
                
                # 回帰モデルの情報を追加（最高性能の単体指標とのDeLong検定のp-value）
                regression_ap = average_precision_score(y_test, y_test_proba)
                model_name = {
                    'l1': 'L1 Logistic Regression',
                    'l2': 'L2 Logistic Regression',
                    'elasticnet': 'E-Net',
                    'none': 'No Penalty Logistic Regression',
                    'bolasso': 'BOLASSO',
                    'lars_traps': 'LARS-Traps',
                    'lars_cv': 'LARS-CV',
                    'randomforest': 'Random Forest'
                }.get(args.model_type, f'{args.model_type.upper()} Logistic Regression')
                
                # 最高性能の単体指標とのDeLong検定
                # p値の表示：0.0000になる場合は科学記法で表示
                if np.isnan(delong_result['p_value']):
                    p_value_str = '-'
                elif delong_result['p_value'] < 0.0001:
                    p_value_str = f"{delong_result['p_value']:.2e}"
                else:
                    p_value_str = f"{delong_result['p_value']:.4f}"
                
                comparison_data.append({
                    'Metric': model_name,
                    'AUC-ROC': f"{test_auc:.4f}",
                    'AUC-PR': f"{regression_ap:.4f}",
                    'p-value': p_value_str
                })
                
                # 表を表示
                comparison_df = pd.DataFrame(comparison_data)
                # タブ区切り形式で出力（スプレッドシートにコピペしやすい）
                print_and_save("\n" + comparison_df.to_csv(sep='\t', index=False))
                
                # 結果を保存（delongディレクトリに）
                delong_dir = feature_output_dir / "delong"
                delong_dir.mkdir(parents=True, exist_ok=True)
                delong_result_path = delong_dir / "delong_test_single_metric.csv"
                result_df = pd.DataFrame([{
                    'best_single_metric': best_metric,
                    'best_single_metric_auc': best_metric_auc,
                    'regression_auc': test_auc,
                    **delong_result
                }])
                result_df.to_csv(delong_result_path, index=False)
                print_and_save(f"  - DeLongの検定結果を保存: {delong_result_path}")
                
                # 比較表も保存
                comparison_csv_path = delong_dir / "comparison_table.csv"
                comparison_df.to_csv(comparison_csv_path, index=False)
                print_and_save(f"  - 比較表を保存: {comparison_csv_path}")
            else:
                print_and_save("  ⚠️  警告: 単体指標のROCを計算できませんでした")
            
            # ブートストラップ評価（オプション）
            if args.bootstrap_test:
                if not args.bootstrap_samples_path:
                    print_and_save("  ⚠️  警告: --bootstrap-testを使用する場合は--bootstrap-samples-pathを指定してください")
                else:
                    bootstrap_samples_path = Path(args.bootstrap_samples_path)
                    if not bootstrap_samples_path.exists():
                        print_and_save(f"  ⚠️  警告: ブートストラップサンプルファイルが見つかりません: {bootstrap_samples_path}")
                    else:
                        print_and_save("\n5.7. ブートストラップ評価を実行中（単体指標vs回帰モデル）...")
                        bootstrap_samples = load_bootstrap_samples(bootstrap_samples_path)
                        print_and_save(f"  - ブートストラップサンプルを読み込み: {len(bootstrap_samples)} 反復")
                        
                        # 回帰モデルのブートストラップ評価
                        regression_bootstrap_results = evaluate_bootstrap(
                            y_true=y_test.values,
                            y_pred_proba=y_test_proba,
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
                        bootstrap_results_path = csv_dir / "bootstrap_results_single_vs_regression.csv"
                        save_bootstrap_results(all_bootstrap_results, bootstrap_results_path)
                        print_and_save(f"  - ブートストラップ結果を保存: {bootstrap_results_path}")
                        
                        # サマリーを表示
                        print_and_save(f"\n  ブートストラップ結果のサマリー:")
                        for combination in ["regression", f"single_{best_metric}"]:
                            print_and_save(f"\n  {combination}:")
                            for metric in ['auc', 'ap', 'f1', 'accuracy']:
                                metric_values = all_bootstrap_results[
                                    (all_bootstrap_results['feature_combination'] == combination) &
                                    (all_bootstrap_results['metric_name'] == metric)
                                ]['value'].values
                                mean_val = np.mean(metric_values)
                                std_val = np.std(metric_values)
                                ci_lower = np.percentile(metric_values, 2.5)
                                ci_upper = np.percentile(metric_values, 97.5)
                                print_and_save(f"    {metric.upper()}: {mean_val:.4f} ± {std_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
            
            # 予測結果を保存（回帰モデル同士の比較用）
            predictions_path = csv_dir / "predictions_for_delong.csv"
            pd.DataFrame({
                'y_true': y_test.values,
                'y_pred_proba': y_test_proba
            }).to_csv(predictions_path, index=False)
            print_and_save(f"  - 回帰モデル同士の比較用に予測結果を保存: {predictions_path}")
        
    # クロスバリデーションの場合は、評価結果をスキップ（既にCV結果を表示済み）
    if not args.use_cv:
        print_and_save("\n=== 訓練データの評価 ===")
        print_and_save(f"Accuracy: {train_acc:.4f}")
        print_and_save(f"F1 Score: {train_f1:.4f}")
        print_and_save(f"AUC-ROC: {train_auc:.4f}")
        print_and_save(f"Average Precision: {train_ap:.4f}")
        print_and_save("\n分類レポート:")
        print_and_save(classification_report(y_train, y_train_pred, target_names=['not_clarification', 'clarification']))
        
        print_and_save("\n=== テストデータの評価 ===")
        print_and_save(f"Accuracy: {test_acc:.4f}")
        print_and_save(f"F1 Score: {test_f1:.4f}")
        print_and_save(f"AUC-ROC: {test_auc:.4f}")
        print_and_save(f"Average Precision: {test_ap:.4f}")
        print_and_save("\n分類レポート:")
        print_and_save(classification_report(y_test, y_test_pred, target_names=['not_clarification', 'clarification']))
    else:
        # クロスバリデーションの場合は、CV結果を表示（既に表示済みなのでスキップ）
        print_and_save("\n=== 評価結果 ===")
        print_and_save("（クロスバリデーション結果は既に上記で表示済み）")
        # 可視化用の変数を設定
        y_train_proba = y_train_proba_cv
        y_test_proba = y_test_proba_cv
        y_test = y_test_cv
        train_auc = train_auc_cv
        test_auc = test_auc_cv
        train_ap = train_ap_cv
        test_ap = test_ap_cv
    
    # 訓練データとテストデータの評価指標の比較（クロスバリデーションの場合はスキップ）
    if not args.use_cv:
        print_and_save("\n【訓練データとテストデータの評価指標の比較】")
        print_and_save(f"Accuracy差: {test_acc - train_acc:+.4f} (テスト {'高' if test_acc > train_acc else '低'})")
        print_and_save(f"F1 Score差: {test_f1 - train_f1:+.4f} (テスト {'高' if test_f1 > train_f1 else '低'})")
        print_and_save(f"AUC-ROC差: {test_auc - train_auc:+.4f} (テスト {'高' if test_auc > train_auc else '低'})")
        print_and_save(f"Average Precision差: {test_ap - train_ap:+.4f} (テスト {'高' if test_ap > train_ap else '低'})")
        
        if test_auc > train_auc or test_ap > train_ap:
            print_and_save("\n【注意】テストデータの評価指標が訓練データよりも高い場合、以下の可能性があります:")
        print_and_save("  1. 訓練データとテストデータの分布が異なる（データ分布のシフト）")
        print_and_save("  2. ラベル分布の違い（正例率の違い）")
        print_and_save("  3. モデルが単純で過学習が起きにくい（特徴量が少ない場合など）")
        print_and_save("  4. テストデータの方が「簡単」なケースが多い可能性")
        if args.use_separate_normalization:
            print_and_save("  5. 個別正規化を使用しているため、分布の違いは排除されていますが、")
            print_and_save("     それでもテストが高い場合は、分布以外の要因（データの質、ラベル分布など）が考えられます")
    
    # 6. ROC曲線の描画
    print_and_save("\n6. ROC曲線を描画中...")
    feature_names = list(X_train_norm.columns)
    
    # 特徴量名に基づいて出力ディレクトリとファイルパスを設定
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    # サブフォルダを作成
    graphs_dir = feature_output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = feature_output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    output_file = feature_output_dir / "results.txt"
    
    # baseスコアや複数特徴量がある場合は自動的に単一指標も表示
    has_base_scores = any('logit_clarification' in name for name in feature_names)
    has_multiple_features = len(feature_names) > 1
    show_single = args.show_single_metric_curves or has_base_scores or has_multiple_features
    
    # クロスバリデーションの場合は、モデルと単一指標の曲線にCV結果を使用
    if args.use_cv:
        # CV結果のみを表示（train/devは表示しない）
        plot_roc_curves(
            y_train=None if y_train_proba is None else y_train, 
            y_train_proba=y_train_proba, 
            y_test=y_test, 
            y_test_proba=y_test_proba, 
            train_auc=train_auc, 
            test_auc=test_auc, 
            output_dir=graphs_dir,
            hide_train=True,  # trainは非表示
            X_train=None,  # trainは表示しない
            X_test=X_cv_test_norm if show_single else None,  # CV結果のみ表示
            show_single_metrics=show_single,
            feature_names=feature_names,
            y_test_single=y_cv_test if show_single else None  # CV結果のラベル
        )
    else:
        plot_roc_curves(
            y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc, graphs_dir,
            hide_train=args.hide_train_curves,
            X_train=X_train_norm if show_single else None,
            X_test=X_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names
        )
    print_and_save("  - ROC曲線を保存しました")
    
    # 7. PR曲線の描画
    print_and_save("\n7. Precision-Recall曲線を描画中...")
    if args.use_cv:
        plot_pr_curves(
            y_train=None if y_train_proba is None else y_train,
            y_train_proba=y_train_proba,
            y_test=y_test,
            y_test_proba=y_test_proba,
            train_ap=train_ap,
            test_ap=test_ap,
            output_dir=graphs_dir,
            hide_train=True,
            X_train=None,
            X_test=X_cv_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
            y_test_single=y_cv_test if show_single else None
        )
    else:
        plot_pr_curves(
            y_train, y_train_proba, y_test, y_test_proba, train_ap, test_ap, graphs_dir,
            hide_train=args.hide_train_curves,
            X_train=X_train_norm if show_single else None,
            X_test=X_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names
        )
    print_and_save("  - PR曲線を保存しました")
    
    # 8. 閾値とF1スコアの関係の描画
    print_and_save("\n8. 閾値とF1スコアの関係を描画中...")
    if args.use_cv:
        plot_threshold_f1_curves(
            y_train=None if y_train_proba is None else y_train,
            y_train_proba=y_train_proba,
            y_test=y_test,
            y_test_proba=y_test_proba,
            output_dir=graphs_dir,
            hide_train=True,
            X_train=None,
            X_test=X_cv_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
            y_test_single=y_cv_test if show_single else None
        )
    else:
        plot_threshold_f1_curves(
            y_train, y_train_proba, y_test, y_test_proba, graphs_dir,
            hide_train=args.hide_train_curves,
            X_train=X_train_norm if show_single else None,
            X_test=X_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names
        )
    print_and_save("  - 閾値とF1スコアの関係を保存しました")
    
    # 9. 特徴量のスコア分布の描画
    print_and_save("\n9. 特徴量のスコア分布を描画中...")
    plot_feature_distributions(
        X_train_norm, X_test_norm, graphs_dir,
        hide_train=args.hide_train_curves,
        feature_names=feature_names,
        y_train=y_train,
        y_test=y_test,
        y_train_proba=y_train_proba,
        y_test_proba=y_test_proba
    )
    print_and_save("  - 特徴量のスコア分布を保存しました")
    
    # 10. 相関係数ヒートマップの描画
    print_and_save("\n10. 相関係数ヒートマップを描画中...")
    if args.use_cv:
        # CVの場合は統合データを使用
        plot_correlation_heatmaps(
            X_train=X_train_norm,
            X_test=X_test_norm,
            y_train=y_train,
            y_test=y_test,
            output_dir=graphs_dir,
            feature_names=feature_names,
            use_combined=True
        )
    else:
        plot_correlation_heatmaps(
            X_train=X_train_norm,
            X_test=X_test_norm,
            y_train=y_train,
            y_test=y_test,
            output_dir=graphs_dir,
            feature_names=feature_names,
            use_combined=False
        )
    print_and_save("  - 相関係数ヒートマップを保存しました")
    
    # 11. Confidence Analysis（確信度分析）
    print_and_save("\n11. Confidence Analysis（確信度分析）を実行中...")
    
    # BERTの予測確率を読み込む
    dev_base_probs = load_base_probabilities('dev', args.dataset, BASE_DIR, base_experiment_names=None, use_bert=use_bert, use_roberta=use_roberta, use_transfer=use_transfer)
    
    if len(dev_base_probs) > 0:
        # 最初のBERT確率を使用（複数ある場合は最初のもの）
        bert_prob_key = list(dev_base_probs.keys())[0]
        bert_probs = dev_base_probs[bert_prob_key]
        
        print_and_save(f"  - BERT予測確率を読み込み: {bert_prob_key} ({len(bert_probs)} サンプル)")
        
        # 複数の確率範囲で評価（可視化用）
        confidence_ranges = [
            (0.0, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
            (0.7, 1.0)
        ]
        
        # 0.5付近のクエリを抽出（詳細分析用、デフォルト: 0.4-0.6）
        confidence_threshold_low = 0.4
        confidence_threshold_high = 0.6
        
        # テストデータからconv_idとturn_idを取得
        dev_data = load_json_data(dataset_dir / "dev.json")
        test_keys = []
        for conversation in dev_data:
            for turn in conversation:
                conv_id = str(turn['conv_id'])
                turn_id = int(turn['turn_id'])
                test_keys.append((conv_id, turn_id))
        
        # 全体の評価結果を取得（比較用）
        if args.use_cv:
            overall_auc = cv_test_auc
            overall_ap = cv_test_ap
            overall_f1 = cv_test_f1
            total_samples = len(all_test_labels)
        else:
            overall_auc = test_auc
            overall_ap = test_ap
            overall_f1 = test_f1
            total_samples = len(y_test)
        
        # 単体指標の情報を取得
        has_single_metric_global = (args.delong_test and 
                                   len(single_metric_probas) > 0 and 
                                   best_metric is not None)
        if has_single_metric_global:
            overall_auc_single = best_metric_auc
            overall_auc_improvement = overall_auc - overall_auc_single
        else:
            overall_auc_improvement = None
        
        # 複数の確率範囲で評価（可視化用）
        range_results = []
        
        # CVの場合は統合データのキーを取得
        if args.use_cv:
            train_data = load_json_data(dataset_dir / "train.json")
            combined_keys = []
            for conversation in train_data:
                for turn in conversation:
                    conv_id = str(turn['conv_id'])
                    turn_id = int(turn['turn_id'])
                    combined_keys.append((conv_id, turn_id))
            for conversation in dev_data:
                for turn in conversation:
                    conv_id = str(turn['conv_id'])
                    turn_id = int(turn['turn_id'])
                    combined_keys.append((conv_id, turn_id))
        else:
            combined_keys = test_keys
        
        for range_low, range_high in confidence_ranges:
            range_indices = []
            range_labels = []
            range_proba_regression = []
            range_proba_single = []
            
            # 各範囲でサンプルを抽出
            for idx, key in enumerate(combined_keys):
                if key in bert_probs:
                    bert_prob = bert_probs[key]
                    if range_low <= bert_prob < range_high or (range_high == 1.0 and bert_prob == 1.0):
                        if args.use_cv:
                            if idx < len(all_test_labels):
                                range_indices.append(idx)
                                range_labels.append(all_test_labels[idx])
                                range_proba_regression.append(all_test_proba[idx])
                                if has_single_metric_global:
                                    best_metric_proba = single_metric_probas[best_metric]
                                    if idx < len(best_metric_proba):
                                        range_proba_single.append(best_metric_proba[idx])
                                    else:
                                        range_proba_single.append(None)
                                else:
                                    range_proba_single.append(None)
                        else:
                            if idx < len(y_test):
                                range_indices.append(idx)
                                range_labels.append(y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx])
                                range_proba_regression.append(y_test_proba[idx])
                                if has_single_metric_global:
                                    best_metric_proba = single_metric_probas[best_metric]
                                    if idx < len(best_metric_proba):
                                        range_proba_single.append(best_metric_proba[idx])
                                    else:
                                        range_proba_single.append(None)
                                else:
                                    range_proba_single.append(None)
            
            # 各範囲で評価
            if len(range_indices) > 0:
                range_labels = np.array(range_labels)
                range_proba_regression = np.array(range_proba_regression)
                range_auc_regression = roc_auc_score(range_labels, range_proba_regression)
                
                range_auc_single = None
                if has_single_metric_global and all(p is not None for p in range_proba_single):
                    range_proba_single = np.array(range_proba_single)
                    range_auc_single = roc_auc_score(range_labels, range_proba_single)
                
                range_results.append({
                    'auc_regression': range_auc_regression,
                    'auc_single': range_auc_single,
                    'n_samples': len(range_indices)
                })
            else:
                range_results.append({
                    'auc_regression': None,
                    'auc_single': None,
                    'n_samples': 0
                })
        
        # 可視化（単体指標がある場合のみ）
        if has_single_metric_global and overall_auc_improvement is not None:
            # 有効な範囲のみをフィルタリング
            valid_ranges = []
            valid_results = []
            for (r_low, r_high), result in zip(confidence_ranges, range_results):
                if result['n_samples'] > 0 and result['auc_regression'] is not None and result['auc_single'] is not None:
                    valid_ranges.append((r_low, r_high))
                    valid_results.append(result)
            
            if len(valid_ranges) > 0:
                # feature_namesを取得（可視化セクションより前の場合はX_train_normから取得）
                if args.use_cv:
                    vis_feature_names = list(X_combined.columns) if 'X_combined' in locals() else list(X_train_norm.columns) if 'X_train_norm' in locals() else None
                else:
                    vis_feature_names = list(X_train_norm.columns) if 'X_train_norm' in locals() else None
                
                plot_confidence_analysis(
                    valid_ranges,
                    valid_results,
                    overall_auc_improvement,
                    graphs_dir,
                    feature_names=vis_feature_names
                )
                print_and_save("  - Confidence Analysis可視化を保存しました")
        
        # 0.4-0.6の範囲で詳細分析（既存のコード）
        confidence_indices = []
        confidence_labels = []
        confidence_proba_regression = []
        confidence_proba_single = []
        
        for idx, key in enumerate(combined_keys):
            if key in bert_probs:
                bert_prob = bert_probs[key]
                if confidence_threshold_low <= bert_prob <= confidence_threshold_high:
                    if args.use_cv:
                        if idx < len(all_test_labels):
                            confidence_indices.append(idx)
                            confidence_labels.append(all_test_labels[idx])
                            confidence_proba_regression.append(all_test_proba[idx])
                            if has_single_metric_global:
                                best_metric_proba = single_metric_probas[best_metric]
                                if idx < len(best_metric_proba):
                                    confidence_proba_single.append(best_metric_proba[idx])
                                else:
                                    confidence_proba_single.append(None)
                            else:
                                confidence_proba_single.append(None)
                    else:
                        if idx < len(y_test):
                            confidence_indices.append(idx)
                            confidence_labels.append(y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx])
                            confidence_proba_regression.append(y_test_proba[idx])
                            if has_single_metric_global:
                                best_metric_proba = single_metric_probas[best_metric]
                                if idx < len(best_metric_proba):
                                    confidence_proba_single.append(best_metric_proba[idx])
                                else:
                                    confidence_proba_single.append(None)
                            else:
                                confidence_proba_single.append(None)
        
        if len(confidence_indices) > 0:
            confidence_labels = np.array(confidence_labels)
            confidence_proba_regression = np.array(confidence_proba_regression)
            
            total_samples = len(all_test_labels) if args.use_cv else len(y_test)
            print_and_save(f"\n  Confidence Analysis結果:")
            print_and_save(f"  - BERT確率範囲: [{confidence_threshold_low}, {confidence_threshold_high}]")
            print_and_save(f"  - 抽出サンプル数: {len(confidence_indices)} / {total_samples}")
            print_and_save(f"  - 抽出率: {len(confidence_indices) / total_samples * 100:.2f}%")
            
            # 回帰モデルの評価
            confidence_pred_regression = (confidence_proba_regression >= 0.5).astype(int)
            confidence_acc_regression = accuracy_score(confidence_labels, confidence_pred_regression)
            confidence_f1_regression = f1_score(confidence_labels, confidence_pred_regression)
            confidence_auc_regression = roc_auc_score(confidence_labels, confidence_proba_regression)
            confidence_ap_regression = average_precision_score(confidence_labels, confidence_proba_regression)
            
            print_and_save(f"\n  回帰モデル（QPP統合）の評価:")
            print_and_save(f"    Accuracy: {confidence_acc_regression:.4f}")
            print_and_save(f"    F1 Score: {confidence_f1_regression:.4f}")
            print_and_save(f"    AUC-ROC: {confidence_auc_regression:.4f}")
            print_and_save(f"    Average Precision: {confidence_ap_regression:.4f}")
            
            # 単体指標との比較（DeLong検定が実行されている場合のみ）
            has_single_metric = (args.delong_test and 
                               len(single_metric_probas) > 0 and 
                               best_metric is not None and
                               all(p is not None for p in confidence_proba_single))
            
            if has_single_metric:
                # 単体指標との比較が可能な場合
                confidence_proba_single = np.array(confidence_proba_single)
                confidence_pred_single = (confidence_proba_single >= 0.5).astype(int)
                confidence_acc_single = accuracy_score(confidence_labels, confidence_pred_single)
                confidence_f1_single = f1_score(confidence_labels, confidence_pred_single)
                confidence_auc_single = roc_auc_score(confidence_labels, confidence_proba_single)
                confidence_ap_single = average_precision_score(confidence_labels, confidence_proba_single)
                
                print_and_save(f"\n  単体指標（{format_feature_name(best_metric)}）の評価:")
                print_and_save(f"    Accuracy: {confidence_acc_single:.4f}")
                print_and_save(f"    F1 Score: {confidence_f1_single:.4f}")
                print_and_save(f"    AUC-ROC: {confidence_auc_single:.4f}")
                print_and_save(f"    Average Precision: {confidence_ap_single:.4f}")
                
                # 改善度を計算
                auc_improvement = confidence_auc_regression - confidence_auc_single
                ap_improvement = confidence_ap_regression - confidence_ap_single
                f1_improvement = confidence_f1_regression - confidence_f1_single
                
                print_and_save(f"\n  QPP統合による改善度:")
                print_and_save(f"    AUC-ROC改善: {auc_improvement:+.4f}")
                print_and_save(f"    Average Precision改善: {ap_improvement:+.4f}")
                print_and_save(f"    F1 Score改善: {f1_improvement:+.4f}")
                
                # 全体の評価結果と比較
                if args.use_cv:
                    overall_auc = cv_test_auc
                    overall_ap = cv_test_ap
                    overall_f1 = cv_test_f1
                else:
                    overall_auc = test_auc
                    overall_ap = test_ap
                    overall_f1 = test_f1
                
                overall_auc_single = best_metric_auc
                overall_auc_improvement = overall_auc - overall_auc_single
                
                print_and_save(f"\n  全体データでの改善度との比較:")
                print_and_save(f"    全体データ - AUC改善: {overall_auc_improvement:+.4f}")
                print_and_save(f"    Confidence範囲 - AUC改善: {auc_improvement:+.4f}")
                print_and_save(f"    改善度の差: {auc_improvement - overall_auc_improvement:+.4f}")
                
                if auc_improvement > overall_auc_improvement:
                    print_and_save(f"\n  ✓ QPP統合の効果は、BERTが迷った時（確率0.5付近）により大きくなっています！")
                    print_and_save(f"  → 「BERTが迷った時の最後の一押し（Tie-breaker）」としてQPPが有効であると主張できます。")
                else:
                    print_and_save(f"\n  → Confidence範囲での改善度は全体データと同程度です。")
                
                # 結果をCSVファイルに保存
                confidence_results_path = csv_dir / "confidence_analysis.csv"
                confidence_results_df = pd.DataFrame({
                    'metric': ['regression', 'single_metric'],
                    'accuracy': [confidence_acc_regression, confidence_acc_single],
                    'f1': [confidence_f1_regression, confidence_f1_single],
                    'auc': [confidence_auc_regression, confidence_auc_single],
                    'ap': [confidence_ap_regression, confidence_ap_single],
                    'confidence_range_low': [confidence_threshold_low, confidence_threshold_low],
                    'confidence_range_high': [confidence_threshold_high, confidence_threshold_high],
                    'n_samples': [len(confidence_indices), len(confidence_indices)]
                })
                confidence_results_df.to_csv(confidence_results_path, index=False)
                print_and_save(f"  - Confidence Analysis結果を保存: {confidence_results_path}")
            else:
                # 単体指標との比較ができない場合でも、回帰モデルの結果は保存
                confidence_results_path = csv_dir / "confidence_analysis.csv"
                confidence_results_df = pd.DataFrame({
                    'metric': ['regression'],
                    'accuracy': [confidence_acc_regression],
                    'f1': [confidence_f1_regression],
                    'auc': [confidence_auc_regression],
                    'ap': [confidence_ap_regression],
                    'confidence_range_low': [confidence_threshold_low],
                    'confidence_range_high': [confidence_threshold_high],
                    'n_samples': [len(confidence_indices)]
                })
                confidence_results_df.to_csv(confidence_results_path, index=False)
                print_and_save(f"  - Confidence Analysis結果を保存: {confidence_results_path}")
                
                if not args.delong_test:
                    print_and_save(f"\n  ※ 単体指標との比較を行うには、--delong-testオプションを指定してください")
                else:
                    print_and_save(f"\n  ⚠️  単体指標の予測確率が取得できませんでした")
        else:
            print_and_save(f"  ⚠️  警告: Confidence範囲内のサンプルが見つかりませんでした")
        
        # 12. Overconfidence Analysis（過信分析）
        print_and_save("\n12. Overconfidence Analysis（過信分析）を実行中...")
        
        if len(dev_base_probs) > 0 and has_single_metric_global:
            # 高確信度で間違えた事例を抽出
            # False Positive: BERT確率 > 0.8 で予測=1だが、実際のラベル=0
            # False Negative: BERT確率 < 0.2 で予測=0だが、実際のラベル=1
            
            fp_indices = []
            fp_labels = []
            fp_proba_regression = []
            fp_proba_single = []
            
            fn_indices = []
            fn_labels = []
            fn_proba_regression = []
            fn_proba_single = []
            
            for idx, key in enumerate(combined_keys):
                if key in bert_probs:
                    bert_prob = bert_probs[key]
                    
                    if args.use_cv:
                        if idx < len(all_test_labels):
                            true_label = all_test_labels[idx]
                            bert_pred = 1 if bert_prob >= 0.5 else 0
                            
                            # False Positive: 高確信度で1と予測したが、実際は0
                            if bert_prob >= 0.8 and bert_pred == 1 and true_label == 0:
                                fp_indices.append(idx)
                                fp_labels.append(true_label)
                                fp_proba_regression.append(all_test_proba[idx])
                                best_metric_proba = single_metric_probas[best_metric]
                                if idx < len(best_metric_proba):
                                    fp_proba_single.append(best_metric_proba[idx])
                                else:
                                    fp_proba_single.append(None)
                            
                            # False Negative: 高確信度で0と予測したが、実際は1
                            elif bert_prob <= 0.2 and bert_pred == 0 and true_label == 1:
                                fn_indices.append(idx)
                                fn_labels.append(true_label)
                                fn_proba_regression.append(all_test_proba[idx])
                                best_metric_proba = single_metric_probas[best_metric]
                                if idx < len(best_metric_proba):
                                    fn_proba_single.append(best_metric_proba[idx])
                                else:
                                    fn_proba_single.append(None)
                    else:
                        if idx < len(y_test):
                            true_label = y_test.iloc[idx] if hasattr(y_test, 'iloc') else y_test[idx]
                            bert_pred = 1 if bert_prob >= 0.5 else 0
                            
                            # False Positive
                            if bert_prob >= 0.8 and bert_pred == 1 and true_label == 0:
                                fp_indices.append(idx)
                                fp_labels.append(true_label)
                                fp_proba_regression.append(y_test_proba[idx])
                                best_metric_proba = single_metric_probas[best_metric]
                                if idx < len(best_metric_proba):
                                    fp_proba_single.append(best_metric_proba[idx])
                                else:
                                    fp_proba_single.append(None)
                            
                            # False Negative
                            elif bert_prob <= 0.2 and bert_pred == 0 and true_label == 1:
                                fn_indices.append(idx)
                                fn_labels.append(true_label)
                                fn_proba_regression.append(y_test_proba[idx])
                                best_metric_proba = single_metric_probas[best_metric]
                                if idx < len(best_metric_proba):
                                    fn_proba_single.append(best_metric_proba[idx])
                                else:
                                    fn_proba_single.append(None)
            
            overconfidence_results = {}
            
            # False Positiveの評価
            # FPは全てラベル=0なので、AUCは計算できない
            # 代わりに、予測確率の平均値と標準偏差を比較
            if len(fp_indices) > 0 and all(p is not None for p in fp_proba_single):
                fp_proba_regression = np.array(fp_proba_regression)
                fp_proba_single = np.array(fp_proba_single)
                
                # 理想的な予測確率は0に近い（ラベル=0なので）
                # QPP統合により予測確率が0に近づけば改善
                fp_mean_regression = np.mean(fp_proba_regression)
                fp_mean_single = np.mean(fp_proba_single)
                fp_std_regression = np.std(fp_proba_regression)
                fp_std_single = np.std(fp_proba_single)
                
                # 0からの距離を計算（小さいほど良い）
                fp_distance_regression = fp_mean_regression  # 0からの距離
                fp_distance_single = fp_mean_single
                improvement = fp_distance_single - fp_distance_regression  # 正の値なら改善
                
                overconfidence_results['high_confidence_fp'] = {
                    'mean_regression': fp_mean_regression,
                    'mean_single': fp_mean_single,
                    'std_regression': fp_std_regression,
                    'std_single': fp_std_single,
                    'improvement': improvement,
                    'n_samples': len(fp_indices)
                }
                
                print_and_save(f"\n  False Positive (高確信度で誤分類):")
                print_and_save(f"    - サンプル数: {len(fp_indices)}")
                print_and_save(f"    - 回帰モデル 平均予測確率: {fp_mean_regression:.4f} (std: {fp_std_regression:.4f})")
                print_and_save(f"    - 単体指標 平均予測確率: {fp_mean_single:.4f} (std: {fp_std_single:.4f})")
                print_and_save(f"    - 改善度（0からの距離の減少）: {improvement:+.4f}")
                if improvement > 0:
                    print_and_save(f"    → QPP統合により、誤分類の予測確率が0に近づいています（改善）")
                else:
                    print_and_save(f"    → QPP統合による改善は見られませんでした")
            
            # False Negativeの評価
            # FNは全てラベル=1なので、AUCは計算できない
            # 代わりに、予測確率の平均値と標準偏差を比較
            if len(fn_indices) > 0 and all(p is not None for p in fn_proba_single):
                fn_proba_regression = np.array(fn_proba_regression)
                fn_proba_single = np.array(fn_proba_single)
                
                # 理想的な予測確率は1に近い（ラベル=1なので）
                # QPP統合により予測確率が1に近づけば改善
                fn_mean_regression = np.mean(fn_proba_regression)
                fn_mean_single = np.mean(fn_proba_single)
                fn_std_regression = np.std(fn_proba_regression)
                fn_std_single = np.std(fn_proba_single)
                
                # 1からの距離を計算（小さいほど良い）
                fn_distance_regression = 1.0 - fn_mean_regression  # 1からの距離
                fn_distance_single = 1.0 - fn_mean_single
                improvement = fn_distance_single - fn_distance_regression  # 正の値なら改善
                
                overconfidence_results['high_confidence_fn'] = {
                    'mean_regression': fn_mean_regression,
                    'mean_single': fn_mean_single,
                    'std_regression': fn_std_regression,
                    'std_single': fn_std_single,
                    'improvement': improvement,
                    'n_samples': len(fn_indices)
                }
                
                print_and_save(f"\n  False Negative (高確信度で誤分類):")
                print_and_save(f"    - サンプル数: {len(fn_indices)}")
                print_and_save(f"    - 回帰モデル 平均予測確率: {fn_mean_regression:.4f} (std: {fn_std_regression:.4f})")
                print_and_save(f"    - 単体指標 平均予測確率: {fn_mean_single:.4f} (std: {fn_std_single:.4f})")
                print_and_save(f"    - 改善度（1からの距離の減少）: {improvement:+.4f}")
                if improvement > 0:
                    print_and_save(f"    → QPP統合により、誤分類の予測確率が1に近づいています（改善）")
                else:
                    print_and_save(f"    → QPP統合による改善は見られませんでした")
            
            # 可視化
            if len(overconfidence_results) > 0:
                overconfidence_results['overall_auc_improvement'] = overall_auc_improvement
                
                if args.use_cv:
                    vis_feature_names = list(X_combined.columns) if 'X_combined' in locals() else list(X_train_norm.columns) if 'X_train_norm' in locals() else None
                else:
                    vis_feature_names = list(X_train_norm.columns) if 'X_train_norm' in locals() else None
                
                plot_overconfidence_analysis(
                    overconfidence_results,
                    graphs_dir,
                    feature_names=vis_feature_names
                )
                print_and_save("  - Overconfidence Analysis可視化を保存しました")
                
                # 結果をCSVに保存
                overconfidence_csv_path = csv_dir / "overconfidence_analysis.csv"
                overconfidence_data = []
                if 'high_confidence_fp' in overconfidence_results:
                    fp_data = overconfidence_results['high_confidence_fp']
                    overconfidence_data.append({
                        'error_type': 'False Positive',
                        'mean_regression': fp_data.get('mean_regression'),
                        'mean_single': fp_data.get('mean_single'),
                        'std_regression': fp_data.get('std_regression'),
                        'std_single': fp_data.get('std_single'),
                        'improvement': fp_data.get('improvement'),
                        'n_samples': fp_data.get('n_samples', 0)
                    })
                if 'high_confidence_fn' in overconfidence_results:
                    fn_data = overconfidence_results['high_confidence_fn']
                    overconfidence_data.append({
                        'error_type': 'False Negative',
                        'mean_regression': fn_data.get('mean_regression'),
                        'mean_single': fn_data.get('mean_single'),
                        'std_regression': fn_data.get('std_regression'),
                        'std_single': fn_data.get('std_single'),
                        'improvement': fn_data.get('improvement'),
                        'n_samples': fn_data.get('n_samples', 0)
                    })
                
                if len(overconfidence_data) > 0:
                    overconfidence_df = pd.DataFrame(overconfidence_data)
                    overconfidence_df.to_csv(overconfidence_csv_path, index=False)
                    print_and_save(f"  - Overconfidence Analysis結果を保存: {overconfidence_csv_path}")
            else:
                print_and_save(f"  ⚠️  高確信度で誤分類した事例が見つかりませんでした")
        else:
            if not has_single_metric_global:
                print_and_save(f"  ⚠️  単体指標との比較ができないため、Overconfidence Analysisをスキップします（--delong-testを指定してください）")
            else:
                print_and_save(f"  ⚠️  BERTの予測確率を読み込めませんでした")
    else:
        print_and_save(f"  ⚠️  警告: BERTの予測確率を読み込めませんでした（BASE_EXPERIMENT_NAMESが設定されていない可能性があります）")
    
    # 結果をファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
    print_and_save(f"\n結果をファイルに保存しました: {output_file}")
    
    if args.use_separate_normalization:
        print_and_save("\n" + "="*80)
        print_and_save("⚠️  最終注意: この結果は個別正規化を使用しています！")
        print_and_save("⚠️  実際の予測タスクでは使用できません。")
        print_and_save("="*80)
    elif args.use_combined_normalization:
        print_and_save("\n" + "="*80)
        print_and_save("⚠️  最終警告: この結果はリーク前提の正規化を使用しています！")
        print_and_save("⚠️  実際の予測タスクでは使用できません。")
        print_and_save("="*80)
    
    # 全てのモデルを実行して比較（model_typeが"all"の場合）
    if run_all_models:
        print_and_save("\n" + "="*80)
        print_and_save("全てのモデルタイプを実行して比較します")
        print_and_save("="*80)
        
        models_results = []
        feature_names = list(X_train.columns) if 'X_train' in locals() else list(X_combined.columns) if 'X_combined' in locals() else []
        
        # 正規化済みデータを準備（既に正規化されている場合はそのまま使用）
        if args.use_cv:
            # クロスバリデーションの場合は統合データを使用
            X_data = X_combined
            y_data = y_combined
            # 正規化
            X_data_norm, _ = normalize_features(
                X_data, X_data,
                use_combined_normalization=False,
                use_separate_normalization=False,
                use_minmax_normalization=args.use_minmax_normalization
            )
            # クロスバリデーションで評価
            skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            all_test_proba_all_models = {model_type: [] for model_type in model_types_to_run}
            all_test_labels_all_models = []
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_data_norm, y_data)):
                X_fold_train = X_data_norm.iloc[train_idx].reset_index(drop=True)
                y_fold_train = y_data.iloc[train_idx].reset_index(drop=True)
                X_fold_test = X_data_norm.iloc[test_idx].reset_index(drop=True)
                y_fold_test = y_data.iloc[test_idx].reset_index(drop=True)
                
                for model_type in model_types_to_run:
                    model = create_model(
                        model_type=model_type,
                        max_iter=args.max_iter,
                        random_state=42 + fold_idx,
                        class_weight='balanced',
                        tol=args.tol,
                        non_negative=False,
                        n_bootstrap=args.n_bootstrap,
                        selection_threshold=args.selection_threshold,
                        n_random_traps=args.n_random_traps,
                        cv=args.lars_cv_folds
                    )
                    
                    if model_type in ['bolasso', 'lars_traps', 'lars_cv']:
                        model.fit(X_fold_train, y_fold_train, print_and_save_func=lambda x: None)
                    else:
                        model.fit(X_fold_train, y_fold_train)
                    
                    y_fold_test_proba = model.predict_proba(X_fold_test)[:, 1]
                    all_test_proba_all_models[model_type].extend(y_fold_test_proba)
                
                all_test_labels_all_models.extend(y_fold_test.tolist())
            
            # 各モデルの結果を収集
            for model_type in model_types_to_run:
                y_test_proba = np.array(all_test_proba_all_models[model_type])
                y_test_labels = pd.Series(all_test_labels_all_models)
                
                test_auc = roc_auc_score(y_test_labels, y_test_proba)
                test_ap = average_precision_score(y_test_labels, y_test_proba)
                test_pred = (y_test_proba >= 0.5).astype(int)
                test_acc = accuracy_score(y_test_labels, test_pred)
                test_f1 = f1_score(y_test_labels, test_pred)
                
                model_type_name = {
                    'l1': 'L1',
                    'l2': 'L2',
                    'elasticnet': 'ElasticNet',
                    'none': 'No Penalty',
                    'bolasso': 'BOLASSO',
                    'lars_traps': 'LARS-Traps',
                    'lars_cv': 'LARS-CV',
                    'randomforest': 'RandomForest'
                }.get(model_type, model_type)
                
                models_results.append({
                    'model_name': model_type_name,
                    'y_test_proba': y_test_proba,
                    'test_auc': test_auc,
                    'test_ap': test_ap,
                    'test_acc': test_acc,
                    'test_f1': test_f1
                })
                
                print_and_save(f"\n{model_type_name}:")
                print_and_save(f"  Accuracy: {test_acc:.4f}")
                print_and_save(f"  F1 Score: {test_f1:.4f}")
                print_and_save(f"  AUC-ROC: {test_auc:.4f}")
                print_and_save(f"  Average Precision: {test_ap:.4f}")
                
                # DeLong検定を実行（--delong-testが指定されている場合）
                if args.delong_test:
                    print_and_save(f"\n  --- {model_type_name} モデルのDeLong検定を実行中 ---")
                    
                    # 各単体指標についてロジスティック回帰を実行し、ROCを計算
                    single_metric_aucs = {}
                    single_metric_probas = {}
                    
                    # CV統合データに対して各foldで単体指標の評価を行う
                    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
                    for metric_name in X_combined.columns:
                        metric_aucs = []
                        metric_probas = []
                        
                        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_combined, y_combined)):
                            X_fold_train = X_combined.iloc[train_idx].reset_index(drop=True)
                            y_fold_train = y_combined.iloc[train_idx].reset_index(drop=True)
                            X_fold_test = X_combined.iloc[test_idx].reset_index(drop=True)
                            y_fold_test = y_combined.iloc[test_idx].reset_index(drop=True)
                            
                            # 正規化
                            X_fold_train_norm, X_fold_test_norm = normalize_features(
                                X_fold_train, X_fold_test,
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
                            single_metric_model.fit(X_fold_train_norm[[metric_name]], y_fold_train)
                            y_fold_test_proba = single_metric_model.predict_proba(X_fold_test_norm[[metric_name]])[:, 1]
                            
                            try:
                                fold_auc = roc_auc_score(y_fold_test, y_fold_test_proba)
                                metric_aucs.append(fold_auc)
                                metric_probas.append(y_fold_test_proba)
                            except ValueError:
                                pass
                        
                        if len(metric_aucs) > 0:
                            # 全foldの予測確率を統合
                            single_metric_probas[metric_name] = np.concatenate(metric_probas)
                            # 統合された予測確率からAUC-ROCとAUC-PRを計算（out-of-fold統合）
                            single_metric_aucs[metric_name] = roc_auc_score(y_test_labels, single_metric_probas[metric_name])
                            single_metric_aps[metric_name] = average_precision_score(y_test_labels, single_metric_probas[metric_name])
                    
                    if len(single_metric_aucs) > 0:
                        # 最高ROCを出した指標を見つける
                        best_metric = max(single_metric_aucs, key=single_metric_aucs.get)
                        best_metric_auc = single_metric_aucs[best_metric]
                        best_metric_proba = single_metric_probas[best_metric]
                        
                        print_and_save(f"  単体指標の最高ROC: {best_metric} (AUC = {best_metric_auc:.4f})")
                        print_and_save(f"  回帰モデルのROC: AUC = {test_auc:.4f}")
                        
                        # DeLongの検定
                        delong_result = delong_test(
                            y_true=y_test_labels.values,
                            y_pred_proba_a=y_test_proba,
                            y_pred_proba_b=best_metric_proba
                        )
                        
                        print_and_save(f"\n  DeLongの検定結果:")
                        print_and_save(f"    モデルA ({model_type_name}): AUC = {delong_result['auc_a']:.4f}")
                        print_and_save(f"    モデルB (単体指標 {best_metric}): AUC = {delong_result['auc_b']:.4f}")
                        print_and_save(f"    AUC差: {delong_result['auc_diff']:.4f}")
                        print_and_save(f"    Z統計量: {delong_result['z_stat']:.4f}")
                        print_and_save(f"    p値: {delong_result['p_value']:.4f}")
                        print_and_save(f"    有意差: {'あり' if delong_result['significant'] else 'なし'} (α=0.05)")
                        
                        # 各指標と回帰モデルとの比較表を作成
                        print_and_save("\n  === 各指標とモデルの比較表 ===")
                        comparison_data = []
                        
                        # 各指標のAUC-ROCとAUC-PRを計算
                        for metric_name in single_metric_aucs.keys():
                            metric_proba = single_metric_probas[metric_name]
                            metric_auc = single_metric_aucs[metric_name]
                            metric_ap = average_precision_score(y_test_labels, metric_proba)
                            
                            comparison_data.append({
                                'Metric': format_feature_name(metric_name),
                                'AUC-ROC': f"{metric_auc:.4f}",
                                'AUC-PR': f"{metric_ap:.4f}",
                                'p-value': '-'
                            })
                        
                        # 回帰モデルの情報を追加（最高性能の単体指標とのDeLong検定のp-value）
                        regression_ap = average_precision_score(y_test_labels, y_test_proba)
                        
                        # 最高性能の単体指標とのDeLong検定
                        # p値の表示：0.0000になる場合は科学記法で表示
                        if np.isnan(delong_result['p_value']):
                            p_value_str = '-'
                        elif delong_result['p_value'] < 0.0001:
                            p_value_str = f"{delong_result['p_value']:.2e}"
                        else:
                            p_value_str = f"{delong_result['p_value']:.4f}"
                        
                        comparison_data.append({
                            'Metric': model_type_name,
                            'AUC-ROC': f"{test_auc:.4f}",
                            'AUC-PR': f"{regression_ap:.4f}",
                            'p-value': p_value_str
                        })
                        
                        # 表を表示
                        comparison_df = pd.DataFrame(comparison_data)
                        # タブ区切り形式で出力（スプレッドシートにコピペしやすい）
                        print_and_save("\n" + comparison_df.to_csv(sep='\t', index=False))
                        
                        # 結果を保存（delongディレクトリに）
                        feature_dir_name = get_feature_dir_name(feature_names)
                        delong_dir = output_dir / feature_dir_name / "delong"
                        delong_dir.mkdir(parents=True, exist_ok=True)
                        delong_result_path = delong_dir / f"delong_test_{model_type}.csv"
                        result_df = pd.DataFrame([{
                            'best_single_metric': best_metric,
                            'best_single_metric_auc': best_metric_auc,
                            'regression_auc': test_auc,
                            **delong_result
                        }])
                        result_df.to_csv(delong_result_path, index=False)
                        print_and_save(f"  - DeLongの検定結果を保存: {delong_result_path}")
                        
                        # 比較表も保存
                        comparison_csv_path = delong_dir / f"comparison_table_{model_type}.csv"
                        comparison_df.to_csv(comparison_csv_path, index=False)
                        print_and_save(f"  - 比較表を保存: {comparison_csv_path}")
                    else:
                        print_and_save("  ⚠️  警告: 単体指標のROCを計算できませんでした")
        else:
            # 通常学習の場合
            X_train_norm, X_test_norm = normalize_features(
                X_train, X_test,
                use_combined_normalization=args.use_combined_normalization,
                use_separate_normalization=args.use_separate_normalization,
                use_minmax_normalization=args.use_minmax_normalization
            )
            
            for model_type in model_types_to_run:
                print_and_save(f"\n--- {model_type} モデルを実行中 ---")
                model = create_model(
                    model_type=model_type,
                    max_iter=args.max_iter,
                    random_state=42,
                    class_weight='balanced',
                    tol=args.tol,
                    non_negative=False,
                    n_bootstrap=args.n_bootstrap,
                    selection_threshold=args.selection_threshold,
                    n_random_traps=args.n_random_traps,
                    cv=args.lars_cv_folds
                )
                
                if model_type in ['bolasso', 'lars_traps', 'lars_cv']:
                    model.fit(X_train_norm, y_train, print_and_save_func=print_and_save)
                else:
                    model.fit(X_train_norm, y_train)
                
                y_test_proba = model.predict_proba(X_test_norm)[:, 1]
                y_test_pred = model.predict(X_test_norm)
                
                test_auc = roc_auc_score(y_test, y_test_proba)
                test_ap = average_precision_score(y_test, y_test_proba)
                test_acc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred)
                
                model_type_name = {
                    'l1': 'L1',
                    'l2': 'L2',
                    'elasticnet': 'ElasticNet',
                    'none': 'No Penalty',
                    'bolasso': 'BOLASSO',
                    'lars_traps': 'LARS-Traps',
                    'lars_cv': 'LARS-CV',
                    'randomforest': 'RandomForest'
                }.get(model_type, model_type)
                
                models_results.append({
                    'model_name': model_type_name,
                    'y_test_proba': y_test_proba,
                    'test_auc': test_auc,
                    'test_ap': test_ap,
                    'test_acc': test_acc,
                    'test_f1': test_f1
                })
                
                print_and_save(f"  Accuracy: {test_acc:.4f}")
                print_and_save(f"  F1 Score: {test_f1:.4f}")
                print_and_save(f"  AUC-ROC: {test_auc:.4f}")
                print_and_save(f"  Average Precision: {test_ap:.4f}")
                
                # DeLong検定を実行（--delong-testが指定されている場合）
                if args.delong_test:
                    print_and_save(f"\n  --- {model_type_name} モデルのDeLong検定を実行中 ---")
                    
                    # 各単体指標についてロジスティック回帰を実行し、ROCを計算
                    single_metric_aucs = {}
                    single_metric_aps = {}
                    single_metric_probas = {}
                    
                    for metric_name in X_train_norm.columns:
                        # 正規化
                        X_train_metric = X_train_norm[[metric_name]]
                        X_test_metric = X_test_norm[[metric_name]]
                        
                        # 単一指標でロジスティック回帰
                        single_metric_model = LogisticRegression(
                            penalty='l2',
                            solver='liblinear',
                            C=1.0,
                            max_iter=1000,
                            random_state=42,
                            class_weight='balanced'
                        )
                        single_metric_model.fit(X_train_metric, y_train)
                        y_test_proba_single = single_metric_model.predict_proba(X_test_metric)[:, 1]
                        
                        try:
                            test_auc_single = roc_auc_score(y_test, y_test_proba_single)
                            test_ap_single = average_precision_score(y_test, y_test_proba_single)
                            single_metric_aucs[metric_name] = test_auc_single
                            single_metric_aps[metric_name] = test_ap_single
                            single_metric_probas[metric_name] = y_test_proba_single
                        except ValueError:
                            pass
                    
                    if len(single_metric_aucs) > 0:
                        # 最高ROCを出した指標を見つける
                        best_metric = max(single_metric_aucs, key=single_metric_aucs.get)
                        best_metric_auc = single_metric_aucs[best_metric]
                        best_metric_proba = single_metric_probas[best_metric]
                        
                        print_and_save(f"  単体指標の最高ROC: {best_metric} (AUC = {best_metric_auc:.4f})")
                        print_and_save(f"  回帰モデルのROC: AUC = {test_auc:.4f}")
                        
                        # DeLongの検定
                        delong_result = delong_test(
                            y_true=y_test.values,
                            y_pred_proba_a=y_test_proba,
                            y_pred_proba_b=best_metric_proba
                        )
                        
                        print_and_save(f"\n  DeLongの検定結果:")
                        print_and_save(f"    モデルA ({model_type_name}): AUC = {delong_result['auc_a']:.4f}")
                        print_and_save(f"    モデルB (単体指標 {best_metric}): AUC = {delong_result['auc_b']:.4f}")
                        print_and_save(f"    AUC差: {delong_result['auc_diff']:.4f}")
                        print_and_save(f"    Z統計量: {delong_result['z_stat']:.4f}")
                        print_and_save(f"    p値: {delong_result['p_value']:.4f}")
                        print_and_save(f"    有意差: {'あり' if delong_result['significant'] else 'なし'} (α=0.05)")
                        
                        # 各指標と回帰モデルとの比較表を作成
                        print_and_save("\n  === 各指標とモデルの比較表 ===")
                        comparison_data = []
                        
                        # 各指標のAUC-ROCとAUC-PRを計算
                        for metric_name in single_metric_aucs.keys():
                            metric_proba = single_metric_probas[metric_name]
                            metric_auc = single_metric_aucs[metric_name]
                            metric_ap = single_metric_aps[metric_name]
                            
                            comparison_data.append({
                                'Metric': format_feature_name(metric_name),
                                'AUC-ROC': f"{metric_auc:.4f}",
                                'AUC-PR': f"{metric_ap:.4f}",
                                'p-value': '-'
                            })
                        
                        # 回帰モデルの情報を追加（最高性能の単体指標とのDeLong検定のp-value）
                        regression_ap = average_precision_score(y_test, y_test_proba)
                        
                        # 最高性能の単体指標とのDeLong検定
                        # p値の表示：0.0000になる場合は科学記法で表示
                        if np.isnan(delong_result['p_value']):
                            p_value_str = '-'
                        elif delong_result['p_value'] < 0.0001:
                            p_value_str = f"{delong_result['p_value']:.2e}"
                        else:
                            p_value_str = f"{delong_result['p_value']:.4f}"
                        
                        comparison_data.append({
                            'Metric': model_type_name,
                            'AUC-ROC': f"{test_auc:.4f}",
                            'AUC-PR': f"{regression_ap:.4f}",
                            'p-value': p_value_str
                        })
                        
                        # 表を表示
                        comparison_df = pd.DataFrame(comparison_data)
                        # タブ区切り形式で出力（スプレッドシートにコピペしやすい）
                        print_and_save("\n" + comparison_df.to_csv(sep='\t', index=False))
                        
                        # 結果を保存（delongディレクトリに）
                        feature_dir_name = get_feature_dir_name(feature_names)
                        delong_dir = output_dir / feature_dir_name / "delong"
                        delong_dir.mkdir(parents=True, exist_ok=True)
                        delong_result_path = delong_dir / f"delong_test_{model_type}.csv"
                        result_df = pd.DataFrame([{
                            'best_single_metric': best_metric,
                            'best_single_metric_auc': best_metric_auc,
                            'regression_auc': test_auc,
                            **delong_result
                        }])
                        result_df.to_csv(delong_result_path, index=False)
                        print_and_save(f"  - DeLongの検定結果を保存: {delong_result_path}")
                        
                        # 比較表も保存
                        comparison_csv_path = delong_dir / f"comparison_table_{model_type}.csv"
                        comparison_df.to_csv(comparison_csv_path, index=False)
                        print_and_save(f"  - 比較表を保存: {comparison_csv_path}")
                    else:
                        print_and_save("  ⚠️  警告: 単体指標のROCを計算できませんでした")
        
        # 全モデル統合比較表を作成（--delong-testが指定されている場合）
        if args.delong_test and len(models_results) > 0:
            print_and_save("\n" + "="*80)
            print_and_save("全モデル統合比較表を作成中...")
            print_and_save("="*80)
            
            # 単体指標の結果を収集（最初のモデルで計算したものを使用）
            single_metric_aucs_all = {}
            single_metric_aps_all = {}
            single_metric_probas_all = {}
            best_single_metric_auc = 0.0
            best_single_metric_name = None
            best_single_metric_proba = None
            
            # 単体指標の評価を実行（1回だけ）
            # 非学習指標（QPPスコア）は直接使用、学習指標（BERT等）はロジスティック回帰を使用
            from config import POST_RETRIEVAL_CONFIGS, PRE_RETRIEVAL_CONFIGS
            from module.data_loader import NSP_METRICS, get_nsp_metric_name
            
            # 非学習指標のリスト（QPPスコア）
            non_learning_metrics = set(POST_RETRIEVAL_CONFIGS.keys()) | set(PRE_RETRIEVAL_CONFIGS.keys())
            # NSPメトリクスも非学習指標として扱う（動的に生成されるため、名前で判定）
            nsp_metric_prefixes = [f"nsp_{metric}_topk" for metric in NSP_METRICS.keys()]
            
            skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
            for metric_name in X_combined.columns:
                # 非学習指標かどうかを判定
                is_non_learning = (
                    metric_name in non_learning_metrics or
                    any(metric_name.startswith(prefix) for prefix in nsp_metric_prefixes)
                )
                
                if is_non_learning:
                    # 非学習指標：QPPスコアを直接使用（全foldのテストデータに対応するスコアを使用）
                    metric_scores_list = []
                    metric_labels_list = []
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_combined, y_combined)):
                        X_fold_test = X_combined.iloc[test_idx].reset_index(drop=True)
                        y_fold_test = y_combined.iloc[test_idx].reset_index(drop=True)
                        metric_scores_list.append(X_fold_test[metric_name].values)
                        metric_labels_list.append(y_fold_test.values)
                    
                    # 全foldのスコアとラベルを統合
                    metric_scores_all = np.concatenate(metric_scores_list)
                    metric_labels_all = np.concatenate(metric_labels_list)
                    
                    # Min-Max正規化で[0,1]に変換（予測確率として使用）
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    metric_scores_normalized = scaler.fit_transform(metric_scores_all.reshape(-1, 1)).flatten()
                    
                    try:
                        single_metric_aucs_all[metric_name] = roc_auc_score(metric_labels_all, metric_scores_normalized)
                        single_metric_aps_all[metric_name] = average_precision_score(metric_labels_all, metric_scores_normalized)
                        single_metric_probas_all[metric_name] = metric_scores_normalized
                        
                        # 最高性能の単体指標を記録
                        if single_metric_aucs_all[metric_name] > best_single_metric_auc:
                            best_single_metric_auc = single_metric_aucs_all[metric_name]
                            best_single_metric_name = metric_name
                            best_single_metric_proba = single_metric_probas_all[metric_name]
                    except ValueError:
                        pass
                else:
                    # 学習指標（BERT等）：ロジスティック回帰を使用
                    metric_aucs = []
                    metric_probas = []
                    
                    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_combined, y_combined)):
                        X_fold_train = X_combined.iloc[train_idx].reset_index(drop=True)
                        y_fold_train = y_combined.iloc[train_idx].reset_index(drop=True)
                        X_fold_test = X_combined.iloc[test_idx].reset_index(drop=True)
                        y_fold_test = y_combined.iloc[test_idx].reset_index(drop=True)
                        
                        # 正規化
                        X_fold_train_norm, X_fold_test_norm = normalize_features(
                            X_fold_train, X_fold_test,
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
                        single_metric_model.fit(X_fold_train_norm[[metric_name]], y_fold_train)
                        y_fold_test_proba = single_metric_model.predict_proba(X_fold_test_norm[[metric_name]])[:, 1]
                        
                        try:
                            fold_auc = roc_auc_score(y_fold_test, y_fold_test_proba)
                            metric_aucs.append(fold_auc)
                            metric_probas.append(y_fold_test_proba)
                        except ValueError:
                            pass
                    
                    if len(metric_aucs) > 0:
                        # 全foldの予測確率を統合
                        single_metric_probas_all[metric_name] = np.concatenate(metric_probas)
                        # 統合された予測確率からAUC-ROCとAUC-PRを計算（out-of-fold統合）
                        single_metric_aucs_all[metric_name] = roc_auc_score(
                            pd.Series(all_test_labels_all_models), 
                            single_metric_probas_all[metric_name]
                        )
                        single_metric_aps_all[metric_name] = average_precision_score(
                            pd.Series(all_test_labels_all_models), 
                            single_metric_probas_all[metric_name]
                        )
                    
                    # 最高性能の単体指標を記録
                    if metric_name in single_metric_aucs_all and single_metric_aucs_all[metric_name] > best_single_metric_auc:
                        best_single_metric_auc = single_metric_aucs_all[metric_name]
                        best_single_metric_name = metric_name
                        if metric_name in single_metric_probas_all:
                            best_single_metric_proba = single_metric_probas_all[metric_name]
            
            # 最高性能の単体指標を確認（デバッグ用）
            if best_single_metric_proba is None:
                print_and_save(f"  ⚠️  警告: 最高性能の単体指標の予測確率が設定されていません（best_single_metric_name: {best_single_metric_name}）")
                if len(single_metric_probas_all) > 0:
                    # 代替として、最初の単体指標を使用
                    first_metric = list(single_metric_probas_all.keys())[0]
                    best_single_metric_proba = single_metric_probas_all[first_metric]
                    best_single_metric_name = first_metric
                    print_and_save(f"  → 代替として、最初の単体指標 '{first_metric}' を使用します")
            
            # 統合比較表を作成
            comparison_data_all = []
            
            # 単体指標を追加
            for metric_name in sorted(single_metric_aucs_all.keys()):
                comparison_data_all.append({
                    'Metric': format_feature_name(metric_name),
                    'AUC-ROC': f"{single_metric_aucs_all[metric_name]:.4f}",
                    'AUC-PR': f"{single_metric_aps_all[metric_name]:.4f}",
                    'p-value': '-'
                })
            
            # モデル名のマッピング（実行順序も定義）
            model_name_mapping = {
                'l1': 'L1 Logistic Regression',
                'l2': 'L2 Logistic Regression',
                'elasticnet': 'E-Net',
                'none': 'No Penalty Logistic Regression',
                'bolasso': 'BOLASSO',
                'lars_traps': 'LARS-Traps',
                'lars_cv': 'LARS-CV',
                'randomforest': 'Random Forest'
            }
            
            # models_resultsのmodel_nameからmodel_display_nameへの逆マッピング
            model_name_to_display = {
                'L1': 'L1 Logistic Regression',
                'L2': 'L2 Logistic Regression',
                'ElasticNet': 'E-Net',
                'No Penalty': 'No Penalty Logistic Regression',
                'BOLASSO': 'BOLASSO',
                'LARS-Traps': 'LARS-Traps',
                'LARS-CV': 'LARS-CV',
                'RandomForest': 'Random Forest'
            }
            
            # 実行されたモデルの結果を辞書に格納（model_display_nameをキーとして使用）
            executed_models = {}
            for result in models_results:
                model_name = result['model_name']
                # model_nameをmodel_display_nameに変換
                model_display_name = model_name_to_display.get(model_name, model_name)
                executed_models[model_display_name] = {
                    'auc': result['test_auc'],
                    'ap': result['test_ap'],
                    'proba': result['y_test_proba']
                }
            
            # 全てのモデルタイプを順番に追加（実行されていないものは空欄）
            for model_type_key, model_display_name in model_name_mapping.items():
                if model_display_name in executed_models:
                    # 実行されたモデル
                    model_data = executed_models[model_display_name]
                    model_auc = model_data['auc']
                    model_ap = model_data['ap']
                    model_proba = model_data['proba']
                    
                    # 最高性能の単体指標とのDeLong検定
                    if best_single_metric_proba is not None:
                        delong_result = delong_test(
                            y_true=pd.Series(all_test_labels_all_models),
                            y_pred_proba_a=model_proba,
                            y_pred_proba_b=best_single_metric_proba
                        )
                        # p値の表示：0.0000になる場合は科学記法で表示
                        if np.isnan(delong_result['p_value']):
                            p_value_str = '-'
                        elif delong_result['p_value'] < 0.0001:
                            p_value_str = f"{delong_result['p_value']:.2e}"
                        else:
                            p_value_str = f"{delong_result['p_value']:.4f}"
                    else:
                        p_value_str = '-'
                    
                    comparison_data_all.append({
                        'Metric': model_display_name,
                        'AUC-ROC': f"{model_auc:.4f}",
                        'AUC-PR': f"{model_ap:.4f}",
                        'p-value': p_value_str
                    })
                else:
                    # 実行されていないモデル（空欄）
                    comparison_data_all.append({
                        'Metric': model_display_name,
                        'AUC-ROC': '',
                        'AUC-PR': '',
                        'p-value': ''
                    })
            
            # 表を表示（スプレッドシートにコピペしやすいタブ区切り形式）
            comparison_df_all = pd.DataFrame(comparison_data_all)
            print_and_save("\n=== 全モデル統合比較表 ===")
            # タブ区切り形式で出力（スプレッドシートにコピペしやすい）
            print_and_save(comparison_df_all.to_csv(sep='\t', index=False))
            
            # 表を保存
            feature_dir_name = get_feature_dir_name(feature_names)
            delong_dir = output_dir / feature_dir_name / "delong"
            delong_dir.mkdir(parents=True, exist_ok=True)
            comparison_csv_path = delong_dir / "comprehensive_comparison_table.csv"
            comparison_df_all.to_csv(comparison_csv_path, index=False)
            print_and_save(f"  - 統合比較表（DeLong検定）を保存: {comparison_csv_path}")
            
            # ブートストラップ検定バージョンの表を作成（--bootstrap-testが指定されている場合）
            if args.bootstrap_test and args.bootstrap_samples_path and best_single_metric_proba is not None:
                bootstrap_samples_path = Path(args.bootstrap_samples_path)
                if bootstrap_samples_path.exists():
                    print_and_save("\n" + "="*80)
                    print_and_save("ブートストラップ検定による全モデル統合比較表を作成中...")
                    print_and_save("="*80)
                    
                    # ブートストラップサンプルを読み込み
                    bootstrap_samples = load_bootstrap_samples(bootstrap_samples_path)
                    print_and_save(f"  - ブートストラップサンプルを読み込み: {len(bootstrap_samples)} 反復")
                    
                    # 最高性能の単体指標のブートストラップ評価
                    best_single_metric_bootstrap_results = evaluate_bootstrap(
                        y_true=np.array(all_test_labels_all_models),
                        y_pred_proba=best_single_metric_proba,
                        bootstrap_samples=bootstrap_samples,
                        feature_combination_name=f"single_{best_single_metric_name}"
                    )
                    
                    # ブートストラップ検定バージョンの比較表を作成
                    comparison_data_bootstrap = []
                    
                    # 単体指標を追加
                    for metric_name in sorted(single_metric_aucs_all.keys()):
                        comparison_data_bootstrap.append({
                            'Metric': format_feature_name(metric_name),
                            'AUC-ROC': f"{single_metric_aucs_all[metric_name]:.4f}",
                            'AUC-PR': f"{single_metric_aps_all[metric_name]:.4f}",
                            'p-value': '-'
                        })
                    
                    # 各モデルについてブートストラップ検定を実行
                    for model_type_key, model_display_name in model_name_mapping.items():
                        if model_display_name in executed_models:
                            # 実行されたモデル
                            model_data = executed_models[model_display_name]
                            model_auc = model_data['auc']
                            model_ap = model_data['ap']
                            model_proba = model_data['proba']
                            
                            # モデルのブートストラップ評価
                            model_bootstrap_results = evaluate_bootstrap(
                                y_true=np.array(all_test_labels_all_models),
                                y_pred_proba=model_proba,
                                bootstrap_samples=bootstrap_samples,
                                feature_combination_name=f"model_{model_display_name.replace(' ', '_')}"
                            )
                            
                            # 最高性能の単体指標とのブートストラップ検定
                            bootstrap_comparison = compare_feature_combinations(
                                bootstrap_results=pd.concat([
                                    model_bootstrap_results,
                                    best_single_metric_bootstrap_results
                                ], ignore_index=True),
                                combination_a=f"model_{model_display_name.replace(' ', '_')}",
                                combination_b=f"single_{best_single_metric_name}",
                                metric_name='auc',
                                alpha=0.05
                            )
                            
                            # p値の表示：0.0000になる場合は科学記法で表示
                            if np.isnan(bootstrap_comparison['p_value']):
                                p_value_str = '-'
                            elif bootstrap_comparison['p_value'] < 0.0001:
                                p_value_str = f"{bootstrap_comparison['p_value']:.2e}"
                            else:
                                p_value_str = f"{bootstrap_comparison['p_value']:.4f}"
                            
                            comparison_data_bootstrap.append({
                                'Metric': model_display_name,
                                'AUC-ROC': f"{model_auc:.4f}",
                                'AUC-PR': f"{model_ap:.4f}",
                                'p-value': p_value_str
                            })
                        else:
                            # 実行されていないモデル（空欄）
                            comparison_data_bootstrap.append({
                                'Metric': model_display_name,
                                'AUC-ROC': '',
                                'AUC-PR': '',
                                'p-value': ''
                            })
                    
                    # 表を表示（スプレッドシートにコピペしやすいタブ区切り形式）
                    comparison_df_bootstrap = pd.DataFrame(comparison_data_bootstrap)
                    print_and_save("\n=== 全モデル統合比較表（ブートストラップ検定） ===")
                    # タブ区切り形式で出力（スプレッドシートにコピペしやすい）
                    print_and_save(comparison_df_bootstrap.to_csv(sep='\t', index=False))
                    
                    # 表を保存
                    comparison_bootstrap_csv_path = delong_dir / "comprehensive_comparison_table_bootstrap.csv"
                    comparison_df_bootstrap.to_csv(comparison_bootstrap_csv_path, index=False)
                    print_and_save(f"  - 統合比較表（ブートストラップ検定）を保存: {comparison_bootstrap_csv_path}")
                else:
                    print_and_save(f"  ⚠️  警告: ブートストラップサンプルファイルが見つかりません: {bootstrap_samples_path}")
            elif args.bootstrap_test:
                print_and_save("\n  ⚠️  警告: --bootstrap-testを使用する場合は--bootstrap-samples-pathを指定してください")
        
        # 比較グラフを描画
        if len(models_results) > 0:
            print_and_save("\n" + "="*80)
            print_and_save("モデル比較グラフを描画中...")
            print_and_save("="*80)
            
            y_test_for_comparison = y_test if not args.use_cv else pd.Series(all_test_labels_all_models)
            # 全モデル実行時のグラフ保存先を設定
            feature_dir_name = get_feature_dir_name(feature_names)
            feature_output_dir = output_dir / feature_dir_name
            graphs_dir = feature_output_dir / "graphs"
            graphs_dir.mkdir(parents=True, exist_ok=True)
            
            plot_multiple_models_comparison(
                models_results=models_results,
                y_test=y_test_for_comparison,
                output_dir=graphs_dir,
                feature_names=feature_names
            )
            print_and_save("  - モデル比較グラフを保存しました")
        
        # 全モデル実行時の処理の出力もresults.txtに含める
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_buffer.getvalue())
        print_and_save(f"\n結果をファイルに保存しました（全モデル実行結果を含む）: {output_file}")
    
    print_and_save("\n=== 完了 ===")


if __name__ == "__main__":
    main()

