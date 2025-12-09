"""
L1正則化付きロジスティック回帰によるclarification分類
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
from scipy.optimize import minimize
from scipy.special import expit

from module import (
    load_json_data,
    extract_labels,
    load_qpp_scores,
    load_base_scores,
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
    plot_correlation_heatmaps
)
from module.bootstrap import (
    delong_test
)


# パス設定（main関数内で動的に設定）
BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


class NonNegativeLogisticRegression:
    """非負制約付きL1正則化ロジスティック回帰"""
    
    def __init__(self, C=1.0, max_iter=1000, random_state=42, class_weight='balanced', tol=1e-7):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.n_iter_ = None
        
    def _logistic_loss(self, params, X, y, sample_weights):
        """ロジスティック損失関数（L1正則化付き）"""
        n_features = X.shape[1]
        w = params[:n_features]
        b = params[n_features]
        
        # 予測
        z = X @ w + b
        y_pred = expit(z)
        
        # ロジスティック損失
        loss = -np.sum(sample_weights * (y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15)))
        
        # L1正則化
        l1_penalty = (1.0 / self.C) * np.sum(np.abs(w))
        
        return loss + l1_penalty
    
    def fit(self, X, y):
        """モデルの学習"""
        np.random.seed(self.random_state)
        
        # クラス重みの計算
        if self.class_weight == 'balanced':
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weights = compute_sample_weight('balanced', y)
        else:
            sample_weights = np.ones(len(y))
        
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # 初期値（小さい正の値）
        initial_w = np.random.uniform(0.01, 0.1, n_features)
        initial_b = 0.0
        initial_params = np.concatenate([initial_w, [initial_b]])
        
        # 非負制約（係数のみ、切片は制約なし）
        bounds = [(0, None)] * n_features + [(None, None)]
        
        # 最適化
        result = minimize(
            self._logistic_loss,
            initial_params,
            args=(X.values, y.values, sample_weights),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iter,
                'ftol': self.tol,
                'gtol': self.tol,
                'disp': False
            }
        )
        
        # 収束状況を保存
        self.optimization_result_ = result
        self.coef_ = result.x[:n_features].reshape(1, -1)
        self.intercept_ = result.x[n_features]
        self.n_iter_ = [result.nit]
        
        return self
    
    def predict_proba(self, X):
        """予測確率を返す"""
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        z = X_array @ self.coef_[0] + self.intercept_
        proba_positive = expit(z)
        proba_negative = 1 - proba_positive
        return np.column_stack([proba_negative, proba_positive])
    
    def predict(self, X):
        """予測ラベルを返す"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def main():
    parser = argparse.ArgumentParser(description="L1正則化付きロジスティック回帰によるclarification分類")
    
    # データセット名（必須）
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
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
    
    args = parser.parse_args()
    
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
    
    # config.pyから使用する指標を確認
    import sys
    from pathlib import Path
    import importlib.util
    config_path = Path(__file__).parent / "config.py"
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        use_pre_retrieval = len(config.PRE_RETRIEVAL_CONFIGS) > 0
        use_base = len(config.BASE_EXPERIMENT_NAMES) > 0
    else:
        use_pre_retrieval = False
        use_base = False
    
    print_and_save("=== L1正則化付きロジスティック回帰によるclarification分類 ===\n")
    print_and_save(f"データセット: {args.dataset}")
    print_and_save(f"検索手法: {args.retrieval_method}")
    feature_desc = "QPPスコア（post"
    if use_pre_retrieval:
        feature_desc += " + pre"
    feature_desc += " + nsp"
    if use_base:
        feature_desc += " + base"
    feature_desc += "）"
    print_and_save(f"使用する特徴量: {feature_desc}")
    print_and_save(f"  設定: config.pyで指定")
    print_and_save(f"ラベル分布の調整: {'有効' if args.balance_label_distribution else '無効'}")
    print_and_save(f"非負制約: {'有効' if args.non_negative_coefficients else '無効'}")
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
        pre_retrieval_output_dir=pre_retrieval_output_dir
    )
    train_all_scores.update(train_qpp_scores)
    for metric_name, scores in train_qpp_scores.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    
    # ベーススコアを読み込む（data_loader内のコメントアウトで選択）
    train_base_scores = load_base_scores('train', args.dataset, BASE_DIR, base_experiment_names=None)
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
        pre_retrieval_output_dir=pre_retrieval_output_dir
    )
    dev_all_scores.update(dev_qpp_scores)
    for metric_name, scores in dev_qpp_scores.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    
    # ベーススコアを読み込む（data_loader内のコメントアウトで選択）
    dev_base_scores = load_base_scores('dev', args.dataset, BASE_DIR, base_experiment_names=None)
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
            
            # モデル学習
            if args.non_negative_coefficients:
                fold_model = NonNegativeLogisticRegression(
                    C=1.0,
                    max_iter=args.max_iter,
                    random_state=42,
                    class_weight='balanced',
                    tol=args.tol
                )
                fold_model.fit(X_fold_train_norm, y_fold_train)
            else:
                fold_model = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=1.0,
                    max_iter=args.max_iter,
                    random_state=42,
                    class_weight='balanced',
                    tol=args.tol
                )
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
            single_metric_probas = {}
            
            # CV統合データに対して各foldで単体指標の評価を行う
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
                    single_metric_aucs[metric_name] = np.mean(metric_aucs)
                    # 全foldの予測確率を統合
                    single_metric_probas[metric_name] = np.concatenate(metric_probas)
            
            if len(single_metric_aucs) > 0:
                # 最高ROCを出した指標を見つける
                best_metric = max(single_metric_aucs, key=single_metric_aucs.get)
                best_metric_auc = single_metric_aucs[best_metric]
                best_metric_proba = single_metric_probas[best_metric]
                
                print_and_save(f"  単体指標の最高ROC: {best_metric} (AUC = {best_metric_auc:.4f})")
                print_and_save(f"  回帰モデルのROC: AUC = {cv_test_auc:.4f}")
                
                # DeLongの検定
                delong_result = delong_test(
                    y_true=y_cv_test_labels,
                    y_pred_proba_a=np.array(all_test_proba),
                    y_pred_proba_b=best_metric_proba
                )
                
                print_and_save(f"\n  DeLongの検定結果:")
                print_and_save(f"    モデルA (回帰): AUC = {delong_result['auc_a']:.4f}")
                print_and_save(f"    モデルB (単体指標 {best_metric}): AUC = {delong_result['auc_b']:.4f}")
                print_and_save(f"    AUC差: {delong_result['auc_diff']:.4f}")
                print_and_save(f"    Z統計量: {delong_result['z_stat']:.4f}")
                print_and_save(f"    p値: {delong_result['p_value']:.4f}")
                print_and_save(f"    有意差: {'あり' if delong_result['significant'] else 'なし'} (α=0.05)")
                
                # 結果を保存
                delong_result_path = feature_output_dir / "delong_test_single_metric.csv"
                result_df = pd.DataFrame([{
                    'best_single_metric': best_metric,
                    'best_single_metric_auc': best_metric_auc,
                    'regression_auc': cv_test_auc,
                    **delong_result
                }])
                result_df.to_csv(delong_result_path, index=False)
                print_and_save(f"  - DeLongの検定結果を保存: {delong_result_path}")
            else:
                print_and_save("  ⚠️  警告: 単体指標のROCを計算できませんでした")
        
            # 予測結果を保存（回帰モデル同士の比較用）
            predictions_path = feature_output_dir / "predictions_for_delong.csv"
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
        if args.non_negative_coefficients:
            all_coefs = np.array([model.coef_[0] for model in fold_models])
        else:
            all_coefs = np.array([model.coef_[0] for model in fold_models])
        mean_coefs = np.mean(all_coefs, axis=0)
        
        print_and_save("\n特徴量の係数（各foldの平均）:")
        feature_importance = pd.DataFrame({
            'feature': X_train_norm.columns,
            'coefficient': mean_coefs
        }).sort_values('coefficient', key=abs, ascending=False)
        print_and_save(feature_importance.to_string(index=False))
        
        # 係数をCSVファイルとして保存
        coefficients_csv_path = feature_output_dir / "coefficients.csv"
        feature_importance.to_csv(coefficients_csv_path, index=False)
        print_and_save(f"  - 係数をCSVファイルに保存しました: {coefficients_csv_path}")
        
        # 係数が0の特徴量を表示
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
            predictions_path = feature_output_dir / "predictions_for_delong.csv"
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
    
    # 4. モデル学習（クロスバリデーションの場合は既に完了）
    if not args.use_cv:
        print_and_save("\n4. モデル学習中...")
        if args.non_negative_coefficients:
            print_and_save("  - 非負制約付きモデルを使用します")
            print_and_save(f"  - 最大反復回数: {args.max_iter}")
            model = NonNegativeLogisticRegression(
                C=1.0,
                max_iter=args.max_iter,
                random_state=42,
                class_weight='balanced',
                tol=args.tol
            )
            model.fit(X_train_norm, y_train)
            actual_iter = model.n_iter_[0] if hasattr(model, 'n_iter_') and len(model.n_iter_) > 0 else 'unknown'
            print_and_save(f"  - 実際の反復回数: {actual_iter}")
            
            # 収束状況をチェック
            if hasattr(model, 'optimization_result_'):
                result = model.optimization_result_
                if not result.success:
                    print_and_save(f"  ⚠️  警告: 最適化が収束しませんでした")
                    print_and_save(f"  ⚠️  メッセージ: {result.message}")
                    if actual_iter >= args.max_iter:
                        print_and_save(f"  ⚠️  最大反復回数（{args.max_iter}）に達しました")
                        print_and_save(f"  ⚠️  --max-iterを増やすことを検討してください")
                else:
                    print_and_save("  - 正常に収束しました")
            else:
                print_and_save("  - 学習完了（収束状況の確認ができませんでした）")
        else:
            model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=1.0,
                max_iter=args.max_iter,
                random_state=42,
                class_weight='balanced',
                tol=args.tol
            )
            
            # 警告をキャッチして収束状況を確認
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model.fit(X_train_norm, y_train)
                
                # 実際の反復回数を確認
                actual_iter = model.n_iter_[0] if hasattr(model, 'n_iter_') and len(model.n_iter_) > 0 else 'unknown'
                print_and_save(f"  - 実際の反復回数: {actual_iter}")
                
                # 警告があるかチェック（max_iterに達した場合）
                if w:
                    for warning in w:
                        if "max_iter" in str(warning.message).lower() or "convergence" in str(warning.message).lower():
                            print_and_save(f"  ⚠️  警告: {warning.message}")
                            print_and_save(f"  ⚠️  max_iterを増やすことを検討してください（現在: {model.max_iter}）")
                else:
                    print_and_save("  - 正常に収束しました")
            
            print_and_save("  - 学習完了")
        
        # 特徴量の重要度（係数）を表示
        print_and_save("\n特徴量の係数:")
        feature_importance = pd.DataFrame({
            'feature': X_train_norm.columns,
            'coefficient': model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        print_and_save(feature_importance.to_string(index=False))
        
        # 係数をCSVファイルとして保存
        feature_names = list(X_train_norm.columns)
        feature_dir_name = get_feature_dir_name(feature_names)
        feature_output_dir = output_dir / feature_dir_name
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        coefficients_csv_path = feature_output_dir / "coefficients.csv"
        feature_importance.to_csv(coefficients_csv_path, index=False)
        print_and_save(f"  - 係数をCSVファイルに保存しました: {coefficients_csv_path}")
        
        # 係数が0の特徴量を表示
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
                print_and_save(f"    モデルA (回帰): AUC = {delong_result['auc_a']:.4f}")
                print_and_save(f"    モデルB (単体指標 {best_metric}): AUC = {delong_result['auc_b']:.4f}")
                print_and_save(f"    AUC差: {delong_result['auc_diff']:.4f}")
                print_and_save(f"    Z統計量: {delong_result['z_stat']:.4f}")
                print_and_save(f"    p値: {delong_result['p_value']:.4f}")
                print_and_save(f"    有意差: {'あり' if delong_result['significant'] else 'なし'} (α=0.05)")
                
                # 結果を保存
                delong_result_path = feature_output_dir / "delong_test_single_metric.csv"
                result_df = pd.DataFrame([{
                    'best_single_metric': best_metric,
                    'best_single_metric_auc': best_metric_auc,
                    'regression_auc': test_auc,
                    **delong_result
                }])
                result_df.to_csv(delong_result_path, index=False)
                print_and_save(f"  - DeLongの検定結果を保存: {delong_result_path}")
            else:
                print_and_save("  ⚠️  警告: 単体指標のROCを計算できませんでした")
            
            # 予測結果を保存（回帰モデル同士の比較用）
            predictions_path = feature_output_dir / "predictions_for_delong.csv"
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
            output_dir=output_dir,
            hide_train=True,  # trainは非表示
            X_train=None,  # trainは表示しない
            X_test=X_cv_test_norm if show_single else None,  # CV結果のみ表示
            show_single_metrics=show_single,
            feature_names=feature_names,
            y_test_single=y_cv_test if show_single else None  # CV結果のラベル
        )
    else:
        plot_roc_curves(
            y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc, output_dir,
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
            output_dir=output_dir,
            hide_train=True,
            X_train=None,
            X_test=X_cv_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
            y_test_single=y_cv_test if show_single else None
        )
    else:
        plot_pr_curves(
            y_train, y_train_proba, y_test, y_test_proba, train_ap, test_ap, output_dir,
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
            output_dir=output_dir,
            hide_train=True,
            X_train=None,
            X_test=X_cv_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
            y_test_single=y_cv_test if show_single else None
        )
    else:
        plot_threshold_f1_curves(
            y_train, y_train_proba, y_test, y_test_proba, output_dir,
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
        X_train_norm, X_test_norm, output_dir,
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
            output_dir=output_dir,
            feature_names=feature_names,
            use_combined=True
        )
    else:
        plot_correlation_heatmaps(
            X_train=X_train_norm,
            X_test=X_test_norm,
            y_train=y_train,
            y_test=y_test,
            output_dir=output_dir,
            feature_names=feature_names,
            use_combined=False
        )
    print_and_save("  - 相関係数ヒートマップを保存しました")
    
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
    
    print_and_save("\n=== 完了 ===")


if __name__ == "__main__":
    main()

