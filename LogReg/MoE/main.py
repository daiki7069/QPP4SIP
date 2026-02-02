"""
MoE (Mixture of Experts) による動的ゲーティング統合
LASSOと同じ入力・前処理条件で、ソフトゲーティングによりどの予測子を重視するかを入力に応じて決定
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold

# LASSOと同じmodule（データ読み込み・前処理）を利用
BASE_DIR = Path(__file__).resolve().parent.parent.parent
LASSO_DIR = BASE_DIR / "LogReg" / "LASSO"
LOGREG_DIR = BASE_DIR / "LogReg"
sys.path.insert(0, str(LASSO_DIR))
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
    plot_threshold_f1_curves,
    plot_feature_distributions,
    plot_correlation_heatmaps,
)

# MoE用module（LogRegをpathに追加してからMoE.moduleをimport）
sys.path.insert(0, str(LOGREG_DIR))
from MoE.module import MixtureOfExperts, GatingType


def main():
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    parser = argparse.ArgumentParser(
        description="MoEによる動的ゲーティング統合（LASSOと同一前処理）"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名",
    )
    parser.add_argument(
        "--gating-type",
        type=str,
        default="linear",
        choices=["linear", "mlp", "temperature", "learned_fixed"],
        help="ゲーティング方式: linear=線形+softmax, mlp=2層MLP+softmax, "
        "temperature=温度付きsoftmax, learned_fixed=入力非依存の学習固定重み",
    )
    parser.add_argument(
        "--gate-hidden-size",
        type=int,
        default=32,
        help="MLPゲートの隠れ層サイズ（--gating-type mlp のときのみ）",
    )
    parser.add_argument(
        "--gate-temperature",
        type=float,
        default=1.0,
        help="温度付きsoftmaxの温度（--gating-type temperature のときのみ）",
    )
    parser.add_argument(
        "--balance-label-distribution",
        action="store_true",
        help="訓練データのラベル分布をテストと同じ正例率に調整",
    )
    parser.add_argument(
        "--use-combined-normalization",
        action="store_true",
        help="訓練+テストを結合してから正規化（リーク前提）",
    )
    parser.add_argument(
        "--use-separate-normalization",
        action="store_true",
        help="訓練とテストをそれぞれ個別に正規化",
    )
    parser.add_argument(
        "--use-minmax-normalization",
        action="store_true",
        help="[0,1]正規化を使用（デフォルトはz-score）",
    )
    parser.add_argument(
        "--no-cv",
        action="store_false",
        dest="use_cv",
        default=True,
        help="CVを無効化して train/dev 分離評価",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="クロスバリデーションのfold数",
    )
    parser.add_argument(
        "--retrieval-method",
        type=str,
        default="dpr",
        choices=["dpr", "bm25"],
        help="検索手法",
    )
    parser.add_argument(
        "--feature-types",
        type=str,
        nargs="+",
        choices=["post", "pre", "bert", "roberta", "nsp", "transfer"],
        default=["post", "pre", "nsp"],
        help="使用する特徴量タイプ（LASSOと同一）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="結果出力ディレクトリ（省略時は LogReg/MoE/outputs/<dataset>）",
    )
    args = parser.parse_args()

    use_post = "post" in args.feature_types
    use_pre = "pre" in args.feature_types
    use_bert = "bert" in args.feature_types
    use_roberta = "roberta" in args.feature_types
    use_nsp = "nsp" in args.feature_types
    use_transfer = "transfer" in args.feature_types

    dataset_dir = BASE_DIR / "dataset" / args.dataset
    qpp_output_dir = (
        BASE_DIR / "QPP" / "post_retrieval" / "outputs" / args.dataset / args.retrieval_method
    )
    pre_retrieval_output_dir = BASE_DIR / "QPP" / "pre_retrieval" / "outputs" / args.dataset
    nsp_output_dir = BASE_DIR / "QPP" / "next_sentence_prediction" / "outputs" / args.dataset
    out_dir = args.output_dir or (BASE_DIR / "LogReg" / "MoE" / "outputs" / args.dataset)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nsp_top_k = None
    if nsp_output_dir.exists():
        nsp_top_k = find_common_nsp_top_k(nsp_output_dir, splits=["train", "dev"])

    log_lines = []

    def print_and_save(*a, **kw):
        msg = " ".join(str(x) for x in a)
        print(msg)
        log_lines.append(msg)

    # ヘッダー（LASSO形式）
    if nsp_top_k is not None:
        print_and_save(f"NSP top_k値: {nsp_top_k} (trainとdevで共通)")
    else:
        print_and_save("Warning: trainとdevで共通するNSP top_k値が見つかりませんでした")
    feature_parts = []
    if use_post:
        feature_parts.append("post")
    if use_pre:
        feature_parts.append("pre")
    if use_nsp:
        feature_parts.append("nsp")
    if use_bert or use_roberta or use_transfer:
        feature_parts.append("base")
    print_and_save("使用する特徴量タイプ:", ", ".join(args.feature_types))
    print_and_save("=== MoE（動的ゲーティング）によるclarification分類 ===")
    print_and_save("")
    print_and_save("データセット:", args.dataset)
    print_and_save("検索手法:", args.retrieval_method)
    print_and_save("使用する特徴量: QPPスコア（" + " + ".join(feature_parts) + "）")
    print_and_save("モデルタイプ: MoE（ゲーティング方式: " + args.gating_type + "）")
    print_and_save("ラベル分布の調整:", "有効" if args.balance_label_distribution else "無効")
    print_and_save("クロスバリデーション:", "有効" if args.use_cv else "無効", f"(fold={args.cv_folds})" if args.use_cv else "")
    norm_name = "[0,1]正規化（Min-Max）" if args.use_minmax_normalization else "z-score正規化"
    print_and_save("正規化方法: 訓練データの統計量で" + norm_name + "（通常）")
    print_and_save("")

    # 1. データ読み込み（LASSOと同一）
    print_and_save("1. データ読み込み中...")
    train_data = load_json_data(dataset_dir / "train.json")
    train_labels = extract_labels(train_data)
    train_all_scores = {}
    train_qpp = load_qpp_scores(
        "train",
        qpp_output_dir,
        nsp_output_dir=nsp_output_dir,
        nsp_top_k=nsp_top_k,
        pre_retrieval_output_dir=pre_retrieval_output_dir,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=use_nsp,
    )
    train_all_scores.update(train_qpp)
    train_base = load_base_scores(
        "train",
        args.dataset,
        BASE_DIR,
        use_bert=use_bert,
        use_roberta=use_roberta,
        use_transfer=use_transfer,
    )
    if train_base:
        train_all_scores.update(train_base)
    print_and_save(f"  - 訓練データ: {len(train_labels)} サンプル")
    for metric_name, scores in train_qpp.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    for name, scores in (train_base or {}).items():
        print_and_save(f"  - {name}: {len(scores)} サンプル")

    dev_data = load_json_data(dataset_dir / "dev.json")
    dev_labels = extract_labels(dev_data)
    dev_all_scores = {}
    dev_qpp = load_qpp_scores(
        "dev",
        qpp_output_dir,
        nsp_output_dir=nsp_output_dir,
        nsp_top_k=nsp_top_k,
        pre_retrieval_output_dir=pre_retrieval_output_dir,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=use_nsp,
    )
    dev_all_scores.update(dev_qpp)
    dev_base = load_base_scores(
        "dev",
        args.dataset,
        BASE_DIR,
        use_bert=use_bert,
        use_roberta=use_roberta,
        use_transfer=use_transfer,
    )
    if dev_base:
        dev_all_scores.update(dev_base)
    print_and_save(f"  - テストデータ: {len(dev_labels)} サンプル")
    for metric_name, scores in dev_qpp.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    for name, scores in (dev_base or {}).items():
        print_and_save(f"  - {name}: {len(scores)} サンプル")
    print_and_save("")

    # 2. 特徴量マージ
    print_and_save("2. 特徴量マージ中...")
    X_train, y_train = merge_features(train_all_scores, train_labels)
    X_test, y_test = merge_features(dev_all_scores, dev_labels)
    print_and_save(f"  - 訓練データ: {len(X_train)} サンプル, {len(X_train.columns)} 特徴量")
    print_and_save(f"  - テストデータ: {len(X_test)} サンプル, {len(X_test.columns)} 特徴量")
    print_and_save(f"  - 特徴量: {list(X_train.columns)}")
    print_and_save(f"  - 訓練データのラベル分布: {y_train.value_counts().to_dict()}")
    print_and_save(f"  - テストデータのラベル分布: {y_test.value_counts().to_dict()}")
    print_and_save("")

    if args.balance_label_distribution and not args.use_cv:
        print_and_save("2.5. ラベル分布の調整中...")
        target_rate = y_test.mean()
        print_and_save(f"  - 調整前の訓練データの正例率: {y_train.mean():.4f}")
        print_and_save(f"  - テストデータの正例率（目標）: {target_rate:.4f}")
        X_train, y_train = balance_label_distribution(
            X_train, y_train, target_positive_rate=target_rate, random_state=42
        )
        print_and_save(f"  - 調整後の訓練データの正例率: {y_train.mean():.4f}")
        print_and_save(f"  - 調整後の訓練データのサンプル数: {len(X_train)}")
        print_and_save("")
    else:
        print_and_save("2.5. ラベル分布の調整: スキップ（--balance-label-distribution が指定されていません）")
        print_and_save("")

    # データ分布（正規化前）
    if not args.use_cv:
        print_and_save("【データ分布の確認（正規化前）】")
        print_and_save("訓練データの特徴量統計（正規化前）:")
        print_and_save(X_train.describe().to_string())
        print_and_save("")
        print_and_save("テストデータの特徴量統計（正規化前）:")
        print_and_save(X_test.describe().to_string())
        print_and_save(f"\n訓練データのラベル分布: {y_train.value_counts().to_dict()} (正例率: {y_train.mean():.4f})")
        print_and_save(f"テストデータのラベル分布: {y_test.value_counts().to_dict()} (正例率: {y_test.mean():.4f})")
        print_and_save("")

    # 3. 正規化（LASSOと同一条件）
    norm_type = "[0,1]正規化（Min-Max）" if args.use_minmax_normalization else "z-score正規化"
    print_and_save(f"3. {norm_type}中...")
    print_and_save("  - 方法: 訓練データの統計量でテストデータも正規化（通常）")
    X_train_norm, X_test_norm = normalize_features(
        X_train,
        X_test,
        use_combined_normalization=args.use_combined_normalization,
        use_separate_normalization=args.use_separate_normalization,
        use_minmax_normalization=args.use_minmax_normalization,
    )
    print_and_save("  - 正規化完了")
    print_and_save("")

    if not args.use_cv:
        print_and_save("【データ分布の確認（正規化後）】")
        print_and_save("訓練データの特徴量統計（正規化後）:")
        print_and_save(X_train_norm.describe().to_string())
        print_and_save("")
        print_and_save("テストデータの特徴量統計（正規化後）:")
        print_and_save(X_test_norm.describe().to_string())
        print_and_save("")

    feature_names = list(X_train_norm.columns)
    feature_dir_name = get_feature_dir_name(
        feature_names, retrieval_method=args.retrieval_method
    )
    gating_suffix = args.gating_type
    if args.gating_type == "mlp":
        gating_suffix = f"mlp_h{args.gate_hidden_size}"
    elif args.gating_type == "temperature":
        gating_suffix = f"temperature_t{args.gate_temperature}"
    feature_output_dir = out_dir / f"{feature_dir_name}_{gating_suffix}"
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = feature_output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = feature_output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    results_file = feature_output_dir / "results.txt"

    if args.use_cv:
        X_combined = pd.concat([X_train_norm, X_test_norm], axis=0, ignore_index=True)
        y_combined = pd.concat([y_train, y_test], axis=0, ignore_index=True)
        skf = StratifiedKFold(
            n_splits=args.cv_folds, shuffle=True, random_state=RANDOM_SEED
        )
        fold_results = []
        all_test_proba = []
        all_test_labels = []
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_combined, y_combined)):
            X_fold_train = X_combined.iloc[tr_idx].reset_index(drop=True)
            y_fold_train = y_combined.iloc[tr_idx].reset_index(drop=True)
            X_fold_test = X_combined.iloc[te_idx].reset_index(drop=True)
            y_fold_test = y_combined.iloc[te_idx].reset_index(drop=True)
            model = MixtureOfExperts(
                gating_type=args.gating_type,
                hidden_size=args.gate_hidden_size,
                temperature=args.gate_temperature,
                random_state=RANDOM_SEED,
            )
            model.fit(X_fold_train, y_fold_train)
            y_proba = model.predict_proba(X_fold_test)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            fold_auc = roc_auc_score(y_fold_test, y_proba)
            fold_ap = average_precision_score(y_fold_test, y_proba)
            fold_acc = accuracy_score(y_fold_test, y_pred)
            fold_f1 = f1_score(y_fold_test, y_pred)
            fold_results.append(
                {
                    "fold": fold_idx + 1,
                    "auc": fold_auc,
                    "ap": fold_ap,
                    "acc": fold_acc,
                    "f1": fold_f1,
                }
            )
            all_test_proba.extend(y_proba)
            all_test_labels.extend(y_fold_test.tolist())
            print_and_save(
                f"Fold {fold_idx+1}: AUC={fold_auc:.4f} AP={fold_ap:.4f} Acc={fold_acc:.4f} F1={fold_f1:.4f}"
            )
        all_test_proba = np.array(all_test_proba)
        all_test_labels = np.array(all_test_labels)
        agg_auc = roc_auc_score(all_test_labels, all_test_proba)
        agg_ap = average_precision_score(all_test_labels, all_test_proba)
        agg_acc = accuracy_score(all_test_labels, (all_test_proba >= 0.5).astype(int))
        agg_f1 = f1_score(all_test_labels, (all_test_proba >= 0.5).astype(int))
        print_and_save("4. モデル学習中...")
        print_and_save("  - モデルタイプ: MoE（ゲーティング方式: " + args.gating_type + "）")
        print_and_save("  - 学習完了（各Foldで学習）")
        print_and_save("")
        print_and_save("5. 評価中...（CV統合）")
        print_and_save("")
        print_and_save("=== テストデータの評価（CV統合） ===")
        print_and_save(f"Accuracy: {agg_acc:.4f}")
        print_and_save(f"F1 Score: {agg_f1:.4f}")
        print_and_save(f"AUC-ROC: {agg_auc:.4f}")
        print_and_save(f"Average Precision: {agg_ap:.4f}")
        print_and_save("")
        df_fold = pd.DataFrame(fold_results)
        print_and_save("Fold別:")
        print_and_save(df_fold.to_string(index=False))
        print_and_save("")
        # CV用の描画用変数（単体指標はCV時はインデックスが合わないため表示しない）
        y_train_proba = None
        y_train_for_plot = None
        y_test_proba = np.array(all_test_proba)
        y_test_for_plot = pd.Series(all_test_labels)
        train_auc = train_ap = train_acc = train_f1 = None
        test_auc = agg_auc
        test_ap = agg_ap
        test_acc = agg_acc
        test_f1 = agg_f1
    else:
        # 4. モデル学習
        print_and_save("4. モデル学習中...")
        print_and_save("  - モデルタイプ: MoE（ゲーティング方式: " + args.gating_type + "）")
        model = MixtureOfExperts(
            gating_type=args.gating_type,
            hidden_size=args.gate_hidden_size,
            temperature=args.gate_temperature,
            random_state=RANDOM_SEED,
        )
        model.fit(X_train_norm, y_train)
        print_and_save("  - 学習完了")
        print_and_save("")

        # 5. 評価（LASSO形式）
        print_and_save("5. 評価中...")
        y_train_proba = model.predict_proba(X_train_norm)[:, 1]
        y_test_proba = model.predict_proba(X_test_norm)[:, 1]
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        y_test_pred = (y_test_proba >= 0.5).astype(int)
        train_auc = roc_auc_score(y_train, y_train_proba)
        train_ap = average_precision_score(y_train, y_train_proba)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        test_ap = average_precision_score(y_test, y_test_proba)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        y_train_for_plot = y_train
        y_test_for_plot = y_test

        print_and_save("")
        print_and_save("=== 訓練データの評価 ===")
        print_and_save(f"Accuracy: {train_acc:.4f}")
        print_and_save(f"F1 Score: {train_f1:.4f}")
        print_and_save(f"AUC-ROC: {train_auc:.4f}")
        print_and_save(f"Average Precision: {train_ap:.4f}")
        print_and_save("")
        print_and_save("分類レポート:")
        print_and_save(classification_report(y_train, y_train_pred, target_names=["not_clarification", "clarification"]))
        print_and_save("")
        print_and_save("=== テストデータの評価 ===")
        print_and_save(f"Accuracy: {test_acc:.4f}")
        print_and_save(f"F1 Score: {test_f1:.4f}")
        print_and_save(f"AUC-ROC: {test_auc:.4f}")
        print_and_save(f"Average Precision: {test_ap:.4f}")
        print_and_save("")
        print_and_save("分類レポート:")
        print_and_save(classification_report(y_test, y_test_pred, target_names=["not_clarification", "clarification"]))
        print_and_save("")

    # 6. ROC曲線の描画（LASSOと同一）
    show_single = len(feature_names) > 1 and not args.use_cv  # CV時は単体指標はスキップ
    print_and_save("6. ROC曲線を描画中...")
    if args.use_cv:
        plot_roc_curves(
            y_train=y_train_for_plot,
            y_train_proba=y_train_proba,
            y_test=y_test_for_plot,
            y_test_proba=y_test_proba,
            train_auc=train_auc,
            test_auc=test_auc,
            output_dir=graphs_dir,
            hide_train=True,
            X_train=None,
            X_test=None,
            show_single_metrics=False,
            feature_names=feature_names,
            y_test_single=None,
        )
    else:
        plot_roc_curves(
            y_train_for_plot,
            y_train_proba,
            y_test_for_plot,
            y_test_proba,
            train_auc,
            test_auc,
            graphs_dir,
            hide_train=True,
            X_train=X_train_norm if show_single else None,
            X_test=X_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
        )
    print_and_save("  - ROC曲線を保存しました")
    print_and_save("")

    # 7. PR曲線の描画
    print_and_save("7. Precision-Recall曲線を描画中...")
    if args.use_cv:
        plot_pr_curves(
            y_train=y_train_for_plot,
            y_train_proba=y_train_proba,
            y_test=y_test_for_plot,
            y_test_proba=y_test_proba,
            train_ap=train_ap,
            test_ap=test_ap,
            output_dir=graphs_dir,
            hide_train=True,
            X_train=None,
            X_test=X_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
            y_test_single=y_test_for_plot if show_single else None,
        )
    else:
        plot_pr_curves(
            y_train_for_plot,
            y_train_proba,
            y_test_for_plot,
            y_test_proba,
            train_ap,
            test_ap,
            graphs_dir,
            hide_train=True,
            X_train=X_train_norm if show_single else None,
            X_test=X_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
        )
    print_and_save("  - PR曲線を保存しました")
    print_and_save("")

    # 8. 閾値とF1スコアの関係の描画
    print_and_save("8. 閾値とF1スコアの関係を描画中...")
    if args.use_cv:
        plot_threshold_f1_curves(
            y_train=y_train_for_plot,
            y_train_proba=y_train_proba,
            y_test=y_test_for_plot,
            y_test_proba=y_test_proba,
            output_dir=graphs_dir,
            hide_train=True,
            X_train=None,
            X_test=X_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
            y_test_single=y_test_for_plot if show_single else None,
        )
    else:
        plot_threshold_f1_curves(
            y_train_for_plot,
            y_train_proba,
            y_test_for_plot,
            y_test_proba,
            graphs_dir,
            hide_train=True,
            X_train=X_train_norm if show_single else None,
            X_test=X_test_norm if show_single else None,
            show_single_metrics=show_single,
            feature_names=feature_names,
        )
    print_and_save("  - 閾値とF1スコアの関係を保存しました")
    print_and_save("")

    # 9. 特徴量のスコア分布の描画
    print_and_save("9. 特徴量のスコア分布を描画中...")
    plot_feature_distributions(
        X_train_norm,
        X_test_norm,
        graphs_dir,
        hide_train=True,
        feature_names=feature_names,
        y_train=y_train,
        y_test=y_test,
        y_train_proba=y_train_proba if not args.use_cv else None,
        y_test_proba=y_test_proba if not args.use_cv else None,  # CV時は長さ不一致のため渡さない
    )
    print_and_save("  - 特徴量のスコア分布を保存しました")
    print_and_save("")

    # 10. 相関係数ヒートマップの描画
    print_and_save("10. 相関係数ヒートマップを描画中...")
    if args.use_cv:
        plot_correlation_heatmaps(
            X_train=X_train_norm,
            X_test=X_test_norm,
            y_train=y_train,
            y_test=y_test,
            output_dir=graphs_dir,
            feature_names=feature_names,
            use_combined=True,
        )
    else:
        plot_correlation_heatmaps(
            X_train_norm,
            X_test_norm,
            y_train,
            y_test,
            graphs_dir,
            feature_names=feature_names,
            use_combined=False,
        )
    print_and_save("  - 相関係数ヒートマップを保存しました")
    print_and_save("")

    print_and_save(f"結果を保存: {results_file}")
    results_file.write_text("\n".join(log_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
