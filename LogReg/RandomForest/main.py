"""
Random Forestによるclarification分類
"""
import json
import argparse
import warnings
import sys
from io import StringIO
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, precision_recall_curve, average_precision_score
from typing import Dict, List, Tuple, Any


# パス設定（main関数内で動的に設定）
BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")


def load_json_data(json_path: Path) -> List[List[Dict[str, Any]]]:
    """JSONファイルを読み込む"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_labels(json_data: List[List[Dict[str, Any]]]) -> Dict[Tuple[str, int], int]:
    """
    response_typeからラベルを抽出
    [SEP]で区切られている場合は前方を採用
    clarificationを正例（1）、それ以外を負例（0）とする
    """
    labels = {}
    for conversation in json_data:
        for turn in conversation:
            conv_id = turn['conv_id']
            turn_id = turn['turn_id']
            key = (conv_id, turn_id)
            
            response_type = turn.get('response_type', '')
            # [SEP]で分割した場合は前方を採用
            if ' [SEP] ' in response_type:
                response_type = response_type.split(' [SEP] ')[0]
            
            # clarificationを正例（1）、それ以外を負例（0）
            label = 1 if response_type.strip() == 'clarification' else 0
            labels[key] = label
    
    return labels


def extract_base_scores(json_data: List[List[Dict[str, Any]]]) -> Dict[Tuple[str, int], float]:
    """ベースモデルのlogit_clarificationを抽出"""
    scores = {}
    for conversation in json_data:
        for turn in conversation:
            conv_id = turn['conv_id']
            turn_id = turn['turn_id']
            key = (conv_id, turn_id)
            
            logit = turn.get('logit_clarification')
            if logit is not None:
                scores[key] = float(logit)
    
    return scores


def load_qpp_scores(split: str, qpp_output_dir: Path) -> Dict[str, Dict[Tuple[str, int], float]]:
    """
    QPPスコアを読み込む
    戻り値: {metric_name: {(conv_id, turn_id): score}}
    """
    qpp_scores = {}
    
    # QPPスコアの定義
    qpp_configs = {
        'nqc': ('nqc.csv', 'nqc'),
        'similarity': ('similarity.csv', 'mean_similarity'),
        'wig': ('wig.csv', 'wig'),
        # 'entropy': ('entropy.csv', 'entropy'),
        # 'lci': ('lci.csv', 'lci'),
        # 'unique_titles': ('unique_titles.csv', 'num_unique_titles'),
    }
    
    for metric_name, (filename, column_name) in qpp_configs.items():
        csv_path = qpp_output_dir / f"{split}_{filename}"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {metric_name}")
            continue
        
        df = pd.read_csv(csv_path)
        scores = {}
        for _, row in df.iterrows():
            conv_id = row['conv_id']
            turn_id = int(row['turn_id'])
            key = (conv_id, turn_id)
            
            value = row[column_name]
            if pd.notna(value):
                scores[key] = float(value)
        
        qpp_scores[metric_name] = scores
        print(f"Loaded {len(scores)} {metric_name} scores for {split}")
    
    return qpp_scores


def merge_features(
    base_scores: Dict[Tuple[str, int], float],
    qpp_scores: Dict[str, Dict[Tuple[str, int], float]],
    labels: Dict[Tuple[str, int], int],
    use_base_score: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    全ての特徴量をマージしてDataFrameを作成
    戻り値: (features_df, labels_series)
    
    Args:
        use_base_score: Trueの場合、ベーススコア（logit_clarification）も使用する
    """
    # 全てのキーを収集
    if use_base_score:
        all_keys = set(base_scores.keys())
    else:
        all_keys = set()
    
    for metric_scores in qpp_scores.values():
        all_keys.update(metric_scores.keys())
    all_keys = all_keys.intersection(set(labels.keys()))
    
    # データを収集
    data = []
    for key in all_keys:
        conv_id, turn_id = key
        row = {
            'conv_id': conv_id,
            'turn_id': turn_id,
        }
        
        # ベーススコアを追加（オプション）
        if use_base_score:
            row['logit_clarification'] = base_scores.get(key)
        
        # QPPスコアを追加
        for metric_name in qpp_scores.keys():
            row[metric_name] = qpp_scores[metric_name].get(key)
        
        # ラベルを追加
        row['label'] = labels.get(key)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 特徴量とラベルを分離
    if use_base_score:
        feature_columns = ['logit_clarification'] + list(qpp_scores.keys())
    else:
        feature_columns = list(qpp_scores.keys())
    
    features_df = df[feature_columns].copy()
    labels_series = df['label'].copy()
    
    # NaN値のチェック
    if features_df.isna().any().any():
        nan_counts = features_df.isna().sum()
        print("Warning: NaN values found in features:")
        print(nan_counts[nan_counts > 0])
        raise ValueError("NaN values are not allowed. Please check the data.")
    
    return features_df, labels_series


def plot_roc_curves(
    y_train: pd.Series,
    y_train_proba: np.ndarray,
    y_test: pd.Series,
    y_test_proba: np.ndarray,
    train_auc: float,
    test_auc: float,
    output_dir: Path
) -> None:
    """ROC曲線を描画して保存"""
    # 訓練データのROC曲線
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    
    # テストデータのROC曲線
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    # プロット
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_auc:.4f})', linewidth=2)
    plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Random Forest', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'roc_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def plot_pr_curves(
    y_train: pd.Series,
    y_train_proba: np.ndarray,
    y_test: pd.Series,
    y_test_proba: np.ndarray,
    train_ap: float,
    test_ap: float,
    output_dir: Path
) -> None:
    """Precision-Recall曲線を描画して保存"""
    # 訓練データのPR曲線
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    
    # テストデータのPR曲線
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)
    
    # ベースライン（ランダム分類器）
    baseline = len(y_test[y_test == 1]) / len(y_test)
    
    # プロット
    plt.figure(figsize=(10, 8))
    plt.plot(recall_train, precision_train, label=f'Train (AP = {train_ap:.4f})', linewidth=2)
    plt.plot(recall_test, precision_test, label=f'Test (AP = {test_ap:.4f})', linewidth=2)
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Random (AP = {baseline:.4f})', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Random Forest', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'pr_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    各特徴量を個別にz-score正規化
    訓練データの統計量でテストデータも正規化
    """
    X_train_norm = X_train.copy()
    X_test_norm = X_test.copy()
    
    scalers = {}
    for column in X_train.columns:
        scaler = StandardScaler()
        X_train_norm[column] = scaler.fit_transform(X_train[[column]]).flatten()
        X_test_norm[column] = scaler.transform(X_test[[column]]).flatten()
        scalers[column] = scaler
    
    return X_train_norm, X_test_norm


def main():
    parser = argparse.ArgumentParser(description="Random Forestによるclarification分類")
    
    # データセット名（必須）
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["INSCIT", "AmbigNQ"],
        help="データセット名（INSCIT または AmbigNQ）"
    )
    
    parser.add_argument(
        "--use-base-score",
        action="store_true",
        help="ベーススコア（logit_clarification）も特徴量として使用する（デフォルト: False、QPPスコアのみ使用）"
    )
    
    parser.add_argument(
        "--sip-experiment-name",
        type=str,
        default=None,
        help="SIP実験名（デフォルト: データセット名に基づいて自動設定）"
    )
    
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="決定木の数（デフォルト: 100）"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="決定木の最大深度（デフォルト: None、制限なし）"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）"
    )
    
    args = parser.parse_args()
    
    use_base_score = args.use_base_score
    
    # パスの動的設定
    dataset_dir = BASE_DIR / "dataset" / args.dataset
    qpp_output_dir = BASE_DIR / "QPP" / "post_retrieval" / "outputs" / args.dataset
    
    # SIP実験名の設定
    if args.sip_experiment_name:
        sip_experiment_name = args.sip_experiment_name
    else:
        # デフォルトの実験名（データセット名に基づく）
        sip_experiment_name = f"{args.dataset}_bert-base_lr2e-05_bs16_kfold5"
    
    sip_output_dir = BASE_DIR / "SIP" / "FT-PLM" / "output" / args.dataset / sip_experiment_name
    output_dir = BASE_DIR / "LogReg" / "RandomForest" / "outputs" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 出力ファイルの準備（print内容をファイルにも保存）
    output_file = output_dir / "results.txt"
    output_buffer = StringIO()
    
    def print_and_save(*args, **kwargs):
        """printと同時にファイルにも出力"""
        print(*args, **kwargs)
        print(*args, **kwargs, file=output_buffer)
    
    print_and_save("=== Random Forestによるclarification分類 ===\n")
    print_and_save(f"データセット: {args.dataset}")
    print_and_save(f"使用する特徴量: {'ベーススコア + QPPスコア' if use_base_score else 'QPPスコアのみ'}")
    print_and_save(f"モデルパラメータ: n_estimators={args.n_estimators}, max_depth={args.max_depth}, random_state={args.random_state}\n")
    
    # 1. データ読み込み
    print_and_save("1. データ読み込み中...")
    
    # 訓練データ
    train_json_path = dataset_dir / "train.json"
    train_pred_json_path = sip_output_dir / "train_with_predictions.json"
    
    train_data = load_json_data(train_json_path)
    train_labels = extract_labels(train_data)
    train_qpp_scores = load_qpp_scores('train', qpp_output_dir)
    
    print_and_save(f"  - 訓練データ: {len(train_labels)} サンプル")
    
    # ベーススコアの読み込み（使用する場合のみ）
    if use_base_score:
        train_pred_data = load_json_data(train_pred_json_path)
        train_base_scores = extract_base_scores(train_pred_data)
        print_and_save(f"  - ベーススコア: {len(train_base_scores)} サンプル")
    else:
        train_base_scores = {}  # 空の辞書を渡す（使用しない）
    
    for metric_name, scores in train_qpp_scores.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    
    # テストデータ
    dev_json_path = dataset_dir / "dev.json"
    dev_pred_json_path = sip_output_dir / "dev_with_predictions.json"
    
    dev_data = load_json_data(dev_json_path)
    dev_labels = extract_labels(dev_data)
    
    # ベーススコアの読み込み（使用する場合のみ）
    if use_base_score:
        dev_pred_data = load_json_data(dev_pred_json_path)
        dev_base_scores = extract_base_scores(dev_pred_data)
        print_and_save(f"  - ベーススコア: {len(dev_base_scores)} サンプル")
    else:
        dev_base_scores = {}  # 空の辞書を渡す（使用しない）
    
    dev_qpp_scores = load_qpp_scores('dev', qpp_output_dir)
    
    print_and_save(f"  - テストデータ: {len(dev_labels)} サンプル")
    for metric_name, scores in dev_qpp_scores.items():
        print_and_save(f"  - {metric_name}: {len(scores)} サンプル")
    
    # 2. 特徴量マージ
    print_and_save("\n2. 特徴量マージ中...")
    X_train, y_train = merge_features(train_base_scores, train_qpp_scores, train_labels, use_base_score=use_base_score)
    X_test, y_test = merge_features(dev_base_scores, dev_qpp_scores, dev_labels, use_base_score=use_base_score)
    
    print_and_save(f"  - 訓練データ: {len(X_train)} サンプル, {len(X_train.columns)} 特徴量")
    print_and_save(f"  - テストデータ: {len(X_test)} サンプル, {len(X_test.columns)} 特徴量")
    print_and_save(f"  - 特徴量: {list(X_train.columns)}")
    print_and_save(f"  - 訓練データのラベル分布: {y_train.value_counts().to_dict()}")
    print_and_save(f"  - テストデータのラベル分布: {y_test.value_counts().to_dict()}")
    
    # データ分布の確認（正規化前）
    print_and_save("\n【データ分布の確認（正規化前）】")
    print_and_save("訓練データの特徴量統計（正規化前）:")
    print_and_save(X_train.describe().to_string())
    print_and_save("\nテストデータの特徴量統計（正規化前）:")
    print_and_save(X_test.describe().to_string())
    print_and_save(f"\n訓練データのラベル分布: {y_train.value_counts().to_dict()} (正例率: {y_train.mean():.4f})")
    print_and_save(f"テストデータのラベル分布: {y_test.value_counts().to_dict()} (正例率: {y_test.mean():.4f})")
    
    # 3. 前処理: z-score正規化
    print_and_save("\n3. z-score正規化中...")
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    print_and_save("  - 正規化完了")
    
    # データ分布の確認（正規化後）
    print_and_save("\n【データ分布の確認（正規化後）】")
    print_and_save("訓練データの特徴量統計（正規化後）:")
    print_and_save(X_train_norm.describe().to_string())
    print_and_save("\nテストデータの特徴量統計（正規化後）:")
    print_and_save(X_test_norm.describe().to_string())
    
    # 分布の違いを数値で確認
    print_and_save("\n【分布の違いの分析】")
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
    
    # 4. モデル学習
    print_and_save("\n4. モデル学習中...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        class_weight='balanced',
        n_jobs=-1  # 並列処理を有効化
    )
    
    model.fit(X_train_norm, y_train)
    print_and_save("  - 学習完了")
    
    # 特徴量の重要度を表示
    print_and_save("\n特徴量の重要度:")
    feature_importance = pd.DataFrame({
        'feature': X_train_norm.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print_and_save(feature_importance.to_string(index=False))
    
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
    
    # 6. ROC曲線の描画
    print_and_save("\n6. ROC曲線を描画中...")
    plot_roc_curves(y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc, output_dir)
    print_and_save("  - ROC曲線を保存しました")
    
    # 7. PR曲線の描画
    print_and_save("\n7. Precision-Recall曲線を描画中...")
    plot_pr_curves(y_train, y_train_proba, y_test, y_test_proba, train_ap, test_ap, output_dir)
    print_and_save("  - PR曲線を保存しました")
    
    # 結果をファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
    print_and_save(f"\n結果をファイルに保存しました: {output_file}")
    
    print_and_save("\n=== 完了 ===")


if __name__ == "__main__":
    main()

