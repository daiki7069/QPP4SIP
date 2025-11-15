"""
L1正則化付きロジスティック回帰によるclarification分類
"""
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, roc_curve, precision_recall_curve, average_precision_score
from typing import Dict, List, Tuple, Any


# パス設定
BASE_DIR = Path("/home/daiki_shibata/pj/QPP4SIP")
DATASET_DIR = BASE_DIR / "dataset" / "INSCIT"
QPP_OUTPUT_DIR = BASE_DIR / "QPP" / "post_retrieval" / "outputs"
SIP_OUTPUT_DIR = BASE_DIR / "SIP" / "FT-PLM" / "output" / "bert-base_lr2e-05_bs16_kfold5"
OUTPUT_DIR = BASE_DIR / "LogReg" / "LASSO" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_qpp_scores(split: str) -> Dict[str, Dict[Tuple[str, int], float]]:
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
        'entropy': ('entropy.csv', 'entropy'),
        # 'lci': ('lci.csv', 'lci'),
        # 'unique_titles': ('unique_titles.csv', 'num_unique_titles'),
    }
    
    for metric_name, (filename, column_name) in qpp_configs.items():
        csv_path = QPP_OUTPUT_DIR / f"{split}_{filename}"
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
    test_auc: float
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
    plt.title('ROC Curves - L1 Regularized Logistic Regression', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = OUTPUT_DIR / 'roc_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()


def plot_pr_curves(
    y_train: pd.Series,
    y_train_proba: np.ndarray,
    y_test: pd.Series,
    y_test_proba: np.ndarray,
    train_ap: float,
    test_ap: float
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
    plt.title('Precision-Recall Curves - L1 Regularized Logistic Regression', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = OUTPUT_DIR / 'pr_curves.png'
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
    parser = argparse.ArgumentParser(description="L1正則化付きロジスティック回帰によるclarification分類")
    parser.add_argument(
        "--use-base-score",
        action="store_true",
        help="ベーススコア（logit_clarification）も特徴量として使用する（デフォルト: False、QPPスコアのみ使用）"
    )
    args = parser.parse_args()
    
    use_base_score = args.use_base_score
    
    print("=== L1正則化付きロジスティック回帰によるclarification分類 ===\n")
    print(f"使用する特徴量: {'ベーススコア + QPPスコア' if use_base_score else 'QPPスコアのみ'}\n")
    
    # 1. データ読み込み
    print("1. データ読み込み中...")
    
    # 訓練データ
    train_json_path = DATASET_DIR / "train.json"
    train_pred_json_path = SIP_OUTPUT_DIR / "train_with_predictions.json"
    
    train_data = load_json_data(train_json_path)
    train_labels = extract_labels(train_data)
    train_qpp_scores = load_qpp_scores('train')
    
    print(f"  - 訓練データ: {len(train_labels)} サンプル")
    
    # ベーススコアの読み込み（使用する場合のみ）
    if use_base_score:
        train_pred_data = load_json_data(train_pred_json_path)
        train_base_scores = extract_base_scores(train_pred_data)
        print(f"  - ベーススコア: {len(train_base_scores)} サンプル")
    else:
        train_base_scores = {}  # 空の辞書を渡す（使用しない）
    
    for metric_name, scores in train_qpp_scores.items():
        print(f"  - {metric_name}: {len(scores)} サンプル")
    
    # テストデータ
    dev_json_path = DATASET_DIR / "dev.json"
    dev_pred_json_path = SIP_OUTPUT_DIR / "dev_with_predictions.json"
    
    dev_data = load_json_data(dev_json_path)
    dev_labels = extract_labels(dev_data)
    
    # ベーススコアの読み込み（使用する場合のみ）
    if use_base_score:
        dev_pred_data = load_json_data(dev_pred_json_path)
        dev_base_scores = extract_base_scores(dev_pred_data)
        print(f"  - ベーススコア: {len(dev_base_scores)} サンプル")
    else:
        dev_base_scores = {}  # 空の辞書を渡す（使用しない）
    
    dev_qpp_scores = load_qpp_scores('dev')
    
    print(f"  - テストデータ: {len(dev_labels)} サンプル")
    for metric_name, scores in dev_qpp_scores.items():
        print(f"  - {metric_name}: {len(scores)} サンプル")
    
    # 2. 特徴量マージ
    print("\n2. 特徴量マージ中...")
    X_train, y_train = merge_features(train_base_scores, train_qpp_scores, train_labels, use_base_score=use_base_score)
    X_test, y_test = merge_features(dev_base_scores, dev_qpp_scores, dev_labels, use_base_score=use_base_score)
    
    print(f"  - 訓練データ: {len(X_train)} サンプル, {len(X_train.columns)} 特徴量")
    print(f"  - テストデータ: {len(X_test)} サンプル, {len(X_test.columns)} 特徴量")
    print(f"  - 特徴量: {list(X_train.columns)}")
    print(f"  - 訓練データのラベル分布: {y_train.value_counts().to_dict()}")
    print(f"  - テストデータのラベル分布: {y_test.value_counts().to_dict()}")
    
    # 3. 前処理: z-score正規化
    print("\n3. z-score正規化中...")
    X_train_norm, X_test_norm = normalize_features(X_train, X_test)
    print("  - 正規化完了")
    
    # 4. モデル学習
    print("\n4. モデル学習中...")
    model = LogisticRegression(
        penalty='l1',
        solver='liblinear',
        C=1.0,
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_norm, y_train)
    print("  - 学習完了")
    
    # 特徴量の重要度（係数）を表示
    print("\n特徴量の係数:")
    feature_importance = pd.DataFrame({
        'feature': X_train_norm.columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    print(feature_importance.to_string(index=False))
    
    # 係数が0の特徴量を表示
    zero_coef_features = feature_importance[feature_importance['coefficient'] == 0.0]
    if len(zero_coef_features) > 0:
        print(f"\n係数が0の特徴量（L1正則化により除外）: {len(zero_coef_features)}個")
        print(zero_coef_features[['feature']].to_string(index=False))
    else:
        print("\n係数が0の特徴量はありません（全ての特徴量が使用されています）")
    
    # 5. 評価
    print("\n5. 評価中...")
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
    
    print("\n=== 訓練データの評価 ===")
    print(f"Accuracy: {train_acc:.4f}")
    print(f"F1 Score: {train_f1:.4f}")
    print(f"AUC-ROC: {train_auc:.4f}")
    print(f"Average Precision: {train_ap:.4f}")
    print("\n分類レポート:")
    print(classification_report(y_train, y_train_pred, target_names=['not_clarification', 'clarification']))
    
    print("\n=== テストデータの評価 ===")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")
    print(f"Average Precision: {test_ap:.4f}")
    print("\n分類レポート:")
    print(classification_report(y_test, y_test_pred, target_names=['not_clarification', 'clarification']))
    
    # 6. ROC曲線の描画
    print("\n6. ROC曲線を描画中...")
    plot_roc_curves(y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc)
    print("  - ROC曲線を保存しました")
    
    # 7. PR曲線の描画
    print("\n7. Precision-Recall曲線を描画中...")
    plot_pr_curves(y_train, y_train_proba, y_test, y_test_proba, train_ap, test_ap)
    print("  - PR曲線を保存しました")
    
    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()

