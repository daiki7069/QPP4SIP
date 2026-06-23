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

LASSO_DIR = Path(__file__).resolve().parents[1] / 'LASSO'
sys.path.insert(0, str(LASSO_DIR))
from module.data_loader import extract_base_scores as extract_shared_base_scores
from module.models import RandomForestCVModel


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
            # conv_idとturn_idを文字列/整数に統一
            conv_id = str(turn['conv_id'])
            turn_id = int(turn['turn_id'])
            key = (conv_id, turn_id)
            
            response_type = turn.get('response_type', '')
            # [SEP]で分割した場合は前方を採用
            if ' [SEP] ' in response_type:
                response_type = response_type.split(' [SEP] ')[0]
            
            # clarificationを正例（1）、それ以外を負例（0）
            label = 1 if response_type.strip() == 'clarification' else 0
            labels[key] = label
    
    return labels


def extract_base_scores(
    json_data: List[List[Dict[str, Any]]],
    score_mode: str = 'logit_difference',
) -> Dict[Tuple[str, int], float]:
    """共有ローダーを使ってPLMスコアを抽出する。"""
    return extract_shared_base_scores(json_data, score_mode=score_mode)


def find_common_nsp_top_k(nsp_output_dir: Path, splits: List[str] = None) -> int:
    """
    複数のスプリットで共通して存在するNSPのtop_k値の最大値を返す
    
    Args:
        nsp_output_dir: next_sentence_predictionの出力ディレクトリ
        splits: 確認するスプリットのリスト（Noneの場合は'train'と'dev'を確認）
    
    Returns:
        共通して存在する最大のtop_k値、存在しない場合はNone
    """
    if splits is None:
        splits = ['train', 'dev']
    
    nsp_top_k_values = [10, 20, 50, 100]
    common_top_k = []
    
    for top_k in nsp_top_k_values:
        # 全てのスプリットで存在するか確認
        all_exist = True
        for split in splits:
            csv_path = nsp_output_dir / f"{split}_nsp_graph_topk{top_k}.csv"
            if not csv_path.exists():
                all_exist = False
                break
        
        if all_exist:
            common_top_k.append(top_k)
    
    return max(common_top_k) if common_top_k else None


def load_qpp_scores(split: str, qpp_output_dir: Path, nsp_output_dir: Path = None, nsp_top_k: int = None) -> Dict[str, Dict[Tuple[str, int], float]]:
    """
    QPPスコアを読み込む（post_retrievalとnspの両方）
    戻り値: {metric_name: {(conv_id, turn_id): score}}
    
    Args:
        split: データセットのスプリット（'train' または 'dev'）
        qpp_output_dir: post_retrievalの出力ディレクトリ
        nsp_output_dir: next_sentence_predictionの出力ディレクトリ（オプション）
        nsp_top_k: 使用するNSPのtop_k値（Noneの場合は自動検出）
    """
    qpp_scores = {}
    
    # Post-retrieval QPPスコアの定義
    post_qpp_configs = {
        'nqc': ('nqc.csv', 'nqc'),
        'smv': ('smv.csv', 'smv'),
        'nsv': ('nsv.csv', 'nsv'),
        'similarity': ('similarity.csv', 'mean_similarity'),
        'wig': ('wig.csv', 'wig'),
        'acc': ('coherency.csv', 'acc'),
        'clarity': ('clarity.csv', 'clarity'),
        # 'entropy': ('entropy.csv', 'entropy'),
        # 'lci': ('lci.csv', 'lci'),
        # 'unique_titles': ('unique_titles.csv', 'num_unique_titles'),
        # 'wacc': ('coherency.csv', 'wacc'),
    }
    
    # Post-retrievalスコアを読み込む
    for metric_name, (filename, column_name) in post_qpp_configs.items():
        csv_path = qpp_output_dir / f"{split}_{filename}"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping {metric_name}")
            continue
        
        # conv_idを文字列として読み込む（科学記数法を避けるため）
        df = pd.read_csv(csv_path, dtype={'conv_id': str})
        scores = {}
        for _, row in df.iterrows():
            # conv_idは既に文字列として読み込まれている
            conv_id = str(row['conv_id'])
            turn_id = int(row['turn_id'])
            key = (conv_id, turn_id)
            
            value = row[column_name]
            if pd.notna(value):
                scores[key] = float(value)
        
        qpp_scores[metric_name] = scores
        print(f"Loaded {len(scores)} {metric_name} scores for {split}")
    
    # NSPスコアを読み込む（オプション）
    if nsp_output_dir is not None:
        # top_k値の決定
        if nsp_top_k is None:
            # 自動検出：このスプリットで存在する最大のtop_k値を使用
            nsp_top_k_values = [10, 20, 50, 100]
            available_top_k = []
            
            for top_k in nsp_top_k_values:
                csv_path = nsp_output_dir / f"{split}_nsp_graph_topk{top_k}.csv"
                if csv_path.exists():
                    available_top_k.append(top_k)
            
            if available_top_k:
                top_k = max(available_top_k)
            else:
                print(f"Warning: No NSP score files found in {nsp_output_dir} for {split}")
                return qpp_scores
        else:
            top_k = nsp_top_k
        
        csv_path = nsp_output_dir / f"{split}_nsp_graph_topk{top_k}.csv"
        if not csv_path.exists():
            print(f"Warning: {csv_path} not found, skipping NSP scores for {split}")
            return qpp_scores
        
        # conv_idを文字列として読み込む
        df = pd.read_csv(csv_path, dtype={'conv_id': str})
        
        # NSPメトリクスを読み込む
        nsp_metrics = {
            'node_connectivity': f'nsp_node_connectivity_topk{top_k}',
            'average_node_connectivity': f'nsp_avg_node_connectivity_topk{top_k}',
            'density': f'nsp_density_topk{top_k}',
        }
        
        for column_name, metric_name in nsp_metrics.items():
            if column_name not in df.columns:
                print(f"Warning: {column_name} not found in {csv_path}, skipping")
                continue
            
            scores = {}
            for _, row in df.iterrows():
                conv_id = str(row['conv_id'])
                turn_id = int(row['turn_id'])
                key = (conv_id, turn_id)
                
                value = row[column_name]
                if pd.notna(value):
                    scores[key] = float(value)
            
            qpp_scores[metric_name] = scores
            print(f"Loaded {len(scores)} {metric_name} scores for {split} (top_k={top_k})")
    
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
    output_dir: Path,
    hide_train: bool = True,
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    show_single_metrics: bool = False,
    feature_names: list = None
) -> None:
    """ROC曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    if feature_names is not None and len(feature_names) > 0:
        sorted_features = sorted(feature_names)
        safe_features = [f.replace(' ', '_').replace('/', '_') for f in sorted_features]
        feature_dir_name = 'features_' + '_'.join(safe_features)
        feature_output_dir = output_dir / feature_dir_name
    else:
        feature_output_dir = output_dir
    
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練データのROC曲線
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    
    # テストデータのROC曲線
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    
    # プロット
    figsize = (12, 10) if show_single_metrics else (10, 8)
    plt.figure(figsize=figsize)
    
    # モデルのROC曲線
    if not hide_train:
        plt.plot(fpr_train, tpr_train, label=f'Model Train (AUC = {train_auc:.4f})', linewidth=2.5, color='blue')
    plt.plot(fpr_test, tpr_test, label=f'Model Test (AUC = {test_auc:.4f})', linewidth=2.5, color='red')
    
    # 各特徴量単体のROC曲線（オプション）
    if show_single_metrics and X_train is not None and X_test is not None:
        for feature_name in X_train.columns:
            # テストデータ
            test_scores = X_test[feature_name].values
            fpr_test_single, tpr_test_single, _ = roc_curve(y_test, test_scores)
            test_auc_single = roc_auc_score(y_test, test_scores)
            plt.plot(fpr_test_single, tpr_test_single, 
                    label=f'{feature_name} (Test, AUC = {test_auc_single:.4f})', 
                    linewidth=1.5, alpha=0.7)
    
    # ランダム分類器
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    title = 'ROC Curves - Model and Single Metrics' if show_single_metrics else 'ROC Curves - Random Forest'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9 if show_single_metrics else 11, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'roc_curves.png'
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
    output_dir: Path,
    hide_train: bool = True,
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    show_single_metrics: bool = False,
    feature_names: list = None
) -> None:
    """Precision-Recall曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    if feature_names is not None and len(feature_names) > 0:
        sorted_features = sorted(feature_names)
        safe_features = [f.replace(' ', '_').replace('/', '_') for f in sorted_features]
        feature_dir_name = 'features_' + '_'.join(safe_features)
        feature_output_dir = output_dir / feature_dir_name
    else:
        feature_output_dir = output_dir
    
    feature_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 訓練データのPR曲線
    precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
    
    # テストデータのPR曲線
    precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)
    
    # ベースライン（ランダム分類器）
    baseline = len(y_test[y_test == 1]) / len(y_test)
    
    # プロット
    figsize = (12, 10) if show_single_metrics else (10, 8)
    plt.figure(figsize=figsize)
    
    # モデルのPR曲線
    if not hide_train:
        plt.plot(recall_train, precision_train, label=f'Model Train (AP = {train_ap:.4f})', linewidth=2.5, color='blue')
    plt.plot(recall_test, precision_test, label=f'Model Test (AP = {test_ap:.4f})', linewidth=2.5, color='red')
    
    # 各特徴量単体のPR曲線（オプション）
    if show_single_metrics and X_train is not None and X_test is not None:
        for feature_name in X_train.columns:
            # テストデータ
            test_scores = X_test[feature_name].values
            precision_test_single, recall_test_single, _ = precision_recall_curve(y_test, test_scores)
            test_ap_single = average_precision_score(y_test, test_scores)
            plt.plot(recall_test_single, precision_test_single, 
                    label=f'{feature_name} (Test, AP = {test_ap_single:.4f})', 
                    linewidth=1.5, alpha=0.7)
    
    # ランダム分類器
    plt.axhline(y=baseline, color='k', linestyle='--', label=f'Random (AP = {baseline:.4f})', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    title = 'Precision-Recall Curves - Model and Single Metrics' if show_single_metrics else 'Precision-Recall Curves - Random Forest'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9 if show_single_metrics else 11, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'pr_curves.png'
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
        help="PLMスコアも特徴量として使用する（デフォルト: False、QPPスコアのみ使用）"
    )

    parser.add_argument(
        "--plm-score-mode",
        type=str,
        default="logit_difference",
        choices=["logit_difference", "positive_logit"],
        help="PLMスコア方式（デフォルト: logit_difference=正例logit-負例logit。positive_logitで従来方式）"
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
        help="固定RFで使用する決定木数（--no-rf-search時のみ、デフォルト: 100）"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="固定RFで使用する最大深度（--no-rf-search時のみ、デフォルト: None）"
    )

    parser.add_argument(
        "--no-rf-search",
        action="store_false",
        dest="use_rf_search",
        default=True,
        help="RandomForestのハイパーパラメータ探索を無効化し、従来の固定設定を使用"
    )

    parser.add_argument(
        "--rf-search-iterations",
        type=int,
        default=32,
        help="RandomizedSearchCVの試行数（デフォルト: 32）"
    )

    parser.add_argument(
        "--rf-cv-folds",
        type=int,
        default=5,
        help="RandomForestハイパーパラメータ探索の層化CV fold数（デフォルト: 5）"
    )

    parser.add_argument(
        "--rf-scoring",
        type=str,
        default="roc_auc",
        choices=["roc_auc", "average_precision", "f1", "accuracy"],
        help="RandomForestハイパーパラメータ探索の評価指標（デフォルト: roc_auc）"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="乱数シード（デフォルト: 42）"
    )
    
    parser.add_argument(
        "--show-train-curves",
        action="store_true",
        help="訓練データの曲線も表示する（デフォルト: False、テストデータのみ表示）"
    )
    
    parser.add_argument(
        "--show-single-metric-curves",
        action="store_true",
        help="入力に使った指標単体での曲線も表示する（デフォルト: False）"
    )
    
    args = parser.parse_args()
    
    use_base_score = args.use_base_score
    
    # パスの動的設定
    dataset_dir = BASE_DIR / "dataset" / args.dataset
    qpp_output_dir = BASE_DIR / "QPP" / "post_retrieval" / "outputs" / args.dataset
    nsp_output_dir = BASE_DIR / "QPP" / "next_sentence_prediction" / "outputs" / args.dataset
    
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
    # 注意: output_fileは後で特徴量名に基づくディレクトリに再設定される
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
    
    print_and_save("=== Random Forestによるclarification分類 ===\n")
    print_and_save(f"データセット: {args.dataset}")
    print_and_save(f"使用する特徴量: {'PLMスコア + QPPスコア（post + nsp）' if use_base_score else 'QPPスコア（post + nsp）'}")
    if use_base_score:
        print_and_save(f"PLMスコア方式: {args.plm_score_mode}")
    if args.use_rf_search:
        print_and_save(f"RandomForest探索: RandomizedSearchCV (iterations={args.rf_search_iterations}, cv={args.rf_cv_folds}, scoring={args.rf_scoring}, random_state={args.random_state})\n")
    else:
        print_and_save(f"RandomForest固定設定: n_estimators={args.n_estimators}, max_depth={args.max_depth}, random_state={args.random_state}\n")
    
    # 1. データ読み込み
    print_and_save("1. データ読み込み中...")
    
    # 訓練データ
    train_json_path = dataset_dir / "train.json"
    train_pred_json_path = sip_output_dir / "train_with_predictions.json"
    
    train_data = load_json_data(train_json_path)
    train_labels = extract_labels(train_data)
    train_qpp_scores = load_qpp_scores('train', qpp_output_dir, nsp_output_dir=nsp_output_dir, nsp_top_k=nsp_top_k)
    
    print_and_save(f"  - 訓練データ: {len(train_labels)} サンプル")
    
    # ベーススコアの読み込み（使用する場合のみ）
    if use_base_score:
        train_pred_data = load_json_data(train_pred_json_path)
        train_base_scores = extract_base_scores(train_pred_data, score_mode=args.plm_score_mode)
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
        dev_base_scores = extract_base_scores(dev_pred_data, score_mode=args.plm_score_mode)
        print_and_save(f"  - ベーススコア: {len(dev_base_scores)} サンプル")
    else:
        dev_base_scores = {}  # 空の辞書を渡す（使用しない）
    
    dev_qpp_scores = load_qpp_scores('dev', qpp_output_dir, nsp_output_dir=nsp_output_dir, nsp_top_k=nsp_top_k)
    
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
    if args.use_rf_search:
        model = RandomForestCVModel(
            cv=args.rf_cv_folds,
            scoring=args.rf_scoring,
            random_state=args.random_state,
            class_weight='balanced',
            n_jobs=-1,
            n_iter=args.rf_search_iterations,
        )
        model.fit(X_train_norm, y_train, print_and_save_func=print_and_save)
    else:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            class_weight='balanced',
            n_jobs=-1
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
    feature_names = list(X_train_norm.columns)
    plot_roc_curves(
        y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc, output_dir,
        hide_train=not args.show_train_curves,
        X_train=X_train_norm if args.show_single_metric_curves else None,
        X_test=X_test_norm if args.show_single_metric_curves else None,
        show_single_metrics=args.show_single_metric_curves,
        feature_names=feature_names
    )
    print_and_save("  - ROC曲線を保存しました")
    
    # 7. PR曲線の描画
    print_and_save("\n7. Precision-Recall曲線を描画中...")
    plot_pr_curves(
        y_train, y_train_proba, y_test, y_test_proba, train_ap, test_ap, output_dir,
        hide_train=not args.show_train_curves,
        X_train=X_train_norm if args.show_single_metric_curves else None,
        X_test=X_test_norm if args.show_single_metric_curves else None,
        show_single_metrics=args.show_single_metric_curves,
        feature_names=feature_names
    )
    print_and_save("  - PR曲線を保存しました")
    
    # 結果をファイルに保存（特徴量名に基づくディレクトリに保存）
    if feature_names is not None and len(feature_names) > 0:
        sorted_features = sorted(feature_names)
        safe_features = [f.replace(' ', '_').replace('/', '_') for f in sorted_features]
        feature_dir_name = 'features_' + '_'.join(safe_features)
        feature_output_dir = output_dir / feature_dir_name
        feature_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = feature_output_dir / "results.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_buffer.getvalue())
    print_and_save(f"\n結果をファイルに保存しました: {output_file}")
    
    print_and_save("\n=== 完了 ===")


if __name__ == "__main__":
    main()

