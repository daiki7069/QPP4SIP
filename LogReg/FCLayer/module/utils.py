"""
データ読み込み・前処理ユーティリティ関数
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from typing import Dict, List, Tuple, Any, Optional


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


def extract_base_scores(json_data: List[List[Dict[str, Any]]]) -> Dict[Tuple[str, int], float]:
    """ベースモデルのlogit_clarificationを抽出"""
    scores = {}
    for conversation in json_data:
        for turn in conversation:
            # conv_idとturn_idを文字列/整数に統一
            conv_id = str(turn['conv_id'])
            turn_id = int(turn['turn_id'])
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
        # 'nqc': ('nqc.csv', 'nqc'),
        'similarity': ('similarity.csv', 'mean_similarity'),
        # 'wig': ('wig.csv', 'wig'),
        # 'entropy': ('entropy.csv', 'entropy'),
        # 'lci': ('lci.csv', 'lci'),
        # 'unique_titles': ('unique_titles.csv', 'num_unique_titles'),
    }
    
    for metric_name, (filename, column_name) in qpp_configs.items():
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
    
    # データが空の場合はエラー
    if len(df) == 0:
        raise ValueError("No data found after merging features. Check if keys match between QPP scores and labels.")
    
    # labelカラムが存在するか確認
    if 'label' not in df.columns:
        raise ValueError(f"'label' column not found in DataFrame. Available columns: {df.columns.tolist()}")
    
    # 特徴量とラベルを分離
    # 理論的な特徴量カラム
    if use_base_score:
        theoretical_feature_columns = ['logit_clarification'] + list(qpp_scores.keys())
    else:
        theoretical_feature_columns = list(qpp_scores.keys())
    
    # 実際にDataFrameに存在する特徴量カラムのみを使用
    # conv_id, turn_id, labelは除外
    exclude_columns = {'conv_id', 'turn_id', 'label'}
    available_feature_columns = [col for col in df.columns 
                                 if col not in exclude_columns 
                                 and col in theoretical_feature_columns]
    
    if len(available_feature_columns) == 0:
        raise ValueError(f"No feature columns found. Available columns: {df.columns.tolist()}, "
                         f"Theoretical columns: {theoretical_feature_columns}")
    
    features_df = df[available_feature_columns].copy()
    labels_series = df['label'].copy()
    
    # NaN値のチェックと処理
    if features_df.isna().any().any():
        nan_counts = features_df.isna().sum()
        print("Warning: NaN values found in features:")
        print(nan_counts[nan_counts > 0])
        
        # NaN値を含む行を削除
        valid_mask = ~features_df.isna().any(axis=1)
        num_removed = len(features_df) - valid_mask.sum()
        
        if num_removed > 0:
            print(f"Removing {num_removed} rows with NaN values (keeping {valid_mask.sum()} rows)")
            features_df = features_df[valid_mask].copy()
            labels_series = labels_series[valid_mask].copy()
        
        # 削除後もNaNが残っている場合はエラー
        if features_df.isna().any().any():
            remaining_nan_counts = features_df.isna().sum()
            print("Error: NaN values still remain after removal:")
            print(remaining_nan_counts[remaining_nan_counts > 0])
            raise ValueError("NaN values are not allowed. Please check the data.")
    
    return features_df, labels_series


def balance_label_distribution(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_positive_rate: float,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    訓練データのラベル分布を調整して、指定された正例率に近づける
    
    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        target_positive_rate: 目標とする正例率（0.0-1.0）
        random_state: 乱数シード
    
    Returns:
        調整された訓練データ (X_train_balanced, y_train_balanced)
    """
    current_positive_rate = y_train.mean()
    
    # 既に目標の正例率に近い場合はそのまま返す
    if abs(current_positive_rate - target_positive_rate) < 0.01:
        return X_train.copy(), y_train.copy()
    
    # クラスごとにデータを分離
    positive_indices = y_train[y_train == 1].index
    negative_indices = y_train[y_train == 0].index
    
    n_positive = len(positive_indices)
    n_negative = len(negative_indices)
    
    # 目標の正例率に基づいて、必要なサンプル数を計算
    if target_positive_rate > current_positive_rate:
        # 正例を増やす必要がある場合
        target_n_positive = int(n_negative * target_positive_rate / (1 - target_positive_rate))
        target_n_negative = n_negative
    else:
        # 負例を増やす必要がある場合
        target_n_positive = n_positive
        target_n_negative = int(n_positive * (1 - target_positive_rate) / target_positive_rate)
    
    # サンプリング
    np.random.seed(random_state)
    
    if target_n_positive > n_positive:
        # 正例をオーバーサンプリング（復元抽出）
        positive_sampled_indices = np.random.choice(positive_indices, size=target_n_positive, replace=True)
    else:
        # 正例をアンダーサンプリング
        positive_sampled_indices = np.random.choice(positive_indices, size=target_n_positive, replace=False)
    
    if target_n_negative > n_negative:
        # 負例をオーバーサンプリング（復元抽出）
        negative_sampled_indices = np.random.choice(negative_indices, size=target_n_negative, replace=True)
    else:
        # 負例をアンダーサンプリング
        negative_sampled_indices = np.random.choice(negative_indices, size=target_n_negative, replace=False)
    
    # インデックスを結合してシャッフル
    all_indices = np.concatenate([positive_sampled_indices, negative_sampled_indices])
    np.random.shuffle(all_indices)
    
    # データを抽出
    X_train_balanced = X_train.loc[all_indices].reset_index(drop=True)
    y_train_balanced = y_train.loc[all_indices].reset_index(drop=True)
    
    return X_train_balanced, y_train_balanced


def normalize_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    use_combined_normalization: bool = False,
    use_separate_normalization: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    各特徴量を個別にz-score正規化
    
    Args:
        X_train: 訓練データの特徴量
        X_test: テストデータの特徴量
        use_combined_normalization: Trueの場合、訓練データとテストデータを結合してから正規化（リーク前提）
        use_separate_normalization: Trueの場合、訓練データとテストデータをそれぞれ個別に正規化（各々が平均0、標準偏差1になる）
    
    Returns:
        (X_train_norm, X_test_norm, scaler)
    """
    if use_separate_normalization:
        # 個別正規化：訓練データとテストデータをそれぞれ個別に正規化
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
        
        scalers_train = {}
        scalers_test = {}
        for column in X_train.columns:
            # 訓練データを個別に正規化
            scaler_train = StandardScaler()
            X_train_norm[column] = scaler_train.fit_transform(X_train[[column]]).flatten()
            scalers_train[column] = scaler_train
            
            # テストデータを個別に正規化
            scaler_test = StandardScaler()
            X_test_norm[column] = scaler_test.fit_transform(X_test[[column]]).flatten()
            scalers_test[column] = scaler_test
        
        return X_train_norm, X_test_norm, None
    elif use_combined_normalization:
        # リーク前提：訓練データとテストデータを結合してから正規化
        X_combined = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        X_combined_norm = X_combined.copy()
        
        scalers = {}
        for column in X_train.columns:
            scaler = StandardScaler()
            X_combined_norm[column] = scaler.fit_transform(X_combined[[column]]).flatten()
            scalers[column] = scaler
        
        # 訓練データとテストデータに分割
        n_train = len(X_train)
        X_train_norm = X_combined_norm.iloc[:n_train].reset_index(drop=True)
        X_test_norm = X_combined_norm.iloc[n_train:].reset_index(drop=True)
        
        return X_train_norm, X_test_norm, None
    else:
        # 通常の方法：訓練データの統計量でテストデータも正規化
        X_train_norm = X_train.copy()
        X_test_norm = X_test.copy()
        
        scalers = {}
        for column in X_train.columns:
            scaler = StandardScaler()
            X_train_norm[column] = scaler.fit_transform(X_train[[column]]).flatten()
            X_test_norm[column] = scaler.transform(X_test[[column]]).flatten()
            scalers[column] = scaler
        
        return X_train_norm, X_test_norm, scalers


def get_feature_dir_name(feature_names: Optional[List[str]]) -> str:
    """
    特徴量名のリストからディレクトリ名を生成
    
    Args:
        feature_names: 特徴量名のリスト（Noneの場合は'all'を返す）
    
    Returns:
        ディレクトリ名（例: 'features_similarity', 'features_logit_clarification_similarity'）
    """
    if feature_names is None or len(feature_names) == 0:
        return 'features_all'
    
    # 特徴量名をソートして一貫性を保つ
    sorted_features = sorted(feature_names)
    # ディレクトリ名に使えない文字を置換
    safe_features = [f.replace(' ', '_').replace('/', '_') for f in sorted_features]
    return 'features_' + '_'.join(safe_features)


def plot_roc_curves(
    y_train: pd.Series,
    y_train_proba: np.ndarray,
    y_test: pd.Series,
    y_test_proba: np.ndarray,
    train_auc: float,
    test_auc: float,
    output_dir: Path,
    hide_train: bool = False,
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    show_single_metrics: bool = False,
    feature_names: Optional[List[str]] = None
) -> None:
    """ROC曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
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
            # 訓練データ
            train_scores = X_train[feature_name].values
            if not hide_train:
                fpr_train_single, tpr_train_single, _ = roc_curve(y_train, train_scores)
                train_auc_single = roc_auc_score(y_train, train_scores)
                plt.plot(fpr_train_single, tpr_train_single, 
                        label=f'{feature_name} (Train, AUC = {train_auc_single:.4f})', 
                        linewidth=1.5, linestyle='--', alpha=0.6)
            
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
    title = 'ROC Curves - Model and Single Metrics' if show_single_metrics else 'ROC Curves - Fully Connected Layer'
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
    hide_train: bool = False,
    X_train: pd.DataFrame = None,
    X_test: pd.DataFrame = None,
    show_single_metrics: bool = False,
    feature_names: Optional[List[str]] = None
) -> None:
    """Precision-Recall曲線を描画して保存"""
    # 特徴量名からディレクトリ名を生成
    feature_dir_name = get_feature_dir_name(feature_names)
    feature_output_dir = output_dir / feature_dir_name
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
            # 訓練データ
            train_scores = X_train[feature_name].values
            if not hide_train:
                precision_train_single, recall_train_single, _ = precision_recall_curve(y_train, train_scores)
                train_ap_single = average_precision_score(y_train, train_scores)
                plt.plot(recall_train_single, precision_train_single, 
                        label=f'{feature_name} (Train, AP = {train_ap_single:.4f})', 
                        linewidth=1.5, linestyle='--', alpha=0.6)
            
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
    title = 'Precision-Recall Curves - Model and Single Metrics' if show_single_metrics else 'Precision-Recall Curves - Fully Connected Layer'
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9 if show_single_metrics else 11, ncol=1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存
    output_path = feature_output_dir / 'pr_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  - 保存先: {output_path}")
    plt.close()

