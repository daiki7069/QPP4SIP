"""
ブートストラップ評価用のモジュール
対応のあるブートストラップ（paired bootstrap）とDeLongの検定を実装
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, roc_curve
import pickle
import json
from scipy import stats


def generate_bootstrap_samples(n_samples: int, n_iterations: int, random_state: int = 42) -> List[np.ndarray]:
    """
    ブートストラップサンプル（インデックス）を生成
    
    Args:
        n_samples: 元のサンプル数
        n_iterations: ブートストラップの反復回数
        random_state: 乱数シード
    
    Returns:
        各ブートストラップサンプルのインデックス配列のリスト
    """
    np.random.seed(random_state)
    bootstrap_samples = []
    for i in range(n_iterations):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_samples.append(indices)
    return bootstrap_samples


def save_bootstrap_samples(bootstrap_samples: List[np.ndarray], output_path: Path):
    """
    ブートストラップサンプル（インデックス）を保存
    
    Args:
        bootstrap_samples: ブートストラップサンプルのリスト
        output_path: 保存先パス
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(bootstrap_samples, f)


def load_bootstrap_samples(input_path: Path) -> List[np.ndarray]:
    """
    ブートストラップサンプル（インデックス）を読み込む
    
    Args:
        input_path: 読み込み元パス
    
    Returns:
        ブートストラップサンプルのリスト
    """
    with open(input_path, 'rb') as f:
        return pickle.load(f)


def evaluate_bootstrap(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    bootstrap_samples: List[np.ndarray],
    feature_combination_name: str
) -> pd.DataFrame:
    """
    ブートストラップサンプルごとに評価指標を計算
    
    Args:
        y_true: 真のラベル
        y_pred_proba: 予測確率
        bootstrap_samples: ブートストラップサンプルのリスト
        feature_combination_name: 特徴量組み合わせの名前（ディレクトリ名など）
    
    Returns:
        DataFrame (bootstrap_id, feature_combination, metric_name, value)
    """
    results = []
    
    for bootstrap_id, indices in enumerate(bootstrap_samples):
        # ブートストラップサンプルで評価
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        y_pred_boot = (y_pred_proba_boot >= 0.5).astype(int)
        
        # 各評価指標を計算
        try:
            auc = roc_auc_score(y_true_boot, y_pred_proba_boot)
        except ValueError:
            auc = np.nan
        
        try:
            ap = average_precision_score(y_true_boot, y_pred_proba_boot)
        except ValueError:
            ap = np.nan
        
        f1 = f1_score(y_true_boot, y_pred_boot)
        accuracy = accuracy_score(y_true_boot, y_pred_boot)
        
        # 結果を保存
        for metric_name, value in [
            ('auc', auc),
            ('ap', ap),
            ('f1', f1),
            ('accuracy', accuracy)
        ]:
            results.append({
                'bootstrap_id': bootstrap_id,
                'feature_combination': feature_combination_name,
                'metric_name': metric_name,
                'value': value
            })
    
    return pd.DataFrame(results)


def save_bootstrap_results(bootstrap_results: pd.DataFrame, output_path: Path):
    """
    ブートストラップ結果をCSV形式で保存
    
    Args:
        bootstrap_results: ブートストラップ結果のDataFrame
        output_path: 保存先パス
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_results.to_csv(output_path, index=False)


def load_bootstrap_results(input_path: Path) -> pd.DataFrame:
    """
    ブートストラップ結果を読み込む
    
    Args:
        input_path: 読み込み元パス
    
    Returns:
        ブートストラップ結果のDataFrame
    """
    return pd.read_csv(input_path)


def delong_test(
    y_true: np.ndarray,
    y_pred_proba_a: np.ndarray,
    y_pred_proba_b: np.ndarray
) -> Dict:
    """
    DeLongの検定: 2つのROC曲線のAUCを比較する統計的検定
    
    Args:
        y_true: 真のラベル
        y_pred_proba_a: モデルAの予測確率
        y_pred_proba_b: モデルBの予測確率
    
    Returns:
        検定結果の辞書（auc_a, auc_b, z_stat, p_value, significant）
    
    Reference:
        DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988).
        Comparing the areas under two or more correlated receiver operating
        characteristic curves: a nonparametric approach. Biometrics, 44(3), 837-845.
    """
    # AUCを計算
    try:
        auc_a = roc_auc_score(y_true, y_pred_proba_a)
    except ValueError:
        return {
            'auc_a': np.nan,
            'auc_b': np.nan,
            'z_stat': np.nan,
            'p_value': np.nan,
            'significant': False,
            'error': 'Cannot compute AUC for model A'
        }
    
    try:
        auc_b = roc_auc_score(y_true, y_pred_proba_b)
    except ValueError:
        return {
            'auc_a': auc_a,
            'auc_b': np.nan,
            'z_stat': np.nan,
            'p_value': np.nan,
            'significant': False,
            'error': 'Cannot compute AUC for model B'
        }
    
    # DeLongの検定統計量を計算
    n = len(y_true)
    n_pos = np.sum(y_true == 1)
    n_neg = n - n_pos
    
    if n_pos == 0 or n_neg == 0:
        return {
            'auc_a': auc_a,
            'auc_b': auc_b,
            'z_stat': np.nan,
            'p_value': np.nan,
            'significant': False,
            'error': 'No positive or negative samples'
        }
    
    # V10, V01を計算（DeLong et al., 1988の式）
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    # 効率的な計算: 各正例に対する負例の比較結果を一度に計算
    # V10: 正例iについて、負例の予測確率が正例iの予測確率より大きい割合
    v10_a_list = []
    v10_b_list = []
    for pos_i in pos_indices:
        v10_a_list.append(np.mean(y_pred_proba_a[neg_indices] > y_pred_proba_a[pos_i]))
        v10_b_list.append(np.mean(y_pred_proba_b[neg_indices] > y_pred_proba_b[pos_i]))
    v10_a = np.mean(v10_a_list)
    v10_b = np.mean(v10_b_list)
    
    # V01: 負例jについて、正例の予測確率が負例jの予測確率より大きい割合
    v01_a_list = []
    v01_b_list = []
    for neg_i in neg_indices:
        v01_a_list.append(np.mean(y_pred_proba_a[pos_indices] > y_pred_proba_a[neg_i]))
        v01_b_list.append(np.mean(y_pred_proba_b[pos_indices] > y_pred_proba_b[neg_i]))
    v01_a = np.mean(v01_a_list)
    v01_b = np.mean(v01_b_list)
    
    # AUCの分散共分散行列の要素を計算
    s10_a = (1.0 / (n_pos - 1)) * np.sum([(v10_a_list[i] - v10_a) ** 2 for i in range(len(v10_a_list))])
    s10_b = (1.0 / (n_pos - 1)) * np.sum([(v10_b_list[i] - v10_b) ** 2 for i in range(len(v10_b_list))])
    
    s01_a = (1.0 / (n_neg - 1)) * np.sum([(v01_a_list[i] - v01_a) ** 2 for i in range(len(v01_a_list))])
    s01_b = (1.0 / (n_neg - 1)) * np.sum([(v01_b_list[i] - v01_b) ** 2 for i in range(len(v01_b_list))])
    
    # 共分散を計算
    s10_ab = (1.0 / (n_pos - 1)) * np.sum([
        (v10_a_list[i] - v10_a) * (v10_b_list[i] - v10_b)
        for i in range(len(v10_a_list))
    ])
    
    s01_ab = (1.0 / (n_neg - 1)) * np.sum([
        (v01_a_list[i] - v01_a) * (v01_b_list[i] - v01_b)
        for i in range(len(v01_a_list))
    ])
    
    # AUCの分散
    var_a = s10_a / n_pos + s01_a / n_neg
    var_b = s10_b / n_pos + s01_b / n_neg
    
    # AUCの差の分散
    var_diff = var_a + var_b - 2 * (s10_ab / n_pos + s01_ab / n_neg)
    
    # Z統計量
    if var_diff > 0:
        z_stat = (auc_a - auc_b) / np.sqrt(var_diff)
        # 両側検定のp値
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = 0.0
        p_value = 1.0
    
    significant = p_value < 0.05
    
    return {
        'auc_a': auc_a,
        'auc_b': auc_b,
        'auc_diff': auc_a - auc_b,
        'z_stat': z_stat,
        'p_value': p_value,
        'significant': significant,
        'var_a': var_a,
        'var_b': var_b,
        'var_diff': var_diff
    }


def compare_feature_combinations(
    bootstrap_results: pd.DataFrame,
    combination_a: str,
    combination_b: str,
    metric_name: str = 'auc',
    alpha: float = 0.05
) -> Dict:
    """
    2つの特徴量組み合わせ間で優位性検定を行う（ブートストラップベース）
    """
    # 各組み合わせの結果を抽出し、bootstrap_idでソートして対応を保証
    results_a_df = bootstrap_results[
        (bootstrap_results['feature_combination'] == combination_a) &
        (bootstrap_results['metric_name'] == metric_name)
    ].sort_values('bootstrap_id')
    
    results_b_df = bootstrap_results[
        (bootstrap_results['feature_combination'] == combination_b) &
        (bootstrap_results['metric_name'] == metric_name)
    ].sort_values('bootstrap_id')
    
    # bootstrap_idが一致することを確認
    if not np.array_equal(results_a_df['bootstrap_id'].values, results_b_df['bootstrap_id'].values):
        raise ValueError("bootstrap_idが一致しません。対応のあるブートストラップが正しく実行されていません。")
    
    results_a = results_a_df['value'].values
    results_b = results_b_df['value'].values
    
    # 差を計算（対応のあるブートストラップ）
    differences = results_a - results_b
    
    # 統計量を計算
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)
    
    # 信頼区間（パーセンタイル法）
    ci_lower = np.percentile(differences, alpha / 2 * 100)
    ci_upper = np.percentile(differences, (1 - alpha / 2) * 100)
    
    # p値（両側検定: H0: mean_diff = 0）
    # ブートストラップ分布から直接p値を計算（パーセンタイル法）
    # 0より大きい差の割合と0より小さい差の割合の小さい方を2倍
    p_value = 2 * min(
        np.mean(differences > 0),
        np.mean(differences < 0)
    )
    # 差が0の場合の処理
    if p_value == 0:
        p_value = 1.0 / n  # 最小のp値
    
    significant = p_value < alpha
    
    return {
        'combination_a': combination_a,
        'combination_b': combination_b,
        'metric_name': metric_name,
        'mean_a': np.mean(results_a),
        'mean_b': np.mean(results_b),
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': significant,
        'n_bootstrap': n
    }

