"""
BOLASSO特有の関数
"""
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from .preprocessing import normalize_features


def bootstrap_sample(X: pd.DataFrame, y: pd.Series, random_state: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    ブートストラップサンプルを作成
    元のデータサイズと同じサイズのサンプルを復元抽出で作成
    """
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)


def bolasso_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_bootstrap: int,
    C: float,
    selection_threshold: float,
    random_state: int,
    print_and_save_func
) -> List[str]:
    """
    BOLASSOによる特徴量選択
    複数のブートストラップサンプルでLASSOを実行し、選択頻度が閾値以上の特徴量を返す
    
    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        n_bootstrap: ブートストラップサンプルの数
        C: LASSOの正則化パラメータ
        selection_threshold: 特徴量選択の閾値（0.0-1.0、例：0.5なら50%以上のモデルで選択された特徴量）
        random_state: 乱数シード
        print_and_save_func: 出力用の関数
    
    Returns:
        選択された特徴量名のリスト
    """
    print_and_save_func(f"\n  - ブートストラップサンプル数: {n_bootstrap}")
    print_and_save_func(f"  - 特徴量選択閾値: {selection_threshold:.2f} ({selection_threshold*100:.0f}%以上のモデルで選択)")
    
    feature_selection_counts = Counter()
    all_features = set(X_train.columns)
    
    # 各ブートストラップサンプルでLASSOを実行
    for i in range(n_bootstrap):
        # ブートストラップサンプルを作成
        X_boot, y_boot = bootstrap_sample(X_train, y_train, random_state=random_state + i)
        
        # LASSOモデルを学習
        model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=C,
            max_iter=1000,
            random_state=random_state + i,
            class_weight='balanced'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_boot, y_boot)
        
        # 選択された特徴量（係数が0でない特徴量）をカウント
        selected_features = [feature for feature, coef in zip(X_train.columns, model.coef_[0]) if abs(coef) > 1e-6]
        feature_selection_counts.update(selected_features)
        
        if (i + 1) % 10 == 0:
            print_and_save_func(f"    - {i + 1}/{n_bootstrap} サンプル完了")
    
    # 選択頻度を計算
    selection_frequencies = {feature: count / n_bootstrap for feature, count in feature_selection_counts.items()}
    
    # 閾値以上の特徴量を選択
    selected_features = [feature for feature, freq in selection_frequencies.items() if freq >= selection_threshold]
    
    # 選択されなかった特徴量も表示
    unselected_features = [feature for feature in all_features if feature not in selection_frequencies or selection_frequencies[feature] < selection_threshold]
    
    print_and_save_func(f"\n  特徴量選択結果:")
    print_and_save_func(f"    - 選択された特徴量: {len(selected_features)}/{len(all_features)}")
    print_and_save_func(f"    - 選択頻度:")
    for feature in sorted(selection_frequencies.keys(), key=lambda x: selection_frequencies[x], reverse=True):
        freq = selection_frequencies[feature]
        status = "✓" if freq >= selection_threshold else "✗"
        print_and_save_func(f"      {status} {feature}: {freq:.2%} ({feature_selection_counts[feature]}/{n_bootstrap})")
    
    if unselected_features:
        print_and_save_func(f"    - 選択されなかった特徴量: {len(unselected_features)}個")
        for feature in unselected_features:
            freq = selection_frequencies.get(feature, 0.0)
            print_and_save_func(f"      ✗ {feature}: {freq:.2%} ({feature_selection_counts.get(feature, 0)}/{n_bootstrap})")
    
    return selected_features


def cross_validate_bolasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_folds: int,
    n_bootstrap: int,
    C: float,
    selection_threshold: float,
    random_state: int,
    print_and_save_func
) -> Tuple[List[str], Dict[str, float]]:
    """
    交差検証を使用したBOLASSO特徴量選択
    
    Args:
        X_train: 訓練データの特徴量
        y_train: 訓練データのラベル
        n_folds: 交差検証のフォールド数
        n_bootstrap: 各フォールドでのブートストラップサンプル数
        C: LASSOの正則化パラメータ
        selection_threshold: 特徴量選択の閾値
        random_state: 乱数シード
        print_and_save_func: 出力用の関数
    
    Returns:
        (選択された特徴量のリスト, 各フォールドの評価結果)
    """
    print_and_save_func(f"\n  - 交差検証フォールド数: {n_folds}")
    print_and_save_func(f"  - 各フォールドのブートストラップサンプル数: {n_bootstrap}")
    
    # StratifiedKFoldを使用（クラス不均衡に対応）
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    fold_feature_selections = []
    fold_results = {}
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print_and_save_func(f"\n  Fold {fold_idx + 1}/{n_folds}:")
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # このフォールドで特徴量選択
        selected_features = bolasso_feature_selection(
            X_fold_train, y_fold_train, n_bootstrap, C, selection_threshold,
            random_state + fold_idx * 1000, print_and_save_func
        )
        fold_feature_selections.append(set(selected_features))
        
        # 選択された特徴量でモデルを学習して評価
        if len(selected_features) > 0:
            X_fold_train_selected = X_fold_train[selected_features]
            X_fold_val_selected = X_fold_val[selected_features]
            
            # 正規化
            X_fold_train_norm, X_fold_val_norm, _ = normalize_features(X_fold_train_selected, X_fold_val_selected)
            
            # モデル学習
            model = LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=C,
                max_iter=1000,
                random_state=random_state + fold_idx,
                class_weight='balanced'
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_fold_train_norm, y_fold_train)
            
            # 評価
            y_val_pred = model.predict(X_fold_val_norm)
            y_val_proba = model.predict_proba(X_fold_val_norm)[:, 1]
            
            fold_results[f'fold_{fold_idx + 1}'] = {
                'accuracy': accuracy_score(y_fold_val, y_val_pred),
                'f1': f1_score(y_fold_val, y_val_pred),
                'auc': roc_auc_score(y_fold_val, y_val_proba),
                'ap': average_precision_score(y_fold_val, y_val_proba),
                'n_features': len(selected_features)
            }
            
            print_and_save_func(f"    - Accuracy: {fold_results[f'fold_{fold_idx + 1}']['accuracy']:.4f}")
            print_and_save_func(f"    - F1 Score: {fold_results[f'fold_{fold_idx + 1}']['f1']:.4f}")
            print_and_save_func(f"    - AUC-ROC: {fold_results[f'fold_{fold_idx + 1}']['auc']:.4f}")
            print_and_save_func(f"    - Average Precision: {fold_results[f'fold_{fold_idx + 1}']['ap']:.4f}")
        else:
            print_and_save_func(f"    ⚠️  警告: このフォールドで選択された特徴量がありません")
            fold_results[f'fold_{fold_idx + 1}'] = {
                'accuracy': 0.0,
                'f1': 0.0,
                'auc': 0.5,
                'ap': y_fold_val.mean(),
                'n_features': 0
            }
    
    # 全フォールドで選択された特徴量の積集合または和集合を計算
    all_selected_features = set.intersection(*fold_feature_selections) if fold_feature_selections else set()
    
    # より緩い基準: 50%以上のフォールドで選択された特徴量
    feature_fold_counts = Counter()
    for fold_features in fold_feature_selections:
        feature_fold_counts.update(fold_features)
    
    threshold_folds = max(1, int(n_folds * selection_threshold))
    relaxed_selected_features = [feature for feature, count in feature_fold_counts.items() 
                                 if count >= threshold_folds]
    
    print_and_save_func(f"\n  交差検証結果の集約:")
    print_and_save_func(f"    - 全フォールドで選択された特徴量（積集合）: {len(all_selected_features)}個")
    print_and_save_func(f"    - {threshold_folds}フォールド以上で選択された特徴量: {len(relaxed_selected_features)}個")
    
    # デフォルトは緩い基準を使用
    final_selected_features = relaxed_selected_features if len(relaxed_selected_features) > 0 else list(all_selected_features)
    
    # 各フォールドの平均評価
    if fold_results:
        avg_results = {
            'accuracy': np.mean([r['accuracy'] for r in fold_results.values()]),
            'f1': np.mean([r['f1'] for r in fold_results.values()]),
            'auc': np.mean([r['auc'] for r in fold_results.values()]),
            'ap': np.mean([r['ap'] for r in fold_results.values()]),
            'n_features': np.mean([r['n_features'] for r in fold_results.values()])
        }
        print_and_save_func(f"\n  交差検証の平均結果:")
        print_and_save_func(f"    - Accuracy: {avg_results['accuracy']:.4f}")
        print_and_save_func(f"    - F1 Score: {avg_results['f1']:.4f}")
        print_and_save_func(f"    - AUC-ROC: {avg_results['auc']:.4f}")
        print_and_save_func(f"    - Average Precision: {avg_results['ap']:.4f}")
        print_and_save_func(f"    - 平均特徴量数: {avg_results['n_features']:.1f}")
    
    return final_selected_features, fold_results

