"""
特徴量選択に関するモジュール
BOLASSO、LARS-Trapsなどの特徴量選択手法を実装
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LassoLars
from collections import Counter
import warnings


def bootstrap_sample(X, y, random_state):
    """ブートストラップサンプルを作成"""
    np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    if isinstance(X, pd.DataFrame):
        return X.iloc[indices].reset_index(drop=True), y.iloc[indices].reset_index(drop=True)
    else:
        return X[indices], y[indices]


def bolasso_feature_selection(X_train, y_train, n_bootstrap=100, C=1.0, selection_threshold=0.5, 
                              random_state=42, print_and_save_func=None):
    """BOLASSOによる特徴量選択"""
    if print_and_save_func:
        print_and_save_func(f"\n  - ブートストラップサンプル数: {n_bootstrap}")
        print_and_save_func(f"  - 特徴量選択閾値: {selection_threshold:.2f} ({selection_threshold*100:.0f}%以上のモデルで選択)")
    
    feature_selection_counts = Counter()
    all_features = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else list(range(X_train.shape[1]))
    
    # 各ブートストラップサンプルでLASSOを実行
    for i in range(n_bootstrap):
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
        coefs = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        selected_features = [all_features[j] for j, coef in enumerate(coefs) if abs(coef) > 1e-6]
        feature_selection_counts.update(selected_features)
        
        if print_and_save_func and (i + 1) % 10 == 0:
            print_and_save_func(f"    - {i + 1}/{n_bootstrap} サンプル完了")
    
    # 選択頻度を計算
    selection_frequencies = {feature: count / n_bootstrap for feature, count in feature_selection_counts.items()}
    
    # 閾値以上の特徴量を選択
    selected_features = [feature for feature, freq in selection_frequencies.items() if freq >= selection_threshold]
    
    if print_and_save_func:
        print_and_save_func(f"\n  特徴量選択結果:")
        print_and_save_func(f"    - 選択された特徴量: {len(selected_features)}/{len(all_features)}")
        print_and_save_func(f"    - 選択頻度:")
        for feature in sorted(selection_frequencies.keys(), key=lambda x: selection_frequencies[x], reverse=True):
            freq = selection_frequencies[feature]
            status = "✓" if freq >= selection_threshold else "✗"
            print_and_save_func(f"      {status} {feature}: {freq:.2%} ({feature_selection_counts[feature]}/{n_bootstrap})")
    
    return selected_features


def lars_traps_feature_selection(X_train, y_train, n_random_traps=10, random_state=42, print_and_save_func=None):
    """LARS-Traps: ランダムに予測子を追加してパスを探索"""
    if print_and_save_func:
        print_and_save_func(f"\n  - ランダムトラップ数: {n_random_traps}")
    
    np.random.seed(random_state)
    n_features = X_train.shape[1]
    feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else list(range(n_features))
    
    # 各ランダムトラップでLARSを実行
    selected_features_all = []
    
    for trap_idx in range(n_random_traps):
        # ランダムに特徴量の順序をシャッフル
        feature_order = np.random.permutation(n_features)
        
        # 順次特徴量を追加しながらLARSを実行
        X_array = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_array = y_train.values if isinstance(y_train, pd.Series) else y_train
        
        # ランダムな順序で特徴量を追加
        X_ordered = X_array[:, feature_order]
        
        # LARSを実行
        # LassoLarsはrandom_stateパラメータを持たないため、numpyのランダムシードを設定
        np.random.seed(random_state + trap_idx)
        lars = LassoLars(alpha=0.1, max_iter=1000)
        lars.fit(X_ordered, y_array)
        
        # 選択された特徴量（係数が0でない特徴量）
        selected_indices = np.where(np.abs(lars.coef_) > 1e-6)[0]
        selected_features = [feature_names[feature_order[idx]] for idx in selected_indices]
        selected_features_all.extend(selected_features)
        
        if print_and_save_func and (trap_idx + 1) % 5 == 0:
            print_and_save_func(f"    - {trap_idx + 1}/{n_random_traps} トラップ完了")
    
    # 最も頻繁に選択された特徴量を返す
    feature_counts = Counter(selected_features_all)
    # 50%以上のトラップで選択された特徴量を返す
    threshold = n_random_traps * 0.5
    selected_features = [feature for feature, count in feature_counts.items() if count >= threshold]
    
    if print_and_save_func:
        print_and_save_func(f"\n  特徴量選択結果:")
        print_and_save_func(f"    - 選択された特徴量: {len(selected_features)}/{n_features}")
        print_and_save_func(f"    - 選択頻度:")
        for feature in sorted(feature_counts.keys(), key=lambda x: feature_counts[x], reverse=True):
            count = feature_counts[feature]
            status = "✓" if count >= threshold else "✗"
            print_and_save_func(f"      {status} {feature}: {count}/{n_random_traps} ({count/n_random_traps:.2%})")
    
    return selected_features if len(selected_features) > 0 else feature_names

