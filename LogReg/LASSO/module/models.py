"""
モデル作成と管理に関するモジュール
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LassoLarsCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from scipy.optimize import minimize
from scipy.special import expit
import warnings
from collections import Counter
from .feature_selection import bolasso_feature_selection, lars_traps_feature_selection


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
        bounds = [(0, None) for _ in range(n_features)] + [(None, None)]
        
        # 最適化
        result = minimize(
            self._logistic_loss,
            initial_params,
            args=(X.values if isinstance(X, pd.DataFrame) else X, 
                  y.values if isinstance(y, pd.Series) else y, 
                  sample_weights),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )
        
        self.optimization_result_ = result
        self.coef_ = np.array([result.x[:n_features]])
        self.intercept_ = np.array([result.x[n_features]])
        self.n_iter_ = np.array([result.nit])
        
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


class BOLASSOModel:
    """BOLASSOモデル（特徴量選択 + ロジスティック回帰）"""
    def __init__(self, n_bootstrap=100, C=1.0, selection_threshold=0.5, random_state=42, 
                 max_iter=1000, class_weight='balanced', tol=1e-8):
        self.n_bootstrap = n_bootstrap
        self.C = C
        self.selection_threshold = selection_threshold
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.tol = tol
        self.selected_features_ = None
        self.model_ = None
        
    def fit(self, X, y, print_and_save_func=None):
        # BOLASSOで特徴量選択
        self.selected_features_ = bolasso_feature_selection(
            X, y, self.n_bootstrap, self.C, self.selection_threshold,
            self.random_state, print_and_save_func
        )
        
        # 選択された特徴量でロジスティック回帰を学習
        if isinstance(X, pd.DataFrame):
            X_selected = X[self.selected_features_]
        else:
            X_selected = X[:, [list(X.columns).index(f) if isinstance(X, pd.DataFrame) else f for f in self.selected_features_]]
        
        self.model_ = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            tol=self.tol
        )
        self.model_.fit(X_selected, y)
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(self._get_selected_features(X))
    
    def predict(self, X):
        return self.model_.predict(self._get_selected_features(X))
    
    def _get_selected_features(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X[:, [list(X.columns).index(f) if isinstance(X, pd.DataFrame) else f for f in self.selected_features_]]


class LARSTrapsModel:
    """LARS-Trapsモデル（特徴量選択 + ロジスティック回帰）"""
    def __init__(self, n_random_traps=10, random_state=42, max_iter=1000, 
                 class_weight='balanced', tol=1e-8):
        self.n_random_traps = n_random_traps
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.tol = tol
        self.selected_features_ = None
        self.model_ = None
        
    def fit(self, X, y, print_and_save_func=None):
        # LARS-Trapsで特徴量選択
        self.selected_features_ = lars_traps_feature_selection(
            X, y, self.n_random_traps, self.random_state, print_and_save_func
        )
        
        # 選択された特徴量でロジスティック回帰を学習
        if isinstance(X, pd.DataFrame):
            X_selected = X[self.selected_features_]
        else:
            X_selected = X[:, [list(X.columns).index(f) if isinstance(X, pd.DataFrame) else f for f in self.selected_features_]]
        
        self.model_ = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            tol=self.tol
        )
        self.model_.fit(X_selected, y)
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(self._get_selected_features(X))
    
    def predict(self, X):
        return self.model_.predict(self._get_selected_features(X))
    
    def _get_selected_features(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            return X[:, [list(X.columns).index(f) if isinstance(X, pd.DataFrame) else f for f in self.selected_features_]]


class L1CVModel:
    """L1-CVモデル（クロスバリデーションでCを最適化）"""
    def __init__(self, cv=5, random_state=42, max_iter=1000, class_weight='balanced', tol=1e-8,
                 C_range=None, scoring='roc_auc', n_jobs=-1):
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.tol = tol
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None
        
        # デフォルトのパラメータ範囲
        if C_range is None:
            # Cの範囲: 0.01から100まで対数スケールで10個
            self.C_range = np.logspace(-2, 2, 10)
        else:
            self.C_range = C_range
    
    def fit(self, X, y, print_and_save_func=None):
        """GridSearchCVでCを最適化してモデルを学習"""
        # パラメータグリッド
        param_grid = {
            'C': self.C_range
        }
        
        # ベースモデル
        base_model = LogisticRegression(
            penalty='l1',
            solver='liblinear',
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            tol=self.tol
        )
        
        # StratifiedKFoldでCV
        cv_fold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_fold,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # 学習
        grid_search.fit(X, y)
        
        # 最適なパラメータを保存
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.model_ = grid_search.best_estimator_
        
        if print_and_save_func:
            print_and_save_func(f"\n  - L1-CVで最適化されたパラメータ:")
            print_and_save_func(f"    C: {self.best_params_['C']:.4f}")
            print_and_save_func(f"    最適化スコア ({self.scoring}): {self.best_score_:.4f}")
        
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def predict(self, X):
        return self.model_.predict(X)


class L2CVModel:
    """L2-CVモデル（クロスバリデーションでCを最適化）"""
    def __init__(self, cv=5, random_state=42, max_iter=1000, class_weight='balanced', tol=1e-8,
                 C_range=None, scoring='roc_auc', n_jobs=-1):
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.tol = tol
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None
        
        # デフォルトのパラメータ範囲
        if C_range is None:
            # Cの範囲: 0.01から100まで対数スケールで10個
            self.C_range = np.logspace(-2, 2, 10)
        else:
            self.C_range = C_range
    
    def fit(self, X, y, print_and_save_func=None):
        """GridSearchCVでCを最適化してモデルを学習"""
        # パラメータグリッド
        param_grid = {
            'C': self.C_range
        }
        
        # ベースモデル
        base_model = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            tol=self.tol
        )
        
        # StratifiedKFoldでCV
        cv_fold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_fold,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # 学習
        grid_search.fit(X, y)
        
        # 最適なパラメータを保存
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.model_ = grid_search.best_estimator_
        
        if print_and_save_func:
            print_and_save_func(f"\n  - L2-CVで最適化されたパラメータ:")
            print_and_save_func(f"    C: {self.best_params_['C']:.4f}")
            print_and_save_func(f"    最適化スコア ({self.scoring}): {self.best_score_:.4f}")
        
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def predict(self, X):
        return self.model_.predict(X)


class ElasticNetCVModel:
    """ElasticNet-CVモデル（クロスバリデーションでCとl1_ratioを最適化）"""
    def __init__(self, cv=5, random_state=42, max_iter=1000, class_weight='balanced', tol=1e-8,
                 C_range=None, l1_ratio_range=None, scoring='roc_auc', n_jobs=-1):
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.tol = tol
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None
        
        # デフォルトのパラメータ範囲
        if C_range is None:
            # Cの範囲: 0.01から100まで対数スケールで10個
            self.C_range = np.logspace(-2, 2, 10)
        else:
            self.C_range = C_range
            
        if l1_ratio_range is None:
            # l1_ratioの範囲: 0.1から0.9まで0.1刻み
            self.l1_ratio_range = np.arange(0.1, 1.0, 0.1)
        else:
            self.l1_ratio_range = l1_ratio_range
    
    def fit(self, X, y, print_and_save_func=None):
        """GridSearchCVでCとl1_ratioを最適化してモデルを学習"""
        # パラメータグリッド
        param_grid = {
            'C': self.C_range,
            'l1_ratio': self.l1_ratio_range
        }
        
        # ベースモデル
        base_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            max_iter=self.max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
            tol=self.tol
        )
        
        # StratifiedKFoldでCV
        cv_fold = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        
        # GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_fold,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # 学習
        grid_search.fit(X, y)
        
        # 最適なパラメータを保存
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.model_ = grid_search.best_estimator_
        
        if print_and_save_func:
            print_and_save_func(f"\n  - ElasticNet-CVで最適化されたパラメータ:")
            print_and_save_func(f"    C: {self.best_params_['C']:.4f}")
            print_and_save_func(f"    l1_ratio: {self.best_params_['l1_ratio']:.4f}")
            print_and_save_func(f"    最適化スコア ({self.scoring}): {self.best_score_:.4f}")
        
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(X)
    
    def predict(self, X):
        return self.model_.predict(X)


class LARSCVModel:
    """LARS-CVモデル（クロスバリデーションで正則化パラメータを選択）"""
    def __init__(self, cv=5, random_state=42, max_iter=1000, class_weight='balanced', tol=1e-8):
        self.cv = cv
        self.random_state = random_state
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.tol = tol
        self.lars_model_ = None
        self.model_ = None
        self.selected_features_ = None
        
    def fit(self, X, y, print_and_save_func=None):
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # LARS-CVで特徴量選択
        # LassoLarsCVはrandom_stateパラメータを持たないため、numpyのランダムシードを設定
        np.random.seed(self.random_state)
        self.lars_model_ = LassoLarsCV(cv=self.cv, max_iter=1000)
        self.lars_model_.fit(X_array, y_array)
        
        # 選択された特徴量（係数が0でない特徴量）
        coefs = self.lars_model_.coef_
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
        self.selected_features_ = [feature_names[i] for i in np.where(np.abs(coefs) > 1e-6)[0]]
        
        if print_and_save_func:
            print_and_save_func(f"\n  - LARS-CVで選択された特徴量: {len(self.selected_features_)}/{len(feature_names)}")
            print_and_save_func(f"  - 選択された特徴量: {self.selected_features_}")
        
        # 選択された特徴量でロジスティック回帰を学習
        if len(self.selected_features_) > 0:
            if isinstance(X, pd.DataFrame):
                X_selected = X[self.selected_features_]
            else:
                X_selected = X[:, [i for i, name in enumerate(feature_names) if name in self.selected_features_]]
            
            self.model_ = LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                tol=self.tol
            )
            self.model_.fit(X_selected, y)
        else:
            # 特徴量が選択されなかった場合は全ての特徴量を使用
            self.selected_features_ = feature_names
            self.model_ = LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                tol=self.tol
            )
            self.model_.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        return self.model_.predict_proba(self._get_selected_features(X))
    
    def predict(self, X):
        return self.model_.predict(self._get_selected_features(X))
    
    def _get_selected_features(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        else:
            feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else list(range(X.shape[1]))
            return X[:, [i for i, name in enumerate(feature_names) if name in self.selected_features_]]


class RandomForestCVModel:
    """層化CVでハイパーパラメータを選択するRandomForest分類器。"""

    def __init__(self, cv=5, scoring='roc_auc', random_state=42,
                 class_weight='balanced', n_jobs=-1, n_iter=32,
                 param_distributions=None):
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.param_distributions = param_distributions
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.best_train_score_ = None

    def _default_param_distributions(self):
        class_weights = [self.class_weight]
        if self.class_weight == 'balanced':
            class_weights.append('balanced_subsample')
        return {
            'n_estimators': [200, 500, 800],
            'max_depth': [3, 5, 8, 12, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'max_features': ['sqrt', 'log2', 0.5],
            'class_weight': class_weights,
        }

    def fit(self, X, y, print_and_save_func=None):
        labels, counts = np.unique(np.asarray(y), return_counts=True)
        if len(labels) < 2:
            raise ValueError("RandomForest tuning requires at least two classes")
        effective_cv = min(self.cv, int(counts.min()))
        if effective_cv < 2:
            raise ValueError("RandomForest tuning requires at least two samples in every class")

        param_distributions = self.param_distributions or self._default_param_distributions()
        total_candidates = int(np.prod([len(values) for values in param_distributions.values()]))
        n_iter = min(self.n_iter, total_candidates)
        cv = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=self.random_state)
        base_model = RandomForestClassifier(
            random_state=self.random_state,
            class_weight=self.class_weight,
            n_jobs=1,
        )
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=cv,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            refit=True,
            return_train_score=True,
            error_score='raise',
        )
        search.fit(X, y)

        self.search_ = search
        self.model_ = search.best_estimator_
        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = float(search.best_score_)
        self.best_train_score_ = float(search.cv_results_['mean_train_score'][search.best_index_])
        self.feature_importances_ = self.model_.feature_importances_
        self.classes_ = self.model_.classes_
        self.n_features_in_ = self.model_.n_features_in_
        if hasattr(self.model_, 'feature_names_in_'):
            self.feature_names_in_ = self.model_.feature_names_in_

        log = print_and_save_func or print
        log("RandomForest hyperparameter search completed:")
        log(f"  best params: {self.best_params_}")
        log(f"  best mean CV {self.scoring}: {self.best_score_:.4f}")
        log(f"  corresponding mean train score: {self.best_train_score_:.4f}")
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        return self.model_.score(X, y)


def create_model(model_type, max_iter=1000, random_state=42, class_weight='balanced', tol=1e-8, 
                 non_negative=False, n_bootstrap=100, selection_threshold=0.5, n_random_traps=10, cv=5,
                 use_l1_cv=False, use_l2_cv=False, use_elasticnet_cv=False, C_range=None, l1_ratio_range=None, scoring='roc_auc', n_jobs=-1):
    """モデルを作成する関数
    
    Args:
        use_l1_cv: L1の場合にCVで最適化するかどうか
        use_l2_cv: L2の場合にCVで最適化するかどうか
        use_elasticnet_cv: ElasticNetの場合にCVで最適化するかどうか
        C_range: CVで使用するCの範囲（Noneの場合はデフォルト）
        l1_ratio_range: ElasticNet-CVで使用するl1_ratioの範囲（Noneの場合はデフォルト）
        scoring: GridSearchCVで使用する評価指標
        n_jobs: GridSearchCVで使用する並列ジョブ数
    """
    if model_type == 'l1':
        if use_l1_cv:
            return L1CVModel(
                cv=cv,
                random_state=random_state,
                max_iter=max_iter,
                class_weight=class_weight,
                tol=tol,
                C_range=C_range,
                scoring=scoring,
                n_jobs=n_jobs
            )
        elif non_negative:
            return NonNegativeLogisticRegression(
                C=1.0,
                max_iter=max_iter,
                random_state=random_state,
                class_weight=class_weight,
                tol=tol
            )
        else:
            return LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=1.0,
                max_iter=max_iter,
                random_state=random_state,
                class_weight=class_weight,
                tol=tol
            )
    elif model_type == 'l2':
        if use_l2_cv:
            return L2CVModel(
                cv=cv,
                random_state=random_state,
                max_iter=max_iter,
                class_weight=class_weight,
                tol=tol,
                C_range=C_range,
                scoring=scoring,
                n_jobs=n_jobs
            )
        else:
            return LogisticRegression(
                penalty='l2',
                solver='lbfgs',
                C=1.0,
                max_iter=max_iter,
                random_state=random_state,
                class_weight=class_weight,
                tol=tol
            )
    elif model_type == 'elasticnet':
        if use_elasticnet_cv:
            return ElasticNetCVModel(
                cv=cv,
                random_state=random_state,
                max_iter=max_iter,
                class_weight=class_weight,
                tol=tol,
                C_range=C_range,
                l1_ratio_range=l1_ratio_range,
                scoring=scoring,
                n_jobs=n_jobs
            )
        else:
            return LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                C=1.0,
                l1_ratio=0.5,
                max_iter=max_iter,
                random_state=random_state,
                class_weight=class_weight,
                tol=tol
            )
    elif model_type == 'randomforest':
        return RandomForestCVModel(
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
    elif model_type == 'none':
        # ペナルティなしロジスティック回帰
        return LogisticRegression(
            penalty=None,
            solver='lbfgs',
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
            tol=tol
        )
    elif model_type == 'bolasso':
        return BOLASSOModel(
            n_bootstrap=n_bootstrap,
            C=1.0,
            selection_threshold=selection_threshold,
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            tol=tol
        )
    elif model_type == 'lars_traps':
        return LARSTrapsModel(
            n_random_traps=n_random_traps,
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            tol=tol
        )
    elif model_type == 'lars_cv':
        return LARSCVModel(
            cv=cv,
            random_state=random_state,
            max_iter=max_iter,
            class_weight=class_weight,
            tol=tol
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_coefficients(model, feature_names, model_type):
    """モデルから係数を取得する関数"""
    if model_type == 'randomforest':
        # RandomForestの場合はfeature_importances_を使用
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        else:
            return np.zeros(len(feature_names))
    elif model_type in ['bolasso', 'lars_traps', 'lars_cv']:
        # 特徴量選択モデルの場合は、内部モデルの係数を使用
        if hasattr(model, 'model_') and model.model_ is not None:
            coefs = model.model_.coef_[0] if len(model.model_.coef_.shape) > 1 else model.model_.coef_
            # 選択された特徴量のみの係数なので、全特徴量にマッピング
            full_coefs = np.zeros(len(feature_names))
            if hasattr(model, 'selected_features_') and model.selected_features_ is not None:
                for i, feature in enumerate(model.selected_features_):
                    if feature in feature_names:
                        idx = feature_names.index(feature)
                        full_coefs[idx] = coefs[i] if i < len(coefs) else 0.0
            return full_coefs
        else:
            return np.zeros(len(feature_names))
    elif (model_type in ['l1', 'l2', 'elasticnet']) and hasattr(model, 'model_') and model.model_ is not None:
        # L1-CV、L2-CV、ElasticNet-CVの場合は、内部モデルの係数を使用
        if hasattr(model.model_, 'coef_') and model.model_.coef_ is not None:
            return model.model_.coef_[0] if len(model.model_.coef_.shape) > 1 else model.model_.coef_
        else:
            return np.zeros(len(feature_names))
    else:
        # 線形モデルの場合はcoef_を使用
        if hasattr(model, 'coef_') and model.coef_ is not None:
            return model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        else:
            return np.zeros(len(feature_names))

