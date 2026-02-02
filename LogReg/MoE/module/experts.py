"""
MoE用Expertモジュール
各予測子（QPP指標）を1次元入力のロジスティック回帰Expertとして学習
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class PerFeatureExperts:
    """
    各特徴量（予測子）ごとに1変数ロジスティック回帰を学習し、
    各Expertの予測確率を返す
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        class_weight: str = "balanced",
        tol: float = 1e-8,
    ):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.class_weight = class_weight
        self.tol = tol
        self.experts_ = []  # list of LogisticRegression
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PerFeatureExperts":
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.feature_names_ = list(X.columns)
        self.experts_ = []
        for col in self.feature_names_:
            X_j = X[[col]].values
            lr = LogisticRegression(
                penalty="l2",
                C=self.C,
                solver="lbfgs",
                max_iter=self.max_iter,
                random_state=self.random_state,
                class_weight=self.class_weight,
                tol=self.tol,
            )
            lr.fit(X_j, y)
            self.experts_.append(lr)
        return self

    def predict_proba_per_expert(self, X: pd.DataFrame) -> np.ndarray:
        """
        各Expertの予測確率（正例の確率）を (n_samples, n_experts) で返す
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
        n_samples = len(X)
        n_experts = len(self.experts_)
        proba = np.zeros((n_samples, n_experts))
        for j, (col, lr) in enumerate(zip(self.feature_names_, self.experts_)):
            X_j = X[[col]].values
            proba[:, j] = lr.predict_proba(X_j)[:, 1]
        return proba
