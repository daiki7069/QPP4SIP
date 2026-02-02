"""
Mixture of Experts (MoE) 統合モデル
動的ゲーティングで各予測子の予測をソフトに統合
"""
import numpy as np
import pandas as pd
from typing import Literal

from .experts import PerFeatureExperts
from .gating import (
    create_gate,
    LinearGate,
    MLPGate,
    TemperatureGate,
    LearnedFixedGate,
    GatingType,
)


class MixtureOfExperts:
    """
    MoE: 各Expert（1変数ロジスティック回帰）の予測を
    ゲートで重み付けして統合。予測 = Σ g_i * p_i
    """

    def __init__(
        self,
        gating_type: GatingType = "linear",
        expert_C: float = 1.0,
        expert_max_iter: int = 1000,
        gate_max_iter: int = 500,
        hidden_size: int = 32,
        temperature: float = 1.0,
        random_state: int = 42,
        class_weight: str = "balanced",
    ):
        self.gating_type = gating_type
        self.expert_C = expert_C
        self.expert_max_iter = expert_max_iter
        self.gate_max_iter = gate_max_iter
        self.hidden_size = hidden_size
        self.temperature = temperature
        self.random_state = random_state
        self.class_weight = class_weight
        self.experts_ = None
        self.gate_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MixtureOfExperts":
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        self.feature_names_ = list(X.columns)
        n_input = X.shape[1]
        n_experts = n_input

        # 1. Experts を各特徴量1次元で学習
        self.experts_ = PerFeatureExperts(
            C=self.expert_C,
            max_iter=self.expert_max_iter,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )
        self.experts_.fit(X, y)

        # 2. 各Expertの予測確率を計算
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        expert_proba = self.experts_.predict_proba_per_expert(X)

        # 3. ゲートを学習（統合予測の交差エントロピーを最小化）
        if self.gating_type == "learned_fixed":
            self.gate_ = create_gate(
                "learned_fixed",
                n_input,
                n_experts,
                random_state=self.random_state,
            )
        else:
            self.gate_ = create_gate(
                self.gating_type,
                n_input,
                n_experts,
                hidden_size=self.hidden_size,
                temperature=self.temperature,
                random_state=self.random_state,
            )
        self.gate_.fit(
            X_arr,
            expert_proba,
            y.values if isinstance(y, pd.Series) else y,
            max_iter=self.gate_max_iter,
        )
        return self

    def predict_weights(self, X: pd.DataFrame) -> np.ndarray:
        """各サンプルに対するExpertへの重み (n_samples, n_experts)"""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names_)
        X_arr = X.values
        return self.gate_.predict_weights(X_arr)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """統合予測確率 [P(0), P(1)] を (n_samples, 2) で返す"""
        expert_proba = self.experts_.predict_proba_per_expert(X)
        weights = self.predict_weights(X)
        p_pos = (weights * expert_proba).sum(axis=1)
        p_pos = np.clip(p_pos, 1e-15, 1 - 1e-15)
        return np.column_stack([1 - p_pos, p_pos])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
