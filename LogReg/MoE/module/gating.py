"""
MoE用ゲーティングモジュール
入力特徴量から各Expertへの重み（ソフトマックス）を計算する複数方式を提供
"""
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from typing import Literal

GatingType = Literal["linear", "mlp", "temperature", "learned_fixed"]


class LinearGate:
    """線形ゲート: g = softmax(W @ x + b)"""

    def __init__(self, n_input: int, n_experts: int, random_state: int = 42):
        self.n_input = n_input
        self.n_experts = n_experts
        self.random_state = random_state
        self.W_ = None  # (n_input, n_experts)
        self.b_ = None  # (n_experts,)

    def _params_to_Wb(self, params: np.ndarray) -> tuple:
        n = self.n_input * self.n_experts
        W = params[:n].reshape(self.n_input, self.n_experts)
        b = params[n:]
        return W, b

    def _Wb_to_params(self, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([W.ravel(), b])

    def forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        # (n_samples, n_experts)
        logits = X @ W + b
        return softmax(logits, axis=1)

    def fit(
        self,
        X: np.ndarray,
        expert_proba: np.ndarray,
        y: np.ndarray,
        max_iter: int = 500,
    ) -> "LinearGate":
        """統合予測 p = sum_i g_i * p_i の交差エントロピーを最小化"""
        np.random.seed(self.random_state)
        W0 = np.random.randn(self.n_input * self.n_experts) * 0.01
        b0 = np.zeros(self.n_experts)
        params0 = self._Wb_to_params(W0, b0)

        def loss(params):
            W, b = self._params_to_Wb(params)
            g = self.forward(X, W, b)
            p = (g * expert_proba).sum(axis=1)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(loss, params0, method="L-BFGS-B", options={"maxiter": max_iter})
        self.W_, self.b_ = self._params_to_Wb(res.x)
        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, self.W_, self.b_)


class MLPGate:
    """2層MLPゲート: g = softmax(W2 @ relu(W1 @ x + b1) + b2)"""

    def __init__(
        self,
        n_input: int,
        n_experts: int,
        hidden_size: int = 32,
        random_state: int = 42,
    ):
        self.n_input = n_input
        self.n_experts = n_experts
        self.hidden_size = hidden_size
        self.random_state = random_state
        self.W1_ = None
        self.b1_ = None
        self.W2_ = None
        self.b2_ = None

    def _params_to_weights(self, params: np.ndarray) -> tuple:
        i1 = self.n_input * self.hidden_size
        i2 = i1 + self.hidden_size
        i3 = i2 + self.hidden_size * self.n_experts
        W1 = params[:i1].reshape(self.n_input, self.hidden_size)
        b1 = params[i1:i2]
        W2 = params[i2:i3].reshape(self.hidden_size, self.n_experts)
        b2 = params[i3:]
        return W1, b1, W2, b2

    def _weights_to_params(self, W1, b1, W2, b2) -> np.ndarray:
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def forward(
        self,
        X: np.ndarray,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray,
    ) -> np.ndarray:
        h = np.maximum(0, X @ W1 + b1)
        logits = h @ W2 + b2
        return softmax(logits, axis=1)

    def fit(
        self,
        X: np.ndarray,
        expert_proba: np.ndarray,
        y: np.ndarray,
        max_iter: int = 500,
    ) -> "MLPGate":
        np.random.seed(self.random_state)
        W1 = np.random.randn(self.n_input, self.hidden_size) * 0.1
        b1 = np.zeros(self.hidden_size)
        W2 = np.random.randn(self.hidden_size, self.n_experts) * 0.1
        b2 = np.zeros(self.n_experts)
        params0 = self._weights_to_params(W1, b1, W2, b2)

        def loss(params):
            W1, b1, W2, b2 = self._params_to_weights(params)
            g = self.forward(X, W1, b1, W2, b2)
            p = (g * expert_proba).sum(axis=1)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(loss, params0, method="L-BFGS-B", options={"maxiter": max_iter})
        self.W1_, self.b1_, self.W2_, self.b2_ = self._params_to_weights(res.x)
        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        return self.forward(
            X, self.W1_, self.b1_, self.W2_, self.b2_
        )


class TemperatureGate:
    """温度付きソフトマックスゲート: g = softmax((W @ x + b) / T)"""

    def __init__(
        self,
        n_input: int,
        n_experts: int,
        temperature: float = 1.0,
        random_state: int = 42,
    ):
        self.n_input = n_input
        self.n_experts = n_experts
        self.temperature = temperature
        self.random_state = random_state
        self.W_ = None
        self.b_ = None

    def _params_to_Wb(self, params: np.ndarray) -> tuple:
        n = self.n_input * self.n_experts
        W = params[:n].reshape(self.n_input, self.n_experts)
        b = params[n:]
        return W, b

    def _Wb_to_params(self, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.concatenate([W.ravel(), b])

    def forward(
        self,
        X: np.ndarray,
        W: np.ndarray,
        b: np.ndarray,
        T: float = None,
    ) -> np.ndarray:
        T = T if T is not None else self.temperature
        logits = (X @ W + b) / T
        return softmax(logits, axis=1)

    def fit(
        self,
        X: np.ndarray,
        expert_proba: np.ndarray,
        y: np.ndarray,
        max_iter: int = 500,
    ) -> "TemperatureGate":
        np.random.seed(self.random_state)
        W0 = np.random.randn(self.n_input * self.n_experts) * 0.01
        b0 = np.zeros(self.n_experts)
        params0 = self._Wb_to_params(W0, b0)

        def loss(params):
            W, b = self._params_to_Wb(params)
            g = self.forward(X, W, b)
            p = (g * expert_proba).sum(axis=1)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(loss, params0, method="L-BFGS-B", options={"maxiter": max_iter})
        self.W_, self.b_ = self._params_to_Wb(res.x)
        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, self.W_, self.b_)


class LearnedFixedGate:
    """入力に依存しない学習可能な固定重み: g = softmax(learned_logits)"""

    def __init__(self, n_experts: int, random_state: int = 42):
        self.n_experts = n_experts
        self.random_state = random_state
        self.logits_ = None  # (n_experts,)

    def fit(
        self,
        X: np.ndarray,
        expert_proba: np.ndarray,
        y: np.ndarray,
        max_iter: int = 500,
    ) -> "LearnedFixedGate":
        np.random.seed(self.random_state)
        logits0 = np.zeros(self.n_experts)

        def loss(logits):
            g = softmax(logits)[np.newaxis, :]
            p = (g * expert_proba).sum(axis=1)
            p = np.clip(p, 1e-15, 1 - 1e-15)
            return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        res = minimize(
            loss,
            logits0,
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        self.logits_ = res.x
        return self

    def predict_weights(self, X: np.ndarray) -> np.ndarray:
        g = softmax(self.logits_)
        return np.tile(g, (X.shape[0], 1))


def create_gate(
    gating_type: GatingType,
    n_input: int,
    n_experts: int,
    *,
    hidden_size: int = 32,
    temperature: float = 1.0,
    random_state: int = 42,
):
    """ゲートインスタンスを生成"""
    if gating_type == "linear":
        return LinearGate(n_input, n_experts, random_state=random_state)
    if gating_type == "mlp":
        return MLPGate(
            n_input,
            n_experts,
            hidden_size=hidden_size,
            random_state=random_state,
        )
    if gating_type == "temperature":
        return TemperatureGate(
            n_input,
            n_experts,
            temperature=temperature,
            random_state=random_state,
        )
    if gating_type == "learned_fixed":
        return LearnedFixedGate(n_experts, random_state=random_state)
    raise ValueError(f"Unknown gating type: {gating_type}")
