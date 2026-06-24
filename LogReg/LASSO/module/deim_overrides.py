"""DEIM実験で使用するPLMスコアとRandomForestの既定設定。"""
import os
from collections import Counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import data_loader as _data_loader
from . import models as _models


PLM_SCORE_MODES = ("logit_diff", "positive_logit")
RANDOM_FOREST_PARAM_GRID = {
    "n_estimators": [200, 500],
    "max_depth": [3, 5, 10],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 4],
    "max_features": ["sqrt", 0.5],
}


def _score_mode(score_mode=None):
    mode = score_mode or os.getenv("DEIM_PLM_SCORE_MODE", "logit_diff")
    if mode not in PLM_SCORE_MODES:
        raise ValueError(
            f"Unknown PLM score mode: {mode}. "
            f"Choose one of {', '.join(PLM_SCORE_MODES)}."
        )
    return mode


def _score_turn(turn, score_mode):
    positive = turn.get("logit_clarification")
    if positive is None:
        return None
    if score_mode == "positive_logit":
        return float(positive)
    negative = turn.get("logit_not_clarification")
    if negative is None:
        raise ValueError(
            "logit_not_clarification is required for logit_diff. "
            "Use DEIM_PLM_SCORE_MODE=positive_logit for legacy files."
        )
    return float(positive) - float(negative)


def extract_base_scores(json_data, score_mode=None):
    """PLMスコアを抽出する。デフォルトは正例logitと負例logitの差。"""
    score_mode = _score_mode(score_mode)
    return {
        (str(turn["conv_id"]), int(turn["turn_id"])): score
        for conversation in json_data
        for turn in conversation
        if (score := _score_turn(turn, score_mode)) is not None
    }


def load_base_scores(*args, score_mode=None, **kwargs):
    """既存の読み込み処理を保ったままPLMスコア方式を切り替える。"""
    score_mode = _score_mode(score_mode)
    if score_mode == "positive_logit":
        return _data_loader.load_base_scores(*args, **kwargs)

    original_loader = _data_loader.load_json_data

    def load_with_logit_diff(path):
        data = original_loader(path)
        for conversation in data:
            for turn in conversation:
                score = _score_turn(turn, score_mode)
                if score is not None:
                    turn["logit_clarification"] = score
        return data

    _data_loader.load_json_data = load_with_logit_diff
    try:
        return _data_loader.load_base_scores(*args, **kwargs)
    finally:
        _data_loader.load_json_data = original_loader


class RandomForestCVModel:
    """層化交差検証でハイパーパラメータを選択するRandomForest。"""

    def __init__(
        self,
        cv=5,
        random_state=42,
        class_weight="balanced",
        scoring="roc_auc",
        n_jobs=-1,
        param_grid=None,
    ):
        self.cv = cv
        self.random_state = random_state
        self.class_weight = class_weight
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.param_grid = param_grid or RANDOM_FOREST_PARAM_GRID
        self.model_ = None
        self.best_params_ = None
        self.best_score_ = None

    def fit(self, X, y, print_and_save_func=None):
        class_counts = Counter(y)
        if len(class_counts) < 2:
            raise ValueError("RandomForest tuning requires two classes.")
        n_splits = min(self.cv, min(class_counts.values()))
        if n_splits < 2:
            raise ValueError(
                "RandomForest tuning requires at least two samples per class."
            )

        self.search_ = GridSearchCV(
            RandomForestClassifier(
                random_state=self.random_state,
                class_weight=self.class_weight,
                n_jobs=1,
            ),
            self.param_grid,
            cv=StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_state,
            ),
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True,
        ).fit(X, y)
        self.model_ = self.search_.best_estimator_
        self.best_params_ = self.search_.best_params_
        self.best_score_ = self.search_.best_score_

        message = (
            f"RandomForest best params: {self.best_params_} "
            f"({self.scoring}={self.best_score_:.4f})"
        )
        (print_and_save_func or print)(message)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    @property
    def intercept_(self):
        # main.pyのCV可視化処理との互換性を維持するためのダミー切片。
        return np.array([0.0])


def create_model(model_type, *args, **kwargs):
    """既存APIを維持し、RandomForestのみCV探索版へ差し替える。"""
    if model_type != "randomforest":
        return _models.create_model(model_type, *args, **kwargs)
    return RandomForestCVModel(
        cv=kwargs.get("cv", 5),
        random_state=kwargs.get("random_state", 42),
        class_weight=kwargs.get("class_weight", "balanced"),
        scoring=kwargs.get("scoring", "roc_auc"),
        n_jobs=kwargs.get("n_jobs", -1),
    )
