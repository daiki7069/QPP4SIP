"""DEIM実験で使用するPLMスコアとRandomForestの既定設定。"""
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

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


def _resolve_score_mode(score_mode: str = None) -> str:
    mode = score_mode or os.getenv("DEIM_PLM_SCORE_MODE", "logit_diff")
    if mode not in PLM_SCORE_MODES:
        raise ValueError(
            f"Unknown PLM score mode: {mode}. "
            f"Choose one of {', '.join(PLM_SCORE_MODES)}."
        )
    return mode


def _score_turn(turn: dict, score_mode: str):
    positive_logit = turn.get("logit_clarification")
    if positive_logit is None:
        return None
    if score_mode == "positive_logit":
        return float(positive_logit)

    negative_logit = turn.get("logit_not_clarification")
    if negative_logit is None:
        raise ValueError(
            "logit_not_clarification is required for the default "
            "logit_diff mode. Use DEIM_PLM_SCORE_MODE=positive_logit "
            "for legacy prediction files."
        )
    return float(positive_logit) - float(negative_logit)


def extract_base_scores(json_data, score_mode: str = None):
    """PLMスコアを抽出する。デフォルトは正例logitと負例logitの差。"""
    score_mode = _resolve_score_mode(score_mode)
    scores = {}
    for conversation in json_data:
        for turn in conversation:
            score = _score_turn(turn, score_mode)
            if score is not None:
                scores[(str(turn["conv_id"]), int(turn["turn_id"]))] = score
    return scores


def load_base_scores(
    split: str,
    dataset: str,
    base_dir: Path,
    base_experiment_names: List[str] = None,
    use_bert: bool = True,
    use_roberta: bool = True,
    use_transfer: bool = False,
    use_full_train_model_for_dev: bool = False,
    score_mode: str = None,
) -> Dict[str, Dict[Tuple[str, int], float]]:
    """既存の読み込み処理を保ったままPLMスコア方式を切り替える。"""
    score_mode = _resolve_score_mode(score_mode)
    if score_mode == "positive_logit":
        return _data_loader.load_base_scores(
            split,
            dataset,
            base_dir,
            base_experiment_names=base_experiment_names,
            use_bert=use_bert,
            use_roberta=use_roberta,
            use_transfer=use_transfer,
            use_full_train_model_for_dev=use_full_train_model_for_dev,
        )

    original_load_json_data = _data_loader.load_json_data

    def load_json_data_with_logit_diff(json_path):
        json_data = original_load_json_data(json_path)
        for conversation in json_data:
            for turn in conversation:
                score = _score_turn(turn, score_mode)
                if score is not None:
                    turn["logit_clarification"] = score
        return json_data

    _data_loader.load_json_data = load_json_data_with_logit_diff
    try:
        return _data_loader.load_base_scores(
            split,
            dataset,
            base_dir,
            base_experiment_names=base_experiment_names,
            use_bert=use_bert,
            use_roberta=use_roberta,
            use_transfer=use_transfer,
            use_full_train_model_for_dev=use_full_train_model_for_dev,
        )
    finally:
        _data_loader.load_json_data = original_load_json_data


class RandomForestCVModel:
    """層化交差検証でハイパーパラメータを選択するRandomForest。"""

    def __init__(
        self,
        cv: int = 5,
        random_state: int = 42,
        class_weight="balanced",
        scoring: str = "roc_auc",
        n_jobs: int = -1,
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

        search = GridSearchCV(
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
        )
        search.fit(X, y)
        self.model_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.best_score_ = search.best_score_

        message = (
            f"RandomForest best params: {self.best_params_} "
            f"({self.scoring}={self.best_score_:.4f})"
        )
        if print_and_save_func:
            print_and_save_func(message)
        else:
            print(message)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_

    @property
    def classes_(self):
        return self.model_.classes_

    @property
    def intercept_(self):
        # main.pyのCV可視化処理との互換性を維持するためのダミー切片。
        return np.array([0.0])


def create_model(
    model_type,
    max_iter=1000,
    random_state=42,
    class_weight="balanced",
    tol=1e-8,
    non_negative=False,
    n_bootstrap=100,
    selection_threshold=0.5,
    n_random_traps=10,
    cv=5,
    use_l1_cv=False,
    use_l2_cv=False,
    use_elasticnet_cv=False,
    C_range=None,
    l1_ratio_range=None,
    scoring="roc_auc",
    n_jobs=-1,
):
    """既存APIを維持し、RandomForestのみCV探索版へ差し替える。"""
    if model_type == "randomforest":
        return RandomForestCVModel(
            cv=cv,
            random_state=random_state,
            class_weight=class_weight,
            scoring=scoring,
            n_jobs=n_jobs,
        )
    return _models.create_model(
        model_type=model_type,
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight,
        tol=tol,
        non_negative=non_negative,
        n_bootstrap=n_bootstrap,
        selection_threshold=selection_threshold,
        n_random_traps=n_random_traps,
        cv=cv,
        use_l1_cv=use_l1_cv,
        use_l2_cv=use_l2_cv,
        use_elasticnet_cv=use_elasticnet_cv,
        C_range=C_range,
        l1_ratio_range=l1_ratio_range,
        scoring=scoring,
        n_jobs=n_jobs,
    )
