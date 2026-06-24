import json

import numpy as np
import pytest

from module import extract_base_scores, load_base_scores
from module.deim_overrides import RandomForestCVModel


def prediction_data():
    return [[{
        "conv_id": "conv-1",
        "turn_id": 0,
        "logit_clarification": 2.0,
        "logit_not_clarification": 0.5,
    }]]


def test_plm_score_defaults_to_logit_difference():
    scores = extract_base_scores(prediction_data())
    assert scores[("conv-1", 0)] == pytest.approx(1.5)


def test_positive_logit_mode_is_available():
    scores = extract_base_scores(
        prediction_data(), score_mode="positive_logit"
    )
    assert scores[("conv-1", 0)] == pytest.approx(2.0)


def test_load_base_scores_applies_logit_difference(tmp_path, monkeypatch):
    experiment_name = "AmbigNQ_bert-base_test"
    output_dir = (
        tmp_path
        / "SIP"
        / "FT-PLM"
        / "output"
        / "AmbigNQ"
        / experiment_name
    )
    output_dir.mkdir(parents=True)
    with open(output_dir / "train_with_predictions.json", "w") as f:
        json.dump(prediction_data(), f)

    from module import deim_overrides
    monkeypatch.setattr(
        deim_overrides._data_loader,
        "BASE_EXPERIMENT_NAMES",
        [experiment_name],
    )

    scores = load_base_scores(
        "train",
        "AmbigNQ",
        tmp_path,
        use_bert=True,
        use_roberta=False,
    )
    assert scores["bert_logit_clarification"][("conv-1", 0)] == pytest.approx(1.5)


def test_randomforest_selects_hyperparameters_with_cv():
    X = np.array([[i, i % 3] for i in range(24)])
    y = np.array([0] * 12 + [1] * 12)
    model = RandomForestCVModel(
        cv=2,
        random_state=0,
        n_jobs=1,
        param_grid={
            "n_estimators": [10],
            "max_depth": [2, 4],
            "min_samples_split": [2],
            "min_samples_leaf": [1],
            "max_features": ["sqrt"],
        },
    )

    model.fit(X, y)

    assert model.best_params_["max_depth"] in {2, 4}
    assert model.predict_proba(X).shape == (24, 2)
    assert len(model.feature_importances_) == X.shape[1]
