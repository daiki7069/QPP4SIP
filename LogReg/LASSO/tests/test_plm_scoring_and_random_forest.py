import json

import numpy as np
import pytest

from module import data_loader
from module.models import RandomForestCVModel, create_model, get_coefficients


def prediction_data():
    return [[{
        "conv_id": "conv-1",
        "turn_id": 0,
        "logit_clarification": 2.0,
        "logit_not_clarification": 0.5,
    }]]


def test_plm_score_defaults_to_logit_difference():
    scores = data_loader.extract_base_scores(prediction_data())
    assert scores[("conv-1", 0)] == pytest.approx(1.5)


def test_positive_logit_mode_is_retained():
    scores = data_loader.extract_base_scores(
        prediction_data(), score_mode="positive_logit"
    )
    assert scores[("conv-1", 0)] == pytest.approx(2.0)


def test_load_base_scores_uses_selected_mode(tmp_path, monkeypatch):
    experiment = "AmbigNQ_bert-base_test"
    output_dir = (
        tmp_path / "SIP" / "FT-PLM" / "output" /
        "AmbigNQ" / experiment
    )
    output_dir.mkdir(parents=True)
    with open(output_dir / "train_with_predictions.json", "w") as f:
        json.dump(prediction_data(), f)
    monkeypatch.setattr(
        data_loader, "BASE_EXPERIMENT_NAMES", [experiment]
    )

    scores = data_loader.load_base_scores(
        "train", "AmbigNQ", tmp_path,
        use_bert=True, use_roberta=False,
    )
    assert scores["bert_logit_clarification"][("conv-1", 0)] == pytest.approx(1.5)


def test_random_forest_hyperparameters_are_selected_by_cv():
    X = np.array([[i, i % 3] for i in range(24)])
    y = np.array([0] * 12 + [1] * 12)
    model = create_model(
        "randomforest", cv=2, n_jobs=1, random_state=0
    )
    assert isinstance(model, RandomForestCVModel)
    model.param_grid = {
        "n_estimators": [10],
        "max_depth": [2, 4],
        "min_samples_split": [2],
        "min_samples_leaf": [1],
        "max_features": ["sqrt"],
    }
    model.fit(X, y)

    assert model.best_params_["max_depth"] in {2, 4}
    importances = get_coefficients(
        model, ["feature_1", "feature_2"], "randomforest"
    )
    assert len(importances) == X.shape[1]
