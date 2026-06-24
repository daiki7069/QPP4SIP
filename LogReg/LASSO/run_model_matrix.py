#!/usr/bin/env python3
"""Evaluate one DEIM feature/model condition from existing PLM predictions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from module import (
    create_model,
    extract_labels,
    get_coefficients,
    load_base_scores,
    load_json_data,
    load_qpp_scores,
    merge_features,
    normalize_features,
)

FEATURE_SETS = {
    "pre_post": (True, True, None),
    "bert": (False, False, "bert"),
    "roberta": (False, False, "roberta"),
    "pre_post_bert": (True, True, "bert"),
    "pre_post_roberta": (True, True, "roberta"),
}
EXPERIMENTS = {
    "bert": "AmbigNQ_bert-base_lr2e-05_bs16_kfold5",
    "roberta": "AmbigNQ_roberta-base_lr2e-05_bs16_earlystop_kfold5",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="AmbigNQ")
    parser.add_argument("--retrieval-method", default="dpr", choices=["dpr", "bm25"])
    parser.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SETS))
    parser.add_argument(
        "--model",
        required=True,
        choices=["randomforest", "l1", "l2", "elasticnet"],
    )
    parser.add_argument(
        "--plm-score-mode",
        default="logit_diff",
        choices=["logit_diff", "positive_logit"],
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_condition(args, repo_root):
    use_pre, use_post, plm = FEATURE_SETS[args.feature_set]
    dataset_dir = repo_root / "dataset" / args.dataset
    post_dir = (
        repo_root
        / "QPP"
        / "post_retrieval"
        / "outputs"
        / args.dataset
        / args.retrieval_method
    )
    pre_dir = repo_root / "QPP" / "pre_retrieval" / "outputs" / args.dataset

    train_labels = extract_labels(load_json_data(dataset_dir / "train.json"))
    dev_labels = extract_labels(load_json_data(dataset_dir / "dev.json"))
    train_scores = load_qpp_scores(
        "train",
        post_dir,
        pre_retrieval_output_dir=pre_dir,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=False,
    )
    dev_scores = load_qpp_scores(
        "dev",
        post_dir,
        pre_retrieval_output_dir=pre_dir,
        use_post=use_post,
        use_pre=use_pre,
        use_nsp=False,
    )

    if plm:
        experiment = EXPERIMENTS[plm].replace("AmbigNQ", args.dataset, 1)
        kwargs = {
            "base_experiment_names": [experiment],
            "use_bert": plm == "bert",
            "use_roberta": plm == "roberta",
            "score_mode": args.plm_score_mode,
        }
        train_scores.update(load_base_scores("train", args.dataset, repo_root, **kwargs))
        dev_scores.update(load_base_scores("dev", args.dataset, repo_root, **kwargs))
        if not any(plm in name for name in train_scores):
            raise FileNotFoundError(f"PLM prediction files were not loaded: {experiment}")

    X_train, y_train = merge_features(train_scores, train_labels)
    X_dev, y_dev = merge_features(dev_scores, dev_labels)
    X_train, X_dev = normalize_features(
        X_train,
        X_dev,
        use_minmax_normalization=True,
    )
    return X_train, y_train, X_dev, y_dev


def fit_model(args, X_train, y_train):
    kwargs = {
        "model_type": args.model,
        "max_iter": 5000,
        "random_state": args.random_state,
        "class_weight": "balanced",
        "cv": args.cv_folds,
        "scoring": "roc_auc",
        "n_jobs": -1,
    }
    if args.model == "l1":
        kwargs["use_l1_cv"] = True
    elif args.model == "l2":
        kwargs["use_l2_cv"] = True
    elif args.model == "elasticnet":
        kwargs["use_elasticnet_cv"] = True
    model = create_model(**kwargs)
    model.fit(X_train, y_train, print_and_save_func=print)
    return model


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
    }
    return metrics, predictions


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def search_metadata(model):
    metadata = {}
    for attribute in ("best_params_", "best_score_", "best_train_score_"):
        if hasattr(model, attribute):
            metadata[attribute.rstrip("_")] = json_value(getattr(model, attribute))
    return metadata


def save_curves(y_true, probabilities, output_dir):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap = average_precision_score(y_true, probabilities)
    plt.figure(figsize=(7, 7))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    X_train, y_train, X_dev, y_dev = load_condition(args, repo_root)
    model = fit_model(args, X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    dev_proba = model.predict_proba(X_dev)[:, 1]
    train_metrics, train_pred = evaluate(y_train, train_proba)
    dev_metrics, dev_pred = evaluate(y_dev, dev_proba)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "dataset": args.dataset,
        "retrieval_method": args.retrieval_method,
        "feature_set": args.feature_set,
        "model": args.model,
        "plm_score_mode": args.plm_score_mode,
        "features": list(X_train.columns),
        "train_samples": len(X_train),
        "dev_samples": len(X_dev),
        "cv_folds": args.cv_folds,
        "search": search_metadata(model),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    weights = get_coefficients(model, list(X_train.columns), args.model)
    pd.DataFrame(
        {"feature": X_train.columns, "coefficient_or_importance": weights}
    ).sort_values("coefficient_or_importance", key=abs, ascending=False).to_csv(
        args.output_dir / "feature_weights.csv", index=False
    )

    for split, X, y, predictions, probabilities in (
        ("train", X_train, y_train, train_pred, train_proba),
        ("dev", X_dev, y_dev, dev_pred, dev_proba),
    ):
        frame = X.reset_index(drop=True).copy()
        frame.insert(0, "true_label", np.asarray(y))
        frame.insert(1, "predicted_label", predictions)
        frame.insert(2, "probability", probabilities)
        frame.to_csv(args.output_dir / f"{split}_predictions.csv", index=False)

    (args.output_dir / "dev_classification_report.txt").write_text(
        classification_report(y_dev, dev_pred), encoding="utf-8"
    )
    save_curves(y_dev, dev_proba, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
