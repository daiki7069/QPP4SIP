#!/usr/bin/env python3
"""Run one DEIM feature/model condition and persist reproducible results."""
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


FEATURE_SET_CONFIG = {
    "pre_post": {"pre": True, "post": True, "plm": None},
    "bert": {"pre": False, "post": False, "plm": "bert"},
    "roberta": {"pre": False, "post": False, "plm": "roberta"},
    "pre_post_bert": {"pre": True, "post": True, "plm": "bert"},
    "pre_post_roberta": {"pre": True, "post": True, "plm": "roberta"},
}

EXPERIMENT_NAMES = {
    "bert": "AmbigNQ_bert-base_lr2e-05_bs8_kfold5",
    "roberta": "AmbigNQ_roberta-base_lr2e-05_bs8_earlystop_kfold5",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="AmbigNQ", choices=["AmbigNQ", "INSCIT"])
    parser.add_argument("--retrieval-method", default="dpr", choices=["dpr", "bm25"])
    parser.add_argument("--feature-set", required=True, choices=sorted(FEATURE_SET_CONFIG))
    parser.add_argument(
        "--model",
        required=True,
        choices=["randomforest", "l1", "l2", "elasticnet"],
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--plm-score-mode", default="logit_diff", choices=["logit_diff", "positive_logit"])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def fit_model(model_name, X_train, y_train, cv_folds, random_state):
    common = dict(
        model_type=model_name,
        cv=cv_folds,
        random_state=random_state,
        class_weight="balanced",
        scoring="roc_auc",
        n_jobs=-1,
    )
    if model_name == "l1":
        model = create_model(**common, use_l1_cv=True)
    elif model_name == "l2":
        model = create_model(**common, use_l2_cv=True)
    elif model_name == "elasticnet":
        model = create_model(**common, use_elasticnet_cv=True)
    else:
        model = create_model(**common)

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


def load_features(args, repo_root):
    config = FEATURE_SET_CONFIG[args.feature_set]
    dataset_dir = repo_root / "dataset" / args.dataset
    post_dir = repo_root / "QPP" / "post_retrieval" / "outputs" / args.dataset / args.retrieval_method
    pre_dir = repo_root / "QPP" / "pre_retrieval" / "outputs" / args.dataset

    train_labels = extract_labels(load_json_data(dataset_dir / "train.json"))
    dev_labels = extract_labels(load_json_data(dataset_dir / "dev.json"))
    train_scores = load_qpp_scores(
        "train",
        post_dir,
        pre_retrieval_output_dir=pre_dir,
        use_post=config["post"],
        use_pre=config["pre"],
        use_nsp=False,
    )
    dev_scores = load_qpp_scores(
        "dev",
        post_dir,
        pre_retrieval_output_dir=pre_dir,
        use_post=config["post"],
        use_pre=config["pre"],
        use_nsp=False,
    )

    plm = config["plm"]
    if plm:
        experiment_name = EXPERIMENT_NAMES[plm].replace("AmbigNQ", args.dataset, 1)
        use_bert = plm == "bert"
        use_roberta = plm == "roberta"
        train_scores.update(
            load_base_scores(
                "train",
                args.dataset,
                repo_root,
                base_experiment_names=[experiment_name],
                use_bert=use_bert,
                use_roberta=use_roberta,
                score_mode=args.plm_score_mode,
            )
        )
        dev_scores.update(
            load_base_scores(
                "dev",
                args.dataset,
                repo_root,
                base_experiment_names=[experiment_name],
                use_bert=use_bert,
                use_roberta=use_roberta,
                score_mode=args.plm_score_mode,
            )
        )

    X_train, y_train = merge_features(train_scores, train_labels)
    X_dev, y_dev = merge_features(dev_scores, dev_labels)
    X_train, X_dev = normalize_features(
        X_train,
        X_dev,
        use_minmax_normalization=True,
    )
    return X_train, y_train, X_dev, y_dev


def model_search_metadata(model):
    metadata = {}
    for name in ("best_params_", "best_score_", "best_train_score_"):
        if hasattr(model, name):
            value = getattr(model, name)
            if isinstance(value, (np.floating, np.integer)):
                value = value.item()
            metadata[name.rstrip("_")] = value
    return metadata


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    X_train, y_train, X_dev, y_dev = load_features(args, repo_root)
    model = fit_model(
        args.model,
        X_train,
        y_train,
        args.cv_folds,
        args.random_state,
    )

    train_probabilities = model.predict_proba(X_train)[:, 1]
    dev_probabilities = model.predict_proba(X_dev)[:, 1]
    train_metrics, train_predictions = evaluate(y_train, train_probabilities)
    dev_metrics, dev_predictions = evaluate(y_dev, dev_probabilities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": args.dataset,
        "retrieval_method": args.retrieval_method,
        "feature_set": args.feature_set,
        "model": args.model,
        "plm_score_mode": args.plm_score_mode,
        "features": list(X_train.columns),
        "train_samples": int(len(X_train)),
        "dev_samples": int(len(X_dev)),
        "cv_folds": args.cv_folds,
        "search": model_search_metadata(model),
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    coefficients = get_coefficients(model, list(X_train.columns), args.model)
    pd.DataFrame(
        {
            "feature": X_train.columns,
            "coefficient_or_importance": coefficients,
        }
    ).sort_values(
        "coefficient_or_importance", key=abs, ascending=False
    ).to_csv(args.output_dir / "feature_weights.csv", index=False)

    pd.DataFrame(
        {
            "true_label": np.asarray(y_train),
            "predicted_label": train_predictions,
            "probability": train_probabilities,
        }
    ).to_csv(args.output_dir / "train_predictions.csv", index=False)
    pd.DataFrame(
        {
            "true_label": np.asarray(y_dev),
            "predicted_label": dev_predictions,
            "probability": dev_probabilities,
        }
    ).to_csv(args.output_dir / "dev_predictions.csv", index=False)
    (args.output_dir / "dev_classification_report.txt").write_text(
        classification_report(y_dev, dev_predictions), encoding="utf-8"
    )
    save_curves(y_dev, dev_probabilities, args.output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
