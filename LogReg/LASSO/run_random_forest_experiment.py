#!/usr/bin/env python3
"""Run the DEIM RandomForest experiment and persist all core results."""
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
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from module import (
    create_model,
    extract_labels,
    load_json_data,
    load_qpp_scores,
    merge_features,
    normalize_features,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run tuned RandomForest on DEIM QPP features"
    )
    parser.add_argument("--dataset", default="AmbigNQ", choices=["AmbigNQ", "INSCIT"])
    parser.add_argument("--retrieval-method", default="dpr", choices=["dpr", "bm25"])
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=["pre", "post"],
        choices=["pre", "post"],
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--scoring",
        default="roc_auc",
        choices=["roc_auc", "average_precision", "f1", "accuracy"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def evaluate(y_true, probabilities):
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "f1": float(f1_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
    }, predictions


def save_curves(y_true, probabilities, output_dir):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f"RandomForest (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("RandomForest ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    ap = average_precision_score(y_true, probabilities)
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, label=f"RandomForest (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("RandomForest Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=200)
    plt.close()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
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

    use_pre = "pre" in args.feature_types
    use_post = "post" in args.feature_types
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
    X_train, y_train = merge_features(train_scores, train_labels)
    X_dev, y_dev = merge_features(dev_scores, dev_labels)
    X_train_norm, X_dev_norm = normalize_features(
        X_train,
        X_dev,
        use_minmax_normalization=True,
    )

    model = create_model(
        "randomforest",
        cv=args.cv_folds,
        scoring=args.scoring,
        random_state=args.random_state,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train_norm, y_train)

    train_probabilities = model.predict_proba(X_train_norm)[:, 1]
    dev_probabilities = model.predict_proba(X_dev_norm)[:, 1]
    train_metrics, train_predictions = evaluate(y_train, train_probabilities)
    dev_metrics, dev_predictions = evaluate(y_dev, dev_probabilities)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "dataset": args.dataset,
        "retrieval_method": args.retrieval_method,
        "feature_types": args.feature_types,
        "features": list(X_train_norm.columns),
        "train_samples": int(len(X_train_norm)),
        "dev_samples": int(len(X_dev_norm)),
        "cv_folds": args.cv_folds,
        "search_scoring": args.scoring,
        "best_cv_score": float(model.best_score_),
        "best_params": model.best_params_,
        "train_metrics": train_metrics,
        "dev_metrics": dev_metrics,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )

    pd.DataFrame(
        {
            "feature": X_train_norm.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False).to_csv(
        args.output_dir / "feature_importances.csv", index=False
    )

    prediction_frame = X_dev_norm.reset_index(drop=True).copy()
    prediction_frame.insert(0, "true_label", np.asarray(y_dev))
    prediction_frame.insert(1, "predicted_label", dev_predictions)
    prediction_frame.insert(2, "probability", dev_probabilities)
    prediction_frame.to_csv(args.output_dir / "dev_predictions.csv", index=False)

    train_frame = pd.DataFrame(
        {
            "true_label": np.asarray(y_train),
            "predicted_label": train_predictions,
            "probability": train_probabilities,
        }
    )
    train_frame.to_csv(args.output_dir / "train_predictions.csv", index=False)
    save_curves(y_dev, dev_probabilities, args.output_dir)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
