from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

import os
import numpy as np


def compute_metrics(logits, labels):
    threshold = 0.5
    probabilities = 1 / (1 + np.exp(-logits))
    binary_predictions = probabilities >= threshold

    roc_auc = roc_auc_score(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)
    f1 = f1_score(labels, binary_predictions)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
    }


def save_metrics(metrics_dict, output_path, filename):
    metrics = ["roc_auc", "pr_auc", "f1"]

    metrics_path = os.path.join(output_path, f"{filename}.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "a") as f:
        if os.stat(metrics_path).st_size == 0:
            f.write(",".join(metrics) + "\n")
        f.write(",".join([str(metrics_dict[metric]) for metric in metrics]) + "\n")


def print_metrics(metrics_dict, field: str):
    metrics = ["roc_auc", "pr_auc", "f1"]
    metrics_str = " - ".join(
        [f"{metric}: {metrics_dict[metric]:.4f}" for metric in metrics]
    )
    print(field + metrics_str)


def save_predictions(logits, labels, output_path, filename):
    predictions_path = os.path.join(output_path, f"{filename}.csv")
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    with open(predictions_path, "a") as f:
        if os.stat(predictions_path).st_size == 0:
            f.write("logits,labels\n")
        for logit, label in zip(logits, labels):
            f.write(f"{logit},{label}\n")
