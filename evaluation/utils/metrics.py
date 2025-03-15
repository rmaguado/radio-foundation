from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
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

    accuracy = accuracy_score(labels, binary_predictions)
    precision = precision_score(labels, binary_predictions)
    recall = recall_score(labels, binary_predictions)
    roc_auc = roc_auc_score(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)
    f1 = f1_score(labels, binary_predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
    }


def save_metrics(val_metrics, output_path, label):
    metrics = ["roc_auc", "pr_auc", "f1"]

    metrics_path = os.path.join(output_path, f"{label}.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "a") as f:
        if os.stat(metrics_path).st_size == 0:
            f.write(",".join(metrics) + "\n")
        f.write(",".join([str(val_metrics[metric]) for metric in metrics]) + "\n")


def print_metrics(val_metrics):
    metrics = ["roc_auc", "pr_auc", "f1"]
    metrics_str = " - ".join(
        [f"{metric}: {val_metrics[metric]:.4f}" for metric in metrics]
    )
    print(metrics_str)
