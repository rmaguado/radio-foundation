from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

import os
import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(logits, labels):
    threshold = 0.5
    probabilities = 1 / (1 + np.exp(-logits))
    binary_predictions = probabilities >= threshold

    accuracy = accuracy_score(labels, binary_predictions)
    precision = precision_score(labels, binary_predictions)
    recall = recall_score(labels, binary_predictions)
    f1 = f1_score(labels, binary_predictions)
    roc_auc = roc_auc_score(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def save_metrics(val_metrics, output_path, label):
    metrics = [(m["roc_auc"], m["f1_score"], m["pr_auc"]) for m in val_metrics]

    metrics_path = os.path.join(output_path, "metrics", f"{label}.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "w") as f:
        f.write("roc_auc,f1_score,pr_auc\n")
        for m in metrics:
            f.write(f"{m[0]},{m[1]},{m[2]}\n")

    sorted_metrics = sorted(metrics, key=lambda x: x[0])
    final_scores = {
        "roc_auc": np.mean([m[0] for m in sorted_metrics[-5:]]),
        "f1_score": np.mean([m[1] for m in sorted_metrics[-5:]]),
        "pr_auc": np.mean([m[2] for m in sorted_metrics[-5:]]),
    }

    summary_path = os.path.join(output_path, "metrics", "summary.csv")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "a") as f:
        if os.stat(summary_path).st_size == 0:
            f.write("label,roc_auc,f1_score,pr_auc\n")
        f.write(
            f"{label},{final_scores['roc_auc']},{final_scores['f1_score']},{final_scores['pr_auc']}\n"
        )

    figures_path = os.path.join(output_path, "figures")
    os.makedirs(figures_path, exist_ok=True)

    plt.figure(figsize=(10, 10))
    plt.plot([m[0] for m in sorted_metrics], label="ROC AUC")
    plt.plot([m[1] for m in sorted_metrics], label="F1 Score")
    plt.plot([m[2] for m in sorted_metrics], label="PR AUC")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(label)
    plt.ylim(0, 1)

    plt.legend()
    plt.savefig(os.path.join(figures_path, f"{label}.png"))
    plt.close()
