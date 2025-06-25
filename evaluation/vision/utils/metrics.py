from sklearn.metrics import roc_auc_score, average_precision_score

import os
import numpy as np


def compute_metrics(logits, labels):
    probabilities = 1 / (1 + np.exp(-logits))

    roc_auc = roc_auc_score(labels, probabilities)
    pr_auc = average_precision_score(labels, probabilities)

    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def save_metrics(metrics_dict, output_path, filename):
    metrics = ["roc_auc", "pr_auc"]

    metrics_path = os.path.join(output_path, f"{filename}.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    with open(metrics_path, "a") as f:
        if os.stat(metrics_path).st_size == 0:
            f.write(",".join(metrics) + "\n")
        f.write(",".join([str(metrics_dict[metric]) for metric in metrics]) + "\n")


def print_metrics(metrics_dict, field: str):
    metrics = ["roc_auc", "pr_auc"]
    metrics_str = " - ".join(
        [f"{metric}: {metrics_dict[metric]:.4f}" for metric in metrics]
    )
    print(field + metrics_str)


def save_predictions(map_ids, logits, labels, output_path, filename):
    predictions_path = os.path.join(output_path, f"{filename}.csv")
    os.makedirs(os.path.dirname(predictions_path), exist_ok=True)

    with open(predictions_path, "a") as f:
        if os.stat(predictions_path).st_size == 0:
            f.write("map_id,logits,labels\n")
        for map_id, logit, label in zip(map_ids, logits, labels):
            f.write(f"{map_id},{logit},{label}\n")


def save_attention_maps(attention_maps, output_path):
    os.makedirs(output_path, exist_ok=True)
    for map_id, attn_map in attention_maps.items():
        attention_maps_path = os.path.join(output_path, f"{map_id}.npy")
        np.save(attention_maps_path, attention_maps)
