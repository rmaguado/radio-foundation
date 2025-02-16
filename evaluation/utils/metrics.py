from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import numpy as np


def compute_metrics(logits, labels):
    threshold = 0.5
    probabilities = 1 / (1 + np.exp(-logits))
    binary_predictions = probabilities >= threshold

    accuracy = accuracy_score(labels, binary_predictions)
    precision = precision_score(labels, binary_predictions)
    recall = recall_score(labels, binary_predictions)
    f1 = f1_score(labels, binary_predictions)
    aucroc = roc_auc_score(labels, probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "aucroc": aucroc,
    }
