from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_metrics(predictions, labels, threshold=0.5):
    binary_predictions = [x >= threshold for x in predictions]

    accuracy = accuracy_score(labels, binary_predictions)
    precision = precision_score(labels, binary_predictions)
    recall = recall_score(labels, binary_predictions)
    f1 = f1_score(labels, binary_predictions)
    aucroc = roc_auc_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "aucroc": aucroc,
    }
