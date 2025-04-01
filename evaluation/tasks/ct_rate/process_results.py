import pandas as pd
import os
import random
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt


METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_test",
        help="Name of experiment. Used to differentiate save files.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="vitb-14.10",
        help="Name of folder in /runs used to generate embeddings",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="unnamed_checkpoint",
        help="Name of checkpoint in /runs/{$run_name}/checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/tasks/ct_rate/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        help="Number of folds in cross-validation",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs in training. ",
    )
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="roc_auc",
        help="Metric to use for selecting best model from validation",
    )
    parser.add_argument(
        "--z_score",
        type=float,
        default=1.96,
        help="Z-score for confidence intervals",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="evaluation/tasks/ct_rate/processed_results",
        help="Path to save processed results and plots",
    )
    return parser.parse_args()


def generate_val_epoch_plot(args, train_results, label, metric):

    output_dir = os.path.join(args.output_path, "plots", f"{label}_val_epochs.png")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()

    for fold_idx in range(1, args.num_folds + 1):
        epoch_metrics = train_results[label][metric][fold_idx - 1]
        plt.plot(epoch_metrics, label=f"Fold {fold_idx}")

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.title(f"Training Curve {metric} for {label}")

    plt.savefig(output_dir)


def generate_cv_roc_plot(args, results_path, label, epoch):
    label_result_path = os.path.join(results_path, label)
    output_dir = os.path.join(args.output_path, "plots", f"{label}.png")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()

    for fold_idx in range(1, args.num_folds + 1):
        logits, labels = get_logits(label_result_path, fold_idx, epoch)

        fpr, tpr, _ = roc_curve(labels, logits)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"Fold {fold_idx} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve for {label}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir)


def get_roc_auc_se(auc, labels):
    n1 = np.sum(labels)
    n2 = len(labels) - n1
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = np.sqrt(
        (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2))
        / (n1 * n2)
    )
    return se


def generate_test_roc_plot(args, results_path, label, epoch):
    label_result_path = os.path.join(results_path, label)
    output_dir = os.path.join(args.output_path, "plots", f"{label}.png")
    os.makedirs(output_dir, exist_ok=True)

    plt.figure()

    logits, labels = get_logits(label_result_path, "test", epoch)

    fpr, tpr, _ = roc_curve(labels, logits)
    roc_auc = auc(fpr, tpr)

    se = get_roc_auc_se(roc_auc, labels)
    ci = args.z_score * se

    plt.fill_between(fpr, tpr - ci, tpr + ci, color="grey", alpha=0.5)
    plt.plot(fpr, tpr, label=f"Test (AUC = {roc_auc:.4f} ± {ci:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Test ROC Curve for {label}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_dir)


def get_pr_f1(logits, labels):
    probabilities = 1 / (1 + np.exp(-logits))
    thresholds = np.linspace(0, 1, 100)
    best_threshold = 0.5
    best_youden = 0

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        sensitivity = np.sum((predictions == 1) & (labels == 1)) / np.sum(labels == 1)
        specificity = np.sum((predictions == 0) & (labels == 0)) / np.sum(labels == 0)
        youden = sensitivity + specificity - 1

        if youden > best_youden:
            best_youden = youden
            best_threshold = threshold

    final_predictions = (probabilities >= best_threshold).astype(int)
    final_precision = precision_score(labels, final_predictions)
    final_recall = recall_score(labels, final_predictions)
    best_f1 = 2 * final_precision * final_recall / (final_precision + final_recall)

    return {"precision": final_precision, "recall": final_recall, "f1": best_f1}


def get_logits(label_result_path, fold_idx, epoch):
    """
    Folds and Epochs are indexed from 1.
    """
    epoch_predictions_path = os.path.join(
        label_result_path, f"fold_{fold_idx}", "predictions", f"epoch_{epoch:02d}.csv"
    )
    epoch_predictions = pd.read_csv(epoch_predictions_path)
    logits = epoch_predictions["logits"].values
    labels = epoch_predictions["labels"].values

    return logits, labels


def get_cv_results(args, results_path: str, labels: List[str]):
    """
    Get cross-validation results for each label.

    Args:
        args: Command line arguments.
        results_path: Path to results directory.
        labels: List of labels.

    Returns:
        confidence_intervals: Dictionary of confidence intervals for each metric, label.
        metrics_epoch_fold: Dictionary of metrics for each epoch and fold for each metric, label, epoch, and cv fold.
        best_epochs: Dictionary of best epochs for each label.
    """
    n_folds = args.num_folds
    n_epochs = args.epochs

    confidence_intervals = {}
    metrics_epoch_fold = {}
    best_epochs = {}

    print("Processing cross-validation results.")
    for label in tqdm(labels):
        label_result_path = os.path.join(results_path, label)

        epochs_performance = {
            m: {ep: [] for ep in range(1, n_epochs + 1)} for m in METRICS
        }

        for i in range(1, n_folds + 1):
            cv_fold_path = os.path.join(label_result_path, f"fold_{i}", "summary")
            summary_df = pd.read_csv(cv_fold_path)
            assert len(summary_df) == n_epochs

            for ep in range(1, n_epochs + 1):
                logits, labels = get_logits(label_result_path, i, ep)

                pr_f1 = get_pr_f1(logits, labels)
                for m in ["precision", "recall", "f1"]:
                    epochs_performance[m][ep].append(pr_f1[m])

            for ep, row in summary_df.iterrows():
                for m in ["roc_auc", "pr_auc"]:
                    epochs_performance[m][ep + 1].append(row[m])

        label_epoch_mean_selection = {
            m: [np.mean(epochs_performance[m][ep]) for ep in range(1, n_epochs + 1)]
            for m in METRICS
        }
        label_best_epoch = (
            np.argmax(label_epoch_mean_selection[args.selection_metric]) + 1
        )

        label_summary = {
            m: (
                np.mean(epochs_performance[m][label_best_epoch]),
                np.std(epochs_performance[m][label_best_epoch]),
            )
            for m in METRICS
        }

        confidence_intervals[label] = label_summary
        metrics_epoch_fold[label] = epochs_performance
        best_epochs[label] = label_best_epoch

    return confidence_intervals, metrics_epoch_fold, best_epochs


def get_test_results(args, results_path, best_epochs):
    results = {}
    roc_auc_se = {}

    for label, best_epochs in best_epochs.items():
        label_results = {}
        label_result_path = os.path.join(results_path, label)

        logits, labels = get_logits(label_result_path, "test", best_epochs[label])

        pr_f1 = get_pr_f1(logits, labels)
        for m in ["precision", "recall", "f1"]:
            label_results[m] = pr_f1[m]

        summary_path = os.path.join(label_result_path, "test", "summary.csv")
        summary_df = pd.read_csv(summary_path)
        for m in ["roc_auc", "pr_auc"]:
            label_results[m] = summary_df[m].values[0]

        se = get_roc_auc_se(label_results["roc_auc"], labels)

        results[label] = label_results
        roc_auc_se[label] = se

        return results, roc_auc_se


def save_cv_results(args, results: Dict[str, Dict[str, Tuple[float, float]]]):
    """
    Save results to a csv file.

    Args:
        args: Command line arguments.
        results: Dictionary of results:

    Format of results dict:
    {
        "label": {
            "metric": (mean, std),
            ...
        },
    }
    """
    output_dir = os.path.join(args.output_path, "cv_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    labels = list(results.keys())
    labels.sort()
    table = {"label": [], **{m: [] for m in METRICS}}

    print("Saving cross-validation results.")
    for label in labels:
        table["label"].append(label)
        for metric in METRICS:
            mean, std = results[label][metric]
            ci = args.z_score * std
            table[metric].append(f"{mean:.4f} ± {ci:.4f}")

    df = pd.DataFrame(table)
    df.to_csv(output_dir, index=False)


def save_test_results(args, results: Dict[str, Dict[str, float]]):
    output_dir = os.path.join(args.output_path, "test_results.csv")
    os.makedirs(output_dir, exist_ok=True)

    labels = list(results.keys())
    labels.sort()
    table = {"label": [], **{m: [] for m in METRICS}}

    print("Saving test results.")
    for label in labels:
        table["label"].append(label)
        for metric in METRICS:
            result = results[label][metric]
            table[metric].append(f"{result:.4f}")

    df = pd.DataFrame(table)
    df.to_csv(output_dir, index=False)


def main(args):
    results_path = os.path.join(
        "results", args.run_name, args.checkpoint_name, args.test_name
    )
    labels = os.listdir(results_path)

    cv_summary, _, cv_best_epochs = get_cv_results(args, results_path, labels)
    test_summary, _ = get_test_results(args, results_path, cv_best_epochs)

    save_cv_results(args, cv_summary)
    save_test_results(args, test_summary)

    print("Generating plots...")
    for label in tqdm(labels):

        generate_cv_roc_plot(args, results_path, label, cv_best_epochs[label])
        generate_test_roc_plot(args, results_path, label, cv_best_epochs[label])


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    args = get_args()
    main(args)
