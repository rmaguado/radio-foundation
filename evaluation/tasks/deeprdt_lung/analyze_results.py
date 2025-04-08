import pandas as pd
import os
import random
import numpy as np
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logit, expit
from scipy.stats import t


METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall"]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outcome",
        type=str,
        default="response",
        help="Outcome variable to be predicted.",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default="True",
        help="Category to be predicted.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="outcome",
        help="Name of experiment. Used to differentiate save files.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="base10pat",
        help="Name of folder in /runs used to generate embeddings",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="f",
        help="Name of checkpoint in /runs/{$run_name}/checkpoints",
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
        default=100,
        help="Number of epochs in training. ",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="evaluation/tasks/deeprdt_lung/results",
        help="Path of training results",
    )

    return parser.parse_args()


def get_cv_results(args):

    label = args.outcome
    results_path = os.path.join(
        args.results_path,
        args.run_name,
        args.outcome,
        args.experiment_name,
    )
    n_folds = args.num_folds
    n_epochs = args.epochs

    epochs_performance = {m: {ep: [] for ep in range(n_epochs)} for m in ["roc_auc"]}

    for fold in range(n_folds):
        cv_fold_path = os.path.join(results_path, f"fold_{fold}", "summary.csv")
        summary_df = pd.read_csv(cv_fold_path)
        assert len(summary_df) == n_epochs

        for ep, row in summary_df.iterrows():
            for m in ["roc_auc"]:
                epochs_performance[m][ep].append(row[m])

    generate_val_epoch_plot(args, results_path, label, epochs_performance)

    return epochs_performance


def auc_ci_logit(auc_values, confidence=0.95):
    """
    Compute the logit-transformed confidence interval for AUC values.

    Args:
        auc_values (list or np.array): AUC values from cross-validation or epochs.
        confidence (float): Confidence level (default 0.95).

    Returns:
        mean_auc: Mean AUC.
        (lower_ci, upper_ci): Tuple containing lower and upper bounds of the CI.
    """
    auc_values = np.clip(auc_values, 1e-6, 1 - 1e-6)  # Prevent logit issues at 0 or 1
    logit_vals = logit(auc_values)

    mean_logit = np.mean(logit_vals)
    se_logit = np.std(logit_vals, ddof=1) / np.sqrt(len(logit_vals))

    df = len(logit_vals) - 1
    t_val = t.ppf((1 + confidence) / 2, df)

    ci_lower_logit = mean_logit - t_val * se_logit
    ci_upper_logit = mean_logit + t_val * se_logit

    mean_auc = expit(mean_logit)
    ci_lower = expit(ci_lower_logit)
    ci_upper = expit(ci_upper_logit)

    return mean_auc, (ci_lower, ci_upper)


def get_roc_auc(args, results_path: str, label: str):
    n_folds = args.num_folds
    n_epochs = args.epochs

    auc_roc = []
    auc_se = []

    for epoch in range(n_epochs):
        pred = []
        for fold in range(n_folds):
            cv_fold_path = os.path.join(
                results_path, f"fold_{fold}", "predictions", f"epoch_{epoch:02d}.csv"
            )
            pred.append(pd.read_csv(cv_fold_path))
        pred = pd.concat(pred)
        aroc = roc_auc_score(pred["labels"], pred["logits"])
        aroc_se = get_roc_auc_se(aroc, pred["labels"])
        auc_roc.append(aroc)
        auc_se.append(aroc_se)

    pred["probs"] = 1 / (1 + np.exp(-pred["logits"]))

    prob1 = pred[pred["labels"] == True]["probs"]
    prob0 = pred[pred["labels"] == False]["probs"]

    return auc_roc, auc_se, prob0, prob1


def get_roc_auc_se(auc, label):
    n1 = np.sum(label)
    n2 = len(label) - n1
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    se = np.sqrt(
        (auc * (1 - auc) + (n1 - 1) * (q1 - auc**2) + (n2 - 1) * (q2 - auc**2))
        / (n1 * n2)
    )
    return se


def generate_val_epoch_plot(args, results_path, label, train_results, density=True):

    metric = "roc_auc"

    output_dir = os.path.join(results_path, f"{metric}_{label}.png")

    nfolds = args.num_folds

    plt.figure(1, figsize=(10, 25))

    plt.subplot(3, 1, 1)
    for fold_idx in range(nfolds):
        epoch_metrics = [x[fold_idx] for x in list(train_results[metric].values())]

        plt.plot(range(len(epoch_metrics)), epoch_metrics, label=f"Fold {fold_idx}")

    mean = []
    cilo = []
    cihi = []

    for epoch_aucs in list(train_results[metric].values()):
        mean_auc, (ci_lower, ci_upper) = auc_ci_logit(epoch_aucs)
        mean.append(mean_auc)
        cilo.append(ci_lower)
        cihi.append(ci_upper)

    plt.plot(range(len(epoch_metrics)), mean, label="Mean", color="black", linewidth=3)
    plt.plot(
        range(len(epoch_metrics)),
        cihi,
        label="_CI hi",
        color="black",
        linewidth=1,
        linestyle=":",
    )
    plt.plot(
        range(len(epoch_metrics)),
        cilo,
        label="_CI low",
        color="black",
        linewidth=1,
        linestyle=":",
    )

    plt.ylabel("AUC ROC")
    plt.legend()
    plt.axhline(y=0.5, color="black", linestyle="--")
    plt.ylim((0.4, 1))
    plt.title(f"Test AUC ROC for {label} = {args.predict} (folds and mean)")
    plt.axhline(y=max(mean), color="yellow", linestyle="--")

    maxauc = max(mean)
    idx_max = mean.index(maxauc)
    final = args.epochs - 1
    plt.text(
        0,
        0.98,
        f"AUC ROC ({idx_max:4d}): {mean[idx_max]:.3f} [{cilo[idx_max]:.3f}; {cihi[idx_max]:.3f}]",
    )
    plt.text(
        0,
        0.96,
        f"AUC ROC ({final+1:4d}): {mean[final]:.3f} [{cilo[final]:.3f}; {cihi[final]:.3f}]",
    )

    plt.subplot(3, 1, 2)

    roc_auc, roc_se, prob0, prob1 = get_roc_auc(args, results_path, label)
    cilo = np.asarray(roc_auc) - np.asarray(roc_se) * 1.96
    cihi = np.asarray(roc_auc) + np.asarray(roc_se) * 1.96

    plt.plot(range(len(roc_auc)), roc_auc, label=f"ROC", color="black", linewidth=3)
    plt.plot(
        range(len(roc_auc)),
        cihi,
        label="_CI hi",
        color="black",
        linewidth=1,
        linestyle=":",
    )
    plt.plot(
        range(len(roc_auc)),
        cilo,
        label="_CI low",
        color="black",
        linewidth=1,
        linestyle=":",
    )

    plt.xlabel("Epoch")
    plt.ylabel("AUC ROC")
    plt.legend()
    plt.axhline(y=0.5, color="black", linestyle="--")
    plt.ylim((0.4, 1))
    plt.title(f"Folds combined")
    plt.axhline(y=max(roc_auc), color="yellow", linestyle="--")

    plt.subplot(3, 1, 3)

    if density:
        # Plot density
        sns.kdeplot(prob0, label="Reference", fill=True, alpha=0.5)
        sns.kdeplot(prob1, label=args.predict, fill=True, alpha=0.5)
    else:
        # Plot histograms for each class
        bins = np.linspace(0, 1, 41)
        plt.hist(prob0, bins=bins, alpha=0.5, label="Reference", density=True)
        plt.hist(prob1, bins=bins, alpha=0.5, label=args.predict, density=True)

    # Plot means
    mean0 = np.mean(prob0)
    mean1 = np.mean(prob1)
    plt.axvline(mean0, color="blue", linestyle="--", label=f"Mean Ref: {mean0:.2f}")
    plt.axvline(
        mean1, color="orange", linestyle="--", label=f"Mean {args.predict}: {mean1:.2f}"
    )

    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Density of Predicted Probabilities by Class")
    plt.xlim((0, 1))
    plt.legend()
    plt.grid(True)

    plt.savefig(output_dir)

    print(f"plot saved: {output_dir} max AUC: {round(max(roc_auc),3)}")


def main(args):
    train_results = get_cv_results(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
