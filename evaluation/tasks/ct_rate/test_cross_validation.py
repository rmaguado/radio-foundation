import numpy as np
import random
import os
import torch
from dotenv import load_dotenv
import argparse
import time

from evaluation.utils.networks import (
    FullScanClassPredictor,
    FullScanPatchPredictor,
    FullScanClassPatchPredictor,
)
from evaluation.extended_datasets import CachedEmbeddings
from evaluation.tasks.ct_rate.datasets import CT_RATE

from evaluation.utils.dataset import (
    BalancedSampler,
    cross_validation_split,
    collate_sequences,
)
from evaluation.utils.train import train, evaluate
from evaluation.utils.metrics import save_metrics, save_predictions, print_metrics

ALL_LABELS = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


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
        "--embed_dim",
        type=int,
        default=768,
        help="Dimension of embeddings. ",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Dimension of classifier model. ",
    )
    parser.add_argument(
        "--patch_resample_dim",
        type=int,
        default=16,
        help="Number of patches to resample from each scan. ",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of samples in each batch. ",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=8,
        help="Number of steps to accumulate gradients before updating model parameters.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of times to validate between training.",
    )
    parser.add_argument(
        "--cls_only",
        type=bool,
        default=False,
        help="Wether to use only the CLS token or CLS + Patch tokens.",
    )
    parser.add_argument(
        "--patch_only",
        type=bool,
        default=False,
        help="Wether to use only the patch token or CLS + Patch tokens.",
    )
    return parser.parse_args()


def get_datasets(args, label):
    project_path = os.getenv("PROJECTPATH")
    data_path = os.getenv("DATAPATH")

    train_embeddings_path = os.path.join(
        project_path,
        "evaluation/cache/CT-RATE_train_eval",
        args.run_name,
        args.checkpoint_name,
    )
    test_embeddings_path = os.path.join(
        project_path,
        "evaluation/cache/CT-RATE_valid_eval",
        args.run_name,
        args.checkpoint_name,
    )
    train_embeddings_provider = CachedEmbeddings(
        train_embeddings_path, cls_only=args.cls_only, patch_only=args.patch_only
    )
    test_embeddings_provider = CachedEmbeddings(
        test_embeddings_path, cls_only=args.cls_only, patch_only=args.patch_only
    )

    train_metadata_path = os.path.join(
        data_path, "niftis/CT-RATE/multi_abnormality_labels/train_predicted_labels.csv"
    )
    test_metadata_path = os.path.join(
        data_path, "niftis/CT-RATE/multi_abnormality_labels/valid_predicted_labels.csv"
    )

    train_dataset = CT_RATE(
        train_embeddings_provider,
        train_metadata_path,
        label,
    )

    test_dataset = CT_RATE(
        test_embeddings_provider,
        test_metadata_path,
        label,
    )

    return train_dataset, test_dataset


def get_model(args, device):
    if args.cls_only:
        classifier_model = FullScanClassPredictor(
            args.embed_dim, args.hidden_dim, num_labels=1
        )
    elif args.patch_only:
        classifier_model = FullScanPatchPredictor(
            args.embed_dim,
            args.hidden_dim,
            num_labels=1,
            patch_resample_dim=args.patch_resample_dim,
        )
    else:
        classifier_model = FullScanClassPatchPredictor(
            args.embed_dim,
            args.hidden_dim,
            num_labels=1,
            patch_resample_dim=args.patch_resample_dim,
        )
    classifier_model.to(device)

    return classifier_model


def train_and_evaluate_model(
    args,
    train_dataset,
    val_dataset,
    device,
    output_path,
):
    logits_path = os.path.join(output_path, "predictions")
    os.makedirs(logits_path, exist_ok=True)

    classifier_model = get_model(args, device)
    optimizer = torch.optim.SGD(
        classifier_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        sampler=BalancedSampler(train_dataset),
        num_workers=8,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=8,
        persistent_workers=True,
    )

    positive_n = sum(train_dataset.get_target(i) for i in range(len(train_dataset)))
    negative_n = len(train_dataset) - positive_n

    epoch_iterations = len(train_loader)

    print(f"P: {positive_n}, N: {negative_n}")

    t0 = time.time()

    for i in range(args.epochs):
        print(f"Epoch: {i + 1}/{args.epochs}")

        train_metrics, _ = train(
            classifier_model,
            optimizer,
            criterion,
            train_loader,
            epoch_iterations,
            args.accum_steps,
            device,
            verbose=False,
        )

        print_metrics(train_metrics, "train | ")

        eval_metrics, (eval_logits, eval_labels) = evaluate(
            classifier_model,
            val_loader,
            device=device,
            max_eval_n=None,
            verbose=False,
        )

        print_metrics(eval_metrics, "valid | ")

        save_metrics(eval_metrics, output_path, "summary")
        save_predictions(eval_logits, eval_labels, logits_path, f"epoch_{i:02d}")

    tf = time.time() - t0

    print("Final validation:")
    print_metrics(eval_metrics, "valid | ")
    print(f"Finished training in {tf:.4f} seconds. \n")


def run_evaluation(args, label):
    project_path = os.getenv("PROJECTPATH")

    output_path = os.path.join(
        project_path,
        args.output_dir,
        args.run_name,
        args.checkpoint_name,
        args.experiment_name,
        label,
    )
    os.makedirs(output_path, exist_ok=True)

    print(f'Running cross validation for "{label}"\n')

    device = torch.device("cuda")

    train_val_dataset, test_dataset = get_datasets(args, label)

    cv_folds = cross_validation_split(train_val_dataset, 5)

    for fold_idx, (train_dataset, val_dataset) in enumerate(cv_folds):

        print(f"Running fold {fold_idx + 1}/5\n")

        train_and_evaluate_model(
            args,
            train_dataset,
            val_dataset,
            device,
            os.path.join(output_path, f"fold_{fold_idx}"),
        )

    print(f"Running test evaluation for {label}\n")

    train_and_evaluate_model(
        args, train_val_dataset, test_dataset, device, os.path.join(output_path, "test")
    )


def run_benchmarks(args):
    labels = ALL_LABELS

    for label in labels:
        run_evaluation(args, label)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    load_dotenv()

    assert torch.cuda.is_available(), "CUDA is not available"
    args = get_args()
    run_benchmarks(args)
