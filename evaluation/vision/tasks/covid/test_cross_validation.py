import numpy as np
import random
import os
import torch
import gc
from dotenv import load_dotenv
import argparse
import time
import logging
import json

from dinov2.inference.networks import AttnPoolPredictor
from dionv2.inference.extended_datasets import CachedEmbeddings
from evaluation.tasks.covid.datasets import CCCCII

from evaluation.utils.dataset import (
    BalancedSampler,
    split_dataset,
    cross_validation_split,
    collate_sequences,
)
from evaluation.utils.train import train, evaluate, save_classifier
from evaluation.utils.metrics import save_metrics, save_predictions


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
        default=1,
        help="Number of samples in each batch. ",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=32,
        help="Number of steps to accumulate gradients before updating model parameters.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of times to see all training data.",
    )
    parser.add_argument(
        "--select_feature",
        type=str,
        default="cls",
        help="What features to use. Options: cls, patch, or cls_patch",
    )
    parser.add_argument(
        "--save_final",
        action="store_true",
        default=False,
        help="Whether to save the final checkpoint.",
    )
    parser.add_argument(
        "--cross_validation",
        action="store_true",
        default=False,
        help="Whether to perform cross validation in addition. ",
    )

    return parser.parse_args()


def get_datasets(args, label="covid"):
    project_path = os.getenv("PROJECTPATH")
    data_path = os.getenv("DATAPATH")

    embeddings_path = os.path.join(
        project_path,
        "evaluation/cache/ccccii_eval",
        args.run_name,
        args.checkpoint_name,
    )

    embeddings_provider = CachedEmbeddings(
        embeddings_path, select_feature=args.select_feature
    )

    labels_path = os.path.join(
        project_path, f"evaluation/tasks/covid/labels_{label}.csv"
    )

    dataset = CCCCII(
        embeddings_provider,
        labels_path,
    )

    # logging.info(dataset.targets)

    train_dataset, test_dataset = split_dataset(dataset, 0.9)

    return train_dataset, test_dataset


def get_model(args, device):
    classifier_model = AttnPoolPredictor(args.embed_dim, args.hidden_dim)

    classifier_model = torch.nn.DataParallel(classifier_model)
    classifier_model.to(device)

    return classifier_model


def train_and_evaluate_model(
    args, train_dataset, val_dataset, device, output_path, save_final=False
):
    classifier_model = get_model(args, device)
    logging.info(f"Using devices: {classifier_model.device_ids}")

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
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
    )

    epoch_iterations = len(train_loader)

    for i in range(args.epochs):
        logging.info(f"Epoch: {i + 1}/{args.epochs}")

        train(
            classifier_model,
            optimizer,
            criterion,
            train_loader,
            epoch_iterations,
            args.accum_steps,
            device,
            verbose=True,
        )

        eval_dict = evaluate(classifier_model, val_loader, device, verbose=True)

        save_metrics(eval_dict["metrics"], output_path, "summary")

        logits_path = os.path.join(output_path, "predictions")
        save_predictions(
            eval_dict["map_ids"],
            eval_dict["logits"],
            eval_dict["labels"],
            logits_path,
            f"epoch_{i:02d}",
        )

    if save_final:
        final_model_path = os.path.join(output_path, "model")
        save_classifier(final_model_path, classifier_model)

    del classifier_model, optimizer, train_loader, val_loader, criterion
    torch.cuda.empty_cache()
    gc.collect()


def cross_validation(args, train_val_dataset, output_path, device):
    cv_folds = cross_validation_split(train_val_dataset, 5)

    for fold_idx, (train_dataset, val_dataset) in enumerate(cv_folds):

        logging.info(f"Running fold {fold_idx + 1}/5\n")

        fold_output_path = os.path.join(output_path, f"fold_{fold_idx}")
        train_and_evaluate_model(
            args,
            train_dataset,
            val_dataset,
            device,
            fold_output_path,
        )

        torch.cuda.empty_cache()
        gc.collect()


def run_evaluation(args, label):
    project_path = os.getenv("PROJECTPATH")

    output_path = os.path.join(
        project_path,
        args.output_dir,
        "results",
        label,
    )
    os.makedirs(output_path, exist_ok=True)

    device = torch.device("cuda")

    train_val_dataset, test_dataset = get_datasets(args, label)

    if args.cross_validation:
        logging.info(f'Running cross validation for "{label}"\n')
        cross_validation(args, train_val_dataset, output_path, device)

    logging.info(f"Running test evaluation for {label}\n")

    test_output_path = os.path.join(output_path, "test")
    train_and_evaluate_model(
        args,
        train_val_dataset,
        test_dataset,
        device,
        test_output_path,
        save_final=args.save_final,
    )


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting")

    assert torch.cuda.is_available(), "CUDA is not available"
    args = get_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    for label in ["covid", "pneumonia"]:
        run_evaluation(args, label)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    load_dotenv()

    main()
