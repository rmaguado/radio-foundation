import numpy as np
import random
import os
import torch
from dotenv import load_dotenv
import argparse

from evaluation.utils.networks import FullScanPatchPredictor
from evaluation.tasks.ct_rate.datasets import CT_RATE
from evaluation.utils.dataset import BalancedSampler, split_dataset
from evaluation.utils.train import train, evaluate
from evaluation.utils.metrics import save_metrics


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
    "Interlobular septal thickening"
]


def get_args():
    parser = argparse.ArgumentParser()
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
        "--train_iters",
        type=int,
        default=320,
        help="Number of batches to iterate over the training set.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of batches to iterate over the training set.",
    )
    return parser.parse_args()


def get_dataloaders(args, label):
    data_path = os.getenv("DATAPATH")
    metadata_path = os.path.join(
        data_path, "CT-RATE/multi_abnormality_labels/train_predicted_labels.csv"
    )

    dataset = CT_RATE(args.run_name, args.checkpoint_name, metadata_path, label)

    train_dataset, val_dataset = split_dataset(dataset, 0.8)

    def collate_fn(batch):
        embeddings, labels = zip(*batch)
        batch_size = len(embeddings)

        _, num_tokens, embed_dim = embeddings[0].shape

        max_axial_dim = max([embedding.shape[0] for embedding in embeddings])

        padded_embeddings = torch.zeros(
            batch_size, max_axial_dim, num_tokens, embed_dim
        )
        mask = torch.zeros(batch_size, max_axial_dim)

        for i, embedding in enumerate(embeddings):
            padded_embeddings[i, : embedding.shape[0]] = embedding
            mask[i, : embedding.shape[0]] = 1

        return padded_embeddings, mask, torch.tensor(labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        sampler=BalancedSampler(train_dataset),
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        sampler=BalancedSampler(val_dataset),
        num_workers=1,
        persistent_workers=True,
    )

    return train_loader, val_loader


def get_model(args, device):
    classifier_model = FullScanPatchPredictor(
        args.embed_dim,
        args.hidden_dim,
        num_labels=1,
        patch_resample_dim=args.patch_resample_dim,
    )
    classifier_model.to(device)

    return classifier_model


def run_evaluation(args, label):

    print(f'Running evaluation for "{label}"')

    project_path = os.getenv("PROJECTPATH")

    device = torch.device("cuda")

    train_loader, val_loader = get_dataloaders(args, label)

    classifier_model = get_model(args, device)
    optimizer = torch.optim.AdamW(classifier_model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    output_path = os.path.join(
        project_path, args.output_dir, args.run_name, args.checkpoint_name
    )
    os.makedirs(output_path, exist_ok=True)

    for i in range(args.epochs):
        train_metrics = train(
            classifier_model,
            optimizer,
            criterion,
            train_loader,
            args.train_iters,
            args.accum_steps,
            device,
        )
        print(
            f"PR AUC: {train_metrics['pr_auc']:.4f} - ROC AUC: {train_metrics['roc_auc']:.4f}\n"
        )

        eval_metrics = evaluate(
            classifier_model, val_loader, device=device, max_eval_n=64
        )
        print(
            f"PR AUC: {eval_metrics['pr_auc']:.4f} - ROC AUC: {eval_metrics['roc_auc']:.4f}\n"
        )

        save_metrics(eval_metrics, output_path, label)


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
