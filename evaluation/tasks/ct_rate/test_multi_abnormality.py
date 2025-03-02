import numpy as np
import random
import os
import torch
from dotenv import load_dotenv
import argparse

from evaluation.utils.networks import FullScanClassPredictor
from evaluation.tasks.ct_rate.datasets import CT_RATE
from evaluation.utils.dataset import BalancedSampler, split_dataset
from evaluation.utils.train import train
from evaluation.utils.metrics import save_metrics


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
        "--embed_dim",
        type=int,
        default=512,
        help="Dimension of embeddings",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=1024,
        help="Dimension of classifier model",
    )
    parser.add_argument(
        "--train_epochs",
        type=int,
        default=30,
        help="Number of epochs to train classifier model",
    )
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=32,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/tasks/ct_rate/results",
        help="Directory to save results",
    )

    return parser.parse_args()


def get_dataloaders(args, label):
    data_path = os.getenv("DATAPATH")
    metadata_path = os.path.join(
        data_path, "CT-RATE/multi_abnormality_labels/valid_predicted_labels.csv"
    )

    dataset = CT_RATE(args.run_name, args.checkpoint_name, metadata_path, label)

    train_dataset, val_dataset = split_dataset(dataset, 0.8)

    def collate_fn(batch):
        embeddings, labels = zip(*batch)

        _, num_tokens, embed_dim = embeddings[0].shape

        max_length = max([embedding.shape[0] for embedding in embeddings])

        padded_embeddings = torch.zeros(
            len(embeddings), max_length, num_tokens, embed_dim
        )
        mask = torch.zeros(len(embeddings), max_length)

        for i, embedding in enumerate(embeddings):
            padded_embeddings[i, : embedding.shape[0]] = embedding
            mask[i, : embedding.shape[0]] = 1

        return padded_embeddings, mask, torch.tensor(labels)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        sampler=BalancedSampler(train_dataset),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader


def get_model(args, device):
    classifier_model = FullScanClassPredictor(
        args.embed_dim, args.hidden_dim, num_labels=1
    ).to(device)

    return classifier_model


def run_evaluation(args, label):

    project_path = os.getenv("PROJECTPATH")

    device = torch.device("cuda")

    train_loader, val_loader = get_dataloaders(args, label)

    classifier_model = get_model(args, device)
    optimizer = torch.optim.SGD(
        classifier_model.parameters(), momentum=0.9, weight_decay=0.01, lr=1e-3
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    _, _, val_metrics = train(
        classifier_model,
        optimizer,
        criterion,
        train_loader,
        val_loader,
        args.train_epochs,
        args.accum_steps,
        device,
    )

    output_path = os.path.join(
        project_path, args.output_dir, args.run_name, args.checkpoint_name
    )
    os.makedirs(output_path, exist_ok=True)

    save_metrics(val_metrics, output_path, label)


def run_benchmarks(args):
    labels = [
        "Lung nodule",
        "Lung opacity",
        "Arterial wall calcification",
        "Pulmonary fibrotic sequela",
    ]

    for label in labels:
        run_evaluation(args, label)


if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)

    load_dotenv()

    assert torch.cuda.is_available(), "CUDA is not available"
    args = get_args()
    run_benchmarks(args)
