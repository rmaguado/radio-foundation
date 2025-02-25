from evaluation.utils.metrics import compute_metrics

from tqdm import tqdm
import torch
import numpy as np


def train(
    classifier_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    train_epochs: int,
    accum_steps: int,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
) -> int:
    def evaluate():
        classifier_model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            val_iter = iter(val_loader)
            for idx, (embeddings, _, labels) in enumerate(tqdm(val_iter)):
                embeddings = embeddings.to(device)
                logits = classifier_model(embeddings)

                all_logits.append(logits.flatten().cpu().numpy())
                all_labels.append(labels.float().cpu().numpy())

        all_logits = np.array(all_logits).flatten()
        all_labels = np.array(all_labels).flatten()

        return compute_metrics(all_logits, all_labels)

    classifier_model.train()
    losses = []
    grad_norms = []
    val_metrics = []

    for train_epoch in range(train_epochs):

        total_loss = 0.0
        total_grad_norm = 0.0

        optimizer.zero_grad()

        train_iter = iter(train_loader)

        for idx, (embeddings, _, labels) in enumerate(tqdm(train_iter)):
            logits = classifier_model(embeddings.to(device))
            loss = criterion(logits.flatten(), labels.float().to(device))
            total_loss += loss.item()
            loss.backward()

            total_grad_norm += torch.nn.utils.clip_grad_norm_(
                classifier_model.parameters(), 1.0
            ).item()

            if ((idx + 1) % accum_steps == 0) or ((idx + 1) == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

        losses.append(total_loss / len(train_loader))
        grad_norms.append(total_grad_norm / len(train_loader))

        epoch_metrics = evaluate()
        val_metrics.append(epoch_metrics)

        pr_auc = epoch_metrics["pr_auc"]
        roc_auc = epoch_metrics["roc_auc"]

        print(
            f"Epoch {train_epoch + 1}/{train_epochs} - PR AUC: {pr_auc:.4f} - ROC AUC: {roc_auc:.4f}"
        )

    return losses, grad_norms, val_metrics
