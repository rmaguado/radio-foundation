import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


def get_predictions(model, dataloader, device):
    all_labels = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for embeddings, labels in tqdm(dataloader, total=len(dataloader)):
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            predictions = model(embeddings).flatten()
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_labels, all_predictions


def get_predictions_stack(model, dataloader, device):
    all_labels = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for embeddings, mask, labels in tqdm(dataloader, total=len(dataloader)):
            embeddings = embeddings.to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            predictions = model(embeddings, mask).flatten()
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_labels, all_predictions


def train_classifier(
    model,
    optimizer,
    loss_fn,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device,
    select_criteria="f1",
    threshold=0.5,
):
    """Main function to train and validate the classifier."""
    assert select_criteria in ["loss", "rocauc", "precision", "recall", "f1"]

    history = {
        "train_loss": [],
        "train_rocauc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_loss": [],
        "val_rocauc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_val_metric = -1 if select_criteria != "loss" else float("inf")
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_all_labels = []
        train_all_predictions = []

        for embeddings, labels in tqdm(
            train_dataloader, desc=f"Train {epoch+1}/{num_epochs}", leave=False
        ):
            embeddings, labels = embeddings.to(device), labels.to(device)
            predictions = model(embeddings).flatten()
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_all_labels.append(labels.detach().cpu())
            train_all_predictions.append(predictions.detach().cpu())

        train_labels_cat = torch.cat(train_all_labels)
        train_predictions_cat = torch.cat(train_all_predictions)
        train_probabilities = torch.sigmoid(train_predictions_cat)
        train_binary_predictions = (train_probabilities >= threshold).long()

        history["train_loss"].append(train_loss / len(train_dataloader))
        history["train_rocauc"].append(
            roc_auc_score(train_labels_cat, train_probabilities)
        )
        history["train_precision"].append(
            precision_score(train_labels_cat, train_binary_predictions, zero_division=0)
        )
        history["train_recall"].append(
            recall_score(train_labels_cat, train_binary_predictions, zero_division=0)
        )
        history["train_f1"].append(
            f1_score(train_labels_cat, train_binary_predictions, zero_division=0)
        )

        model.eval()
        val_loss = 0.0
        val_all_labels = []
        val_all_predictions = []
        with torch.no_grad():
            for embeddings, labels in tqdm(
                val_dataloader, desc=f"Valid {epoch+1}/{num_epochs}", leave=False
            ):
                embeddings, labels = embeddings.to(device), labels.to(device)
                predictions = model(embeddings).flatten()
                loss = loss_fn(predictions, labels)
                val_loss += loss.item()
                val_all_labels.append(labels.detach().cpu())
                val_all_predictions.append(predictions.detach().cpu())

        val_labels_cat = torch.cat(val_all_labels)
        val_predictions_cat = torch.cat(val_all_predictions)
        val_probabilities = torch.sigmoid(val_predictions_cat)
        val_binary_predictions = (val_probabilities >= threshold).long()

        history["val_loss"].append(val_loss / len(val_dataloader))
        history["val_rocauc"].append(roc_auc_score(val_labels_cat, val_probabilities))
        history["val_precision"].append(
            precision_score(val_labels_cat, val_binary_predictions, zero_division=0)
        )
        history["val_recall"].append(
            recall_score(val_labels_cat, val_binary_predictions, zero_division=0)
        )
        history["val_f1"].append(
            f1_score(val_labels_cat, val_binary_predictions, zero_division=0)
        )

        current_metric = history[f"val_{select_criteria}"][-1]
        if (select_criteria == "loss" and current_metric < best_val_metric) or (
            select_criteria != "loss" and current_metric > best_val_metric
        ):
            best_val_metric = current_metric
            best_model_state = model.state_dict()

    history["state_dict"] = best_model_state
    return history


def train_classifier_stack(
    model,
    optimizer,
    loss_fn,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device,
    select_criteria="f1",
    threshold=0.5,
):
    """Main function to train and validate the classifier."""
    assert select_criteria in ["loss", "rocauc", "precision", "recall", "f1"]

    history = {
        "train_loss": [],
        "train_rocauc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "val_loss": [],
        "val_rocauc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    best_val_metric = -1 if select_criteria != "loss" else float("inf")
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_all_labels = []
        train_all_predictions = []

        for embeddings, mask, labels in tqdm(
            train_dataloader, desc=f"Train {epoch+1}/{num_epochs}", leave=False
        ):
            embeddings, mask, labels = (
                embeddings.to(device),
                mask.to(device),
                labels.to(device),
            )
            predictions = model(embeddings, mask).flatten()
            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_all_labels.append(labels.detach().cpu())
            train_all_predictions.append(predictions.detach().cpu())

        train_labels_cat = torch.cat(train_all_labels)
        train_predictions_cat = torch.cat(train_all_predictions)
        train_probabilities = torch.sigmoid(train_predictions_cat)
        train_binary_predictions = (train_probabilities >= threshold).long()

        history["train_loss"].append(train_loss / len(train_dataloader))
        history["train_rocauc"].append(
            roc_auc_score(train_labels_cat, train_probabilities)
        )
        history["train_precision"].append(
            precision_score(train_labels_cat, train_binary_predictions, zero_division=0)
        )
        history["train_recall"].append(
            recall_score(train_labels_cat, train_binary_predictions, zero_division=0)
        )
        history["train_f1"].append(
            f1_score(train_labels_cat, train_binary_predictions, zero_division=0)
        )

        model.eval()
        val_loss = 0.0
        val_all_labels = []
        val_all_predictions = []
        with torch.no_grad():
            for embeddings, mask, labels in tqdm(
                val_dataloader, desc=f"Valid {epoch+1}/{num_epochs}", leave=False
            ):
                embeddings, mask, labels = (
                    embeddings.to(device),
                    mask.to(device),
                    labels.to(device),
                )
                predictions = model(embeddings, mask).flatten()
                loss = loss_fn(predictions, labels)
                val_loss += loss.item()
                val_all_labels.append(labels.detach().cpu())
                val_all_predictions.append(predictions.detach().cpu())

        val_labels_cat = torch.cat(val_all_labels)
        val_predictions_cat = torch.cat(val_all_predictions)
        val_probabilities = torch.sigmoid(val_predictions_cat)
        val_binary_predictions = (val_probabilities >= threshold).long()

        history["val_loss"].append(val_loss / len(val_dataloader))
        history["val_rocauc"].append(roc_auc_score(val_labels_cat, val_probabilities))
        history["val_precision"].append(
            precision_score(val_labels_cat, val_binary_predictions, zero_division=0)
        )
        history["val_recall"].append(
            recall_score(val_labels_cat, val_binary_predictions, zero_division=0)
        )
        history["val_f1"].append(
            f1_score(val_labels_cat, val_binary_predictions, zero_division=0)
        )

        current_metric = history[f"val_{select_criteria}"][-1]
        if (select_criteria == "loss" and current_metric < best_val_metric) or (
            select_criteria != "loss" and current_metric > best_val_metric
        ):
            best_val_metric = current_metric
            best_model_state = model.state_dict()

    history["state_dict"] = best_model_state
    return history
