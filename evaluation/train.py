import torch
from sklearn.metrics import roc_auc_score


def train_classifier(
    model,
    optimizer,
    loss_fn,
    train_dataloader,
    val_dataloader,
    num_epochs,
    device,
):
    train_loss_list = []
    train_rocauc_list = []
    val_loss_list = []
    val_rocauc_list = []

    best_val_loss = float("inf")
    best_model_state = model.state_dict()

    for epoch in range(num_epochs):

        train_all_labels = []
        train_all_predictions = []
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for embeddings, labels, masks in train_dataloader:
            embeddings, labels, masks = (
                embeddings.to(device),
                labels.to(device),
                masks.to(device),
            )

            predictions = model(embeddings, masks)
            loss = loss_fn(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_all_labels.append(labels.detach().cpu())
            train_all_predictions.append(predictions.detach().cpu())

        train_labels_cat = torch.cat(train_all_labels)
        train_predictinos_cat = torch.cat(train_all_predictions)
        train_predictinos_cat = torch.nn.functional.softmax(
            train_predictinos_cat, dim=1
        )
        train_predictinos_cat = torch.argmax(train_predictinos_cat, dim=1)

        train_rocauc_list.append(roc_auc_score(train_labels_cat, train_predictinos_cat))

        val_all_labels = []
        val_all_predictions = []
        val_loss = 0.0

        model.eval()
        with torch.no_grad():
            for embeddings, labels, masks in val_dataloader:
                embeddings, labels, masks = (
                    embeddings.to(device),
                    labels.to(device),
                    masks.to(device),
                )

                predictions = model(embeddings, masks)
                loss = loss_fn(predictions, labels)

                val_loss += loss.item()
                val_all_labels.append(labels.detach().cpu())
                val_all_predictions.append(predictions.detach().cpu())

        val_labels_cat = torch.cat(val_all_labels)
        val_predictinos_cat = torch.cat(val_all_predictions)
        val_predictinos_cat = torch.nn.functional.softmax(val_predictinos_cat, dim=1)
        val_predictinos_cat = torch.argmax(val_predictinos_cat, dim=1)
        val_rocauc_list.append(roc_auc_score(val_labels_cat, val_predictinos_cat))

        avg_train_loss = train_loss / len(train_labels_cat)
        avg_val_loss = val_loss / len(val_labels_cat)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

    return {
        "train_loss": train_loss_list,
        "train_rocauc": train_rocauc_list,
        "val_loss": val_loss_list,
        "val_rocauc": val_rocauc_list,
        "state_dict": best_model_state,
    }
