import torch


def get_predictions(model, dataloader, device):
    all_labels = []
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = (
                embeddings.to(device),
                labels.to(device),
            )

            predictions = model(embeddings)
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
        for embeddings, masks, labels in dataloader:
            embeddings, masks, labels = (
                embeddings.to(device),
                masks.to(device),
                labels.to(device),
            )

            predictions = model(embeddings, masks)
            all_labels.append(labels.cpu())
            all_predictions.append(predictions.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_labels, all_predictions
