import torch
from torch.utils.data import Dataset


def collate_regression(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    embeddings_list, labels_list = zip(*batch)

    max_len = embeddings_list[0].shape[0]
    padded_embeddings = torch.zeros(
        len(embeddings_list), max_len, embeddings_list[0].shape[1]
    )
    masks = torch.zeros(len(embeddings_list), max_len, dtype=torch.bool)

    for i, embedding in enumerate(embeddings_list):
        seq_len = embedding.shape[0]
        padded_embeddings[i, :seq_len, :] = embedding
        masks[i, :seq_len] = True

    labels = torch.tensor(labels_list, dtype=torch.float32).unsqueeze(1)

    return padded_embeddings, labels, masks


def collate_classification(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)

    embeddings_list, labels_list = zip(*batch)

    max_len = embeddings_list[0].shape[0]
    padded_embeddings = torch.zeros(
        len(embeddings_list), max_len, embeddings_list[0].shape[1]
    )
    masks = torch.zeros(len(embeddings_list), max_len, dtype=torch.bool)

    for i, embedding in enumerate(embeddings_list):
        seq_len = embedding.shape[0]
        padded_embeddings[i, :seq_len, :] = embedding
        masks[i, :seq_len] = True

    labels = torch.tensor(labels_list, dtype=torch.float32)

    return padded_embeddings, labels, masks


def get_class_weights(labels):
    unique_labels = list(set(labels))
    unique_labels.sort()
    assert all(isinstance(l, int) and l >= 0 for l in unique_labels)

    num_classes = len(unique_labels)
    return torch.tensor(
        [len(labels) / num_classes / labels.count(x) for x in range(num_classes)]
    )


def get_pos_weights(labels):
    unique_labels = list(set(labels))
    unique_labels.sort()
    assert all(isinstance(l, int) and l >= 0 for l in unique_labels)

    num_classes = len(unique_labels)
    assert num_classes == 2

    return torch.tensor([labels.count(0) / labels.count(1)])


class EmbeddingDataset(Dataset):
    def __init__(
        self, patient_ids, id_to_path, labels, add_noise=False, sigma=0.05, p=0.5
    ):
        self.patient_ids = patient_ids
        self.id_to_path = id_to_path
        self.labels = labels
        self.add_noise = add_noise
        self.sigma = sigma
        self.p = p

    def add_embedding_noise(self, embeddings):
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(embeddings) * self.sigma
            embeddings = embeddings + noise
        return embeddings

    def __len__(self):
        return len(self.patient_ids)

    def get_labels(self):
        return [self.labels[x] for x in self.patient_ids]

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        embedding_path = self.id_to_path[patient_id]

        embedding_data = torch.load(embedding_path, mmap=True)
        embeddings = embedding_data["cls"]
        label = float(self.labels[patient_id])

        if self.add_noise:
            embeddings = self.add_embedding_noise(embeddings)

        return embeddings, label
