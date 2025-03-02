import os

os.environ["OPENBLAS_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
os.environ["OMP_NUM_THREADS"] = "16"

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import tqdm
import math
from scipy.stats import zscore
import torch.nn.functional as F
import pickle
import time
import argparse


ABNORMALITIES = [
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


def map_accessions_to_labels(accession, df):
    accession = accession.replace(".npy", ".nii.gz")
    row = df[df["VolumeName"] == accession]
    if not row.empty:
        try:
            return row.iloc[0, 1:]
        except:
            return 0
    else:
        print(f"Label not found for {accession}")
        return np.zeros(df.shape[1] - 1)


def map_accessions_to_meta(accession, df):
    # Manufacturer: 0: 'Siemens Healthineers' or 'SIEMENS', 1: 'Philips', 2: 'PNMS'
    # PatientSex: 0: 'F', 1: 'M', -1.0: nan
    # PatientAge: '049Y' -> 49.0, nan -> -1.0
    # XYSpacing: '0.703125' -> 0.703125, nan -> -1.0
    # ZSpacing: '1.0' -> 1.0, nan -> -1.0
    accession = accession.replace(".npy", ".nii.gz")
    row = df[df["VolumeName"] == accession]

    labels = np.ndarray(5)
    if not row.empty:
        try:
            manufacturer = row["Manufacturer"].iloc[0]
            sex = row["PatientSex"].iloc[0]
            age = row["PatientAge"].iloc[0]
            xyspacing = row["XYSpacing"].iloc[0]
            zspacing = row["ZSpacing"].iloc[0]

            labels[0] = (
                0
                if manufacturer == "Siemens Healthineers" or manufacturer == "SIEMENS"
                else 1 if manufacturer == "Philips" else 2
            )
            labels[1] = 0 if sex == "F" else 1 if sex == "M" else -1.0
            labels[2] = float(age[:-1]) if not pd.isna(age) else -1.0
            labels[3] = float(eval(xyspacing)[0]) if not pd.isna(xyspacing) else -1.0
            labels[4] = float(zspacing) if not pd.isna(zspacing) else -1.0

            return labels
        except Exception as e:
            print("failed", e)
            return np.zeros(5)


def process_file(file_name, directory, abnormalities_df, metadata_df):
    if file_name.endswith(".npy"):
        file_path = os.path.join(directory, file_name)
        data = np.load(file_path, mmap_mode="r").mean(axis=1).flatten()
        labels = map_accessions_to_labels(file_name, abnormalities_df)
        meta = map_accessions_to_meta(file_name, metadata_df)
        return data, labels, meta
    return None, None, None


def load_latents_and_labels(directory, abnormalities_df, metadata_df):
    files = [f for f in os.listdir(directory) if f.endswith(".npy")]

    latents = []
    labels = []
    metas = []
    for file_name in tqdm.tqdm(files):
        data, label, meta = process_file(
            file_name, directory, abnormalities_df, metadata_df
        )
        if data is not None:
            latents.append(data)
            labels.append(label)
            metas.append(meta)

    latents = np.vstack(latents)

    return latents, labels, metas


def tsne_projection(data, n_components=2, perplexity=30, n_iter=300):
    t0 = time.time()
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=n_iter,
        random_state=41,
    )
    embedding = tsne.fit_transform(data)
    tf = time.time() - t0
    print(f"Computed tsne in {tf} seconds.")
    return embedding


def remove_outliers_zscore(data, labels, threshold=4.0):
    z_scores = np.abs(zscore(data, axis=0))
    mask = (z_scores < threshold).all(axis=1)
    return data[mask], labels[mask]


def plot_category_embeddings(binary_labels, embeddings, fname="categories"):
    num_categories = binary_labels.shape[1]
    plt.figure(figsize=(8, 6))

    for i in range(num_categories):
        labels = binary_labels[:, i]
        plt.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=["black" if l == 0 else "red" for l in labels],
            s=1.0,
            alpha=0.8,
        )

        plt.title(f"{ABNORMALITIES[i]}")
        plt.savefig(f"{fname}{ABNORMALITIES[i]}.png", dpi=300)


def plot_tsne_num_abnormalities(embedding, labels, fname="new_image_latents"):
    num_abnormalities = labels.sum(axis=1)
    color_list = ["#000000", "#ff0066", "#117f80", "#ab66ff", "#66ccfc", "#FF7F50"]
    annots = {
        "# of abnormalities = 0": (0, 1),
        "1 <= # of abnormalities < 4": (1, 4),
        "4 <= # of abnormalities < 7": (4, 7),
        "7 <= # of abnormalities < 10": (7, 10),
        "10 <= # of abnormalities < 13": (10, 13),
        "13 <= # of abnormalities": (13, 19),
    }

    plt.figure(figsize=(8, 6))
    scatter_plots = []

    for i, (key, value) in enumerate(annots.items()):
        start, end = value
        indices = np.where((num_abnormalities >= start) & (num_abnormalities < end))[0]
        sc = plt.scatter(
            embedding[indices, 0],
            embedding[indices, 1],
            c=color_list[i],
            s=1.0,
            alpha=0.4,
            label=key,
        )
        scatter_plots.append(sc)

    plt.title(f"t-SNE (Image Latents)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color_list[i],
            markersize=10,
        )
        for i in range(len(annots))
    ]

    plt.legend(
        handles=legend_handles,
        labels=annots.keys(),
        loc="upper left",
        bbox_to_anchor=(1, 1),
        fontsize=10,
    )
    plt.savefig(f"{fname}.png", dpi=600, bbox_inches="tight")
    plt.show()
    plt.clf()


def plot_tsne_metadata(embedding, metas, fname="new_image_latents"):
    cmap = plt.get_cmap("viridis")
    class_colors = ["#ff0066", "#117f80", "#ab66ff", "#66ccfc", "#FF7F50"]

    metadata_names = [
        (0, "Manufacturer", ["Siemens Healthineers", "Philips", "PNMS"]),
        (1, "PatientSex", ["F", "M"]),
        (2, "PatientAge", "continuous"),
        (3, "XYSpacing", "continuous"),
        (4, "ZSpacing", "continuous"),
    ]

    for i, name, label_names in metadata_names:
        non_nans = np.where(metas[:, i] != -1.0)[0]
        labeled_embedding = embedding[non_nans]
        labeled_metas = metas[non_nans]
        if label_names == "continuous":
            plt.figure(figsize=(8, 6))
            plt.scatter(
                labeled_embedding[:, 0],
                labeled_embedding[:, 1],
                c=labeled_metas[:, i],
                s=1.0,
                cmap=cmap,
            )
            plt.colorbar()
            plt.title(f"t-SNE (Image Latents) - {name}")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.savefig(f"{fname}_{name}.png", dpi=600)
            plt.show()
            plt.clf()

        else:
            plt.figure(figsize=(8, 6))
            for j, label_name in enumerate(label_names):
                indices = np.where(labeled_metas[:, i] == j)[0]
                plt.scatter(
                    labeled_embedding[indices, 0],
                    labeled_embedding[indices, 1],
                    c=class_colors[j],
                    s=1.0,
                    label=label_name,
                    alpha=0.6,
                )

            plt.title(f"t-SNE (Image Latents) - {name}")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=class_colors[j],
                    markersize=10,
                )
                for j in range(len(label_names))
            ]
            plt.legend(
                handles=legend_handles,
                labels=label_names,
                loc="upper left",
                bbox_to_anchor=(1, 1),
                fontsize=10,
            )
            plt.savefig(f"{fname}_{name}.png", dpi=600, bbox_inches="tight")
            plt.show()
            plt.clf()


def load_latents(args):
    latent_path = args.latent_path
    abnormalities_path = args.abnormalities_path
    metadata_path = args.metadata_path

    abnormalities_df = pd.read_csv(abnormalities_path)
    metadata_df = pd.read_csv(metadata_path)

    latents, labels, metadatas = load_latents_and_labels(
        latent_path, abnormalities_df, metadata_df
    )
    labels = np.array(labels)
    metadatas = np.array(metadatas)
    latents = np.vstack([latents])

    return latents, labels, metadatas


def save_all(latents, labels, metadata, embedding, fname):
    with open(f"{fname}.pkl", "wb") as file:
        pickle.dump(
            {
                "latents": latents,
                "labels": labels,
                "metadata": metadata,
                "embedding": embedding,
            },
            file,
        )


def load_save(path="tsne/save.pkl"):
    with open(path, "rb") as file:
        data = pickle.load(file)
        labels = data["labels"]
        metadata = data["metadata"]
        embedding = data["embedding"]
    return labels, metadata, embedding


def main(args):
    os.makedirs("tsne", exist_ok=True)
    os.makedirs("tsne/categories", exist_ok=True)
    os.makedirs("tsne/metadata", exist_ok=True)

    latents, labels, metadata = load_latents(args)

    perplexity = int(math.sqrt(labels.shape[0]))
    print("perplexity:", perplexity)

    embedding = tsne_projection(latents, perplexity=perplexity, n_iter=1000)

    save_all(latents, labels, metadata, embedding, "tsne/save")

    plot_tsne_metadata(embedding, metadata, "tsne/metadata/")

    plot_tsne_num_abnormalities(embedding, labels, "tsne/abnormalities")

    plot_category_embeddings(labels, embedding, "tsne/categories/")


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--latent_path",
        type=str,
        required=True,
        help="Path to cache folder contain .npy files of embeddings.",
    )
    parser.add_argument(
        "--abnormalities_path",
        type=str,
        default="dicom_datasets",
        required=False,
        help="Path to .csv file containing multiclass abnormality labels. ",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to .csv file containing scan metadata.",
    )
    return parser


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    main(args)
