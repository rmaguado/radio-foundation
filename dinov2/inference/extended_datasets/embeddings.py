import os
import torch
import numpy as np
from dotenv import load_dotenv
from einops import rearrange

from evaluation.utils.finetune import (
    load_model,
    ImageTransform,
    ImageTransformResampleSlices,
    extract_class_tokens,
    extract_patch_tokens,
    extract_all_tokens,
)
from evaluation.extended_datasets.dicoms import DicomFullVolumeEval
from evaluation.extended_datasets.niftis import NiftiFullVolumeEval
from evaluation.extended_datasets.npz import NpzFullVolumeEval


def get_model(project_path, run_name, checkpoint_name):
    load_dotenv()
    project_path = os.getenv("PROJECTPATH")

    path_to_run = os.path.join(project_path, "runs", run_name)
    device = torch.device("cuda")
    feature_model, config = load_model(path_to_run, checkpoint_name, device, 1)

    return feature_model, config


class CachedEmbeddings:
    def __init__(self, embeddings_path: str, select_feature: str = "cls"):
        assert select_feature in ["cls", "patch", "cls_patch", "3d_patch"]
        self.embeddings_path = embeddings_path
        self.select_feature = select_feature

        self.map_ids = [
            file.split(".npy")[0]
            for file in os.listdir(self.embeddings_path)
            if ".npy" in file
        ]

    def get_embeddings(self, map_id: str) -> torch.Tensor:
        embeddings = np.load(
            os.path.join(self.embeddings_path, f"{map_id}.npy"), mmap_mode="r"
        )
        if self.select_feature == "cls":
            return torch.from_numpy(embeddings[:, :1, :].copy()).float()
        if self.select_feature == "patch":
            return torch.from_numpy(embeddings[:, 1:, :].copy()).float()
        if self.select_feature == "3d_patch":
            features = torch.from_numpy(embeddings[:, 1:, :].copy()).float().squeeze(0)
            features = rearrange(features, "a w h e -> a (w h) 1 e")
            return torch.mean(features, dim=1)
        return torch.from_numpy(embeddings.copy()).float()


class EmbeddingsGenerator:
    def __init__(
        self,
        project_path,
        run_name,
        checkpoint_name,
        root_path,
        dataset_name,
        db_storage,
        device,
        select_feature,
        max_batch_size=64,
        max_workers=16,
        resample_slices=False,
    ):
        self.model, config = get_model(project_path, run_name, checkpoint_name)
        full_image_size = config.student.full_image_size  # 504
        data_mean = -573.8
        data_std = 461.3
        channels = config.student.channels

        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.device = device

        if resample_slices:
            img_processor = ImageTransformResampleSlices(
                full_image_size, data_mean, data_std, channels=channels
            )
        else:
            img_processor = ImageTransform(
                full_image_size, data_mean, data_std, channels
            )

        db_params = {
            "root_path": root_path,
            "dataset_name": dataset_name,
            "channels": channels,
            "transform": img_processor,
        }
        if db_storage == "dicom":
            self.dataset = DicomFullVolumeEval(**db_params, max_workers=max_workers)
        elif db_storage == "nifti":
            self.dataset = NiftiFullVolumeEval(**db_params)
        elif db_storage == "npz":
            self.dataset = NpzFullVolumeEval(**db_params)
        else:
            raise ValueError(
                "Invalid database storage type (db_storage should be dicom, nifti, or npz)."
            )

        self.map_ids = self.dataset.entries["map_id"]

        if select_feature == "all":
            self.embed_fcn = extract_all_tokens
        elif select_feature == "patch":
            self.embed_fcn = extract_patch_tokens
        elif select_feature == "cls":
            self.embed_fcn = extract_class_tokens
        else:
            raise RuntimeError

    def get_embeddings(self, map_id: str) -> torch.Tensor:
        index = self.dataset.get_index_from_map_id(map_id)
        volume, _ = self.dataset[index]

        embeddings = []
        for i in range(0, volume.shape[0], self.max_batch_size):
            upper_range = min(i + self.max_batch_size, volume.shape[0])
            with torch.no_grad():
                x_tokens_list = self.model(volume[i:upper_range].to(self.device))
            embeddings.append(self.embed_fcn(x_tokens_list).cpu())

        return torch.cat(embeddings, dim=0)
