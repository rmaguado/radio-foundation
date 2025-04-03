import os
from dotenv import load_dotenv
import numpy as np
import torch
import time
import logging
import pydicom
import nibabel as nib

from evaluation.utils.finetune import (
    load_model,
    ImageTransform,
    extract_class_tokens,
    extract_patch_tokens,
    extract_all_tokens,
)


def get_image_processor(config, mean=None, std=None):
    full_image_size = config.student.full_image_size
    if mean is None or std is None:
        norm = config.datasets[0].norm
        mean = norm.mean
        std = norm.std
    img_processor = ImageTransform(full_image_size, data_mean, data_std)


def get_model(project_path, run_name, checkpoint_name):
    load_dotenv()
    project_path = os.getenv("PROJECTPATH")

    path_to_run = os.path.join(project_path, "runs", run_name)
    device = torch.device("cuda")
    feature_model, config = load_model(path_to_run, checkpoint_name, device, 1)

    return feature_model, config


def generate_embeddings(
    model, volume, embed_patches=True, embed_cls=True, channels=10, max_batch_size=64
):
    if embed_patches and embed_cls:
        embed_fcn = extract_all_tokens
    elif embed_patches:
        embed_fcn = extract_patch_tokens
    elif embed_cls:
        embed_fcn = extract_class_tokens
    else:
        raise ValueError("Must extract patch token and/or class token embeddings.")

    max_slices = volume.shape[0] // channels * channels
    volume = rearrange(volume[:max_slices], "(b c) w h -> b c w h", c=channels)

    embeddings = []
    for i in range(0, volume.shape[0], max_batch_size):
        upper_range = min(i + max_batch_size, volume.shape[0])
        with torch.no_grad():
            x_tokens_list = model(volume[i:upper_range].to(device))
        embeddings.append(embed_fcn(x_tokens_list).cpu().numpy())

    embeddings = torch.concatenate(embeddings, axis=0)

    return embeddings


def load_dicom(dicom_folder):
    dcm_paths = [
        os.path.join(dicom_folder, x)
        for x in os.listdir(dicom_folder)
        if x.endswith(".dcm")
    ]
    dicoms = [pydicom.dcmread(x) for x in dcm_paths]
    series_ids = [x.get("SeriesInstanceUID") for x in dicoms]
    unique_series_ids = list(set(series_ids))
    assert len(unique_series_ids) == 1, "All series ids in folder must be the same."

    images_position = [
        (x.pixel_array * x.RescaleSlope + x.RescaleIntercept, x.ImagePositionPatient)
        for x in dicoms
    ]

    sorted_images = images_position.sort(key=lambda x: x[1])
    volume = np.stack([x[0] for x in sorted_images]).astype(np.float32)

    return torch.from_numpy(volume)
