import os
import argparse
from dotenv import load_dotenv
import numpy as np
import torch
from tqdm import tqdm

from evaluation.utils.finetune import (
    load_model,
    ImageTransform,
    extract_class_tokens,
    extract_patch_tokens,
    extract_all_tokens,
)
from evaluation.extended_datasets import DicomFullVolumeEval, NiftiFullVolumeEval


def get_model(project_path, run_name, checkpoint_name, device):
    load_dotenv()
    project_path = os.getenv("PROJECTPATH")

    path_to_run = os.path.join(project_path, "runs", run_name)
    feature_model, config = load_model(path_to_run, checkpoint_name, device)

    return feature_model, config


def generate_embeddings(
    model, dataset, output_path, embed_patches, embed_cls, device, max_batch_size
):
    if embed_patches and embed_cls:
        embed_fcn = extract_all_tokens
    elif embed_patches:
        embed_fcn = extract_patch_tokens
    elif embed_cls:
        embed_fcn = extract_class_tokens
    else:
        raise ValueError("Must save at least path or class token embeddings.")

    for i in tqdm(range(len(dataset))):
        volume, map_id = dataset[i]
        output_file = os.path.join(output_path, f"{map_id}.npy")

        embeddings = []
        for i in range(0, volume.shape[0], max_batch_size):
            upper_range = min(i + max_batch_size, volume.shape[0])
            with torch.no_grad():
                x_tokens_list = model(volume[i:upper_range].to(device))
            embeddings.append(embed_fcn(x_tokens_list).cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        np.save(output_file, embeddings)


def main():
    parser = get_argpase()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available.")
    device = torch.device("cuda:0")

    load_dotenv()
    project_path = os.getenv("PROJECTPATH")

    model, config = get_model(project_path, args.run_name, args.checkpoint_name, device)
    full_image_size = config.student.full_image_size  # 504
    data_mean = -573.8
    data_std = 461.3
    channels = 4
    max_workers = 4

    img_processor = ImageTransform(full_image_size, data_mean, data_std)

    output_path = os.path.join(
        project_path,
        "evaluation/cache",
        args.db_name,
        args.run_name,
        args.checkpoint_name,
    )
    os.makedirs(output_path, exist_ok=True)

    db_params = {
        "root_path": args.root_path,
        "dataset_name": args.db_name,
        "channels": channels,
        "transform": img_processor,
    }
    if args.db_storage == "dicom":
        dataset = DicomFullVolumeEval(**db_params, max_workers=max_workers)
    elif args.db_storage == "nifti":
        dataset = NiftiFullVolumeEval(**db_params)
    else:
        raise ValueError(
            "Invalid database storage type (--db_storage should be dicom or nifti)."
        )

    generate_embeddings(
        model,
        dataset,
        output_path,
        args.embed_patches,
        args.embed_cls,
        device,
        args.max_batch_size,
    )


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        required=True,
        help="The name of the .db file found in data/index.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="The name of the run to load. Should exist in /runs",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        required=True,
        help="The name of the checkpoint to load.",
    )
    parser.add_argument(
        "--embed_patches", action="store_true", help="Wether to save patch embeddings."
    )
    parser.add_argument(
        "--embed_cls", action="store_true", help="Wether to save class embeddings."
    )
    parser.add_argument(
        "--max_batch_size", type=int, default=64, help="The maximum batch size."
    )
    parser.add_argument(
        "--db_storage",
        type=str,
        default="dicom",
        help="The type of database (dicom or nifti).",
    )
    return parser


if __name__ == "__main__":
    main()
