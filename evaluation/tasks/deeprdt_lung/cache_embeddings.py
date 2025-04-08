import os
import argparse
from dotenv import load_dotenv
import numpy as np
import torch
from tqdm import tqdm
import logging
from einops import rearrange

from evaluation.utils.finetune import load_model, extract_all_tokens
from evaluation.extended_datasets import DicomFullVolumeEval


class DeepRDTCrop(DicomFullVolumeEval):
    def __init__(self, root_path, dataset_name, channels, img_processor, metadata_path):
        super().__init__(
            root_path=root_path,
            dataset_name=dataset_name,
            channels=channels,
            transform=img_processor,
        )

        self.metadata = pd.read_csv(metadata_path)

    def get_center(self, mapid):
        raise NotImplementedError

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, map_id = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        center = self.get_center(map_id)
        transformed_image = self.transform(image, center)

        return transformed_image, map_id


class ImageTransform:
    def __init__(
        self, img_size, mean, std, enable_crop, crop_radius_xy, crop_radius_z, pad_value
    ):
        self.img_size = img_size
        self.resize = transforms.Resize((img_size, img_size))
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.enable_crop = enable_crop
        self.crop_xy = crop_radius_xy
        self.crop_z = crop_radius_z
        self.pad_value = pad_value

    def crop(self, image, center):
        g, c, w, h = image.shape
        s = g * c
        x, y, z = center

        image = rearrange(image, "g c w h -> (g c) w h")

        def get_range(dim):
            return x - self.crop_r, x + self.crop_r

        z1, z2 = z - self.crop_z, z + self.crop_z
        x1, x2 = x - self.crop_xy, x + self.crop_xy
        y1, y2 = y - self.crop_xy, y + self.crop_xy

        pad_z1, pad_z2 = -min(0, z1), max(s, z2) - s
        pad_x1, pad_x2 = -min(0, x1), max(w, x2) - w
        pad_y1, pad_y2 = -min(0, y1), max(h, y2) - h

        image = image[
            max(0, z1) : min(z2, s), max(0, x1) : min(x2, w), max(0, y1) : min(y2, h)
        ]

        image = torch.nn.functional.pad(
            image,
            (pad_y1, pad_y2, pad_x1, pad_x2, pad_z1, pad_z2),
            mode="constant",
            value=self.pad_value,
        )
        return rearrange(image, "(g c) w h -> g c w h", c=c)

    def __call__(self, image, center):
        image = self.crop(image, center)
        imgage = self.resize(image)
        imgage = self.normalize(imgage)

        return imgage


def get_model(project_path, run_name, checkpoint_name):
    load_dotenv()
    project_path = os.getenv("PROJECTPATH")

    path_to_run = os.path.join(project_path, "runs", run_name)
    device = torch.device("cuda")
    feature_model, config = load_model(path_to_run, checkpoint_name, device, 1)

    return feature_model, config


def generate_embeddings(model, dataset, output_path, max_batch_size, get_center):
    device = torch.device("cuda")

    model.to(device)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )

    for i, (volume, map_id) in tqdm(enumerate(dataloader), total=len(dataloader)):
        map_id = map_id[0]
        volume = volume[0]

        logging.info(f"Processing volume {i+1}/{len(dataloader)} ({map_id})")
        output_file = os.path.join(output_path, f"{map_id}.npy")

        embeddings = []
        for i in range(0, volume.shape[0], max_batch_size):

            upper_range = min(i + max_batch_size, volume.shape[0])
            with torch.no_grad():
                x_tokens_list = model(volume[i:upper_range].to(device))
            embeddings.append(extract_all_tokens(x_tokens_list).cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        np.save(output_file, embeddings)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="cache_embeddings.log",
    )
    logging.info("Starting cache_embeddings.py")

    parser = get_argpase()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available.")

    load_dotenv()
    project_path = os.getenv("PROJECTPATH")

    model, config = get_model(project_path, args.run_name, args.checkpoint_name)
    full_image_size = config.student.full_image_size  # 504
    data_mean = -573.8
    data_std = 461.3
    channels = config.student.channels

    img_processor = ImageTransform(
        full_image_size,
        data_mean,
        data_std,
        enable_crop=True,
        crop_radius_xy=64,
        crop_radius_z=40,
        pad_value=-1000,
    )

    output_path = os.path.join(
        project_path,
        "evaluation/cache",
        args.dataset_name,
        args.run_name,
        args.checkpoint_name,
    )
    os.makedirs(output_path, exist_ok=True)

    dataset = DeepRDTCrop(
        root_path=args.root_path,
        dataset_name=args.dataset_name,
        channels=channels,
        transform=img_processor,
        metadata_path=args.metadata_path,
    )

    generate_embeddings(model, dataset, output_path, args.max_batch_size)


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset folder in the root path. Also the base name of the database file.",
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
        "--max_batch_size", type=int, default=64, help="The maximum batch size."
    )
    parser.add_argument(
        "--db_storage",
        type=str,
        default="dicom",
        help="The type of database (dicom, nifti, or npz).",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to csv with tumor centers.",
    )
    return parser


if __name__ == "__main__":
    main()
