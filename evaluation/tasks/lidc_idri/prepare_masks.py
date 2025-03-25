import os
import numpy as np
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
import argparse
from dotenv import load_dotenv
import logging
from torchvision import transforms
import torch
from einops import rearrange
import pandas as pd


np.int = int
logger = logging.getLogger("dataprep")


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--image_size",
        type=int,
        default=504,
        help="The target size to resize the image.",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=14,
        help="Number of pixels that make each patch.",
    )
    return parser


def main(args):
    nodule_locations = {"map_id": [], "c": [], "h": [], "v": []}

    load_dotenv()
    project_path = os.getenv("PROJECTPATH")

    output_path = os.path.join(project_path, "evaluation/cache/LIDC-IDRI/masks")
    os.makedirs(output_path, exist_ok=True)

    scans = pl.query(pl.Scan).all()
    patient_ids = list(set(scan.patient_id for scan in scans))
    patient_ids.sort()

    resize = transforms.Resize((args.image_size, args.image_size))

    for patient_id in tqdm(patient_ids):
        patient_scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        for scan in patient_scans:

            try:
                scan_path = scan.get_path_to_dicom_files()
                map_id = scan.id

            except Exception as e:
                logger.error(f"failed to get dicom files for patient {patient_id}.")
                continue

            dicom_files = [x for x in os.listdir(scan_path) if x.endswith(".dcm")]

            if not dicom_files:
                logger.error(f"dicom folder is empty for patient {patient_id}")
                continue

            num_slices = len(dicom_files)
            segmentation_mask = np.zeros((num_slices, 512, 512), dtype=np.uint8)

            nods = scan.cluster_annotations(verbose=False)
            for _, anns in enumerate(nods, 1):
                cmask, bbox, _ = consensus(anns, clevel=0.5)

                cmask = np.transpose(cmask, (2, 0, 1))
                bbox = (bbox[2], bbox[0], bbox[1])

                segmentation_mask[bbox] = cmask

            segmentation_mask = resize(torch.from_numpy(segmentation_mask))
            segmentation_mask = rearrange(
                segmentation_mask,
                "s (w wp) (h hp) -> s w h wp hp",
                wp=args.patch_size,
                hp=args.patch_size,
            )
            segmentation_mask = segmentation_mask.sum(dim=3).sum(dim=3).numpy()

            for c, h, v in zip(*np.where(segmentation_mask)):
                nodule_locations["map_id"].append(map_id)
                nodule_locations["c"].append(c)
                nodule_locations["h"].append(h)
                nodule_locations["v"].append(v)

            mask_path = os.path.join(output_path, f"{map_id}.npy")
            np.save(mask_path, segmentation_mask)

    locations_df = pd.DataFrame(nodule_locations)
    locations_df.to_csv(os.path.join(output_path, "locations.csv"))


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    main(args)
