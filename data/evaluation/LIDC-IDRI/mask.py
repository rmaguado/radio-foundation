import os
import numpy as np
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus
import argparse
import logging


np.int = int

logger = logging.getLogger("dataprep")


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--output_path", type=str, required=True, help="The output path of the masks."
    )
    return parser


def rle_encode(x: np.ndarray):
    x_ = x.flatten()

    dif = np.concatenate(([0], x_))[:-1] != x_
    pos = np.nonzero(dif)[0]

    run_lengths = np.diff(pos, prepend=0)

    return run_lengths, x.shape


def rle_decode(run_lengths, shape):
    pos = np.cumsum(run_lengths)
    decoded = np.zeros(np.prod(shape), dtype=np.int32)
    current_value = 1
    for i in range(len(pos)):
        decoded[pos[i] : pos[i + 1] if i + 1 < len(pos) else None] = current_value
        current_value = 1 - current_value

    return decoded.reshape(shape)


def main(args):
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    scans = pl.query(pl.Scan).all()
    patient_ids = list(set(scan.patient_id for scan in scans))
    patient_ids.sort()

    for patient_id in tqdm(patient_ids):
        patient_scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        for scan in patient_scans:

            try:
                scan_path = scan.get_path_to_dicom_files()
                scan_id = scan.id

            except Exception as e:
                logger.exception(
                    f"failed to get dicom files for patient {patient_id}: {e}"
                )
                continue

            dicom_files = [x for x in os.listdir(scan_path) if x.endswith(".dcm")]

            if not dicom_files:
                logger.exception(f"dicom folder is empty for patient {patient_id}: {e}")
                continue

            num_slices = len(dicom_files)
            segmentation_mask = np.zeros((num_slices, 512, 512), dtype=np.uint8)

            nods = scan.cluster_annotations(verbose=False)
            for _, anns in enumerate(nods, 1):
                cmask, bbox, _ = consensus(anns, clevel=0.5)

                cmask = np.transpose(cmask, (2, 0, 1))
                bbox = (bbox[2], bbox[0], bbox[1])

                segmentation_mask[bbox] = cmask

            rle_mask, shape = rle_encode(segmentation_mask)
            mask_path = os.path.join(output_path, f"{scan_id}.npz")
            np.savez(mask_path, rle_mask=rle_mask, shape=shape)


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    main(args)
