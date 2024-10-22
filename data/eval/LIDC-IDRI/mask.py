import os
import numpy as np
from tqdm import tqdm
import pylidc as pl
from pylidc.utils import consensus


def main(output_path: str):
    scans = pl.query(pl.Scan).all()
    patient_ids = list(set(scan.patient_id for scan in scans))
    patient_ids.sort()
    
    for patient_id in tqdm(patient_ids):
        patient_scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        for scan in patient_scans:
            scan_path = scan.get_path_to_dicom_files()
            
            scan_id = scan.id

            segmentation_mask = np.zeros(scan.volume.shape, dtype=np.uint8)

            nods = scan.cluster_annotations(verbose=False)
            for _, anns in enumerate(nods, 1):
                cmask, bbox, _ = consensus(anns, clevel=0.5)

                cmask = np.transpose(cmask, (2, 0, 1))
                bbox = (bbox[2], bbox[0], bbox[1])

                segmentation_mask[bbox] = cmask

            mask_path = os.path.join(output_path, f"{scan_id}.npy")
            np.save(mask_path, segmentation_mask)


if __name__ == "__main__":
    main()