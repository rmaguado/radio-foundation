import argparse
import logging
import os
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import polars as pl
import SimpleITK as sitk
from tqdm import tqdm

sitk.ProcessObject_SetGlobalWarningDisplay(False)

# ---------- NIFTI ----------
def _process_one_nifti(file_path: str) -> Dict:
    """Executed in a child process."""
    try:
        image = sitk.ReadImage(file_path)
        meta = {"path": file_path}
        meta.update(get_fields(image))
        return meta
    except Exception as e:
        # Returning None is cheaper than raising across processes
        return {"path": file_path, "error": str(e)}


# ---------- DICOM ----------
def _process_one_dicom_folder(dirpath: str, target_modality: str) -> Dict:
    """Executed in a child process."""
    try:
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(dirpath)
        assert series_IDs, f"No DICOM series found in: {dirpath}"
        assert len(series_IDs) == 1, f"Expected one series, found {len(series_IDs)}"

        series_file_names = reader.GetGDCMSeriesFileNames(dirpath, series_IDs[0])
        reader.SetFileNames(series_file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()

        if reader.HasMetaDataKey(slice=0, key="0008|0060"):
            modality = reader.GetMetaData(slice=0, key="0008|0060")
            assert modality == target_modality, f"Expected {target_modality}, got {modality}"
        else:
            logging.info(f"{dirpath} has no modality info, assuming OK")

        if target_modality == "CT":
            

        meta = {"path": dirpath}
        meta.update(get_fields(image))

        if target_modality == "MR":
            meta.update(get_fields_mri(reader))
        return meta

    except Exception as e:
        return {"path": dirpath, "error": str(e)}


def walk(root_dir):
    """
    Walks through the directory tree and yields directories and files.
    Ignores directories containing "ignore" in their names and skips folders
    that contain a file named "ignore".
    """
    ignorewords = ["ignore"]
    ignore_folders = []

    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in ignore_folders]

        if any(x in filenames for x in ignorewords):
            dirnames[:] = []

        yield dirpath, dirnames, filenames


def get_fields(image) -> Dict:
    spacing = image.GetSpacing()
    return {
        "width": image.GetWidth(),
        "height": image.GetHeight(),
        "depth": image.GetDepth(),
        "dimensions": image.GetDimension(),
        "x_spacing": round(spacing[0], 6),
        "y_spacing": round(spacing[1], 6),
        "z_spacing": round(spacing[2], 6),
    }


def get_fields_mri(reader) -> Dict:
    mri_metadata = {}
    existing_keys = reader.GetMetaDataKeys(slice=0)
    key_names = {
        "0018|0080": "repetition_time",
        "0018|0081": "echo_time",
        "0018|0082": "inversion_time",
        "0018|0087": "magnetic_field_strength",
    }

    for key, name in key_names.items():
        if key in existing_keys:
            value = reader.GetMetaData(slice=0, key=key)
        else:
            value = None
        mri_metadata[name] = value

    return mri_metadata

def index_niftis(root_path: str, output_path: str, max_workers: int = None) -> None:
    # Collect file paths first (cheap I/O)
    file_paths = []
    for dirpath, _, filenames in walk(root_path):
        for f in filenames:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                file_paths.append(os.path.join(dirpath, f))

    metadata: List[Dict] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_process_one_nifti, fp) for fp in file_paths]
        for f in tqdm(as_completed(futures), total=len(futures), desc="NIfTI"):
            result = f.result()
            if "error" in result:
                logging.error(f"Error processing {result['path']}: {result['error']}")
            else:
                metadata.append(result)

    pl.DataFrame(metadata).write_csv(output_path)
    logging.info(f"NIfTI metadata saved to {output_path}")


def index_dicoms(root_path: str, output_path: str, target_modality: str,
                 max_workers: int = None) -> None:
    # Collect candidate folders
    folders = []
    for dirpath, _, filenames in walk(root_path):
        if any(x.endswith(".dcm") for x in filenames):
            folders.append(dirpath)

    metadata: List[Dict] = []
    worker = partial(_process_one_dicom_folder, target_modality=target_modality)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, d) for d in folders]
        for f in tqdm(as_completed(futures), total=len(futures), desc="DICOM"):
            result = f.result()
            if "error" in result:
                logging.error(f"Error processing {result['path']}: {result['error']}")
            else:
                metadata.append(result)

    pl.DataFrame(metadata).write_csv(output_path)
    logging.info(f"DICOM metadata saved to {output_path}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--modality", type=str, default='CT')
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = get_args()
    if args.dataset_type == "nifti":
        index_niftis(args.root_path, args.output_path, args.workers)
    elif args.dataset_type == "dicom":
        assert args.modality in ("CT", "MR"), f"Modality must be CT or MR, got {args.modality}"
        index_dicoms(args.root_path, args.output_path, args.modality, args.workers)
    else:
        raise ValueError(f"Unrecognized dataset_type: {args.dataset_type}")