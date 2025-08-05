import argparse
import logging
from typing import Dict
import os
from tqdm import tqdm
import polars as pl
import SimpleITK as sitk

sitk.ProcessObject_SetGlobalWarningDisplay(False)


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


def index_niftis(root_path, output_path) -> None:
    nifti_files = []
    for dirpath, _, filenames in tqdm(
        walk(root_path), desc=f"Walking through directories of {root_path}"
    ):
        for filename in filenames:
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                file_path = os.path.join(dirpath, filename)

                try:
                    image = sitk.ReadImage(file_path)

                    metadata = {"path": file_path}
                    metadata.update(get_fields(image))

                    nifti_files.append(metadata)

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    continue

    df = pl.DataFrame(nifti_files)
    df.write_csv(output_path)
    logging.info(f"Metadata saved to {output_path}")


def index_dicoms(root_path, output_path, target_modality) -> None:
    dicom_folders = []
    for dirpath, _, filenames in tqdm(
        walk(root_path), desc="Walking through directories"
    ):
        has_dcm = any(x.endswith(".dcm") for x in os.listdir(dirpath))
        if not has_dcm:
            continue

        try:
            reader = sitk.ImageSeriesReader()
            series_IDs = reader.GetGDCMSeriesIDs(dirpath)
            assert series_IDs, f"No DICOM series found in: {dirpath}"
            assert (
                len(series_IDs) == 1
            ), f"Expected only one dicom series, found{len(series_IDs)}."
            series_file_names = reader.GetGDCMSeriesFileNames(dirpath, series_IDs[0])

            reader.SetFileNames(series_file_names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            image = reader.Execute()

            if reader.HasMetaDataKey(slice=0, key="0008|0060"):
                modality = reader.GetMetaData(slice=0, key="0008|0060")
                assert (
                    modality == target_modality
                ), f"Expected modality == {target_modality}, found {modality}."
            else:
                logging.info(f"{dirpath} has no modality info, assuming OK")

            metadata = {"path": dirpath}
            metadata.update(get_fields(image))

            if target_modality == "MR":
                metadata.update(get_fields_mri(reader))

            dicom_folders.append(metadata)

        except Exception as e:
            logging.error(f"Error processing {dirpath}: {e}")
            continue
    df = pl.DataFrame(dicom_folders)
    df.write_csv(output_path)
    logging.info(f"Metadata saved to {output_path}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--modality", type=str, default="CT")
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = get_args()
    if args.dataset_type == "nifti":
        index_niftis(args.root_path, args.output_path)
    elif args.dataset_type == "dicom":
        assert (
            args.modality == "CT" or args.modality == "MR"
        ), f"Modality should be CT or MR, found {args.modality}."
        index_dicoms(args.root_path, args.output_path, args.modality)
    else:
        raise ValueError(f"Unrecognized dataset_type: {args.dataset_type}.")
