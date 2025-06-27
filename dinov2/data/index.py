import os
import nibabel as nib
import pydicom
from typing import Dict
from tqdm import tqdm
import polars as pl
import logging


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


def CTValidation(metadata: Dict) -> str:
    """
    Validates CT metadata.
    Args:
        metadata (Dict): Metadata extracted from the DICOM file.
    Returns:
        str: Validation results.
    """
    errors = []

    if metadata["slice_thickness"] > 5.0:
        errors.append(f"Slice thickness is too high: {metadata['slice_thickness']}")

    return "\n".join(errors)


def index_niftis():
    """
    Main function to validate Nifti files in a directory.
    """
    root_dir = "path/to/dicom/files"
    output_file = "path/to/output/nifti_files.csv"

    # column names for the output CSV file
    # "path", "rows", "columns", "slices", "slice_thickness", "xyspacing", "zspacing"

    nifti_files = []
    for dirpath, dirnames, filenames in tqdm(
        walk(root_dir), desc="Walking through directories"
    ):
        for filename in filenames:
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                file_path = os.path.join(dirpath, filename)

                try:
                    nifti_image = nib.load(file_path)
                    metadata = {
                        "path": file_path,
                        "rows": nifti_image.shape[0],
                        "columns": nifti_image.shape[1],
                        "slices": nifti_image.shape[2],
                        "slice_thickness": nifti_image.header.get_zooms()[2],
                        "xyspacing": nifti_image.header.get_zooms()[:2],
                        "zspacing": nifti_image.header.get_zooms()[2],
                    }

                    errors = CTValidation(metadata)
                    if errors:
                        logging.error(f"Validation failed for {file_path}: {errors}")
                        continue

                    nifti_files.append(metadata)

                except Exception as e:
                    logging.error(f"Error processing {file_path}: {e}")
                    continue

    df = pl.DataFrame(nifti_files)
    df.write_csv(output_file)
    logging.info(f"Metadata saved to {output_file}")


def index_dicoms():
    """
    Main function to validate DICOM files in a directory.
    """
    root_dir = "path/to/dicom/files"
    output_file = "path/to/output/dicom_folders.csv"

    dicom_folders = []
    for dirpath, dirnames, filenames in tqdm(
        walk(root_dir), desc="Walking through directories"
    ):
        dicom_files = [f for f in filenames if f.endswith(".dcm")]
        if not dicom_files:
            continue

        try:
            first_file = os.path.join(dirpath, dicom_files[0])
            ds = pydicom.dcmread(first_file, stop_before_pixels=True)
            metadata = {
                "path": dirpath,
                "rows": ds.Rows,
                "columns": ds.Columns,
                "slices": len(dicom_files),
                "slice_thickness": (
                    float(ds.SliceThickness) if "SliceThickness" in ds else None
                ),
                "xyspacing": (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1])),
                "zspacing": (
                    float(ds.SliceThickness) if "SliceThickness" in ds else None
                ),
            }

            errors = CTValidation(metadata)
            if errors:
                logging.error(f"Validation failed for {dirpath}: {errors}")
                continue

            dicom_folders.append(metadata)

        except Exception as e:
            logging.error(f"Error processing {dirpath}: {e}")
            continue
    df = pl.DataFrame(dicom_folders)
    df.write_csv(output_file)
    logging.info(f"Metadata saved to {output_file}")
