import os
import nibabel as nib
import pydicom
from typing import Dict
from tqdm import tqdm
import polars as pl
import logging
import argparse


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

    # Check for 'slice_thickness' before accessing it
    if (
        "slice_thickness" in metadata
        and metadata["slice_thickness"] is not None
        and metadata["slice_thickness"] > 5.0
    ):
        errors.append(f"Slice thickness is too high: {metadata['slice_thickness']}")

    return "\n".join(errors)


def index_niftis(root_path, output_path):
    """
    Main function to validate Nifti files in a directory.
    """
    nifti_files = []
    for dirpath, dirnames, filenames in tqdm(
        walk(root_path), desc="Walking through directories"
    ):
        for filename in filenames:
            if filename.endswith(".nii") or filename.endswith(".nii.gz"):
                file_path = os.path.join(dirpath, filename)

                try:
                    nifti_image = nib.load(file_path)
                    zooms = nifti_image.header.get_zooms()
                    rows, cols, slices = nifti_image.shape
                    metadata = {
                        "path": file_path,
                        "rows": rows,
                        "columns": cols,
                        "slices": slices,
                        "slice_thickness": zooms[2],
                        "x_spacing": zooms[0],
                        "y_spacing": zooms[1],
                        "z_spacing": zooms[2],
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
    df.write_csv(output_path)
    logging.info(f"Metadata saved to {output_path}")


def index_dicoms(root_path, output_path):
    """
    Main function to validate DICOM files in a directory.
    """
    dicom_folders = []
    for dirpath, dirnames, filenames in tqdm(
        walk(root_path), desc="Walking through directories"
    ):
        dicom_files = [f for f in filenames if f.endswith(".dcm")]
        if not dicom_files:
            continue

        try:
            first_file = os.path.join(dirpath, dicom_files[0])
            ds = pydicom.dcmread(first_file, stop_before_pixels=True)

            # Handle PixelSpacing and SliceThickness potentially being missing or single values
            pixel_spacing_x = (
                float(ds.PixelSpacing[0])
                if "PixelSpacing" in ds and len(ds.PixelSpacing) > 0
                else None
            )
            pixel_spacing_y = (
                float(ds.PixelSpacing[1])
                if "PixelSpacing" in ds and len(ds.PixelSpacing) > 1
                else None
            )
            slice_thickness_val = (
                float(ds.SliceThickness) if "SliceThickness" in ds else None
            )

            metadata = {
                "path": dirpath,
                "rows": ds.Rows if "Rows" in ds else None,
                "columns": ds.Columns if "Columns" in ds else None,
                "slices": len(dicom_files),
                "slice_thickness": slice_thickness_val,
                "x_spacing": pixel_spacing_x,
                "y_spacing": pixel_spacing_y,
                "z_spacing": slice_thickness_val,
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
    df.write_csv(output_path)
    logging.info(f"Metadata saved to {output_path}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--root_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if args.dataset_type == "nifti":
        index_niftis(args.root_path, args.output_path)
    elif args.dataset_type == "dicom":
        index_dicoms(args.root_path, args.output_path)
    else:
        raise ValueError(f"{args.dataset_type} not a recognized data type. ")
