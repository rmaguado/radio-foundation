"""
Indexing script for evaluation with DeepRDT-lung
"""

from typing import List, Tuple
import os
import argparse
import logging

from data.evaluation.base import DicomEvalBase, DicomProcessorBase


logger = logging.getLogger("dataprep")


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    return parser


class DicomProcessorDeepRDT(DicomProcessorBase):
    def __init__(self, config: dict, database):
        super().__init__(config, database)

    def get_paths_and_mapids(self) -> List[Tuple[str, str]]:
        """
        Get paths to dicom files and their corresponding mapids.
        """
        paths_ids = []

        mapids = [
            folder
            for folder in os.listdir(self.absolute_dataset_path)
            if os.path.isdir(os.path.join(self.absolute_dataset_path, folder))
        ]

        for mapid in mapids:
            scan_path = os.path.join(self.absolute_dataset_path, mapid, "pCT")

            if os.path.isdir(scan_path):
                paths_ids.append((scan_path, mapid))

        return paths_ids


def main(args):
    dataset_name = "DeepRDT-lung"
    db_name = "DeepRDT-lung_eval"
    dataset_path = os.path.join(args.root_path, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    db_dir = os.path.join("data/index", db_name)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "index.db")

    config = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "db_path": db_path,
    }
    database = DicomEvalBase(config)
    processor = DicomProcessorBase(config, database)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e
