"""
Indexing script for evaluation with COVID-19-NY-SBU
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


def main(args):
    dataset_name = "COVID-19-NY-SBU"
    db_name = "COVID-19-NY-SBU_eval"
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
