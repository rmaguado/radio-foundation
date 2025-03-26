"""
Indexing script for indexing the CT-RATE niftis with the reports.
"""

import os
import argparse
import logging
import pandas as pd

from data.mllm.base import NiftiEvalBase, NiftiProcessorBase


logger = logging.getLogger("dataprep")


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    return parser


class CT_RATE_Processor(NiftiProcessorBase):
    def __init__(self, config: dict, database):
        super().__init__(config, database)

        self.reports_df = pd.read_csv(config["reports_path"])

    def get_text_data(self, map_id: int) -> str:
        return self.reports_df.loc[
            self.reports_df["map_id"] == map_id, "report"
        ].values[0]


def main(args):
    dataset_name = "CT-RATE_train"
    db_name = "CT-RATE_train_reports"
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
    database = NiftiEvalBase(config)
    processor = NiftiProcessorBase(config, database)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argpase()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e
