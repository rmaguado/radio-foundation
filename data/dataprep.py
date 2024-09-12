# reads database file if exists
# finds already processed datasets
# calls dataset.py on each unprocessed datasets
# calls statistics.py on final database

from dataset import Processor
import sqlite3
import argparse
import os


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="data/radiomics_datasets.db",
        required=False,
        help="The path to the database.",
    )
    parser.add_argument(
        "--derived_okay",
        action="store_true",
        help="Specify if derived images are allowed.",
    )
    return parser


def get_processed_datasets(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    datasets = [table[0] for table in cursor.fetchall() if table[0] != "global"]

    conn.close()
    return datasets


def get_unprocessed_datasets(root_path, db_path):
    datasets_in_root = [
        x for x in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, x))
    ]
    if not os.path.exists(db_path):
        return datasets_in_root

    processed_datasets = get_processed_datasets(db_path)
    unprocessed_datasets = [x for x in datasets_in_root if x not in processed_datasets]

    return unprocessed_datasets


def main(args):
    root_path = args.root_path
    db_path = args.db_path
    derived_okay = args.derived_okay

    unprocessed_datasets = get_unprocessed_datasets(root_path, db_path)

    for dataset in unprocessed_datasets:
        dataset_path = os.path.join(root_path, dataset)
        config = {
            "dataset_name": dataset,
            "dataset_path": dataset_path,
            "target_path": db_path,
            "derived_okay": derived_okay,
        }
        processor = Processor(config)
        processor.prepare_dataset()


if __name__ == "__main__":
    args = get_argpase().parse_args()
    main(args)
