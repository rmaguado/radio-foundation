import os
import argparse
import logging
from tqdm import tqdm

from data.utils import walk, set_logging
from data.dataprep import CtDatabase


logger = logging.getLogger("dataprep")


class NpzDatabase(CtDatabase):
    def __init__(self, config):
        super().__init__(config, storage="npz")

        self.create_global_table()
        self.add_dataset(config["dataset_name"])

    def create_global_table(self) -> None:
        """
        Creates a global table in the database for Npz files.
        """
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS global (
                dataset TEXT,
                volume_path TEXT
            )
            """
        )

    def insert_global_data(self, dataset_name: str, volume_path: str) -> None:
        self.cursor.execute(
            """
            INSERT INTO "global" (dataset, volume_path)
            VALUES (?, ?)
            """,
            (
                dataset_name,
                volume_path,
            ),
        )


class NpzProcessor:
    def __init__(self, config: dict):
        self.dataset_name = config["dataset_name"]
        self.absolute_dataset_path = os.path.abspath(config["dataset_path"])

        self.database = NpzDatabase(config)

        log_path = f"data/log/{self.dataset_name}.log"
        set_logging(log_path)

    def process_volume(self, volume_path: str) -> None:
        """
        Process a npz volume, extract metadata, and store it in the database.

        Args:
            volume_path (str): Path to the Npz file.
        """

        rel_path = os.path.relpath(volume_path, self.absolute_dataset_path)
        self.database.insert_global_data(self.dataset_name, rel_path)

        logger.info(f"Processed nifti: {volume_path}.")

    def prepare_dataset(self) -> None:
        """
        Prepares the dataset by scanning the directories and processing NIfTI files.
        """

        logger.info(f"Processing dataset: {self.dataset_name}")
        print("Walking dataset directories.")
        total_dirs = sum(1 for _ in walk(self.absolute_dataset_path))
        logger.info(f"{self.dataset_name} total directories: {total_dirs}")
        volume_counts = 0

        for data_folder, dirs, files in tqdm(
            walk(self.absolute_dataset_path), total=total_dirs
        ):
            paths = [os.path.join(data_folder, f) for f in files if f.endswith(".npz")]
            if not paths:
                continue

            for path in paths:
                try:
                    self.process_volume(path)
                except Exception as e:
                    logger.exception(f"Error processing nifti {path}: {e}")
                    continue
                volume_counts += 1

        logger.info(
            f"Finished processing {self.dataset_name}. Added {volume_counts} volumes."
        )

    def close_db(self) -> None:
        """
        Closes the database connection.
        """
        self.database.close()


def get_argparse():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--root_path", type=str, required=True, help="The root path of the dataset."
    )
    parser.add_argument(
        "--dataset_name", type=str, required=True, help="The name of the dataset."
    )
    parser.add_argument(
        "--db_name",
        type=str,
        default="npz_datasets",
        required=False,
        help="The name of the database. Will be saved in data/index/<db_name>/index.db",
    )
    return parser


def main(args):
    dataset_path = os.path.join(args.root_path, args.dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")

    db_dir = os.path.join("data/index", args.db_name)
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "index.db")

    config = {
        "dataset_name": args.dataset_name,
        "dataset_path": dataset_path,
        "db_path": db_path,
    }
    processor = NpzProcessor(config)
    processor.prepare_dataset()


if __name__ == "__main__":
    parser = get_argparse()
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        logger.exception(e)
        raise e
