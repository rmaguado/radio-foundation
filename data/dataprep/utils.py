import sqlite3
import os

import warnings
import logging


logger = logging.getLogger("dataprep")
logger.setLevel(logging.INFO)
os.makedirs("data/log", exist_ok=True)

warnings.filterwarnings("ignore")


def walk(root_dir):
    ignorewords = ["ignore"]
    ignore_folders = ["labels", "segmentations"]

    for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in ignore_folders]

        if any(x in filenames for x in ignorewords):
            dirnames[:] = []

        yield dirpath, dirnames, filenames


def set_logging(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


class Database:
    def __init__(self, config):
        self.conn = sqlite3.connect(config["db_path"])
        self.cursor = self.conn.cursor()
        self.is_open = True
        self.create_metadata_table()
        self.create_datasets_table()

    def create_metadata_table(self):
        """
        Creates a table for storing metadata.
        """
        self.cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )

    def create_datasets_table(self) -> None:
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset TEXT,
                PRIMARY KEY (dataset)
            )
            """
        )

    def add_dataset(self, dataset: str) -> None:
        self.cursor.execute(
            "INSERT OR REPLACE INTO datasets (dataset) VALUES (?);", (dataset,)
        )

    def close(self):
        """
        Closes the database connection.
        """
        if self.is_open:
            self.conn.commit()
            self.conn.close()
            self.is_open = False

    def __del__(self):
        self.close()
