import logging
import numpy as np
import torch
from typing import Tuple, Any
import os

from dinov2.data.datasets.npz import NpzCtVolumesFull


logger = logging.getLogger("dinov2")


class NpzFullVolumeEval(NpzCtVolumesFull):
    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        channels: int = 1,
        transform=None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            root_path=root_path,
            channels=channels,
            transform=transform,
        )

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of paths.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """
        logger.info(f"Creating entries for {self.dataset_name}.")

        entries_dtype = [
            ("rowid", np.uint32),
            ("map_id", "U256"),
        ]
        entries = []
        row_map_ids = self.cursor.execute(
            f"SELECT rowid, map_id FROM global"
        ).fetchall()

        for rowid, map_id in row_map_ids:
            entries.append((rowid, map_id))

        logger.info(f"Total number of scans: {len(entries)}.")

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        logger.info(f"Saving entries to {entries_dir}.")
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_image_data(self, index: int) -> torch.Tensor:
        rowid, map_id = self.entries[index]
        self.cursor.execute(
            """
            SELECT dataset, volume_path FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, volume_path = self.cursor.fetchone()

        abs_path_to_npz = os.path.join(self.root_path, dataset, volume_path)
        npz_data = np.load(abs_path_to_npz)
        volume_data = npz_data["image_data"].astype(np.float32)

        num_slices = volume_data.shape[0]
        num_stacks = num_slices // self.channels
        num_slices = num_stacks * self.channels

        image_width = volume_data.shape[1]
        image_height = volume_data.shape[2]

        volume_data = volume_data[:num_slices]
        volume_data = torch.tensor(volume_data).view(
            num_stacks, self.channels, image_width, image_height
        )

        return self.process_ct(volume_data), map_id

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, map_id = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        transformed_image = self.transform(image)

        return transformed_image, map_id
