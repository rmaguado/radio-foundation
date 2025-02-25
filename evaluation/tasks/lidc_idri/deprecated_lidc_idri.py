import os
import torch
from typing import Tuple, Any
import numpy as np
import logging

logger = logging.getLogger("dinov2")


def rle_decode(run_lengths, shape):
    pos = np.cumsum(run_lengths)
    decoded = np.zeros(np.prod(shape), dtype=np.int32)
    current_value = 1
    for i in range(len(pos)):
        decoded[pos[i] : pos[i + 1] if i + 1 < len(pos) else None] = current_value
        current_value = 1 - current_value

    return decoded.reshape(shape)


class LidcIdri:
    def __init__(
        self,
        mask_path: str,
        project_path: str,
        run_name: str,
        checkpoint_name: str,
        num_patches: int,
    ):
        self.embeddings_path = os.path.join(
            project_path,
            "evaluation/cache",
            "deeprdt_lung",
            run_name,
            checkpoint_name,
        )
        self.map_ids = os.listdir(self.embeddings_path)
        self.root_mask_path = mask_path
        self.num_patches = num_patches

    def get_embeddings(self, map_id: str):
        return np.load(os.path.join(self.embeddings_path, f"{map_id}.npy"))

    def get_mask(self, map_id: str):
        mask_path = os.path.join(self.root_mask_path, f"{map_id}.npz")

        with np.load(mask_path) as data:
            rle_mask = data["rle_mask"]
            shape = data["shape"]

        mask = rle_decode(rle_mask, shape)

        return torch.tensor(mask, dtype=torch.float32)

    def mask_to_patch(self, mask: torch.Tensor):
        pass
