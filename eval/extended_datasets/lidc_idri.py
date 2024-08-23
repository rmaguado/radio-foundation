import os
import h5py
import torch

from dinov2.data.datasets import CtDataset


class LidcIdriNodules(CtDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, enable_targets=True)

    def get_target(self, index: int):
        entries = self._get_entries()

        series_id = entries[index]["series_id"]
        slice_index = entries[index]["slice_index"]

        mask_full_path = os.path.join(self.root, series_id, "mask.h5")

        with h5py.File(mask_full_path, "r") as f:
            data = f["data"]
            loaded_mask = data[slice_index]

        return torch.from_numpy(loaded_mask).unsqueeze(0)
