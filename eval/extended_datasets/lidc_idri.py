import os
import torch
from typing import Tuple, Any

from dinov2.data.datasets.dicoms import DicomCTVolumesFull


def rle_decode(run_lengths, shape):
    pos = np.cumsum(run_lengths)
    decoded = np.zeros(np.prod(shape), dtype=np.int32)
    current_value = 1
    for i in range(len(pos)):
        decoded[pos[i] : pos[i + 1] if i + 1 < len(pos) else None] = current_value
        current_value = 1 - current_value

    return decoded.reshape(shape)


class LidcIdriNodules(DicomCTVolumesFull):
    def __init__(
        self,
        mask_path: str,
        root_path: str,
        transform,
    ):
        super().__init__(
            dataset_name="LIDC-IDRI",
            root_path=root_path,
            transform=transform,
        )

        self.root_mask_path = mask_path

    def get_target(self, scanid: str) -> torch.Tensor:

        mask_path = os.path.join(self.root_mask_path, f"{scanid}.npz")

        with load(mask_path) as data:
            rle_mask = data["rle_mask"]
            shape = data["shape"]

        mask = rle_decode(rle_mask, shape)

        return mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, scanid = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(scanid)

        return self.apply_transforms(image, target)
