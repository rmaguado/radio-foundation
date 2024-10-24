import os
import h5py
import torch
import sqlite3

from dinov2.data.datasets.dicoms import DicomCTVolumesFull


def rle_decode(run_lengths, shape):
    pos = np.cumsum(run_lengths)
    decoded = np.zeros(np.prod(shape), dtype=np.int32)
    current_value = 1
    for i in range(len(pos)):
        decoded[pos[i]:pos[i+1] if i+1 < len(pos) else None] = current_value
        current_value = 1 - current_value
    
    return decoded.reshape(shape)


class LidcIdriNodules(DicomCTVolumesFull):
    def __init__(self, mask_path,*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.root_mask_path = mask_path

    def get_target(self, scanid: str) -> torch.Tensor:
        
        mask_path = os.path.join(self.root_mask_path, f"{scanid}.npz")

        with load(mask_path) as data:
            rle_mask = data['rle_mask']
            shape = data['shape']
        
        mask = rle_decode(rle_mask, shape)

        return mask

    def get_image_data(self, index: int) -> Tuple[torch.Tensor, str]:
        """default copied from dinov2/data/datasets/dicoms.py

        returns image tensor and scanid
        """
        dataset_name, rowid = self.entries[index]
        series_id, scanid = self.cursor.execute(
            f"SELECT series_id, scanid FROM '{dataset_name}' WHERE rowid = ?",
            (rowid),
        ).fetchone()

        self.cursor.execute(
            """
            SELECT slice_index, dataset, dicom_path
            FROM global 
            WHERE series_id = ? 
            AND dataset = ?
            """,
            (series_id, dataset_name),
        )
        slice_indexes_rowid = self.cursor.fetchall()
        slice_indexes_rowid.sort(key=lambda x: x[0])

        try:
            stack_data = self.create_stack_data(slice_indexes_rowid)
        except Exception as e:
            logger.exception(f"Error processing stack. Seriesid: {series_id} \n{e}")
            stack_data = torch.zeros((10, 512, 512))

        return stack_data, scanid

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            image, scanid = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image for sample {index}") from e

        target = self.get_target(scanid)

        return self.apply_transforms(image, target)

    