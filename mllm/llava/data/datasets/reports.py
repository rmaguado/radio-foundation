import os
import copy
import torch
from torchvision import transforms
import numpy as np
import nibabel as nib
from typing import Tuple, Any
from einops import rearrange
from typing import Dict, Sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from dinov2.data.datasets.niftis import NiftiCtVolumesFull
from mllm.llava.mm_utils import tokenizer_image_token
from mllm.llava import conversation as conversation_lib
from mllm.llava.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


class ImageProcessor:
    def __init__(
        self,
        img_size,
        mean,
        std,
        min_zspacing=1.5,
        channels=10,
        pad_value=-1000,
        max_slices=300,
    ):
        self.img_size = img_size
        self.normalize = transforms.Normalize(mean=mean, std=std)

        self.min_zspacing = min_zspacing
        self.channels = channels
        self.pad_value = pad_value
        self.max_slices = max_slices

        assert max_slices % channels == 0, "max_slices must be divisible by channels"

    def pad_square(self, image):
        s, w, h = image.shape

        if w == h:
            return image

        if w > h:
            pad_1 = (w - h) // 2
            pad_2 = (w - h) - pad_1
            image = torch.nn.functional.pad(
                image, (pad_1, pad_2, 0, 0, 0, 0), value=self.pad_value
            )
        else:
            pad_1 = (h - w) // 2
            pad_2 = (h - w) - pad_1
            image = torch.nn.functional.pad(
                image, (0, 0, pad_1, pad_2, 0, 0), value=self.pad_value
            )

        return image

    def clip_slices(self, image):
        slices, w, h = image.shape

        if slices > self.max_slices:
            start = (slices - self.max_slices) // 2
            end = start + self.max_slices

            image = image[start:end]

        return image

    def resize(self, image, slice_thickness):
        slices, w, h = image.shape

        target_width = self.img_size if w >= h else self.img_size * w // h
        target_height = self.img_size if h >= w else self.img_size * h // w

        if slice_thickness < self.min_zspacing:
            target_slices = int(slices * slice_thickness / self.min_zspacing)
        else:
            target_slices = slices

        groups = target_slices // self.channels
        target_slices = groups * self.channels

        image = image.unsqueeze(0).unsqueeze(0)
        image = torch.nn.functional.interpolate(
            image, size=(target_slices, target_width, target_height), mode="trilinear"
        ).squeeze()
        image = self.pad_square(image)
        image = self.clip_slices(image)
        return rearrange(
            image,
            "(g c) w h -> g c w h",
            c=self.channels,
            w=self.img_size,
            h=self.img_size,
        )

    def __call__(self, image, slice_thickness):
        image = self.resize(image, slice_thickness)
        image = self.normalize(image)
        return image


class _ReportDataset(NiftiCtVolumesFull):
    def __init__(
        self,
        root_path: str,
        dataset_name: str,
        channels: int = 10,
        img_size: int = 504,
        mean: float = 0.0,
        std: float = 1.0,
        min_zspacing=1.5,
        max_slices=300,
    ):
        super().__init__(
            dataset_name=dataset_name,
            root_path=root_path,
            channels=channels,
            transform=None,
        )

        self.image_processor = ImageProcessor(
            img_size=img_size,
            mean=mean,
            std=std,
            min_zspacing=min_zspacing,
            channels=channels,
            max_slices=max_slices,
        )

    def create_entries(self) -> np.ndarray:
        """
        Generates a numpy memmap object pointing to the sqlite database rows of dicom paths.

        Returns:
            np.ndarray: The entries dataset (memmap).
        """

        entries_dtype = [
            ("rowid", np.uint32),
            ("map_id", "U256"),
            ("length", np.uint32),
            ("slices", np.uint32),
        ]
        entries = []
        row_id_lengths = self.cursor.execute(
            f"SELECT rowid, map_id, length, num_slices FROM global"
        ).fetchall()

        for rowid, map_id, length, slices in row_id_lengths:
            entries.append((rowid, map_id, length, slices))

        entries_array = np.array(entries, dtype=entries_dtype)

        entries_dir = self.get_entries_dir()
        np.save(entries_dir, entries_array)
        return np.load(entries_dir, mmap_mode="r")

    def get_lengths(self):
        return self.entries["length"]

    def get_slices(self):
        return self.entries["slices"]

    def process_report(self, report_text):
        return report_text

    def get_image_data(self, index: int) -> torch.Tensor:
        rowid, map_id, _, _ = self.entries[index]
        self.cursor.execute(
            """
            SELECT dataset, axial_dim, nifti_path, text, slice_thickness FROM global WHERE rowid = ?
            """,
            (int(rowid),),
        )
        dataset, axial_dim, nifti_path, report, slice_thickness = self.cursor.fetchone()

        abs_path_to_nifti = os.path.join(self.root_path, dataset, nifti_path)
        nifti_file = nib.load(abs_path_to_nifti)

        volume_data = nifti_file.get_fdata().astype(np.float32)
        volume_data = np.moveaxis(volume_data, axial_dim, 0)
        volume_data = torch.from_numpy(volume_data)
        volume_data = self.process_ct(volume_data)
        volume_data = self.image_processor(volume_data, slice_thickness)

        return map_id, volume_data, report

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        try:
            map_id, image, report = self.get_image_data(index)
        except Exception as e:
            raise RuntimeError(f"can not read image/report for sample {index}") from e

        return map_id, image, report


def preprocess(
    sources: Sequence[str],
    tokenizer: PreTrainedTokenizer,
    conversation: conversation_lib.Conversation,
) -> Dict:
    conversation.messages = []
    for source in sources:
        conversation.append_message(source["from"], source["value"])
    prompt_chunks = conversation.get_prompt()

    input_ids = []
    targets = []
    for prompt, is_target in prompt_chunks:
        ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids.append(ids)
        if not is_target:
            targets.append(torch.tensor([IGNORE_INDEX] * len(ids)))
        else:
            targets.append(ids.clone())

    input_ids = torch.cat(input_ids, dim=0)
    targets = torch.cat(targets, dim=0)

    return dict(input_ids=input_ids, labels=targets)


class RadiologyReportDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: PreTrainedTokenizer, data_args):
        super().__init__()

        self.tokenizer = tokenizer

        self.debug_mode = data_args.debug_mode

        self.dataset = _ReportDataset(
            root_path=data_args.root_path,
            dataset_name=data_args.db_name,
            channels=data_args.channels,
            mean=data_args.data_mean,
            std=data_args.data_std,
        )
        self.data_args = data_args
        self.image_tokens = self.data_args.image_tokens

        if data_args.conv_template in conversation_lib.conv_templates:
            self.conversation_template = conversation_lib.conv_templates[
                data_args.conv_template
            ]
        else:
            raise ValueError(
                f"Unknown conversation template: {data_args.conv_template}"
            )

    def __len__(self):
        if self.debug_mode:
            return min(8, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        data_dict = {}

        map_id, image, text = self.dataset[i]
        sources = [
            {
                "from": "human",
                "value": f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}",
            },
            {"from": "gpt", "value": text},
        ]

        data_dict = preprocess(
            copy.deepcopy(sources),
            self.tokenizer,
            self.conversation_template,
        )

        return dict(
            input_ids=data_dict["input_ids"],
            labels=data_dict["labels"],
            image=image,
            map_id=map_id,
        )
