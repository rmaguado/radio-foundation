from typing import Union

from .dicoms import DicomCtDataset
from .niftis import NiftiCtDataset
from .multidataset import MultiDataset

MedicalImageDataset = Union[DicomCtDataset, NiftiCtDataset, MultiDataset]
