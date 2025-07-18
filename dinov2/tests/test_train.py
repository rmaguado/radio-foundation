import pytest
import torch
import logging
import time
from functools import partial

from dinov2.configs import get_cfg_from_path
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.data import collate_data_and_cast, MaskingGenerator, DataAugmentationDINO

logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)


@pytest.fixture
def cfg():
    return get_cfg_from_path("dinov2/configs/minimal_model.yaml")


@pytest.fixture
def device():
    return torch.device("cuda")


@pytest.fixture
def arch(cfg, device):
    arch = SSLMetaArch(cfg)
    arch.to(device)
    return arch
