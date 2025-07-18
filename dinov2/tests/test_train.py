import pytest
import torch
import logging

from dinov2.configs import get_cfg_from_path
from dinov2.train.ssl_meta_arch import SSLMetaArch

logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)


@pytest.fixture
def cfg():
    return get_cfg_from_path("dinov2/configs/tests/minimal.yaml")


@pytest.fixture
def device():
    return torch.device("cuda")


@pytest.fixture
def arch(cfg, device):
    arch = SSLMetaArch(cfg)
    arch.to(device)
    return arch
