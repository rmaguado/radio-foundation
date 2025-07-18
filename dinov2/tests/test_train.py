import pytest
import torch
import time
import logging

from dinov2.configs import get_cfg_from_path
from dinov2.train.setup import setup_dataloader
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
def inputs_dtype():
    return torch.bfloat16


@pytest.fixture
def dataloader(cfg, inputs_dtype):
    return setup_dataloader(cfg, inputs_dtype)


@pytest.fixture
def model(cfg, device):
    model = SSLMetaArch(cfg)
    model.to(device)
    return model


def test_train_speed(model, dataloader, inputs_dtype):
    dataloader_iter = iter(dataloader)

    for idx in range(16):
        data = next(dataloader_iter)

        t0 = time.time()
        with torch.autocast(device_type="cuda", enabled=True, dtype=inputs_dtype):
            loss_accumulator, loss_dict = model.forward(data, teacher_temp=0.99)
        tf = time.time() - t0

        logger.info(f"Batch {idx}: forward took {tf:.06f} seconds.")

        t0 = time.time()
        loss_accumulator.backward()
        tf = time.time() - t0

        logger.info(f"Batch {idx}: backward took {tf:.06f} seconds.")
