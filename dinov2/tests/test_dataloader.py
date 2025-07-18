import pytest
import torch
import os
import matplotlib.pyplot as plt
import time
import logging

from dinov2.configs import get_cfg_from_path
from dinov2.train.setup import setup_dataloader

logger = logging.getLogger("test")
logger.setLevel(logging.DEBUG)


@pytest.fixture
def cfg():
    return get_cfg_from_path("dinov2/configs/tests/minimal.yaml")

@pytest.fixture
def dataloader(cfg):
    inputs_dtype = torch.bfloat16
    return setup_dataloader(cfg, inputs_dtype)

def test_dataloader_speed(cfg, dataloader):
    num_workers = cfg.train.num_workers
    logger.info(f"Using {num_workers} workers.")
    dataloader_iter = iter(dataloader)
    
    for idx in range(32):
        t0 = time.time()
        data = next(dataloader_iter)
        tf = time.time() - t0
        logger.info(f"Batch {idx}: waited {tf:.06f} seconds.")

    assert True

def test_dataloader_output(cfg, dataloader):
    dataloader_iter = iter(dataloader)
    data = next(dataloader_iter)

    assert all(x in data.keys() for x in ["global_3d", "local_3d", "global_2d", "local_2d"])

def test_dataloader_inspect(cfg, dataloader):
    dataloader_iter = iter(dataloader)
    data = next(dataloader_iter)
    output_path = "dinov2/tests/out"

    global_3d = data["global_3d"]
    _, V, D, W, H = global_3d.shape

    plt.figure()
    plt.imshow(global_3d[0,0,D//2,:,:], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_3d_viewA.png"))

    plt.figure()
    plt.imshow(global_3d[0,0,:,W//2,:], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_3d_viewB.png"))

    plt.figure()
    plt.imshow(global_3d[0,0,:,:,H//2], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_3d_viewC.png"))