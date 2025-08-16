import pytest
import torch
import os
import matplotlib.pyplot as plt
import time
import logging

from dino.configs import get_cfg_from_path
from dino.train.setup import setup_dataloader

logger = logging.getLogger("dinov2")
logger.setLevel(logging.DEBUG)


@pytest.fixture
def cfg():
    return get_cfg_from_path("dinov2/configs/tests/minimal.yaml")


@pytest.fixture
def dataloader(cfg):
    inputs_dtype = torch.bfloat16
    return setup_dataloader(cfg, inputs_dtype, 0)


def test_dataloader_speed(cfg):
    num_workers = cfg.train.num_workers
    logger.info(f"Using {num_workers} workers.")

    inputs_dtype = torch.bfloat16
    t0 = time.time()
    dataloader = setup_dataloader(cfg, inputs_dtype, 0)
    dataloader_iter = iter(dataloader)
    tf = time.time() - t0
    logger.info(f"Created dataloader in {tf:.06f} seconds.")

    for idx in range(128):
        t0 = time.time()
        data = next(dataloader_iter)
        tf = time.time() - t0
        logger.info(f"Batch {idx}: waited {tf:.06f} seconds.")
        time.sleep(0.6)

    assert all(
        x in data.keys() for x in ["global_3d", "local_3d", "global_2d", "local_2d"]  # type: ignore
    )


def test_dataloader_output(dataloader):
    dataloader_iter = iter(dataloader)
    data = next(dataloader_iter)
    output_path = "dinov2/tests/out"

    global_3d = data["global_3d"]["images"].float().numpy()
    _, V, D, W, H = global_3d.shape

    plt.figure()
    plt.imshow(global_3d[0, 0, D // 2, :, :], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_3d_viewA.png"))

    plt.figure()
    plt.imshow(global_3d[0, 0, :, W // 2, :], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_3d_viewB.png"))

    plt.figure()
    plt.imshow(global_3d[0, 0, :, :, H // 2], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_3d_viewC.png"))

    global_2d = data["global_2d"]["images"].float().numpy()
    _, V, D, W, H = global_2d.shape

    plt.figure()
    plt.imshow(global_2d[0, 0, D // 2, :, :], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_2d_viewA.png"))

    plt.figure()
    plt.imshow(global_2d[0, 0, :, W // 2, :], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_2d_viewB.png"))

    plt.figure()
    plt.imshow(global_2d[0, 0, :, :, H // 2], cmap="gray")
    plt.colorbar()
    plt.savefig(os.path.join(output_path, "global_2d_viewC.png"))
