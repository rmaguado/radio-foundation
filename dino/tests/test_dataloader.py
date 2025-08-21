import pytest
import torch
import os
import matplotlib.pyplot as plt
import time
import logging

from dino.configs import get_cfg_from_path
from dino.train.setup import setup_dataloader

logger = logging.getLogger("dino")
logger.setLevel(logging.DEBUG)


@pytest.fixture
def cfg():
    return get_cfg_from_path("dino/configs/tests/minimal.yaml")


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


def test_dataloader_output(dataloader):
    def extract_imgs(view, output_path, idx):
        plt.figure()
        plt.imshow(view[0, 0, D // 2, :, :], cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(output_path, f"{idx}A.png"))

        plt.figure()
        plt.imshow(view[0, 0, :, W // 2, :], cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(output_path, f"{idx}B.png"))

        plt.figure()
        plt.imshow(view[0, 0, :, :, H // 2], cmap="gray")
        plt.colorbar()
        plt.savefig(os.path.join(output_path, f"{idx}C.png"))

    dataloader_iter = iter(dataloader)
    output_path = "dino/tests/out"

    for idx, data in enumerate(dataloader_iter):

        global_view = data["global"].float().numpy()
        _, V, D, W, H = global_view.shape

        extract_imgs(global_view, output_path, f"g_{idx:02}")

        local_view = data["global"].float().numpy()
        _, V, D, W, H = local_view.shape

        extract_imgs(local_view, output_path, f"l_{idx:02}")

        if idx == 5:
            break
