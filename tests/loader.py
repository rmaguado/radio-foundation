import pytest
from omegaconf import OmegaConf
import torch

from dinov2.configs import dinov2_default_config
from dinov2.train.setup import setup_dataloader

@pytest.fixture
def prep_cfg():
    config_file = "configs/tests/vitb_cnn_depth.yaml"
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli())
    return cfg

def test_loader(prep_cfg):
    dataloader = setup_dataloader(cfg, torch.half, use_full_image=False)
    get_iter = iter(dataloader)
    example_data = next(iterable_dataloader)