import torch
import pytest
from dinov2.data.augmentations import DataAugmentationDINO
from dinov2.configs import get_cfg_from_path


@pytest.fixture
def dummy_image3d():
    return torch.rand(1, 512, 512, 512)

@pytest.fixture
def config():
    return get_cfg_from_path("dinov2/tests/configs/minimal_model.yaml")


def test_recursive_hierarchy(dummy_image3d, config):
    dataset_cfg = config.datasets[0]
    augmenter = DataAugmentationDINO(config, dataset_cfg)

    output = augmenter(dummy_image3d)

    assert output["3d_global_ld"]["images"].shape == (1, 1, 112, 112, 112)
    assert output["3d_local_ld"]["images"].shape == (1, 3, 1, 56, 56, 56)
    assert output["2d_global_hd"]["images"].shape == (1, 3, 1, 1, 224, 224)
    assert output["2d_local_hd"]["images"].shape == (1, 3, 2, 1, 1, 112, 112)

    assert output["3d_global_ld"]["view_shape"] == [1]
    assert output["3d_local_ld"]["view_shape"] == [1, 3]
    assert output["2d_global_hd"]["view_shape"] == [1, 3]
    assert output["2d_local_hd"]["view_shape"] == [1, 3, 2]
