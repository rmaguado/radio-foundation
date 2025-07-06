import torch
import pytest
from dinov2.data.augmentations import DataAugmentationDINO
from dinov2.configs import get_cfg_from_path


@pytest.fixture
def dummy_image():
    return torch.rand(512, 512, 512)

@pytest.fixture
def config():
    return get_cfg_from_path("dinov2/tests/configs/minimal_model.yaml")

def test_initialization(config):
    dataset_cfg = config.datasets[0]
    augmenter = DataAugmentationDINO(config, dataset_cfg)

    # Check that transform_groups are parsed
    assert isinstance(augmenter.transforms, dict)
    assert len(augmenter.transforms) > 0

    # Each transform group should have corresponding info
    for group_name in augmenter.transforms:
        assert group_name in augmenter.group_info
        info = augmenter.group_info[group_name]
        assert "size" in info
        assert "num_crops" in info
        assert "embed_layer" in info
        assert "mask_shape" in info


def test_output_structure(dummy_image, config):
    dataset_cfg = config.datasets[0]
    augmenter = DataAugmentationDINO(config, dataset_cfg)

    output = augmenter(dummy_image)

    expected_groups = {g["name"] for g in config.transform_groups}
    for key in output:
        assert key in expected_groups

    for group_name, group_data in output.items():
        assert "images" in group_data
        assert "size" in group_data
        assert "num_crops" in group_data
        assert isinstance(group_data["images"], torch.Tensor)

        num_crops = group_data["num_crops"]
        assert group_data["images"].shape[0] == num_crops


def test_recursive_hierarchy(dummy_image, config):
    dataset_cfg = config.datasets[0]
    augmenter = DataAugmentationDINO(config, dataset_cfg)

    output = augmenter(dummy_image)
    print(output)