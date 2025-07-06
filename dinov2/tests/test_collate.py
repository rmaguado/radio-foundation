import torch
import pytest


@pytest.fixture
def dummy_batch():
    return [
        {
            "3d_global_ld": {
                "images": torch.rand(1, 1, 112, 112, 112),
                "is_target": True,
                "targets": [],
                "embed_layer": "patch3d",
                "mask_shape": (8, 8, 8),
                "view_shape": (1,),
            },
            "2d_global_hd": {
                "images": torch.rand(1, 3, 1, 1, 224, 224),
                "is_target": True,
                "targets": ["3d_global_ld"],
                "embed_layer": "patch2d",
                "mask_shape": (16, 16),
                "view_shape": (1, 3),
            },
            "2d_local_hd": {
                "images": torch.rand(1, 3, 2, 1, 1, 112, 112),
                "is_target": False,
                "targets": ["3d_global_ld", "2d_global_hd"],
                "embed_layer": "patch2d",
                "mask_shape": (8, 8),
                "view_shape": (1, 3, 2),
            },
        } for _ in range(5)
    ]

def test_collate(dummy_batch):
    from dinov2.data.collate import collate_data_and_cast
    from dinov2.data.masking import MaskingGenerator

    mask_ratio_tuple = (0.1, 0.5)
    mask_probability = 0.5
    dtype = torch.float32
    mask_generator = MaskingGenerator()

    collated_data = collate_data_and_cast(
        dummy_batch,
        mask_ratio_tuple,
        mask_probability,
        dtype,
        mask_generator
    )

    assert isinstance(collated_data, dict)
    assert "3d_global_ld" in collated_data
    assert "2d_global_hd" in collated_data
    assert "2d_local_hd" in collated_data
    assert collated_data["3d_global_ld"]["images"].dtype == dtype
    assert collated_data["2d_global_hd"]["images"].dtype == dtype
    assert collated_data["2d_local_hd"]["images"].dtype == dtype
    assert collated_data["3d_global_ld"]["masks"].shape == (5, 1, 8, 8, 8)
    assert collated_data["2d_global_hd"]["masks"].shape == (5, 1, 3, 16, 16)
    