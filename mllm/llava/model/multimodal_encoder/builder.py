from .dino_encoder import DINOVisionTower


def build_vision_tower(vision_tower_cfg, torch_dtype, delay_load=False):
    vision_tower = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )

    return DINOVisionTower(
        vision_tower,
        args=vision_tower_cfg,
        torch_dtype=torch_dtype,
        delay_load=delay_load,
    )
