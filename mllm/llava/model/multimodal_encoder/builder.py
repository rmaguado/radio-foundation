import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .dino_encoder import DINOVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower_name = getattr(
        vision_tower_cfg,
        "mm_vision_tower",
        getattr(vision_tower_cfg, "vision_tower", None),
    )

    vision_tower_type = getattr(vision_tower_cfg, "vision_tower_type", None)

    if vision_tower_type == "dino":
        return DINOVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)
    elif vision_tower_type == "clip":
        use_s2 = getattr(vision_tower_cfg, "s2", False)
        if use_s2:
            return CLIPVisionTowerS2(vision_tower_name, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs)

    raise ValueError(f"Unknown vision tower: {vision_tower_type}")
