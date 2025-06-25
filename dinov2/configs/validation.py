import yaml
import os
from omegaconf import OmegaConf
from pydantic import BaseModel, validator, ValidationError, Field
from typing import List, Optional, Literal, Tuple, Dict, Any


class ModelConfig(BaseModel):
    WEIGHTS: str


class MixedPrecisionConfig(BaseModel):
    param_dtype: Literal["fp16", "fp32", "bf16"]
    reduce_dtype: Literal["fp16", "fp32", "bf16"]
    buffer_dtype: Literal["fp16", "fp32", "bf16"]


class PrecisionComponentConfig(BaseModel):
    sharding_strategy: Literal[
        "SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD", "HYBRID_SHARD"
    ]
    mixed_precision: MixedPrecisionConfig


class ComputePrecisionConfig(BaseModel):
    grad_scaler: bool
    teacher: Dict[str, PrecisionComponentConfig]
    student: Dict[str, PrecisionComponentConfig]


class DinoConfig(BaseModel):
    loss_weight: float
    head_n_prototypes: int
    head_bottleneck_dim: int
    head_nlayers: int
    head_hidden_dim: int
    koleo_loss_weight: float


class IbotConfig(BaseModel):
    loss_weight: float
    mask_sample_probability: float
    mask_ratio_min_max: Tuple[float, float]
    separate_head: bool
    head_n_prototypes: int
    head_bottleneck_dim: int
    head_nlayers: int
    head_hidden_dim: int

    @validator("mask_ratio_min_max")
    def validate_mask_ratio(cls, v):
        if not 0.0 <= v[0] <= v[1] <= 1.0:
            raise ValueError(
                "mask_ratio_min_max must be a tuple of two floats [min, max] between 0 and 1, with min <= max"
            )
        return v


class TrainStageConfig(BaseModel):
    epochs: int
    batch_size_total: int
    batch_size_per_gpu: int
    grad_accum_steps: int


class TrainConfig(BaseModel):
    output_dir: str
    seed: int
    num_workers: int
    OFFICIAL_EPOCH_LENGTH: int
    cache_dataset: bool
    centering: Literal["centering", "sinkhorn_knopp"]
    stage1: TrainStageConfig
    stage2: TrainStageConfig


class CheckpointsConfig(BaseModel):
    print_iterations: int
    save_checkpoint_iterations: int
    save_teacher_iterations: int


class StudentConfig(BaseModel):
    arch: str
    patch_size: int
    full_image_size: int
    channels: int
    drop_path_rate: float
    layerscale: float
    drop_path_uniform: bool
    pretrained_weights: str
    ffn_layer: str
    block_chunks: int
    qkv_bias: bool
    proj_bias: bool
    ffn_bias: bool
    num_register_tokens: int
    interpolate_antialias: bool
    interpolate_offset: float


class TeacherConfig(BaseModel):
    momentum_teacher: float
    final_momentum_teacher: float
    warmup_teacher_temp: float
    teacher_temp: float
    warmup_teacher_temp_epochs: int

    @validator("momentum_teacher", "final_momentum_teacher")
    def validate_momentum(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Momentum values must be between 0.0 and 1.0")
        return v


class OptimConfig(BaseModel):
    weight_decay: float
    weight_decay_end: float
    base_lr: float
    lr: float
    warmup_epochs: int
    min_lr: float
    clip_grad: float
    freeze_last_layer_epochs: int
    patch_embed_lr_mult: float
    layerwise_decay: float
    adamw_beta1: float
    adamw_beta2: float


class CropsStageConfig(BaseModel):
    global_crops_size: int
    local_crops_size: int


class CropsConfig(BaseModel):
    global_crops_scale: Tuple[float, float]
    local_crops_number: int
    local_crops_scale: Tuple[float, float]
    stage1: CropsStageConfig
    stage2: CropsStageConfig

    @validator("global_crops_scale", "local_crops_scale")
    def validate_scales(cls, v):
        if not 0.0 <= v[0] <= v[1] <= 1.0:
            raise ValueError(
                "Crop scales must be a tuple of two floats [min, max] between 0 and 1, with min <= max"
            )
        return v


class PixelRangeConfig(BaseModel):
    lower: float
    upper: float


class NormConfig(BaseModel):
    mean: float
    std: float


class DatasetConfig(BaseModel):
    name: str
    weight: float
    root_path: str
    type: str
    storage: str
    augmentation: str
    channels: int
    pixel_range: PixelRangeConfig
    norm: NormConfig


class AugmentationStepConfig(BaseModel):
    name: str
    p: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None


class AugmentationCollection(BaseModel):
    global_1: List[AugmentationStepConfig]
    global_2: List[AugmentationStepConfig]
    local: List[AugmentationStepConfig]


class AugmentationsConfig(BaseModel):
    default_ct: AugmentationCollection


class MainConfig(BaseModel):
    MODEL: ModelConfig
    compute_precision: ComputePrecisionConfig
    dino: DinoConfig
    ibot: IbotConfig
    train: TrainConfig
    checkpoints: CheckpointsConfig
    student: StudentConfig
    teacher: TeacherConfig
    optim: OptimConfig
    crops: CropsConfig
    datasets: List[DatasetConfig]
    augmentations: AugmentationsConfig


def validate_config(conf) -> bool:
    conf_dict = OmegaConf.to_container(conf, resolve=True)
    MainConfig(**conf_dict)
    return True
