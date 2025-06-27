from omegaconf import OmegaConf
from pydantic import BaseModel, field_validator
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

    @field_validator("koleo_loss_weight", mode="before")
    @classmethod
    def validate_koleo_loss_weight(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Koleo Loss weight must be between 0.0 and 1.0")
        return v

    @field_validator("loss_weight", mode="before")
    @classmethod
    def validate_dino_loss_weight_nonzero(cls, v):
        if not 0.0 < v <= 1.0:
            raise ValueError(
                "Dino loss weight must be non-zero and between 0.0 and 1.0"
            )
        return v

    @field_validator(
        "head_n_prototypes",
        "head_bottleneck_dim",
        "head_nlayers",
        "head_hidden_dim",
        mode="before",
    )
    @classmethod
    def validate_head_params(cls, v):
        if v <= 0:
            raise ValueError("Head parameters must be positive integers")
        return v


class IbotConfig(BaseModel):
    loss_weight: float
    mask_sample_probability: float
    mask_ratio_min_max: Tuple[float, float]
    separate_head: bool
    head_n_prototypes: int
    head_bottleneck_dim: int
    head_nlayers: int
    head_hidden_dim: int

    @field_validator("mask_sample_probability", mode="before")
    @classmethod
    def validate_mask_sample_probability(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("mask_sample_probability must be between 0.0 and 1.0")
        return v

    @field_validator("mask_ratio_min_max", mode="before")
    @classmethod
    def validate_mask_ratio(cls, v):
        if not 0.0 <= v[0] <= v[1] <= 1.0:
            raise ValueError(
                "mask_ratio_min_max must be a tuple of two floats [min, max] between 0 and 1, with min <= max"
            )
        return v

    @field_validator("loss_weight", mode="before")
    @classmethod
    def validate_ibot_loss_weight(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Ibot loss weight must between 0.0 and 1.0")
        return v

    @field_validator(
        "head_n_prototypes",
        "head_bottleneck_dim",
        "head_nlayers",
        "head_hidden_dim",
        mode="before",
    )
    @classmethod
    def validate_ibot_head_params(cls, v):
        if v <= 0:
            raise ValueError("Ibot head parameters must be positive integers")
        return v


class TrainStageConfig(BaseModel):
    epochs: int
    batch_size_total: int
    batch_size_per_gpu: int
    grad_accum_steps: int

    @field_validator(
        "epochs",
        "batch_size_total",
        "batch_size_per_gpu",
        "grad_accum_steps",
        mode="before",
    )
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v


class TrainConfig(BaseModel):
    output_dir: str
    seed: int
    num_workers: int
    OFFICIAL_EPOCH_LENGTH: int
    cache_dataset: bool
    centering: Literal["centering", "sinkhorn_knopp"]
    stage1: TrainStageConfig
    stage2: TrainStageConfig

    @field_validator("OFFICIAL_EPOCH_LENGTH", mode="before")
    @classmethod
    def validate_epoch_length(cls, v):
        if v <= 0:
            raise ValueError("OFFICIAL_EPOCH_LENGTH must be a positive integer")
        return v


class CheckpointsConfig(BaseModel):
    print_iterations: int
    save_checkpoint_iterations: int
    save_teacher_iterations: int

    @field_validator(
        "print_iterations",
        "save_checkpoint_iterations",
        "save_teacher_iterations",
        mode="before",
    )
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v


class StudentConfig(BaseModel):
    arch: str
    patch_size: int
    full_image_size: int
    channels: int
    drop_path_rate: float
    layerscale: float
    drop_path_uniform: bool
    pretrained_weights: str
    ffn_layer: Literal["mlp", "swiglu"]
    block_chunks: int
    qkv_bias: bool
    proj_bias: bool
    ffn_bias: bool
    num_register_tokens: int
    interpolate_antialias: bool
    interpolate_offset: float

    @field_validator("drop_path_rate", "layerscale", mode="before")
    @classmethod
    def validate_float_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Value must be between 0.0 and 1.0")
        return v

    @field_validator("patch_size", "full_image_size", "channels", mode="before")
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v


class TeacherConfig(BaseModel):
    momentum_teacher: float
    final_momentum_teacher: float
    warmup_teacher_temp: float
    teacher_temp: float
    warmup_teacher_temp_epochs: int

    @field_validator("momentum_teacher", "final_momentum_teacher", mode="before")
    @classmethod
    def validate_momentum(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Momentum values must be between 0.0 and 1.0")
        return v

    @field_validator("final_momentum_teacher", mode="before")
    @classmethod
    def validate_final_momentum(cls, v, values):
        if "momentum_teacher" in values and v < values["momentum_teacher"]:
            raise ValueError(
                "final_momentum_teacher must be greater than or equal to momentum_teacher"
            )
        return v

    @field_validator("warmup_teacher_temp", "teacher_temp", mode="before")
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 < v <= 1.0:
            raise ValueError("Temperature values must be between 0.0 and 1.0")
        return v

    @field_validator("warmup_teacher_temp_epochs", mode="before")
    @classmethod
    def validate_warmup_epochs(cls, v):
        if v < 0:
            raise ValueError(
                "warmup_teacher_temp_epochs must be a non negative integer"
            )
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

    @field_validator(
        "weight_decay",
        "weight_decay_end",
        "base_lr",
        "min_lr",
        "patch_embed_lr_mult",
        "layerwise_decay",
        "adamw_beta1",
        "adamw_beta2",
        mode="before",
    )
    @classmethod
    def validate_positive_floats(cls, v):
        if v <= 0.0:
            raise ValueError("Value must be a positive float")
        return v

    @field_validator("freeze_last_layer_epochs", "warmup_epochs", mode="before")
    @classmethod
    def validate_non_negative_integers(cls, v):
        if v < 0:
            raise ValueError("Value must be a non-negative integer")
        return v


class CropsStageConfig(BaseModel):
    small_size: int
    medium_size: int
    large_size: int

    @field_validator("small_size", "medium_size", "large_size", mode="before")
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Crop sizes must be positive integers")
        return v


class CropsConfig(BaseModel):
    global_crop_scale: Tuple[float, float]
    local_crop_scale: Tuple[float, float]

    teacher_3d_global_crops: int
    student_3d_local_crops: int
    student_2d_global_crops: int
    teacher_2d_global_crops: int
    student_2d_local_crops: int

    stage1: CropsStageConfig
    stage2: CropsStageConfig

    @field_validator("global_crop_scale", "local_crop_scale", mode="before")
    @classmethod
    def validate_scales(cls, v):
        if not 0.0 <= v[0] <= v[1] <= 1.0:
            raise ValueError(
                "Crop scales must be a tuple of two floats [min, max] between 0 and 1, with min <= max"
            )
        return v


class PixelRangeConfig(BaseModel):
    lower: float
    upper: float

    @field_validator("lower", "upper", mode="before")
    @classmethod
    def validate_pixel_lower_upper(cls, v, values):
        if not values["lower"] < values["upper"]:
            raise ValueError("Pixel range lower must be less than upper")
        return v


class NormConfig(BaseModel):
    mean: float
    std: float


class DatasetConfig(BaseModel):
    name: str
    weight: float
    root_path: str
    type: Literal["ct", "mri"]
    storage: Literal["dicom", "nifti"]
    augmentation: str
    channels: int
    pixel_range: PixelRangeConfig
    norm: NormConfig

    @field_validator("weight", mode="before")
    @classmethod
    def validate_weight(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Dataset weight must be between 0.0 and 1.0")
        return v

    @field_validator("channels", mode="before")
    @classmethod
    def validate_channels(cls, v):
        if v <= 0:
            raise ValueError("Number of channels must be a positive integer")
        return v


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
    MainConfig(**conf_dict)  # type: ignore
    return True
