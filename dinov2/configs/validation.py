from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Dict, Optional, Literal, Tuple


class MixedPrecisionConfig(BaseModel):
    param_dtype: Literal["fp16", "fp32", "bf16"]
    reduce_dtype: Literal["fp16", "fp32", "bf16"]
    buffer_dtype: Literal["fp16", "fp32", "bf16"]


class ModuleMixedPrecisionConfig(BaseModel):
    backbone: MixedPrecisionConfig
    dino_head: MixedPrecisionConfig
    ibot_head: MixedPrecisionConfig


class ComputePrecisionConfig(BaseModel):
    teacher: Dict[str, ModuleMixedPrecisionConfig]
    student: Dict[str, ModuleMixedPrecisionConfig]


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


class TrainConfig(BaseModel):
    output_dir: str
    seed: int
    num_workers: int
    iterations_per_epoch: int
    centering: Literal["centering", "sinkhorn_knopp"]
    epochs: int
    batch_size_total: int
    batch_size_per_gpu: int
    grad_accum_steps: int

    @field_validator(
        "iterations_per_epoch",
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


class EmbedLayerConfig(BaseModel):
    type: Literal["patch2d", "patch3d"]
    patch_size: int
    img_size: int
    in_channels: Optional[int] = None

    @field_validator("patch_size", "img_size", mode="before")
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v

    @field_validator("in_channels", mode="before")
    @classmethod
    def validate_in_channels(cls, v):
        if v is not None and v <= 0:
            raise ValueError("in_channels must be a positive integer")
        return v


class StudentConfig(BaseModel):
    model_name: str
    embed_dim: int
    depth: int
    num_heads: int
    mlp_ratio: int
    embed_layers: List[EmbedLayerConfig]
    drop_path_rate: float
    layerscale: float
    drop_path_uniform: bool
    pretrained_weights: str
    ffn_layer: Literal["mlp", "swiglu"]
    qkv_bias: bool
    proj_bias: bool
    ffn_bias: bool
    num_register_tokens: int

    @field_validator("drop_path_rate", "layerscale", mode="before")
    @classmethod
    def validate_float_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Value must be between 0.0 and 1.0")
        return v

    @field_validator(
        "embed_dim",
        "depth",
        "num_heads",
        "mlp_ratio",
        "num_register_tokens",
        mode="before",
    )
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

    @model_validator(mode="after")
    def validate_final_momentum(self):
        if self.final_momentum_teacher < self.momentum_teacher:
            raise ValueError(
                "final_momentum_teacher must be greater than or equal to momentum_teacher"
            )
        return self

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


class CropsConfig(BaseModel):
    global_crop_scale: Tuple[float, float]
    local_crop_scale: Tuple[float, float]

    @field_validator("global_crop_scale", "local_crop_scale", mode="before")
    @classmethod
    def validate_scales(cls, v):
        if not 0.0 <= v[0] <= v[1] <= 1.0:
            raise ValueError(
                "Crop scales must be a tuple of two floats [min, max] between 0 and 1, with min <= max"
            )
        return v


class TransformConfig(BaseModel):
    name: str
    p: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None


class TransformGroup(BaseModel):
    name: str
    size: int
    num_crops: int
    embed_layer: Literal["patch2d", "patch3d"]
    is_target: bool = False
    targets: Optional[List[str]] = None
    transforms: List[TransformConfig]

    @field_validator("size", "num_crops", mode="before")
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v


class AugmentationsNode(BaseModel):
    name: str
    subgroups: Optional[List["AugmentationsNode"]] = None


AugmentationsNode.model_rebuild()


class PixelRangeConfig(BaseModel):
    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_pixel_lower_upper(self):
        if not self.lower < self.upper:
            raise ValueError("Pixel range lower must be less than upper")
        return self


class NormConfig(BaseModel):
    mean: float
    std: float


class DatasetConfig(BaseModel):
    name: str
    weight: float
    index_path: str
    type: Literal["ct", "mri"]
    storage: Literal["dicom", "nifti"]
    augmentation: str
    pixel_range: PixelRangeConfig
    norm: NormConfig

    @field_validator("weight", mode="before")
    @classmethod
    def validate_weight(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Dataset weight must be between 0.0 and 1.0")
        return v


class MainConfig(BaseModel):
    compute_precision: ComputePrecisionConfig
    dino: DinoConfig
    ibot: IbotConfig
    train: TrainConfig
    checkpoints: CheckpointsConfig
    student: StudentConfig
    teacher: TeacherConfig
    optim: OptimConfig
    crops: CropsConfig
    transform_groups: List[TransformGroup]
    augmentations: Dict[str, List[AugmentationsNode]]
    datasets: List[DatasetConfig]


def validate_config(conf: DictConfig) -> bool:
    """
    Validates the configuration dictionary against the MainConfig model.
    Raises an error if the configuration is invalid.

    Args:
        conf (DictConfig): The configuration dictionary to validate loaded from a YAML file using omegaconf.
    """
    conf_dict = OmegaConf.to_container(conf, resolve=True)
    MainConfig(**conf_dict)
    return True
