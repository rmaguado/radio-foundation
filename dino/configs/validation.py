from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel, field_validator, model_validator
from typing import List, Optional, Literal, Tuple, Dict


class DinoConfig(BaseModel):
    loss_weight: float
    head_n_prototypes: int
    head_bottleneck_dim: int
    head_nlayers: int
    head_hidden_dim: int
    koleo_loss_weight: float
    koleo_loss_distributed: bool
    koleo_topk: int
    koleo_group_size: Optional[int] = None

    @field_validator("koleo_topk", "koleo_group_size", mode="before")
    @classmethod
    def validate_pos_int(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Value must be a positive integer.")
        return v

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
    batch_size_total: int
    batch_size_per_gpu: int
    grad_accum_steps: int

    @field_validator(
        "iterations_per_epoch",
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


class StudentConfig(BaseModel):
    embed_dim: int
    n_blocks: int
    num_heads: int
    ffn_ratio: int
    patch_size: int
    ndims: int
    img_size: int
    in_channels: int
    rope_base: float
    rope_shift_coords: Optional[float] = None
    rope_jitter_coords: Optional[float] = None
    rope_rescale_coords: Optional[float] = None
    drop_path_rate: float
    layerscale_init: float
    norm_layer: Literal["layernorm", "layernormbf16", "rmsnorm"]
    ffn_layer: Literal["mlp", "swiglu"]
    qkv_bias: bool
    proj_bias: bool
    ffn_bias: bool
    num_register_tokens: int
    resume_from_teacher_chkpt: str

    @field_validator("drop_path_rate", "layerscale_init", mode="before")
    @classmethod
    def validate_float_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Value must be between 0.0 and 1.0")
        return v

    @field_validator(
        "embed_dim",
        "n_blocks",
        "num_heads",
        "ffn_ratio",
        "num_register_tokens",
        "patch_size",
        "img_size",
        mode="before",
    )
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v

    @field_validator("ndims", mode="before")
    @classmethod
    def validate_ndims(cls, v):
        if not (v == 2 or v == 3):
            raise ValueError("ndims must be 2 or 3.")
        return v

    @field_validator("in_channels", mode="before")
    @classmethod
    def validate_inchannels(cls, v):
        if v is not None and v <= 0:
            raise ValueError("in_channels must be null or a positive integer.")
        return v

    @model_validator(mode="after")
    def check_channels_dims(self):
        if self.in_channels > 1 and self.ndims == 3:
            raise ValueError("If ndims = 3, then in_channels must be 1.")
        return self

    @model_validator(mode="after")
    def check_embed_dims_multiple(self):
        if not (self.embed_dim % (2 * self.ndims * self.num_heads) == 0):
            raise ValueError("embed_dim must be divisible by (2 * ndims * num_heads).")
        return self


class OptimConfig(BaseModel):
    clip_grad: float
    patch_embed_lr_mult: float
    layerwise_decay: float
    adamw_beta1: float
    adamw_beta2: float

    @field_validator(
        "clip_grad",
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


class ScheduleTemplate(BaseModel):
    start: float
    peak: float
    end: float
    warmup_epochs: int
    freeze_last_layer_epochs: Optional[int] = None
    cosine_epochs: Optional[int] = None

    @field_validator(
        "start",
        "peak",
        "end",
        "warmup_epochs",
        "freeze_last_layer_epochs",
        "cosine_epochs",
        mode="before",
    )
    @classmethod
    def validate_positive_float(cls, v):
        if v is not None and v < 0:
            raise ValueError("Value must be non-negative.")
        return v

    @model_validator(mode="after")
    def check_relationships(self):
        if not (self.start <= self.peak and self.end <= self.peak):
            raise ValueError("Must have start <= peak and end <= peak.")
        return self


class SchedulesConfig(BaseModel):
    lr: ScheduleTemplate
    weight_decay: ScheduleTemplate
    momentum: ScheduleTemplate
    teacher_temp: ScheduleTemplate


class CropsConfig(BaseModel):
    num_crops_global: int
    num_crops_local: int
    size_global: int
    size_local: int
    scale_global: List[float]
    scale_local: List[float]

    @field_validator(
        "num_crops_global",
        "num_crops_local",
        "size_global",
        "size_local",
        mode="before",
    )
    @classmethod
    def validate_positive_integers(cls, v):
        if v <= 0:
            raise ValueError("Value must be a positive integer")
        return v

    @field_validator(
        "scale_global",
        "scale_local",
        mode="before",
    )
    @classmethod
    def validate_scale(cls, v):
        if not len(v) == 2:
            raise ValueError("Value must be a list with two floats.")
        if not (0 < v[0] < v[1] <= 1.0):
            raise ValueError("v[0] and v[1] must be in (0, 1] with v[1] > v[0].")
        return v


class NormConfig(BaseModel):
    mean: float
    std: float


class DatasetConfig(BaseModel):
    name: str
    weight: Optional[float] = None
    index_path: str
    type: Literal["ct", "mri"]
    storage: Literal["torch"]  # "dicom", "nifti",
    bounds: Tuple[float, float]
    norm: NormConfig

    @field_validator("weight", mode="before")
    @classmethod
    def validate_weight(cls, v):
        if v is not None:
            if v <= 0:
                raise ValueError("Dataset weight must be positive floats.")
        return v

    @field_validator("bounds", mode="before")
    @classmethod
    def validate_bounds(cls, v):
        if not v[0] < v[1]:
            raise ValueError("Bounds lower must be less than upper")
        return v


class MainConfig(BaseModel):
    compute_precision: Literal["fp16", "fp32", "bf16"]
    dino: DinoConfig
    ibot: IbotConfig
    train: TrainConfig
    student: StudentConfig
    schedules: SchedulesConfig
    optim: OptimConfig
    crops: CropsConfig
    datasets: List[DatasetConfig]


def validate_config(conf: DictConfig) -> bool:
    """
    Validates the configuration dictionary against the MainConfig model.
    Raises an error if the configuration is invalid.

    Args:
        conf (DictConfig): The configuration dictionary to validate loaded from a YAML file using omegaconf.
    """
    conf_dict: Dict = OmegaConf.to_container(conf, resolve=True)  # type: ignore
    MainConfig(**conf_dict)
    return True
