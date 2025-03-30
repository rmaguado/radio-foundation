from dataclasses import dataclass, field
from typing import Optional
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_vision_config_path: Optional[str] = field(default=None)
    mm_vision_checkpoint_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    root_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    db_name: str = field(
        default="CT-RATE_train_reports",
        metadata={"help": "Name of .db file indexing images and reports."},
    )
    channels: int = field(default=1)
    data_mean: float = field(default=0.0)
    data_std: float = field(default=1.0)
    is_multimodal: bool = True
    image_tokens: int = 128
    conv_template: Optional[str] = field(default="plain")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: float = 1e-4
    group_by_modality_length: bool = field(default=False)
    report_to: str = field(default=None)
    log_dir: str = field(default=None)


def get_args():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model_args.image_tokens = data_args.image_tokens

    if training_args.log_dir is None:
        training_args.log_dir = training_args.output_dir

    return model_args, data_args, training_args
