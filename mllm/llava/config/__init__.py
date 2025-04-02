from dataclasses import dataclass, field
from typing import Optional, List
import transformers
from omegaconf import OmegaConf
import argparse
import os


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=512)
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Root path to the .yaml file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path in runs folder to save. ",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        required=True,
        help="Path to the .json file.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Rank of current process.",
    )
    return parser.parse_args()


def load_config():
    args = get_argpase()

    default_cfg = OmegaConf.create(
        OmegaConf.load("mllm/llava/config/default_config.yaml")
    )
    cfg = OmegaConf.load(args.config_path)
    cfg = OmegaConf.merge(default_cfg, cfg)

    output_dir = os.path.join(args.output_path, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    model_args = cfg["model"]
    data_args = cfg["data"]
    training_args = TrainingArguments(
        **cfg["train"], deepspeed=args.deepspeed, output_dir=output_dir
    )

    model_args.image_tokens = data_args.image_tokens

    with open(os.path.join(args.output_path, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    return model_args, data_args, training_args
