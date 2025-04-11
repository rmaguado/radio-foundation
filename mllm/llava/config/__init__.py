from dataclasses import dataclass, field
from typing import Optional, List
import transformers
from omegaconf import OmegaConf
from datetime import datetime
import argparse
import os


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(default=1024)
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def get_train_args():
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


def get_generate_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Root path to the .yaml file.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint to load from. Should contain a /fina/model.bin.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated text. ",
    )

    return parser.parse_args()


def load_train_config():
    args = get_train_args()

    default_cfg = OmegaConf.create(
        OmegaConf.load("mllm/llava/config/default_config.yaml")
    )
    cfg = OmegaConf.load(args.config_path)
    cfg = OmegaConf.merge(default_cfg, cfg)

    output_dir = os.path.join(args.output_path, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    model_args = cfg["model"]
    data_args = cfg["data"]
    training_args = cfg["train"]
    training_args.logging_dir = os.path.join(
        training_args.logging_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    training_args = TrainingArguments(
        **cfg["train"], deepspeed=args.deepspeed, output_dir=output_dir
    )

    model_args.image_tokens = data_args.image_tokens
    training_args.lora_bias = model_args.lora_backbone.lora_bias
    assert model_args.lora_backbone.lora_bias == model_args.lora_language.lora_bias

    with open(os.path.join(args.output_path, "config.yaml"), "w") as f:
        OmegaConf.save(config=cfg, f=f)

    return model_args, data_args, training_args


def load_generate_config():
    args = get_generate_args()

    model_config_path = os.path.join(args.checkpoint_path, "config.json")
    model_checkpoint_path = os.path.join(args.checkpoint_path, "final/model.bin")
    generate_config_path = args.config_path

    default_cfg = OmegaConf.create(
        OmegaConf.load("mllm/llava/config/default_config.yaml")
    )
    model_cfg = OmegaConf.load(model_config_path)
    generate_cfg = OmegaConf.load(generate_config_path)
    cfg = OmegaConf.merge(default_cfg, model_cfg)
    cfg = OmegaConf.merge(cfg, generate_cfg)

    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)

    model_args = cfg["model"]
    data_args = cfg["data"]
    training_args = cfg["train"]

    training_args = TrainingArguments(**cfg["train"], output_dir=output_dir)

    model_args.checkpoint_path = model_checkpoint_path
    model_args.image_tokens = data_args.image_tokens

    return model_args, data_args, training_args, output_dir
