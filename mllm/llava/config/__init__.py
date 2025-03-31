from dataclasses import dataclass, field
from typing import Optional
import transformers
from omegaconf import OmegaConf
import argparse


def get_argpase():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Root path to the .yaml file.",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default="mllm/configs/deepspeed/zero3.json",
        required=False,
        help="Path to the .json file.",
    )
    return parser


def load_config():
    args = get_argpase()

    default_cfg = OmegaConf.create("mllm/llava/config/default_config.yaml")
    cfg = OmegaConf.load(args.config_path)
    cfg = OmegaConf.merge(default_cfg, cfg)

    model_args = cfg["model"]
    data_args = cfg["data"]
    training_args = transformers.TrainingArguments(
        **cfg["train"], deepspeed=args.deepspeed
    )

    model_args.image_tokens = data_args.image_tokens

    return model_args, data_args, training_args
