"""
Copied and modified from:
https://github.com/2U1/Llama3.2-Vision-Finetune/blob/master/src/training/train.py#L73
"""

import torch
import logging


logger = logging.getLogger("DeepSpeed")


def find_target_linear_names(model, lora_namespan_exclude=[]):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    return lora_module_names


def _configure_lora_module(module, args):
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=find_target_linear_names(module),
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
    )
    return get_peft_model(module, peft_config)


def configure_lora(model, model_args):
    backbone_settings = model_args.lora_backbone

    if backbone_settings.lora_enable:
        _configure_lora_module(model.vision_tower.vision_tower, backbone_settings)
        logger.info("Adding LoRA adapters to vision tower.")

    language_settings = model_args.lora_language
    if language_settings.lora_enable:
        _configure_lora_module(model.layers, language_settings)
        logger.info("Adding LoRA adapters to language model.")
