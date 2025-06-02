"""
Copied and modified from:
https://github.com/2U1/Llama3.2-Vision-Finetune/blob/master/src/training/train.py#L73
"""

import torch
import logging
from peft import LoraConfig, PeftModel, get_peft_model


logger = logging.getLogger("DeepSpeed")


def find_target_linear_names(
    model, exclude_modules=None, include_modules=None, must_have_keywords=None
):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if must_have_keywords is not None:
            if not all(inc_keyword in name for inc_keyword in must_have_keywords):
                continue
        if any(ex_keyword in name for ex_keyword in exclude_modules):
            continue
        if include_modules is not None:
            if not any(inc_keyword in name for inc_keyword in include_modules):
                continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    return lora_module_names


def configure_lora(model, model_args):
    lora_args = model_args.lora
    is_peft_model = hasattr(model, "peft_config")
    if is_peft_model:
        return model

    target_modules = []
    rank_pattern = {}
    alpha_pattern = {}

    if lora_args.lora_vision:
        vision_target_modules = find_target_linear_names(
            model.get_model().vision_tower,
            exclude_modules=lora_args.exclude_modules,
            include_modules=lora_args.lora_vision_modules,
            must_have_keywords=[lora_args.lora_vision_pattern],
        )
        vision_rank_pattern = {
            k: lora_args.lora_vision_rank for k in vision_target_modules
        }
        vision_alpha_pattern = {
            k: lora_args.lora_vision_alpha for k in vision_target_modules
        }
        target_modules += vision_target_modules
        rank_pattern.update(vision_rank_pattern)
        alpha_pattern.update(vision_alpha_pattern)

    if lora_args.lora_language:
        language_exclude_modules = lora_args.exclude_modules + [
            lora_args.lora_vision_pattern
        ]
        language_target_modules = find_target_linear_names(
            model,
            exclude_modules=language_exclude_modules,
            include_modules=lora_args.lora_language_modules,
        )
        language_rank_pattern = {
            k: lora_args.lora_language_rank for k in language_target_modules
        }
        language_alpha_pattern = {
            k: lora_args.lora_language_alpha for k in language_target_modules
        }
        target_modules += language_target_modules
        rank_pattern.update(language_rank_pattern)
        alpha_pattern.update(language_alpha_pattern)

    config = LoraConfig(
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        init_lora_weights=True,
        task_type="CAUSAL_LM",
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        
    )

    model = get_peft_model(model, config)

    return model
