"""
Copied and modified from:
https://github.com/2U1/Llama3.2-Vision-Finetune/blob/master/src/training/train.py#L73
"""

import torch
import logging
from peft import LoraConfig, PeftModel, get_peft_model


logger = logging.getLogger("DeepSpeed")


def find_target_linear_names(model, lora_namespan_exclude=[], include_modules=None):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if include_modules is not None:
            if not any(inc_keyword in name for inc_keyword in include_modules):
                continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    return lora_module_names


def configure_lora(model, model_args):
    vision_args = model_args.lora_vision
    language_args = model_args.lora_language
    pretrain_checkpoint_path = model_args.pretrain_checkpoint_path

    is_peft_model = isinstance(model, PeftModel)

    adapters_config = {}
    if vision_args.lora_enable:
        vision_target_modules = find_target_linear_names(
            model.get_model().vision_tower,
            lora_namespan_exclude=vision_args.exclude_modules,
            include_modules=vision_args.include_modules,
        )

        vision_config = LoraConfig(
            r=vision_args.lora_r,
            lora_alpha=vision_args.lora_alpha,
            target_modules=vision_target_modules,
            lora_dropout=vision_args.lora_dropout,
            bias=vision_args.lora_bias,
            init_lora_weights=True if pretrain_checkpoint_path is None else False,
        )
        adapters_config["vision_adapter"] = vision_config

    if language_args.lora_enable:
        language_target_modules = find_target_linear_names(
            model,
            lora_namespan_exclude=language_args.exclude_modules,
            include_modules=language_args.include_modules,
        )

        language_config = LoraConfig(
            r=language_args.lora_r,
            lora_alpha=language_args.lora_alpha,
            target_modules=language_target_modules,
            lora_dropout=language_args.lora_dropout,
            bias=language_args.lora_bias,
            task_type="CAUSAL_LM",
            init_lora_weights=True if pretrain_checkpoint_path is None else False,
        )
        adapters_config["language_adapter"] = language_config

    if is_peft_model:
        current_adapters = set(model.peft_config.keys())
        logger.info(f"Found existing lora adapters: {current_adapters}")

        for adapter_name, config in adapters_config.items():
            if adapter_name not in current_adapters:
                logger.info(f"Adding new lora adapter: {adapter_name}")
                model.add_adapter(adapter_name, config)
    else:
        if adapters_config:
            first_adapter = list(adapters_config.keys())[0]
            model = get_peft_model(model, adapters_config[first_adapter])

            for adapter_name, config in list(adapters_config.items())[1:]:
                logger.info(f"Adding new lora adapter: {adapter_name}")
                model.add_adapter(adapter_name, config)

    return model
