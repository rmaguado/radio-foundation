"""
Copied and modified from:
https://github.com/2U1/Llama3.2-Vision-Finetune/blob/master/src/training/train.py#L73
"""

import torch
import logging


logger = logging.getLogger("DeepSpeed")


def find_target_linear_names(model):
    lora_namespan_exclude = ["lm_head", "embed_tokens", "mm_projector"]
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    return lora_module_names


def configure_lora(model, training_args):
    from peft import LoraConfig, get_peft_model

    peft_config = LoraConfig(
        r=training_args.lora_r,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_target_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    logger.info("Adding LoRA adapters.")
    return get_peft_model(model, peft_config)
