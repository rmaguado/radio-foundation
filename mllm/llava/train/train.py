# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import pathlib
import torch
import transformers
import os

import torch.distributed as dist

from mllm.llava.train.trainer import LLaVATrainer

from mllm.llava.model import *
from mllm.llava.train.lora import configure_lora
from mllm.llava.train.save import save_model
from mllm.llava.data.data import make_supervised_data_module
from mllm.llava.config import load_train_config


local_rank = None
logger = logging.getLogger("DeepSpeed")


def train(attn_implementation="flash_attention_2"):
    logger.info("Starting training. ")

    model_args, data_args, training_args = load_train_config()

    global local_rank
    local_rank = training_args.local_rank

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else torch.float16),
    )

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    model.freeze_language()

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        fsdp=training_args.fsdp,
    )

    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)

    configure_lora(model.get_model(), model_args)

    if model_args.checkpoint_path is not None:
        pretrained_weights = torch.load(model_args.checkpoint_path, map_location="cpu")
        model.load_state_dict(pretrained_weights, strict=False)

        logger.info(f"Loaded weights from checkpoint: {model_args.checkpoint_path}")

    model.get_vision_tower().to(training_args.device)
    model.get_mm_projector().to(training_args.device)

    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True

    run_dir = os.path.dirname(training_args.output_dir)
    with open(
        os.path.join(run_dir, "gradients.txt"),
        mode="w",
        encoding="utf-8",
    ) as f:
        for name, module in model.named_modules():
            if name:
                has_gradients = any(p.requires_grad for p in module.parameters())
                f.write(f"{name} {has_gradients}\n")

    model.config.mm_projector_lr = training_args.mm_projector_lr
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    final_dir = os.path.join(run_dir, "final")
    os.makedirs(final_dir, exist_ok=True)

    save_model(training_args, model, final_dir)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.exception(e)
    finally:
        torch.distributed.destroy_process_group()
