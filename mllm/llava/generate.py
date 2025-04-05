import os
import logging
import torch
import transformers

from mllm.llava.model import *
from mllm.llava.train.lora import configure_lora
from mllm.llava.data.data import make_supervised_data_module
from mllm.llava.config import load_generate_config


def generate():
    model_args, data_args, training_args, output_dir = load_generate_config()
    logging.basicConfig(
        filename=os.path.join(output_dir, "generate.log"), level=logging.DEBUG
    )

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        fsdp=training_args.fsdp,
    )

    if training_args.lora_enable:
        configure_lora(
            model,
            training_args,
            model_args.lora_backbone,
            model_args.lora_language,
        )

    logging.info(f"Loading from checkpoint: {model_args.checkpoint_path}")
    pretrained_weights = torch.load(model_args.checkpoint_path, map_location="cpu")
    model.load_state_dict(pretrained_weights, strict=False)

    logging.info(f"Loaded weights")
    logging.info(pretrained_weights.keys())

    # model.get_vision_tower().to(training_args.device)
    # model.get_mm_projector().to(training_args.device)

    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    model.requires_grad_(False)
    model.to(training_args.device)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataloader = torch.utils.data.DataLoader(
        data_module["train_dataset"],
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=data_module["data_collator"],
        num_workers=training_args.dataloader_num_workers,
    )

    outdir = os.path.join(
        training_args.output_dir,
        "evaluation",
    )
    os.makedirs(outdir, exist_ok=True)
    logging.info(f"Output directory: {outdir}")

    for batch in dataloader:
        if "image" in batch:
            batch["images"] = batch.pop("image")

        input_ids = batch.pop("input_ids").to(training_args.device)
        attention_mask = batch.pop("attention_mask").to(training_args.device)
        images = [x.to(training_args.device) for x in batch.pop("images")]

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=images,
                max_length=training_args.model_max_length,
                pad_token_id=tokenizer.pad_token_id,
            )

        for i, output in enumerate(outputs):
            output = tokenizer.decode(output, skip_special_tokens=True)
            with open(os.path.join(outdir, f"output_{i}.txt"), "w") as f:
                f.write(output)

        break


if __name__ == "__main__":
    try:
        generate()
    except Exception as e:
        logging.exception(e)
    finally:
        torch.distributed.destroy_process_group()
