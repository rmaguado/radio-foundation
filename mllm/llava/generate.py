import os
import logging
import torch
import transformers
from transformers import AutoConfig
from peft import PeftModel
from tqdm import tqdm

from mllm.llava.model import *
from mllm.llava.data.data import make_supervised_data_module
from mllm.llava.config import load_generate_config

from mllm.llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX


def generate():
    model_args, data_args, training_args, output_dir = load_generate_config()
    logging.basicConfig(
        filename=os.path.join(output_dir, "generate.log"), level=logging.DEBUG
    )

    config = AutoConfig.from_pretrained(
        os.path.join(model_args.pretrain_checkpoint_path, "base_config_dir")
    )

    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.transformers_cache_dir,
        config=config,
        attn_implementation="flash_attention_2",
        torch_dtype=(torch.bfloat16 if training_args.bf16 else torch.float16),
    )

    model.get_model().initialize_vision_modules(
        model_args=model_args,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    model = PeftModel.from_pretrained(
        model,
        os.path.join(model_args.pretrain_checkpoint_path, "lora_adapters_dir"),
    )

    mm_projector_weights = torch.load(
        os.path.join(model_args.pretrain_checkpoint_path, "mm_projector.pth"),
        map_location="cpu",
    )
    model.get_model().mm_projector.load_state_dict(mm_projector_weights)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.transformers_cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    model.requires_grad_(False)
    model.to(
        device=training_args.device,
        dtype=(torch.bfloat16 if training_args.bf16 else torch.float16),
    )
    model.eval()

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataloader = torch.utils.data.DataLoader(
        data_module["eval_dataset"],
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

    for batch in tqdm(dataloader):

        map_id = batch.pop("map_ids")[0]
        input_ids = batch.pop("input_ids").to(training_args.device)
        assert input_ids.shape[0] == 1, "Needs batch size 1"

        if data_args.cache_embed:
            images = None
            image_features = [x for x in batch.pop("image_features")]
        else:
            images = [
                x.to(training_args.device, dtype=torch.bfloat16)
                for x in batch.pop("images")
            ]
            image_features = None

        labels = batch.pop("labels").to(training_args.device)
        input_ids = input_ids[labels == IGNORE_INDEX].unsqueeze(0)

        with torch.inference_mode():
            output = model.generate(
                input_ids,
                images=images,
                image_features=image_features,
                max_length=training_args.model_max_length,
                pad_token_id=tokenizer.pad_token_id,
                # num_beams=5,
                do_sample=False,
                temperature=None,  # 0.0,
                top_p=None,  # 0.3,
                use_cache=True,
            )[0]

        output = tokenizer.decode(output, skip_special_tokens=True)
        with open(os.path.join(outdir, f"{map_id}.txt"), "w") as f:
            f.write(output)


if __name__ == "__main__":
    try:
        generate()
    except Exception as e:
        logging.exception(e)
