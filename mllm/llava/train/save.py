import deepspeed
import torch
import os

from peft import PeftModel


def save_model(training_args, model, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with deepspeed.zero.GatheredParameters(list(model.parameters())):

        if deepspeed.comm.get_rank() == 0:

            if hasattr(model, "save_pretrained"):
                lora_dir = os.path.join(output_dir, "lora_adapters_dir")
                os.makedirs(lora_dir, exist_ok=True)

                model.save_pretrained(lora_dir)

            torch.save(
                model.get_model().mm_projector.state_dict(),
                os.path.join(output_dir, "mm_projector.pth"),
            )

            config_dir = os.path.join(output_dir, "base_config_dir")
            os.makedirs(config_dir, exist_ok=True)
            model.config.save_pretrained(config_dir)
