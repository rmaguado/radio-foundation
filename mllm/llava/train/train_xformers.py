import logging

from mllm.llava.train.llama_xformers_attn_monkey_patch import (
    replace_llama_attn_with_xformers_attn,
)

replace_llama_attn_with_xformers_attn()

from mllm.llava.train.train import train

logger = logging.getLogger("mllm")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.exception(e)
