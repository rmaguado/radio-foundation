import torch

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel.distributed import _MixedPrecision as MixedPrecision


def is_enabled():
    return dist.is_initialized()


def get_world_size():
    if not is_enabled():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_enabled():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def all_reduce(tensor, **kwargs) -> None:
    if not is_enabled():
        return
    dist.barrier()
    dist.all_reduce(tensor, **kwargs)


def barrier() -> None:
    if not is_enabled():
        return
    dist.barrier()


def _get_dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.half
    elif dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def wrap_module_with_mixed_precision(module, mixed_precision_cfg, rank):
    mixed_precision = MixedPrecision(
        param_dtype=_get_dtype_from_str(mixed_precision_cfg.param_dtype),
        reduce_dtype=_get_dtype_from_str(mixed_precision_cfg.reduce_dtype),
        buffer_dtype=_get_dtype_from_str(mixed_precision_cfg.buffer_dtype),
    )
    return DDP(module, mixed_precision=mixed_precision, device_ids=[rank])
