"""
Utility functions for distributed training.
"""

import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group


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
