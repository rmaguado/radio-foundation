import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


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


def setup_distributed_slurm():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    master_addr = os.environ["SLURM_SRUN_COMM_HOST"]
    master_port = os.environ.get("MASTER_PORT", "29500")

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )


def cleanup_distributed():
    dist.destroy_process_group()
