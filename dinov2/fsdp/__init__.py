# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Any

import torch
from functools import partial
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp._runtime_utils import _reshard

import dinov2.distributed as distributed


def get_fsdp_wrapper(model_cfg, modules_to_wrap=set()):
    sharding_strategy_dict = {
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    }

    dtype_dict = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }

    mixed_precision_config = MixedPrecision(
        param_dtype=dtype_dict[model_cfg.mixed_precision.param_dtype],
        reduce_dtype=dtype_dict[model_cfg.mixed_precision.reduce_dtype],
        buffer_dtype=dtype_dict[model_cfg.mixed_precision.buffer_dtype],
    )

    sharding_strategy_config = sharding_strategy_dict[model_cfg.sharding_strategy]

    local_rank = distributed.get_local_rank()

    fsdp_wrapper = partial(
        FSDP,
        sharding_strategy=sharding_strategy_config,
        mixed_precision=mixed_precision_config,
        device_id=local_rank,
        sync_module_states=True,
        use_orig_params=True,
        auto_wrap_policy=ModuleWrapPolicy(modules_to_wrap),
    )
    return fsdp_wrapper


def is_fsdp(x):
    return isinstance(x, FSDP)


def is_sharded_fsdp(x):
    return is_fsdp(x) and x.sharding_strategy is not ShardingStrategy.NO_SHARD


def free_if_fsdp(x):
    if is_sharded_fsdp(x):
        handle = x._handle
        _reshard(x, handle, True)


def get_fsdp_modules(x):
    return FSDP.fsdp_modules(x)


def reshard_fsdp_model(x):
    for m in get_fsdp_modules(x):
        free_if_fsdp(m)


def rankstr():
    return f"rank_{distributed.get_global_rank()}"


class FSDPCheckpointer(Checkpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            data["model"] = self.model.state_dict()

        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.{rankstr()}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(self, *args, **kwargs):
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        # pyre-fixme[6]: For 2nd param expected `Union[PathLike[str], str]` but got
        #  `Union[bytes, str]`.
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint.

        Args:
            last_filename_basename (str): the basename of the last filename.
        """
        if distributed.is_enabled():
            torch.distributed.barrier()
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


class FlexibleFSDPCheckpointer(Checkpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file.

        Args:
            name (str): name of the file.
            kwargs (dict): extra arbitrary data to save.
        """
        if not self.save_dir or not self.save_to_disk:
            return

        self.logger.debug("Saving checkpoint to {} ...".format(self.save_dir))

        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            data["model"] = self.model.state_dict()

        self.logger.debug("Gathered full state dict for model")

        for key, obj in self.checkpointables.items():
            if key == "optimizer":
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                    optim_state_dict = FSDP.optim_state_dict(self.model, obj)
                    data[key] = optim_state_dict

                    self.logger.debug("Gathered full state dict for optimizer")
            else:
                data[key] = obj.state_dict()
        data.update(kwargs)

        if distributed.get_global_rank() != 0:
            self.logger.debug("Waiting for rank 0 to save checkpoint")
            torch.distributed.barrier()
            self.logger.debug("Rank 0 saved checkpoint. Resuming ...")
            return

        basename = f"{name}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

        self.logger.debug("Checkpoint saved")

        torch.distributed.barrier()

    def load(self, path, checkpointables=None):
        if not path:
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("[Checkpointer] Loading from {} ...".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model(checkpoint)
        if incompatible is not None:
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                if key == "optimizer":
                    state_dict = checkpoint.pop("optimizer")
                    self._load_optim(obj, state_dict)
                else:
                    obj.load_state_dict(checkpoint.pop(key))

        return checkpoint

    def _load_optim(self, optim, state_dict):
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            optim_state_dict = FSDP.optim_state_dict_to_load(
                self.model, optim, state_dict
            )
            optim.load_state_dict(optim_state_dict)

    def _load_file(self, f):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = torch.load(
                f, map_location=torch.device("cpu"), weights_only=False
            )
        return state_dict


class AntiFSDPConverter(Checkpointer):
    """
    Used to convert sharded checkpoints to non-sharded checkpoints.
    """

    def save(self, name: str, **kwargs: Any) -> None:
        if not self.save_dir or not self.save_to_disk:
            return

        self.logger.debug("Saving checkpoint to {} ...".format(self.save_dir))

        data = {}
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            data["model"] = self.model.state_dict()

        self.logger.debug("Gathered full state dict for model")

        for key, obj in self.checkpointables.items():
            if key == "optimizer":
                with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                    optim_state_dict = FSDP.optim_state_dict(self.model, obj)
                    data[key] = optim_state_dict

                    self.logger.debug("Gathered full state dict for optimizer")
            else:
                data[key] = obj.state_dict()
        data.update(kwargs)

        if distributed.get_global_rank() != 0:
            self.logger.debug("Waiting for rank 0 to save checkpoint")
            torch.distributed.barrier()
            self.logger.debug("Rank 0 saved checkpoint. Resuming ...")
            return

        basename = f"{name}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        self.logger.info("Saving checkpoint to {}".format(save_file))
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

        self.logger.debug("Checkpoint saved")

        torch.distributed.barrier()

    def load(self, *args, **kwargs):
        with FSDP.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
            return super().load(*args, **kwargs)

    def _load_file(self, f):
        return torch.load(f, map_location=torch.device("cpu"), weights_only=False)

    def has_checkpoint(self) -> bool:
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        save_file = os.path.join(self.save_dir, f"last_checkpoint.{rankstr()}")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            return ""
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        save_file = os.path.join(self.save_dir, f"last_checkpoint")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)  # pyre-ignore


class FlexiblePeriodicCheckpointer(PeriodicCheckpointer):
    def step(self, iteration: int, **kwargs: Any) -> None:
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:
            self.checkpointer.save(
                "{}_{:07d}".format(self.file_prefix, iteration), **additional_state
            )

            if self.max_to_keep is not None:
                self.recent_checkpoints.append(self.checkpointer.get_checkpoint_file())
                if len(self.recent_checkpoints) > self.max_to_keep:
                    file_to_delete = self.recent_checkpoints.pop(0)
                    if self.path_manager.exists(
                        file_to_delete
                    ) and not file_to_delete.endswith(f"{self.file_prefix}_final.pth"):
                        try:
                            self.path_manager.rm(file_to_delete)
                        except FileNotFoundError:
                            pass

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                self.checkpointer.save(f"{self.file_prefix}_final", **additional_state)


ShardedGradScaler = ShardedGradScaler
