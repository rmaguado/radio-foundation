# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict, List, Optional

import torch
from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer

import torch.distributed as dist


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


class DDPCheckpointer(Checkpointer):
    def save(self, name: str, **kwargs: Any) -> None:
        """
        Dump model and checkpointables to a file. Only on rank 0.
        """
        if not self.save_dir or not self.save_to_disk or not is_main_process():
            return

        data = {}
        data["model"] = self.model.state_dict()

        for key, obj in self.checkpointables.items():
            data[key] = obj.state_dict()
        data.update(kwargs)

        basename = f"{name}.pth"
        save_file = os.path.join(self.save_dir, basename)
        assert os.path.basename(save_file) == basename, basename
        if hasattr(self, "logger"):
            self.logger.info(f"Saving checkpoint to {save_file}")
        with self.path_manager.open(save_file, "wb") as f:
            torch.save(data, f)
        self.tag_last_checkpoint(basename)

    def load(
        self, path: str, checkpointables: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        if not path:
            return {}

        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        incompatible = self._load_model({"model": checkpoint["model"]})
        if incompatible is not None:
            self._log_incompatible_keys(incompatible)
        self.model.load_state_dict(checkpoint.pop("model"))

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        return checkpoint

    def resume(self) -> Dict[str, Any]:
        if self.has_checkpoint():
            path = self.get_checkpoint_file()
            return self.load(path)
        else:
            return {}

    def has_checkpoint(self) -> bool:
        """
        Returns:
            bool: whether a checkpoint exists in the target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        return self.path_manager.exists(save_file)

    def get_checkpoint_file(self) -> str:
        """
        Returns:
            str: The latest checkpoint file in target directory.
        """
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        try:
            with self.path_manager.open(save_file, "r") as f:
                last_saved = f.read().strip()
        except IOError:
            # if file doesn't exist, maybe because it has just been
            # deleted by a separate process
            return ""
        return os.path.join(self.save_dir, last_saved)

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        """
        Tag the last checkpoint. Only on rank 0.
        """
        if not is_main_process():
            return
        save_file = os.path.join(self.save_dir, "last_checkpoint")
        with self.path_manager.open(save_file, "w") as f:
            f.write(last_filename_basename)

    def _load_file(self, f: str) -> Dict[str, Any]:
        return torch.load(f, map_location=torch.device("cpu"), weights_only=False)


class DDPPeriodicCheckpointer(PeriodicCheckpointer):
    def step(self, iteration: int, **kwargs: Any) -> None:
        iteration = int(iteration)
        additional_state = {"iteration": iteration}
        additional_state.update(kwargs)

        if (iteration + 1) % self.period == 0:
            if is_main_process():
                self.checkpointer.save(
                    "{}_{:07d}".format(self.file_prefix, iteration),
                    **additional_state,
                )

                if self.max_to_keep is not None:
                    self.recent_checkpoints.append(
                        self.checkpointer.get_checkpoint_file()
                    )
                    if len(self.recent_checkpoints) > self.max_to_keep:
                        file_to_delete = self.recent_checkpoints.pop(0)
                        if self.path_manager.exists(
                            file_to_delete
                        ) and not file_to_delete.endswith(
                            f"{self.file_prefix}_final.pth"
                        ):
                            try:
                                self.path_manager.rm(file_to_delete)
                            except FileNotFoundError:
                                pass

        if self.max_iter is not None:
            if iteration >= self.max_iter - 1:
                if is_main_process():
                    self.checkpointer.save(
                        f"{self.file_prefix}_final", **additional_state
                    )
