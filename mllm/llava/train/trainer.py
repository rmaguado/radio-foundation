import os
import torch
import logging
import numpy as np
import math
import random

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
)
from typing import List, Optional

from mllm.llava.train.save import save_model


logger = logging.getLogger("DeepSpeed")


class LongestFirstSampler(Sampler):
    def __init__(self, lengths, **kwargs):
        self.lengths = lengths

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        indices = torch.argsort(torch.tensor(self.lengths), descending=True).tolist()
        return iter(indices)


class MultimodalBalanceLengthSampler(Sampler):
    def __init__(
        self,
        img_lengths,
        text_lengths,
        balance_text=True,
        balance_images=True,
        bucket_size=128,
    ):
        self.img_lengths = img_lengths
        self.text_lengths = text_lengths
        self.bucket_size = bucket_size

        if balance_text:
            text_ranks = np.argsort(np.argsort(text_lengths))
            text_ranks = len(text_lengths) - text_ranks
        else:
            text_ranks = np.zeros(len(text_lengths))

        if balance_images:
            img_ranks = np.argsort(np.argsort(img_lengths))
            img_ranks = len(img_lengths) - img_ranks
        else:
            img_ranks = np.zeros(len(img_lengths))

        self.weights = (img_ranks + text_ranks).tolist()

    def __len__(self):
        return len(self.img_lengths)

    def __iter__(self):
        sorted_indices = sorted(
            range(len(self.weights)), key=lambda i: self.weights[i], reverse=True
        )

        num_buckets = math.ceil(len(sorted_indices) / self.bucket_size)
        buckets = [[] for _ in range(num_buckets)]
        for i, idx in enumerate(sorted_indices):
            buckets[i % num_buckets].append(idx)

        for bucket in buckets:
            random.shuffle(bucket)
        random.shuffle(buckets)

        indices = []
        for bucket in buckets:
            indices += bucket

        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # return super()._get_train_sampler()
        # return LongestFirstSampler(self.train_dataset.dataset.get_lengths())
        return MultimodalBalanceLengthSampler(
            img_lengths=self.train_dataset.dataset.get_slices(),
            text_lengths=self.train_dataset.dataset.get_lengths(),
            bucket_size=256,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [
                    name
                    for name, _ in opt_model.named_parameters()
                    if "mm_projector" in name
                ]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n not in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (
                                n not in decay_parameters
                                and n in projector_parameters
                                and p.requires_grad
                            )
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        save_model(self.args, self.model, output_dir)

        # self.state.stateful_callbacks["TrainerControl"] = self.control.state()
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        save_model(self.args, self.model, output_dir)
