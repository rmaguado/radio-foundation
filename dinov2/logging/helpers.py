# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from collections import defaultdict
import datetime
import json
import logging
import time

import torch

import dinov2.distributed as distributed


logger = logging.getLogger("dinov2")


class MetricLogger(object):
    def __init__(self, delimiter="\t", output_file=None):
        self.meters = defaultdict(float)
        self.counts = defaultdict(int)
        self.delimiter = delimiter
        self.output_file = output_file
        self.dataloader = None

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k] += v
            self.counts[k] += 1

    def synchronize_between_processes(self):
        """Synchronize metrics between processes in distributed training."""
        if not distributed.is_enabled():
            return
        for meter_name, meter_value in self.meters.items():
            t = torch.tensor(
                [meter_value, self.counts[meter_name]],
                dtype=torch.float64,
                device="cuda",
            )
            torch.distributed.barrier()
            torch.distributed.all_reduce(t)
            self.meters[meter_name], self.counts[meter_name] = t.tolist()

    def __str__(self):
        return self.delimiter.join(
            f"{name}: {value / self.counts[name]:.4f}"
            for name, value in self.meters.items()
        )

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v / self.counts[k] for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(self, print_freq, header=None, n_iterations=None, start_iteration=0):
        iterable = self.dataloader
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time:.4f}",
            "data: {data:.4f}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time = time.time() - end
            yield obj
            iter_time = time.time() - end
            if i % print_freq == 0 or i == n_iterations - 1:
                self.synchronize_between_processes()  # Ensure metrics are synchronized
                self.dump_in_output_file(
                    iteration=i, iter_time=iter_time, data_time=data_time
                )
                eta_seconds = iter_time * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=iter_time,
                            data=data_time,
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=iter_time,
                            data=data_time,
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            f"{header} Total time: {total_time_str} ({total_time / n_iterations:.4f} s / it)"
        )
