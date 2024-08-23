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
        self.meters = defaultdict(RunningAverage)
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
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.avg))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def dump_in_output_file(self, iteration, iter_time, data_time):
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.avg for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def log_every(self, print_freq, header=None, n_iterations=None, start_iteration=0):
        iterable = self.dataloader
        i = start_iteration
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = RunningAverage()
        data_time = RunningAverage()

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = ":" + str(len(str(n_iterations))) + "d"

        log_list = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time:.6f}",
            "data: {data:.6f}",
        ]
        if torch.cuda.is_available():
            log_list += ["max mem: {memory:.0f}"]

        log_msg = self.delimiter.join(log_list)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == n_iterations - 1:
                self.dump_in_output_file(
                    iteration=i, iter_time=iter_time.avg, data_time=data_time.avg
                )
                eta_seconds = iter_time.avg * (n_iterations - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    logger.info(
                        log_msg.format(
                            i,
                            n_iterations,
                            eta=eta_string,
                            meters=str(self),
                            time=iter_time.avg,
                            data=data_time.avg,
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
                            time=iter_time.avg,
                            data=data_time.avg,
                        )
                    )
            i += 1
            end = time.time()
            if i >= n_iterations:
                break
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / n_iterations
            )
        )


class RunningAverage:
    """Track a running average of a series of values."""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, num=1):
        self.count += num
        self.total += value * num

    @property
    def avg(self):
        return self.total / self.count if self.count > 0 else 0.0

    def synchronize_between_processes(self):
        """
        Distributed synchronization of the metric
        """
        if not distributed.is_enabled():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
