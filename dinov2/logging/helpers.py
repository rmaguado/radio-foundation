from collections import defaultdict, deque
import datetime
import json
import logging
import time
from typing import Optional

import torch

import dinov2.distributed as distributed

logger = logging.getLogger("dinov2")


class Metric:
    """Track a series of values and provide the recent average."""

    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)

    def update(self, value):
        self.deque.append(value)

    def avg(self):
        return sum(self.deque) / len(self.deque) if self.deque else 0.0


class MetricLogger(object):
    def __init__(
        self,
        delimiter: str = "    ",
        output_file: Optional[str] = None,
        window_size: int = 20,
    ):
        """
        A general-purpose metric logger that can be used to log the average of a metric over a window of iterations.

        Args:
            delimiter (str): Delimiter to use when formatting the log message.
            output_file (str, optional): Path to the output file to write the logs to.
            window_size (int, optional): The window size to use when computing the average of the metric.
        """
        self.meters = defaultdict(lambda: Metric(window_size))
        self.delimiter = delimiter
        self.output_file = output_file
        self.dataloader = None

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def update(self, **kwargs: dict) -> None:
        """
        Update the values of the meters.

        Args:
            **kwargs: Key-value pairs where the key is the name of the metric and the value is the value to update the metric with.
        """
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

    def __str__(self) -> str:
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f}".format(name, meter.avg()))
        return self.delimiter.join(loss_str)

    def add_meter(self, name: str, meter: Metric) -> None:
        self.meters[name] = meter

    def dump_in_output_file(self, iteration: int, iter_time: float, data_time: float):
        """
        Dump the metrics to the output file.

        Args:
            iteration (int): The current iteration.
            iter_time (float): The average iteration time.
            data_time (float): The average data loading time.
        """
        if self.output_file is None or not distributed.is_main_process():
            return
        dict_to_dump = dict(
            iteration=iteration,
            iter_time=iter_time,
            data_time=data_time,
        )
        dict_to_dump.update({k: v.avg() for k, v in self.meters.items()})
        with open(self.output_file, "a") as f:
            f.write(json.dumps(dict_to_dump) + "\n")

    def build_log_message(self, header: str, space_fmt: str) -> str:
        """
        Build the log message to be printed.

        Args:
            header (str): The header to be printed.
            space_fmt (str): The format to use for the spacing.
        """
        log_list = [
            header,
            f"[{{0{space_fmt}}}/{{1}}]",
            "eta: {eta}",
            "{meters}",
            "time: {time:.4f}",
            "data: {data:.4f}",
        ]
        if torch.cuda.is_available():
            log_list.append("max mem: {memory:.0f}")

        return self.delimiter.join(log_list)

    def log_iteration(
        self,
        i: int,
        n_iterations: int,
        iter_time: Metric,
        data_time: Metric,
        log_msg: str,
        MB: float,
    ):
        """
        Log the metrics for the current iteration.

        Args:
            i (int): The current iteration.
            n_iterations (int): The total number of iterations.
            iter_time (Metric): The average iteration time.
            data_time (Metric): The average data loading time.
            log_msg (str): The log message to be printed.
            MB (float): The number of MBs.
        """
        self.dump_in_output_file(
            iteration=i, iter_time=iter_time.avg(), data_time=data_time.avg()
        )

        eta_seconds = iter_time.avg() * (n_iterations - i)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        log_params = {
            "eta": eta_string,
            "meters": str(self),
            "time": iter_time.avg(),
            "data": data_time.avg(),
        }

        if torch.cuda.is_available():
            log_params["memory"] = torch.cuda.max_memory_allocated() / MB

        logger.info(log_msg.format(i, n_iterations, **log_params))

    def log_total_time(self, header: str, start_time: float, n_iterations: int) -> None:
        """
        Log the total time taken for the training.

        Args:
            header (str): The header to be printed.
            start_time (float): The start time of the training.
            n_iterations (int): The total number of iterations.
        """
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(
            f"{header} Total time: {total_time_str} ({total_time / n_iterations:.6f} s / it)"
        )

    def log_every(
        self,
        print_freq: int,
        header: Optional[str] = None,
        n_iterations: Optional[int] = None,
        start_iteration: int = 0,
    ):
        """
        Log the metrics every `print_freq` iterations.

        Args:
            print_freq (int): The frequency of iterations at which to print the logs.
            header (str, optional): The header to be printed.
            n_iterations (int, optional): The total number of iterations.
            start_iteration (int, optional): The starting iteration.
        """
        iterable = self.dataloader
        i = start_iteration
        header = header or ""

        start_time = time.time()
        end = time.time()
        iter_time = Metric()
        data_time = Metric()

        if n_iterations is None:
            n_iterations = len(iterable)

        space_fmt = f":{len(str(n_iterations))}d"
        log_msg = self.build_log_message(header, space_fmt)

        MB = 1024.0 * 1024.0

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == n_iterations - 1:
                self.log_iteration(i, n_iterations, iter_time, data_time, log_msg, MB)

            end = time.time()
            if i >= n_iterations:
                break

        self.log_total_time(header, start_time, n_iterations)
