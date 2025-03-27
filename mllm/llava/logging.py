import functools
import logging
import os
import sys
from typing import Optional
import torch.distributed as distributed


def get_global_rank():
    is_enabled = distributed.is_available() and distributed.is_initialized()
    return distributed.get_rank() if is_enabled else 0


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
) -> None:
    """
    Setup logging.

    Args:
        output: A file name or a directory to save log files. If None, log
            files will not be saved. If output ends with ".txt" or ".log", it
            is assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
        capture_warnings: Whether warnings should be captured as logs.
    """
    logging.captureWarnings(capture_warnings)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt_prefix = (
        "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    )
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    if get_global_rank() == 0:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        if not get_global_rank() == 0:
            global_rank = get_global_rank()
            filename = filename + ".rank{}".format(global_rank)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a", encoding="utf-8"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
