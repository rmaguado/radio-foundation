import os
import math
import torch
import logging
from typing import Tuple
import torch.distributed as dist


logger = logging.getLogger("dinov2")
