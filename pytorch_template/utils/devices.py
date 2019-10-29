import logging
import random

import gin
import numpy as np
import torch

from pytorch_template.utils.misc import print_cuda_statistics

logger = logging.getLogger()


@gin.configurable
def configure_device(gpu_id=None, seed=None):
    """Wrapper function to call _set_device and _set_random_seed functions"""

    cuda_available = torch.cuda.is_available()
    if not cuda_available and gpu_id is not None:
        logger.info(f"WARNING: You specified gpu_id={gpu_id} but no CUDA device found.")
        gpu_id = None

    device = _set_device(gpu_id)
    _set_random_seed(seed)

    return device


def _set_device(gpu_id):
    if gpu_id is not None:
        device = torch.device(f'cuda:{gpu_id}')
        print_cuda_statistics()
    else:
        device = torch.device("cpu")

    logger.info(f"Program will run on *****{device}*****\n")

    return device


def _set_random_seed(seed=None):
    """
    See https://pytorch.org/docs/stable/notes/randomness.html
    and https://stackoverflow.com/questions/55097671/how-to-save-and-load-random-number-generator-state-in-pytorch
    """

    if seed is not None:
        logger.info(f'Setting manual seed = {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # even when setting the random seed cuda devices can behave nondeterministically
        # setting those flags reduces this nondeterminism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
