# Standard Library
import os
import random
from pathlib import Path
from typing import Optional

# Third-Party Library
import torch

# Torch Library
import numpy as np

ROOT_DIR: Path = Path(__file__).resolve().parent.parent


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    get_device returns the GPU device if available else CPU.

    Args:
        gpu_id (int, optional): gpu id. Defaults to None.

    Returns:
        torch.device of selected gpu
    """
    return torch.device(f"cuda:{0 if gpu_id is None else gpu_id}" if torch.cuda.is_available() else "cpu")


def data_path() -> Path:
    """
    data_path returns the path where datasets are stored.

    Returns:
        Path: path to datasets
    """
    return ROOT_DIR / "data"


def base_path() -> Path:
    """
    base_path returns the base bath where to log accuracies and tensorboard data.

    Returns:
        Path: path to save the logs
    """
    return ROOT_DIR / "log"


def set_random_seed(seed: int) -> None:
    """
    set_random_seed sets the random seeds to a certain value.

    Args:
        seed (int): the value of the random seed
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
