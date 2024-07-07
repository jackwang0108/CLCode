# Standard Library
import inspect
import importlib
from pathlib import Path
from typing import get_args

# My Library
from utils.types import Args
from datasets.utils.gcl_dataset import GCLDataset
from datasets.utils.continual_dataset import ContinualDataset

dataset_dir = Path(__file__).resolve().parent


def get_all_datasets() -> list[str]:
    return [f.stem for f in dataset_dir.iterdir() if f.is_file() and not '__' in f.name and f.suffix == '.py']


NAMES: dict[str, GCLDataset | ContinualDataset] = {}
for dataset_file in dataset_dir.glob("*.py"):

    filename: str = dataset_file.stem
    mod = importlib.import_module(f"datasets.{filename}")

    for cls_name, obj in inspect.getmembers(mod):
        if not inspect.isclass(obj):
            continue

        if (issubclass(obj, ContinualDataset) or issubclass(obj, GCLDataset)) and obj.NAME is not None:
            NAMES[obj.NAME] = obj


def get_dataset(args: Args) -> ContinualDataset:
    """
    get_dataset creates and returns a continual dataset.

    Args:
        args (Args): command line argument

    Returns:
        ContinualDataset: the continual dataset
    """
    assert args.dataset in NAMES.keys(
    ), f"Invalid dataset, should be in {list(NAMES.keys())}"
    return NAMES[args.dataset](args)
