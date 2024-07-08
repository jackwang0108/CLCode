# Standard Library
import importlib
from pathlib import Path
from typing import Callable

# My Library
from .utils.continual_model import ContinualModel
from utils.types import CVBackboneImpl, Args


model_dir = Path(__file__).resolve().parent


def get_all_models() -> list[str]:
    return [f.stem for f in model_dir.iterdir() if f.is_file() and not '__' in f.name and f.suffix == '.py']


models: dict[str, ContinualModel] = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    models[model] = getattr(mod, class_name)


def get_model(args: Args, backbone: CVBackboneImpl, loss: Callable, transform) -> ContinualModel:
    return models[args.model](backbone, loss, args, transform)
