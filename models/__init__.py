# Standard Library
import importlib
from pathlib import Path

# My Library
from utils.types import Args


def get_all_models() -> list[str]:
    model_dir = Path(__file__).resolve().parent
    return [f.stem for f in model_dir.iterdir() if f.is_file() and not '__' in f.name and f.suffix == '.py']


names = {}
for model in get_all_models():
    mod = importlib.import_module('models.' + model)
    class_name = {x.lower(): x for x in mod.__dir__()}[model.replace('_', '')]
    names[model] = getattr(mod, class_name)


def get_model(args: Args, backbone, loss, transform):
    return names[args.model](backbone, loss, args, transform)
