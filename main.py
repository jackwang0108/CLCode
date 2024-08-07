# Standard Library
import os
import sys
import uuid
import socket
import datetime
import importlib
import urllib.request
from argparse import ArgumentParser
from typing import Callable, Optional

# Third-Party Library
import numpy  # needed (don't change it)
import setproctitle

# Torch Library
import torch

# My Library
from models import get_model
from models import get_all_models
from datasets import get_dataset
from datasets import ContinualDataset
from datasets import NAMES as DATASET_NAMES
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.args import add_management_args
from utils.continual_training import train as ctrain
from utils.types import Args, CVBackboneImpl


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
mammoth_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(mammoth_path)
sys.path.append(mammoth_path + '/datasets')
sys.path.append(mammoth_path + '/backbone')
sys.path.append(mammoth_path + '/models')


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description='CL-Codes', allow_abbrev=False)

    # main args
    parser.add_argument('--model', type=str, required=True,
                        help='Model to train', choices=get_all_models())
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='Backbone to use')
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, dataset and memory buffer.')

    # management args
    add_management_args(parser)

    args: Args = parser.parse_known_args()[0]

    torch.set_num_threads(4)

    # load model args
    parser: ArgumentParser
    get_parser: Callable[..., ArgumentParser]
    mod = importlib.import_module('models.' + args.model)
    if args.load_best_args:
        # load from founded best args
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')

        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')

        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if hasattr(mod, 'Buffer'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        to_parse = sys.argv[1:] + ['--' + k +
                                   '=' + str(v) for k, v in best.items()]
        to_parse.remove('--load_best_args')
        args = parser.parse_args(to_parse)
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
    else:

        # load from command line
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        args.seed = int(args.seed)
        set_random_seed(args.seed)

    return args


def main(args: Optional[Args] = None):
    lecun_fix()

    # parse args
    if args is None:
        args = parse_args()

    # if using meta-learning
    if 'meta' in args.backbone:
        from backbone.meta_layers import get_gpu_id
        get_gpu_id(args.gpu_id)

    # add uuid, timestamp and hostname for logging
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()

    # setup dataset
    dataset = get_dataset(args)

    # setup training config
    if args.n_epochs is None and isinstance(dataset, ContinualDataset):
        args.n_epochs = dataset.get_epochs()
    if args.batch_size is None:
        args.batch_size = dataset.get_batch_size()
    if hasattr(importlib.import_module('models.' + args.model), 'Buffer') and not hasattr(args, 'minibatch_size'):
        args.minibatch_size = dataset.get_minibatch_size()

    # get backbone
    backbone: CVBackboneImpl = dataset.get_backbone(args.backbone)

    # get loss
    loss: Callable = dataset.get_loss()

    # get model
    model = get_model(args, backbone, loss, dataset.get_transform())

    # set job name
    setproctitle.setproctitle('{}_{}_{}'.format(
        args.model, args.buffer_size if 'buffer_size' in args else 0, args.dataset))

    # train
    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


if __name__ == '__main__':
    main()
