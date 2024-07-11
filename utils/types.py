# Standard Library
from typing import Literal
from typing import Callable
from argparse import Namespace

# Torch Library
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

# My Library
from backbone.ResNet import ResNet
from backbone.ResNet_meta import MetaResNet

CLModel = Literal[
    'joint', 'bic', 'lwf', 'ewc_on', 'hal',
    'gss', 'joint_gcl', 'gem', 'agem', 'agem_r',
    'icarl', 'lucir', 'pnn', 'rpc', 'sgd', 'fdr',
    'gdumb', 'derpp_cba_online', 'derpp_cba_offline',
    'clser_cba_online', 'clser_cba_offline', 'er_cba_online',
    'er_cba_offline' 'si', 'ssil', 'er', 'er_ace', 'clser',
    'mer', 'der', 'xder', 'derpp',
]

CVBackbone = Literal[
    "resnet18", "resnet34", "resnet50", "resnet50", "resnet152",
    "resnet18-meta", "resnet34-meta", "resnet50-meta", "resnet101-meta", "resnet152-meta"
]

CVBackboneImpl = ResNet | MetaResNet

CVTransforms = Compose

CVDataset = Literal[
    'seq-tinyimg', 'perm-mnist',
    'rot-mnist', 'seq-cifar10-blurry',
    'seq-cifar100', 'seq-mnist', 'mnist-360',
    'seq-cifar10'
]

CLDatasetImpl = MNIST | CIFAR10 | CIFAR100


class Args(Namespace):
    """ Args for type hints """

    # Main args
    load_best_args: bool

    conf_host: str
    conf_jobnum: str
    conf_timestamp: str

    # Experiment Args
    model: CLModel
    dataset: CVDataset
    backbone: CVBackbone

    lr: float
    n_epochs: int
    batch_size: int

    optim_wd: float
    optim_mom: float
    optim_nesterov: float

    # Rehearsal Args
    buffer_size: int
    minibatch_size: int

    # Management Args
    exp: str                    # Experiment name
    seed: int
    notes: str
    gpu_id: int
    csv_log: bool
    validation: bool
    save_model: bool
    tensorboard: bool
    non_verbose: bool

    # Model Args

    # 1. Lucir Args
    lamda_base: float
    lamda_mr: float
    k_mr: float
    mr_margin: float
    fitting_epochs: int
    lr_finetune: float
    imprint_weights: int


class ContinualModelImpl(nn.Module):
    """ ContinualModelImpl is the type hints class for ContinualModel """

    NAME: str
    TITLE: str
    ARXIV: str

    COMPATIBILITY: list[Literal["task-il", "class-il"]]

    args: Args
    loss: Callable
    transform: Compose
    net: CVBackboneImpl

    opt: Optimizer

    device: torch.device

    num_cls: int
    total_classes: int
    n_cls_per_task: int

    current_task: int
    seen_classes: int

    epoch: int
    batch_idx: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward computes a forward pass.

        Args:
            x (torch.Tensor): batch of inputs

        Returns:
            torch.Tensor: the result of the computation
        """
        raise NotImplementedError

    def observe(
            self, inputs: torch.Tensor, labels: torch.Tensor, not_aug_inputs: torch.Tensor) -> torch.Tensor:
        """
        observe computes a training step over a given batch of examples.

        Args:
            inputs (torch.Tensor): batch of examples
            labels (torch.Tensor): ground-truth labels
            not_aug_inputs (torch.Tensor): some methods could require additional parameters

        Returns:
            torch.Tensor: the value of the loss function
        """
        raise NotImplementedError

    def begin_task(self, dataset):
        raise NotImplementedError

    def end_task(self, dataset) -> None:
        raise NotImplementedError
