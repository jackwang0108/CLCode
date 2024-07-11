# Standard Library
from typing import Callable, Literal

# Torch Library
import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD
from torchvision.transforms import Compose

# My Library
from utils.conf import get_device
from utils.types import Args, CVBackboneImpl
from datasets.utils.continual_dataset import ContinualDataset


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """

    NAME: str = None
    TITLE: str = None
    ARXIV: str = None

    COMPATIBILITY: list[Literal["task-il", "class-il"]] = []

    epoch: int = 0              # some model need epoch
    batch_idx: int = 0          # some model need batch_idx

    def __init__(
            self, backbone: CVBackboneImpl, loss: nn.Module, args: Args, transform: Compose) -> None:
        super(ContinualModel, self).__init__()

        self.args: Args = args
        self.loss: Callable = loss
        self.transform: Compose = transform
        self.net: CVBackboneImpl = backbone

        try:
            self.opt: Optimizer = SGD(self.net.parameters(), lr=self.args.lr)
        except:
            self.opt: Optimizer = SGD(self.net.params(), lr=self.args.lr)

        self.device: torch.device = get_device(args.gpu_id)

        self.num_cls: int = self.net.fc.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward computes a forward pass.

        Args:
            x (torch.Tensor): batch of inputs

        Returns:
            torch.Tensor: the result of the computation
        """
        return self.net(x)

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

    def begin_task(self, dataset: ContinualDataset) -> None:
        """
        begin_task prepare the model before learning on next task

        Args:
            dataset (ContinualDataset): continual learning dataset
        """
        raise NotImplementedError

    def end_task(self, dataset: ContinualDataset) -> None:
        """
        end_task setup the model after learning on current task

        Args:
            dataset (ContinualDataset): continual learning dataset
        """
        pass
