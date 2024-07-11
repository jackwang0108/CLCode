# Standard Library
from abc import abstractmethod
from argparse import Namespace
from typing import Tuple, Callable

# Third-Party Library
import numpy as np

# Torch Library
import torch.optim
from torch import nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import VisionDataset

# My Library
from utils.types import Args, CVBackboneImpl, CLDatasetImpl


class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Args) -> None:
        """
        initializes the train and test lists of dataloaders.

        Args:
            args (Args): command line arguments
        """
        self.i: int = 0
        self.args: Args = args
        self.train_loader: DataLoader = None
        self.test_loaders: list[DataLoader] = []

    @abstractmethod
    def get_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """
        get_data_loaders creates and returns the training and test loaders for the current task.

        The current training loader and all test loaders are stored in self.

        Returns:
            tuple[DataLoader, DataLoader]: the current training and test loaders
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_backbone() -> CVBackboneImpl:
        """
        Returns the backbone to be used for to the current dataset.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms.Compose:
        """
        get_transform returns the transform to be used for the current dataset.

        Returns:
            transforms.Compose: the transform to be used for the dataset
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_loss() -> Callable:
        """
        get_loss returns the loss to be used for the current dataset.

        Returns:
            Callable: the loss function defined in nn.functional
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms.Compose:
        """
        get_normalization_transform returns the transform used for normalizing the current dataset.

        Returns:
            transforms.Compose: the normalization transform to be used for the dataset
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms.Compose:
        """
        get_denormalization_transform returns the transform used for denormalizing the current dataset.

        Returns:
            transforms.Compose: the de-normalization transform to be used for the dataset
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def get_scheduler(model, args: Args) -> torch.optim.lr_scheduler.LRScheduler:
        """
        get_scheduler returns the scheduler to be used for to the current dataset.

        Args:
            model (_type_): _description_
            args (Args): command line arguments

        Raises:
            NotImplementedError: _description_

        Returns:
            torch.optim.lr_scheduler.LRScheduler: _description_
        """
        raise NotImplementedError

    @staticmethod
    def get_epochs() -> int:
        raise NotImplementedError

    @staticmethod
    def get_batch_size() -> int:
        raise NotImplementedError

    @staticmethod
    def get_minibatch_size() -> int:
        raise NotImplementedError


def store_masked_loaders(
        train_dataset: CLDatasetImpl, test_dataset: CLDatasetImpl, setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    store_masked_loaders divides the dataset into tasks.

    Args:
        train_dataset (TrainDataset): train dataset
        test_dataset (datasets): test dataset
        setting (ContinualDataset): continual learning setting

    Returns:
        Tuple[DataLoader, DataLoader]: train and test loaders
    """
    train_mask = np.logical_and(
        np.array(train_dataset.targets) >= setting.i,
        np.array(train_dataset.targets) < (
            setting.i + setting.N_CLASSES_PER_TASK)
    )

    test_mask = np.logical_and(
        np.array(test_dataset.targets) >= setting.i,
        np.array(test_dataset.targets) < setting.i + setting.N_CLASSES_PER_TASK
    )

    try:
        train_dataset.data = train_dataset.data[train_mask]
        test_dataset.data = test_dataset.data[test_mask]
    except TypeError:
        train_dataset.data = [train_dataset.data[d]
                              for d in np.where(train_mask > 0)[0]]
        test_dataset.data = [test_dataset.data[d]
                             for d in np.where(test_mask > 0)[0]]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(
        train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=0)

    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def store_blurry_masked_loaders(
        train_dataset: CLDatasetImpl, test_dataset: CLDatasetImpl, setting: ContinualDataset, current_task: int) -> Tuple[DataLoader, DataLoader]:
    """
    store_blurry_masked_loaders divides the dataset into tasks.

    Args:
        train_dataset (TrainDataset): train dataset
        test_dataset (datasets): test dataset
        setting (ContinualDataset): continual learning setting
        current_task (int): current task id

    Returns:
        Tuple[DataLoader, DataLoader]: train and test loaders
    """
    # print(setting.train_idx_all[current_task].max(), len(np.array(train_dataset.targets)))
    train_dataset.data = train_dataset.data[setting.train_idx_all[current_task]]
    test_dataset.data = test_dataset.data[setting.test_idx_all[current_task]]

    train_dataset.targets = np.array(train_dataset.targets)[
        setting.train_idx_all[current_task]]
    test_dataset.targets = np.array(test_dataset.targets)[
        setting.test_idx_all[current_task]]

    train_loader = DataLoader(
        train_dataset, batch_size=setting.args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_dataset, batch_size=setting.args.batch_size, shuffle=False, num_workers=0)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += setting.N_CLASSES_PER_TASK
    return train_loader, test_loader


def get_previous_train_loader(
        train_dataset: CLDatasetImpl, batch_size: int, setting: ContinualDataset) -> DataLoader:
    """
    get_previous_train_loader creates a dataloader for the previous task.

    Args:
        train_dataset (TrainDataset): the entire training set
        batch_size (int): the desired batch size
        setting (ContinualDataset): the continual dataset at hand

    Returns:
        DataLoader: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >= setting.i - setting.N_CLASSES_PER_TASK,
                                np.array(train_dataset.targets) < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
