# Standard Library
from typing import Literal
from argparse import Namespace


class Args(Namespace):
    model: Literal['joint', 'bic', 'lwf', 'ewc_on', 'hal', 'gss', 'joint_gcl',
                   'gem', 'agem', 'agem_r', 'icarl', 'lucir', 'pnn', 'rpc', 'sgd',
                   'fdr', 'gdumb', 'derpp_cba_online', 'derpp_cba_offline',
                   'clser_cba_online', 'clser_cba_offline', 'er_cba_online',
                   'er_cba_offline' 'si', 'ssil', 'er', 'er_ace', 'clser',
                   'mer', 'der', 'xder', 'derpp', ]

    backbone: Literal["resnet18"]

    dataset: Literal['seq-tinyimg', 'perm-mnist', 'rot-mnist',
                     'seq-cifar10-blurry', 'seq-cifar100', 'seq-mnist', 'mnist-360', 'seq-cifar10']

    load_best_args: bool

    # Management Args
    exp: str
    seed: int
    notes: str
    gpu_id: int
    csv_log: bool
    validation: bool
    save_model: bool
    tensorboard: bool
    non_verbose: bool
