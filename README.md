# CL-Codes


## Setup and Training

* Use `python utils/main.py` to run experiments.

* Results will be saved in `./results`.

* Training examples:

  * Baseline **DER++** on **CIFAR-100** with buffer size **M=2000**

    `python utils/main.py --model derpp --backbone resnet18 --dataset seq-cifar100 --lr 0.03 --batch_size 32 --minibatch_size 32 --n_epochs 1 --buffer_size 2000 --seed <5000> --gpu_id <0> --exp <onl-buf2000>`

  * Our method **DER-CBA** on **CIFAR-100** with buffer size **M=2000**

    `python utils/main.py --model derpp_cba_online --backbone resnet18-meta --dataset seq-cifar100 --lr 0.03 --batch_size 32 --minibatch_size 32 --n_epochs 1 --buffer_size 2000 --seed <5000> --gpu_id <0> --exp <onl-buf2000>`

  * We recommend repeating the experiment multiple times with different random seeds to reduce the effect of randomness, especially under the online setting (*i.e.*, `--n_epochs 1`).



## Requirements

* torch==1.7.0
* torchvision=0.9.0
* quadprog=0.1.7


## Acknowledgement

This repository is developed based on the CBA-online-CL