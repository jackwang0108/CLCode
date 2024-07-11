# Standard Library
import sys
from pathlib import Path
from typing import Tuple
from copy import deepcopy

# Torch Library
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# My Library
from utils.loggers import *
from utils.metrics import *
from utils.tb_logger import *
from datasets import get_dataset
from utils.loggers import CsvLogger
from utils.status import progress_bar, create_stash
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from utils.types import Args, ContinualModelImpl


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, task_id: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    , mask_classes masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.

    Args:
        outputs (torch.Tensor): the output tensor
        dataset (ContinualDataset): the continual dataset
        task_id (int): the task index
    """
    outputs[:, 0:task_id * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (task_id + 1) * dataset.N_CLASSES_PER_TASK:
            dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


@torch.no_grad()
def evaluate(model: ContinualModel | ContinualModelImpl, dataset: ContinualDataset,
             last: bool = False, returnt=None) -> Tuple[list, list]:
    """
    evaluate evaluates the accuracy of the model for previous tasks.

    Args:
        model (ContinualModel): the model to be evaluated
        dataset (ContinualDataset): the continual dataset at hand
        last (bool, optional): _description_. Defaults to False.
        returnt (_type_, optional): _description_. Defaults to None.

    Returns:
        Tuple[list, list]: a tuple of lists, containing the class-il and task-il accuracy for each task.
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []

    for task_id, test_loader in enumerate(dataset.test_loaders):

        # skip
        if last and task_id < len(dataset.test_loaders) - 1:
            continue

        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(
                model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, task_id)
            else:
                outputs = model(inputs)

            if returnt == 'CBA':
                res_outputs = model.CBA(F.softmax(outputs, dim=-1))
                outputs = outputs + res_outputs

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, task_id)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total *
                    100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(
        model: ContinualModel | ContinualModelImpl, dataset: ContinualDataset, args: Args) -> None:
    """
    train the model in Continual Learning paradigm, including class incremental setups and task incremental setups. The function also includes evaluations and logging.

    Args:
        model (ContinualModel): the module to be trained
        dataset (ContinualDataset): the continual dataset at hand
        args (Args): the arguments of the current execution, originated from command line arguments
    """

    # logging
    root_dir = Path(__file__).resolve().parent.parent
    save_dir = root_dir / f'results/{model.NAME}-{dataset.NAME}-{args.exp}'

    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)

    # log command-line args
    with (record_file := save_dir / "record.txt").open(mode="a") as f:
        for arg in vars(args):
            f.write('{}:\t{}\n'.format(arg, getattr(args, arg)))

    results, results_mask_classes = [], []

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING)

    dataset_copy = get_dataset(args)

    model.net.to(model.device)
    for task_id in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()

    if model.NAME != 'icarl' and model.NAME != 'pnn' and model.NAME != 'scr':
        random_results_class, random_results_task = evaluate(
            model, dataset_copy)

    model.n_cls_per_task = dataset.N_CLASSES_PER_TASK
    model.total_classes = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS

    print(file=sys.stderr)
    all_accuracy_cls, all_accuracy_tsk = [], []
    all_forward_cls, all_forward_tsk = [], []
    all_backward_cls, all_backward_tsk = [], []
    all_forgetting_cls, all_forgetting_tsk = [], []
    all_acc_auc_cls, all_acc_auc_tsk = [], []

    if hasattr(model, 'CBA'):
        all_CBA_accuracy_cls, all_CBA_accuracy_tsk = [], []

    # Continual Learning
    for task_id in range(dataset.N_TASKS):

        # set current task
        model.net.train()
        model.current_task = task_id
        model.seen_classes = (task_id + 1) * dataset.N_CLASSES_PER_TASK

        # get dataloader
        test_loader: DataLoader
        train_loader: DataLoader
        train_loader, test_loader = dataset.get_data_loaders()

        # if the model needs preparation before learning on current task
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)

        # evaluate on previous tasks
        if task_id:
            accs = evaluate(model, dataset, last=True)
            results[task_id-1] = results[task_id-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[task_id -
                                     1] = results_mask_classes[task_id-1] + accs[1]

        all_acc_auc_cls.append([])
        all_acc_auc_tsk.append([])

        scheduler = dataset.get_scheduler(model, args)

        for epoch in range(model.args.n_epochs):
            model.epoch = epoch  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            batch_data: list[torch.Tensor]
            for batch_idx, batch_data in enumerate(train_loader):
                if model.args.n_epochs == 1 and batch_idx == len(train_loader) - 1:
                    # End the training before the last iteration.
                    # That is because we find that the last few samples would hurt the network training,
                    # which leads to lower performance.
                    continue

                model.batch_idx = batch_idx

                # get the data
                inputs, labels, not_aug_inputs = [
                    t.to(model.device) for t in batch_data[:3]]

                # learn the batch
                if hasattr(dataset.train_loader.dataset, 'logits'):
                    # some model needs logits
                    logits = batch_data[-1]
                    logits = logits.to(model.device)
                    loss = model.observe(
                        inputs, labels, not_aug_inputs, logits)
                else:
                    loss = model.observe(inputs, labels, not_aug_inputs)

                progress_bar(batch_idx, len(train_loader),
                             epoch, task_id, loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, task_id, batch_idx)

                # anytime inference for small datasets
                non_support_models = ["icarl", "scr"]
                if model.args.n_epochs == 1 and args.dataset != 'seq-imgnet1k' and model.NAME not in non_support_models:
                    if batch_idx % 5 == 0:
                        accs = evaluate(deepcopy(model), dataset)
                        all_acc_auc_cls[task_id].append(accs[0])
                        all_acc_auc_tsk[task_id].append(accs[1])

            if scheduler is not None:
                scheduler.step()

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, task_id + 1, dataset.SETTING)
        print('class-il:', accs[0], '\ntask-il:', accs[1])

        # record the results
        all_accuracy_cls.append(accs[0])
        all_accuracy_tsk.append(accs[1])

        # print the fwt, bwt, forgetting
        non_support_models = ["icarl", "pnn", "scr"]
        if model.NAME not in non_support_models:
            fwt = forward_transfer(results, random_results_class)
            fwt_mask_classes = forward_transfer(
                results_mask_classes, random_results_task)
            bwt = backward_transfer(results)
            bwt_mask_classes = backward_transfer(results_mask_classes)
            forget = forgetting(results)
            forget_mask_classes = forgetting(results_mask_classes)

            print('Forward: class-il: {}\ttask-il:{}'.format(fwt, fwt_mask_classes))
            print('Backward: class-il: {}\ttask-il:{}'.format(bwt, bwt_mask_classes))
            print('Forgetting: class-il: {}\ttask-il:{}'.format(forget,
                  forget_mask_classes))

            # record the results
            all_forward_cls.append(fwt)
            all_forward_tsk.append(fwt_mask_classes)
            all_backward_cls.append(bwt)
            all_backward_tsk.append(bwt_mask_classes)
            all_forgetting_cls.append(forget)
            all_forgetting_tsk.append(forget_mask_classes)

        if hasattr(model, 'CBA'):
            print(
                '\n************************ Results of the CBA: ************************')
            accs_bias = evaluate(model, dataset, returnt='CBA')
            print_mean_accuracy(np.mean(accs_bias, axis=1),
                                task_id + 1, dataset.SETTING)
            print('class-il:', accs_bias[0], '\ntask-il:', accs_bias[1])
            print(
                '***********************************************************************************')

            # record the results
            all_CBA_accuracy_cls.append(accs_bias[0])
            all_CBA_accuracy_tsk.append(accs_bias[1])

        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, task_id)

        np.set_printoptions(suppress=True)

    # record the results
    with record_file.open(mode="a") as f:
        f.write('\n== 1. Acc:\n==== 1.1. Class-IL:\n')
        for task_id in range(dataset.N_TASKS):
            f.write(str(all_accuracy_cls[task_id]).strip(
                '[').strip(']') + '\n')
        f.write('\n==== 1.2. Task-IL:\n')
        for task_id in range(dataset.N_TASKS):
            f.write(str(all_accuracy_tsk[task_id]).strip(
                '[').strip(']') + '\n')

        f.write('\n== 2. Forward:')
        f.write('\n==== 2.1. Class-IL:\n' +
                str(all_forward_cls).strip('[').strip(']'))
        f.write('\n==== 2.2. Task-IL:\n' +
                str(all_forward_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 3. Backward:')
        f.write('\n==== 3.1. Class-IL:\n' +
                str(all_backward_cls).strip('[').strip(']'))
        f.write('\n==== 3.2. Task-IL:\n' +
                str(all_backward_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 4. Forgetting:')
        f.write('\n==== 4.1. Class-IL:\n' +
                str(all_forgetting_cls).strip('[').strip(']'))
        f.write('\n==== 4.2. Task-IL:\n' +
                str(all_forgetting_tsk).strip('[').strip(']'))
        f.write('\n')

        f.write('\n== 5. Acc_auc:\n==== 5.1. Class-IL:\n')
        for task_id in range(dataset.N_TASKS):
            f.write('\nTask {}:\n'.format(task_id + 1))
            avg_acc_cls, avg_acc_tsk = [], []
            for acc_cls, acc_tsk in zip(all_acc_auc_cls[task_id], all_acc_auc_tsk[task_id]):
                avg_acc_cls.append(np.mean(acc_cls))
                avg_acc_tsk.append(np.mean(acc_tsk))
                f.write(str(acc_cls).strip('[').strip(
                    ']') + ' - ' + str(np.mean(acc_cls)) + '\n')

        f.write('\nACC_AUC_cls = {}:\n'.format(np.mean(avg_acc_cls)))
        f.write('ACC_AUC_tsk = {}:\n'.format(np.mean(avg_acc_tsk)))

        if hasattr(model, 'CBA') and len(all_CBA_accuracy_cls) > 0:
            f.write('\n== 5. CBA Acc:\n==== 5.1. Class-IL:\n')
            for task_id in range(dataset.N_TASKS):
                f.write(str(all_CBA_accuracy_cls[task_id]).strip(
                    '[').strip(']') + '\n')
            f.write('\n==== 5.2. Task-IL:\n')
            for task_id in range(dataset.N_TASKS):
                f.write(str(all_CBA_accuracy_tsk[task_id]).strip(
                    '[').strip(']') + '\n')

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn' and model.NAME != 'scr':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))
