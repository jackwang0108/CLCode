# Standard Library
import math
from copy import deepcopy

# Third-Party Library
import numpy as np
from models.icarl import fill_buffer
from utils.batch_norm import bn_track_stats

# Torch Library
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

# My Library
from utils.args import *
from datasets import get_dataset
from utils.buffer import Buffer, icarl_replay
from utils.types import Args, CVBackboneImpl, CVTransforms
from models.utils.continual_model import ContinualModel


def lucir_batch_margin_ranking_loss(
        labels: torch.Tensor, probas: torch.Tensor, k: int, margin: float, num_old_classes: int) -> torch.Tensor:
    """
    lucir_batch_margin_ranking_loss calculates margin ranking loss, i.e. L_{mr}(x), defined in Sec. 3.4 Inter-Class Separation

    a good introduction about margin ranking loss: https://gombru.github.io/2019/04/03/ranking_loss/

    Args:
        labels (torch.Tensor): the label of the input
        probas (torch.Tensor): the classification probabilities output by cosine classifier
        k (int): top-K new class embedding chosen as hard negatives
        margin (float): margin threshold
        num_old_classes (int): _description_

    Returns:
        torch.Tensor: lucir batch margin ranking loss L_mr
    """

    # if there is no examples of old classes in the batch
    old_example_index = labels < num_old_classes
    loss = torch.zeros(1).to(probas.device)
    if (old_example_num := old_example_index.sum()) < 0:
        return loss

    # get the ground truth possibility of old examples
    gt_index = torch.zeros_like(probas)
    gt_index = gt_index.scatter(
        dim=1, index=labels.unsqueeze(-1), value=1) == 1
    gt_probas = probas.masked_select(gt_index)

    old_example_gt_probas = gt_probas[old_example_index].view(
        -1, 1).repeat(1, k)

    # get top-K novel classes probability
    top_novel_probas = probas[:, num_old_classes:].topk(k, dim=1)[0]
    old_example_top_novel_probas = top_novel_probas[old_example_index]

    assert (old_example_gt_probas.size() ==
            old_example_top_novel_probas.size()), f"old example probability mismatches, {old_example_gt_probas.size()=}, {old_example_top_novel_probas.size()=}"
    assert (old_example_gt_probas.size(
        0) == old_example_num), f"old example num mismatches, {old_example_gt_probas.size(0)=}, {old_example_num=}"

    # For old examples, pushes novel classes probability (for cosine classifier, distance) away from ground truth old class probability
    loss = nn.MarginRankingLoss(margin=margin)(
        old_example_gt_probas.flatten(),
        old_example_top_novel_probas.flatten(),
        torch.ones(old_example_num * k).to(probas.device)
    )

    return loss


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Learning via Lucir.')

    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)

    parser.add_argument('--lamda_base', type=float, required=False, default=5.,
                        help='Regularization weight for embedding cosine similarity.')
    parser.add_argument('--lamda_mr', type=float, required=False, default=1.,
                        help='Regularization weight for embedding cosine similarity.')
    parser.add_argument('--k_mr', type=int, required=False, default=2,
                        help='K for margin-ranking loss.')
    parser.add_argument('--mr_margin', type=float, default=0.5,
                        required=False, help='Margin for margin-ranking loss.')
    parser.add_argument('--fitting_epochs', type=int, required=False, default=20,
                        help='Number of epochs to finetune on coreset after each task.')
    parser.add_argument('--lr_finetune', type=float, required=False, default=0.01,
                        help='Learning Rate for finetuning.')
    parser.add_argument('--imprint_weights', type=int, choices=[0, 1], required=False, default=1,
                        help='Apply weight imprinting?')
    return parser


class CustomClassifier(nn.Module):
    """ CustomClassifier is the Cosine Similarity Classifier use in Lucir """

    def __init__(self, in_features: int, class_per_task: int, n_tasks: int):
        super().__init__()

        # Parameters
        self.weights: list[nn.Parameter] = nn.ParameterList(
            [
                nn.parameter.Parameter(torch.Tensor(class_per_task, in_features)) for _ in range(n_tasks)
            ]
        )

        self.sigma = nn.parameter.Parameter(torch.Tensor(1))

        # Task settings
        self.n_tasks: int = n_tasks
        self.in_features: int = in_features
        self.class_per_task: int = class_per_task
        self.reset_parameters()

        # set the first task
        self.task = 0
        self.weights[0].requires_grad = True

    def reset_parameters(self):
        """
        reset_parameters resets the parameters of all task
        """
        for i in range(self.n_tasks):
            # Manual Xavier Normalization
            stdv = 1. / math.sqrt(self.weights[i].size(1))
            self.weights[i].data.uniform_(-stdv, stdv)

            self.weights[i].requires_grad = False

        self.sigma.data.fill_(1)

    def forward(self, non_normalized_features: torch.Tensor) -> torch.Tensor:
        return self.noscale_forward(non_normalized_features) * self.sigma

    def reset_weight(self, task_id: int):
        """
        reset_weight resets the weight of specific task

        Args:
            task_id (int): task to reset
        """
        stdv = 1. / math.sqrt(self.weights[task_id].size(1))
        self.weights[task_id].data.uniform_(-stdv, stdv)
        self.weights[task_id].requires_grad = True

        self.weights[task_id-1].requires_grad = False

    def noscale_forward(self, non_normalized_features: torch.Tensor) -> torch.Tensor:

        normalized_features = F.normalize(non_normalized_features, p=2, dim=1).reshape(
            len(non_normalized_features), -1)

        outputs = []
        for t in range(self.n_tasks):
            task_output = F.linear(normalized_features,
                                   F.normalize(self.weights[t], p=2, dim=1))
            outputs.append(task_output)

        outputs = torch.cat(outputs, dim=1)

        return outputs


class Lucir(ContinualModel):
    NAME = 'lucir'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone: CVBackboneImpl, loss, args: Args, transform: CVTransforms):
        super(Lucir, self).__init__(backbone, loss, args, transform)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.eye = torch.eye(self.dataset.N_CLASSES_PER_TASK *
                             self.dataset.N_TASKS).to(self.device)

        self.old_net = None
        self.task = 0
        self.epochs = int(args.n_epochs)
        self.lamda_cos_sim = args.lamda_base

        # self.net.classifier = CustomClassifier(self.net.classifier.in_features, self.dataset.N_CLASSES_PER_TASK, self.dataset.N_TASKS)
        self.net.fc = CustomClassifier(
            self.net.fc.in_features, self.dataset.N_CLASSES_PER_TASK, self.dataset.N_TASKS)

        # upd_weights = [p for n, p in self.net.named_parameters()
        #                if 'classifier' not in n and '_fc' not in n] + [self.net.classifier.weights[0], self.net.classifier.sigma]
        # fix_weights = list(self.net.classifier.weights[1:])
        upd_weights = [p for n, p in self.net.named_parameters()
                       if 'fc' not in n and '_fc' not in n] + [self.net.fc.weights[0], self.net.fc.sigma]
        fix_weights = list(self.net.fc.weights[1:])

        self.opt = torch.optim.SGD([{'params': upd_weights, 'lr': self.args.lr, 'momentum': self.args.optim_mom, 'weight_decay': self.args.optim_wd}, {
            'params': fix_weights, 'lr': 0, 'momentum': self.args.optim_mom, 'weight_decay': 0}])

        self.ft_lr_start = [10]

        self.c_epoch = -1

    def update_classifier(self):
        # self.net.classifier.task += 1
        # self.net.classifier.reset_weight(self.task)
        self.net.fc.task += 1
        self.net.fc.reset_weight(self.task)

    def forward(self, x):
        with torch.no_grad():
            outputs = self.net(x)

        return outputs

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None, fitting=False):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels.long(), self.task)
        loss.backward()

        self.opt.step()

        return loss.item()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int) -> torch.Tensor:
        """
        get_loss computes the loss defined in paper

        Args:
            inputs (torch.Tensor): the images to be fed to the network
            labels (torch.Tensor): the ground-truth labels
            task_idx (int): the task index

        Returns:
            torch.Tensor: the lucir loss on input batch
        """

        pc = task_idx * self.dataset.N_CLASSES_PER_TASK
        ac = (task_idx + 1) * self.dataset.N_CLASSES_PER_TASK

        outputs: torch.Tensor
        outputs = self.net(inputs, returnt='feature').float()

        cos_output = self.net.fc.noscale_forward(outputs)
        outputs = outputs.reshape(outputs.size(0), -1)

        loss = F.cross_entropy(cos_output*self.net.fc.sigma, labels)

        if task_idx > 0:
            with torch.no_grad():
                logits = self.old_net(inputs, returnt='feature')
                logits = logits.reshape(logits.size(0), -1)

            loss2 = F.cosine_embedding_loss(
                outputs, logits.detach(), torch.ones(outputs.shape[0]).to(outputs.device))*self.lamda_cos_sim

            # Remove rescale by sigma before this loss
            loss3 = lucir_batch_margin_ranking_loss(
                labels, cos_output, self.args.k_mr, self.args.mr_margin, pc) * self.args.lamda_mr

            loss = loss+loss2+loss3

        return loss

    def begin_task(self, dataset):

        if self.task > 0:
            icarl_replay(self, dataset)

            with torch.no_grad():
                # Update model classifier
                self.update_classifier()

                if self.args.imprint_weights == 1:
                    self.imprint_weights(dataset)

                # Restore optimizer LR
                # upd_weights = [p for n, p in self.net.named_parameters()
                #                if 'classifier' not in n] + [self.net.classifier.weights[self.task], self.net.classifier.sigma]
                # fix_weights = list(self.net.classifier.weights[:self.task])
                upd_weights = [p for n, p in self.net.named_parameters()
                               if 'fc' not in n] + [self.net.fc.weights[self.task], self.net.fc.sigma]
                fix_weights = list(self.net.fc.weights[:self.task])

                if self.task < self.dataset.N_TASKS-1:
                    # fix_weights += list(self.net.classifier.weights[self.task+1:])
                    fix_weights += list(self.net.fc.weights[self.task+1:])

                self.opt = torch.optim.SGD([{'params': upd_weights, 'lr': self.args.lr,  'weight_decay': self.args.optim_wd}, {
                    'params': fix_weights, 'lr': 0, 'weight_decay': 0}], lr=self.args.lr, momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)

    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())

        self.net.train()
        with torch.no_grad():
            fill_buffer(self, self.buffer, dataset, self.task)

        if self.args.fitting_epochs is not None and self.args.fitting_epochs > 0:
            self.fit_buffer(self.args.fitting_epochs)

        self.task += 1

        # Adapt lambda
        self.lamda_cos_sim = math.sqrt(
            self.task)*float(self.args.lamda_base)

    def imprint_weights(self, dataset):
        self.net.eval()
        # old_embedding_norm = torch.cat([self.net.classifier.weights[i] for i in range(self.task)]).norm(dim=1, keepdim=True)
        old_embedding_norm = torch.cat(
            [self.net.fc.weights[i] for i in range(self.task)]).norm(dim=1, keepdim=True)
        average_old_embedding_norm = torch.mean(
            old_embedding_norm, dim=0).cpu().type(torch.DoubleTensor)
        # num_features = self.net.classifier.in_features
        num_features = self.net.fc.in_features
        novel_embedding = torch.zeros(
            (self.dataset.N_CLASSES_PER_TASK, num_features))
        loader = dataset.train_loader

        cur_dataset = deepcopy(loader.dataset)

        for cls_idx in range(self.task*self.dataset.N_CLASSES_PER_TASK, (self.task+1)*self.dataset.N_CLASSES_PER_TASK):

            cls_indices = np.asarray(
                loader.dataset.targets) == cls_idx
            cur_dataset.data = loader.dataset.data[cls_indices]
            cur_dataset.targets = np.zeros((cur_dataset.data.shape[0]))
            dt = DataLoader(
                cur_dataset, batch_size=self.args.batch_size, num_workers=0)

            num_samples = cur_dataset.data.shape[0]
            cls_features = torch.empty((num_samples, num_features))
            for j, d in enumerate(dt):
                # tt = self.net(d[0].to(self.device), returnt='features').cpu()
                tt = self.net(d[0].to(self.device), returnt='feature').cpu()
                if 'ntu' in self.args.dataset:
                    tt = F.adaptive_avg_pool3d(tt, 1)
                cls_features[j*self.args.batch_size:(
                    j+1)*self.args.batch_size] = tt.reshape(len(tt), -1)

            norm_features = F.normalize(cls_features, p=2, dim=1)
            cls_embedding = torch.mean(norm_features, dim=0)

            novel_embedding[cls_idx-self.task*self.dataset.N_CLASSES_PER_TASK] = F.normalize(
                cls_embedding, p=2, dim=0) * average_old_embedding_norm

        # self.net.classifier.weights[self.task].data = novel_embedding.to(self.device)
        self.net.fc.weights[self.task].data = novel_embedding.to(self.device)
        self.net.train()

    def fit_buffer(self, opt_steps):

        old_opt = self.opt
        # Optimize only final embeddings
        # self.opt = torch.optim.SGD(self.net.classifier.parameters(), self.args.lr_finetune,
        # momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        self.opt = torch.optim.SGD(self.net.fc.parameters(), self.args.lr_finetune,
                                   momentum=self.args.optim_mom, weight_decay=self.args.optim_wd)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.opt, milestones=self.ft_lr_start, gamma=0.1)

        with bn_track_stats(self, False):
            for _ in range(opt_steps):
                examples, labels, _ = self.buffer.get_all_data(self.transform)
                dt = DataLoader([(e, l) for e, l in zip(examples, labels)],
                                shuffle=True, batch_size=self.args.batch_size)
                for inputs, labels in dt:
                    self.observe(inputs, labels, None, fitting=True)
                    lr_scheduler.step()

        self.opt = old_opt
