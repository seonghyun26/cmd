import math
import wandb
import torch
import random

import mdtraj as md
import torch.nn as nn

from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader, random_split

from mlcolvar.core.transform import Normalization

from .data import *
from .model import ModelWrapper
from .md import MDSimulation, SteeredMDSimulation
from .loss import (
    TripletLoss,
    TripletTorchLoss,
    TripletLossNegative,
    TripletLossTest,
    InfoNCELoss,
)


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
        https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(
        self,
        optimizer : torch.optim.Optimizer,
        first_cycle_steps : int,
        cycle_mult : float = 1.,
        max_lr : float = 0.1,
        min_lr : float = 0.001,
        warmup_steps : int = 0,
        gamma : float = 1.,
        last_epoch : int = -1
    ):
        assert warmup_steps < first_cycle_steps
        
        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle
        
        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle
        
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        
        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            

scheduler_dict = {
    "None": None,
    "StepLR": StepLR,
    "LambdaLR": LambdaLR,
    "MultiplicativeLR": MultiplicativeLR,
    "MultiStepLR": MultiStepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineAnnealingWarmupRestarts": CosineAnnealingWarmupRestarts,
}


def load_data(cfg):   
    data_path = f"{cfg.data.dir}/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}/{cfg.data.name}.pt"
    dataset = torch.load(f"{data_path}")
    total_data_mean = None
    total_data_std = None
    
    if cfg.data.molecule == "double-well":
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=cfg.training.loader.batch_size,
            shuffle=cfg.training.loader.shuffle,
            num_workers=cfg.training.loader.num_workers
        )
        test_loader = None
    
    elif cfg.data.molecule == "alanine":
        if cfg.training.loader.test:
            train_dataset, test_dataset = random_split(dataset, cfg.data.train_test_split)
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=cfg.training.loader.batch_size,
                shuffle=cfg.training.loader.shuffle,
                num_workers=cfg.training.loader.num_workers
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=cfg.training.loader.batch_size,
                shuffle=cfg.training.loader.shuffle,
                num_workers=cfg.training.loader.num_workers
            )
        else:
            if cfg.data.normalize:
                total_data = torch.cat([dataset.x, dataset.x_augmented, dataset.x_augmented_hard], dim=0)
                total_data_mean = total_data.mean(dim=0)
                total_data_std = total_data.std(dim=0)
                normalization = Normalization(
                    in_features=total_data.shape[1],
                    mean=total_data_mean,
                    range=total_data_std
                )
                dataset.x = normalization(dataset.x)
                dataset.x_augmented = normalization(dataset.x_augmented)
                dataset.x_augmented_hard = normalization(dataset.x_augmented_hard)
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=cfg.training.loader.batch_size,
                shuffle=cfg.training.loader.shuffle,
                num_workers=cfg.training.loader.num_workers
            )
            test_loader = None
    
    elif cfg.data.molecule == "chignolin":
        raise NotImplementedError("Chignolin dataset TBA")
    
    else:
        raise ValueError(f"Molecule {cfg.data.molecule} not found")
    
    return train_loader, test_loader, total_data_mean, total_data_std
       

def load_model_wrapper(cfg, device):
    model_wrapper = ModelWrapper(cfg, device)
    if cfg.training.train:
        optimizer = load_optimizer(cfg, model_wrapper.parameters())
        scheduler = load_scheduler(cfg, optimizer)
    else:
        optimizer = None
        scheduler = None
    
    return model_wrapper, optimizer, scheduler

def load_optimizer(cfg, model_param):
    optimizer_dict = {
        "Adam": Adam,
        "AdamW": AdamW,
        "SGD": SGD,
    }
    
    if cfg.training.optimizer.name in optimizer_dict.keys():
        optimizer = optimizer_dict[cfg.training.optimizer.name](
            model_param,
            **cfg.training.optimizer.params
        )
    else:
        raise ValueError(f"Optimizer {cfg.training.optimizer} not found")
    
    return optimizer

def load_scheduler(cfg, optimizer):   
    if cfg.training.scheduler.name == "None":
        scheduler = None
    
    elif cfg.training.scheduler.name == "LambdaLR":
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: cfg.training.scheduler.lr_lambda ** epoch
        )
    
    elif cfg.training.scheduler.name == "MultiplicativeLR":
        scheduler = MultiplicativeLR(
            optimizer=optimizer,
            lr_lambda=lambda epoch: 0.95 ** epoch
        )
    
    elif cfg.training.scheduler.name in scheduler_dict.keys():
        scheduler = scheduler_dict[cfg.training.scheduler.name](
            optimizer,
            **cfg.training.scheduler.params
        )
    
    else:
        raise ValueError(f"Scheduler {cfg.training.scheduler.name} not found")
    
    return scheduler

def load_loss(cfg, normalization = None):
    loss_dict = {
        "triplet": TripletLoss,
        "triplet-torch": TripletTorchLoss,
        "triplet-negative": TripletLossNegative,
        "triplet-test": TripletLossTest,
        "infonce": InfoNCELoss,
    }
    
    loss_name = cfg.training.loss.name.lower()
    if loss_name not in loss_dict:
        raise ValueError(f"Loss {loss_name} not found")
    loss_instance = loss_dict[loss_name](cfg)
    
    return loss_instance, loss_instance.loss_types

def load_state_file(cfg, state, device):
    if cfg.job.molecule == "alanine":
        state_dir = f"./data/{cfg.job.molecule}/{state}.pdb"
        state = md.load(state_dir).xyz
        state = torch.tensor(state).to(device)
        states = state.repeat(cfg.job.sample_num, 1, 1)
    
    elif cfg.job.molecule == "double-well":
        if state == "left":
            state = torch.tensor([-1.118, 0], dtype=torch.float32).to(device)
        elif state == "right":
            state = torch.tensor([1.118, 0], dtype=torch.float32).to(device)
        else:
            raise ValueError(f"State {state} not found")
        states = state.repeat(cfg.job.sample_num, 1)
    
    elif cfg.job.molecule == "chignolin":
        raise NotImplementedError("Chignolin state TBA")
    
    else:
        raise ValueError(f"Molecule {cfg.job.molecule} not found")
    
    return states

def load_simulation(cfg, sample_num, device):
    simulation_list = MDSimulation(cfg, sample_num, device)
    
    return simulation_list

def load_steered_simulation(cfg, sample_num, model_wrapper):
    simulation_list = SteeredMDSimulation(cfg, sample_num, model_wrapper)

    return simulation_list

