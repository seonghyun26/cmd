import wandb
import torch

import mdtraj as md
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split

from .data import *
from .model import ModelWrapper
from .md import MDSimulation, SteeredMDSimulation


scheduler_dict = {
    "None": None,
    "LambdaLR": LambdaLR,
    "MultiplicativeLR": MultiplicativeLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ExponentialLR": ExponentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
}


def load_data(cfg):
    data_path = f"/home/shpark/prj-cmd/simulation/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.state}-{cfg.data.version}.pt"
    dataset = torch.load(f"{data_path}")
    
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
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=cfg.training.loader.batch_size,
                shuffle=cfg.training.loader.shuffle,
                num_workers=cfg.training.loader.num_workers
            )
            test_loader = None
    else:
        raise ValueError(f"Molecule {cfg.data.molecule} not found")
    
    return train_loader, test_loader
       

def load_model_wrapper(cfg, device):
    model_wrapper = ModelWrapper(cfg, device)
    optimizer = load_optimizer(cfg, model_wrapper.parameters())
    scheduler = load_scheduler(cfg, optimizer)
    
    return model_wrapper, optimizer, scheduler


def load_optimizer(cfg, model_param):
    optimizer_dict = {
        "Adam": Adam,
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


def load_loss(cfg):    
    def mse_loss(y_true, y_pred, *args):
        loss_list = {
            "mse": nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true),
        }
        
        return loss_list

    def mse_reg_loss(y_true, y_pred, mu, log_var, *args):
        loss_list = {
            "mse": nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true),
            "reg": -0.5 * torch.sum(1 + log_var - log_var.exp())
        }
        
        return loss_list

    def mse_reg2_loss(y_true, y_pred, mu, log_var, *args):
        loss_list = {
            "mse": nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true),
            "reg": torch.square(log_var.exp())
        }
        
        return loss_list

    def mse_reg3_loss(y_true, y_pred, mu, log_var, step, *args):
        mse_loss = nn.MSELoss(reduction="none")(y_pred, y_true).mean(dim=(1))
        reg_loss = torch.square(log_var)
        if cfg.training.repeat:
            step_div = torch.sqrt(step[:, 0]).squeeze()
        else:
            step_div = torch.sqrt(step).squeeze()
        mse_loss /= step_div
        
        if cfg.training.loss.reduction == "mean":
            mse_loss = mse_loss.mean()
            reg_loss = reg_loss.mean()
        elif cfg.training.loss.reduction == "sum":
            mse_loss = mse_loss.sum()
            reg_loss = reg_loss.sum()
        else:
            raise ValueError(f"Reduction {cfg.training.loss.reduction} not found")

        loss_list = {
            "mse": mse_loss,
            "reg": reg_loss
        }
        
        return loss_list
    
    def mse_reg4_loss(y_true, y_pred, mu, log_var, *args):
        loss_list = {
            "mse": nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true),
            "reg": torch.square(log_var).mean()
        }
        
        return loss_list
    
    def mse_reg5_loss(y_true, y_pred, mu, log_var, *args):
        loss_list = {
            "mse": nn.MSELoss(reduction=cfg.training.loss.reduction)(y_pred, y_true),
            "reg": torch.square(mu) + torch.square(log_var)
        }
        
        return loss_list
    
    def mse_reg6_loss(y_true, y_pred, mu, log_var, step, *args):
        mse_loss = nn.MSELoss(reduction="none")(y_pred, y_true).mean(dim=(1))
        reg_loss = torch.square(log_var).mean(dim=(1))
        step_div = step.squeeze()
        mse_loss /= step_div
        reg_loss /= step_div
        
        if cfg.training.loss.reduction == "mean":
            mse_loss = mse_loss.mean()
            reg_loss = reg_loss.mean()
        elif cfg.training.loss.reduction == "sum":
            mse_loss = mse_loss.sum()
            reg_loss = reg_loss.sum()
        else:
            raise ValueError(f"Reduction {cfg.training.loss.reduction} not found")

        loss_list = {
            "mse": mse_loss,
            "reg": reg_loss
        }
        
        return loss_list
    
    def mse_reg7_loss(y_true, y_pred, mu, log_var, step, *args):
        mse_loss = nn.MSELoss(reduction="none")(y_pred, y_true)
        mse_loss /= torch.exp(log_var)
        mse_loss = mse_loss.mean()
        reg_loss = torch.square(log_var).mean()

        loss_list = {
            "mse": mse_loss,
            "reg": reg_loss
        }
        
        return loss_list
        
    
    def mae_loss(y_true, y_pred, *args):
        loss_list = {
            "mae": nn.L1Loss(reduction=cfg.training.loss.reduction)(y_pred, y_true)
        }
        
        return loss_list
    
    def mae_reg4_loss(y_true, y_pred, mu, log_var, *args):
        loss_list = {
            "mae": nn.L1Loss(reduction=cfg.training.loss.reduction)(y_pred, y_true),
            "reg": torch.square(log_var).mean()
        }

        return loss_list
    
    def mae_reg5_loss(result_dict):
        y_pred = result_dict["pred"]
        y_true = result_dict["true"]
        loss_list = {
            "mae": nn.L1Loss(reduction=cfg.training.loss.reduction)(y_pred, y_true),
            "reg": torch.square(log_var).mean(),
            "mu": -torch.square(mu).mean(),
        }
        
        return loss_list
    
    def cl_loss(result_dict):
        current_state_rep = result_dict["current_state_rep"]
        next_state_rep = result_dict["next_state_rep"]
        
        positive_pairs = torch.sum(current_state_rep * next_state_rep, dim=-1)
        negative_pairs = torch.sum(current_state_rep * torch.roll(next_state_rep, shifts=1, dims=0), dim=-1)
        contrastive_loss = -torch.log(torch.exp(positive_pairs) / (torch.exp(positive_pairs) + torch.exp(negative_pairs)))
        contrastive_loss = contrastive_loss.mean()
        
        loss_list = {
            "CL": contrastive_loss
        }
        
        return loss_list
    
    
    loss_func_list = {
        "mse": mse_loss,
        "mse+reg": mse_reg_loss,
        "mse+reg2": mse_reg2_loss,
        "mse+reg3": mse_reg3_loss,
        "mse+reg4": mse_reg4_loss,
        "mse+reg5": mse_reg5_loss,
        "mse+reg7": mse_reg7_loss,
        "mae": mae_loss,
        "mae+reg4": mae_reg4_loss,
        "mae+reg5": mae_reg5_loss,
        "cl": cl_loss,
    }
    
    loss_type_list = {
        "mse": ["mse"],
        "mse+reg": ["mse", "reg"],
        "cl": ["CL"],
    }
    
    loss_name = cfg.training.loss.name.lower()
    if loss_name in loss_func_list.keys():
        loss_func = loss_func_list[loss_name]
    else:
        raise ValueError(f"Loss {loss_name} not found")
    
    if loss_name in loss_type_list.keys():
        loss_type = loss_type_list[loss_name]
    else:
        raise ValueError(f"Loss type {loss_name} not found")
    
    return loss_func, loss_type


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
    else:
        raise ValueError(f"Molecule {cfg.job.molecule} not found")
    
    return states


def load_simulation(cfg, sample_num, device):
    simulation_list = MDSimulation(cfg, sample_num, device)
    
    return simulation_list

def load_steered_simulation(cfg, sample_num, model, device):
    simulation_list = SteeredMDSimulation(cfg, sample_num, model, device)

    return simulation_list