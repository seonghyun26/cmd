import wandb
import torch
import random

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
    data_path = f"{cfg.data.dir}/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.version}.pt"
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
    
    elif cfg.data.molecule == "chignolin":
        raise NotImplementedError("Chignolin dataset TBA")
    
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

def load_similarity(similarity_type):
    
    if similarity_type == "cosine":
        similarity = nn.CosineSimilarity(dim=1)
    else:
        raise ValueError(f"Similarity {similarity_type} not found")

    return similarity

def load_loss(cfg):    
    def mse_loss(result_dict):
        loss_list = {
            "mse": nn.MSELoss(reduction=cfg.training.loss.reduction)(result_dict["pred"], result_dict["true"]),
        }
        
        return loss_list

    def mse_reg_loss(result_dict):
        loss_list = {
            "mse": nn.MSELoss(reduction=cfg.training.loss.reduction)(result_dict["pred"], result_dict["true"]),
            "reg": torch.square(result_dict["log_var"]).mean()
        }
        
        return loss_list
    
    def mae_loss(result_dict):
        loss_list = {
            "mae": nn.L1Loss(reduction=cfg.training.loss.reduction)(result_dict["pred"], result_dict["true"])
        }
        
        return loss_list
    
    def mae_reg_loss(result_dict):
        loss_list = {
            "mae": nn.L1Loss(reduction=cfg.training.loss.reduction)(result_dict["pred"], result_dict["true"]),
            "reg": torch.square(result_dict["log_var"]).mean(),
            "mu": -torch.square(result_dict["mu"]).mean(),
        }
        
        return loss_list
    
    def cl_loss(result_dict):
        current_state_rep = result_dict["current_state_rep"]
        next_state_rep = result_dict["next_state_rep"]
        
        positive_pairs = torch.sum(current_state_rep * next_state_rep, dim=-1)
        negative_pairs = torch.sum(current_state_rep * torch.roll(next_state_rep, shifts=random.randint(1, batch_size), dims=0), dim=-1)
        contrastive_loss = -torch.log(torch.exp(positive_pairs) / (torch.exp(positive_pairs) + torch.exp(negative_pairs)))
        contrastive_loss = contrastive_loss.mean()
        
        loss_list = {
            "CL": contrastive_loss
        }
        
        return loss_list
    
    def nce_loss(result_dict):
        current_state_rep = result_dict["current_state_rep"]
        next_state_rep = result_dict["next_state_rep"]
        batch_size = current_state_rep.shape[0]
        similarity = load_similarity(cfg.training.loss.similarity)
        
        positive_pairs = torch.sigmoid(similarity(current_state_rep, next_state_rep))
        negative_pairs = 1 - torch.sigmoid(similarity(current_state_rep, torch.roll(next_state_rep, shifts=random.randint(1, batch_size), dims=0)))
        contrastive_loss = torch.log(positive_pairs / negative_pairs)
        
        loss_list = {
            "CL": contrastive_loss.mean()
        }
        
        return loss_list
    
    def triplet_loss(result_dict):
        margin = cfg.training.loss.margin
        anchor = result_dict["current_state_rep"]
        positive = result_dict["next_state_rep"]
        batch_size = anchor.shape[0]
        negative = torch.roll(positive, shifts=random.randint(1, batch_size), dims=0)

        distance_positive = torch.nn.functional.pairwise_distance(anchor, positive, p=2)
        distance_negative = torch.nn.functional.pairwise_distance(anchor, negative, p=2)
        triplet_loss = torch.nn.functional.relu(distance_positive - distance_negative + margin)

        loss_list = {
            "CL": triplet_loss.mean()
        }
        
        return  loss_list
    
    loss_func_list = {
        "mse": mse_loss,
        "mse+reg": mse_reg_loss,
        "mae": mae_loss,
        "mae+reg": mae_reg_loss,
        "cl": cl_loss,
        "nce": nce_loss,
        "triplet": triplet_loss,
    }
    
    loss_type_list = {
        "mse": ["mse"],
        "mse+reg": ["mse", "reg"],
        "mae": ["mae"],
        "mae+reg": ["mae", "reg"],
        "cl": ["CL"],
        "nce": ["CL"],
        "triplet": ["CL"],
    }
    
    loss_name = cfg.training.loss.name.lower()
    if loss_name in loss_func_list.keys():
        loss_func = loss_func_list[loss_name]
    else:
        raise ValueError(f"Loss {loss_name} not found")
    
    if loss_name in loss_type_list.keys():
        loss_type = loss_type_list[loss_name]
        loss_type.append("total")
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