import wandb
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from .model import *
from .data import *


def load_model(cfg, device):
    if cfg.model.name == "MLP":
        model = MLP(cfg)
    else:
        raise ValueError(f"Model {cfg.model.name} not found")
    
    model = model.to(device)
    return model

def load_loss(cfg):
    if cfg.training.loss == "MSE":
        loss = nn.MSELoss()
    else:
        raise ValueError(f"Loss {cfg.training.loss} not found")
    
    return loss

def load_data(cfg):
    # data_path = f"/home/shpark/prj-cmd/simulation/dataset/alanine/273.0/c5.pt"
    data_path = f"/home/shpark/prj-cmd/simulation/dataset/alanine/300.0/alpha_P.pt"
    # data_path = f"./dataset/{cfg.data.name}/{cfg.data.temperature}/{cfg.data.state}.pt"
    
    train_dataset = torch.load(f"{data_path}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        num_workers=8,
        shuffle=True
    )
    
    return train_loader