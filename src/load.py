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
    # data_path = f"/home/shpark/prj-cmd/simulation/dataset/alanine/300.0/alpha_P.pt"
    data_path = f"/home/shpark/prj-cmd/simulation/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.state}-{cfg.data.index}.pt"
    
    train_dataset = torch.load(f"{data_path}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        # num_workers=-1,
    )
    
    return train_loader

def load_process_data(cfg):
    def process_data(current_state, goal_state, step, temperature):    
        batch_size = current_state.shape[0]
        
        current_state = current_state.reshape(batch_size, -1)
        goal_state = goal_state.reshape(batch_size, -1)
        step = torch.tensor(step).to(current_state.device).repeat(batch_size, 1)
        temperature = torch.tensor(temperature).to(current_state.device).repeat(batch_size, 1)
        
        processed_state = torch.cat([
            current_state,
            goal_state,
            step,
            temperature
        ], dim=1)
        
        del current_state, goal_state, step, temperature
        return processed_state
    
    return process_data

def load_process_prediction(cfg):
    atom_num = cfg.data.atom
    
    def process_prediction(prediction, shape):
        prediction = prediction.reshape(shape[0], atom_num, 3)
        return prediction
    
    return process_prediction