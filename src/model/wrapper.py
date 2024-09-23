import torch
import torch.nn as nn

import numpy as np
import bgflow as bg

from . import *

class ModelWrapper(nn.Module):
    def __init__(self, cfg, device):
        super(ModelWrapper, self).__init__()
        
        self.device = device
        self.atom_num = cfg.data.atom
        self.batch_size = cfg.training.batch_size
        self.output_dim = cfg.model.hidden_dim[-1]
        self.noise_scale = cfg.training.noise_scale
        self.alpha = 1
        self.cfg = cfg
        # self.alpha = torch.nn.Parameter(torch.ones(1)) .to(device)
        
        self.model = self.load_model(cfg).to(device)

    
    def load_model(self, cfg):
        model_dict = {
            "mlp": MLP,
        }
        
        if cfg.model.transform == "ic":
            self.transform = load_internal_coordinate_transform()
            self.representation_dim = (cfg.data.atom - 5) * 3 + (cfg.data.atom - 7)
        elif cfg.model.transform == "ic2" or cfg.model.transform == "ic4":
            self.transform = load_internal_coordinate_transform()
            self.representation_dim = cfg.data.atom - 5
        elif cfg.model.transform == "ic3":
            self.transform = load_internal_coordinate_transform()
            self.representation_dim = (cfg.data.atom - 5) * 2
        elif cfg.model.transform == "xyz" and cfg.data.molecule == "alanine":
            self.representation_dim = cfg.data.atom * 3
        elif cfg.model.transform == "xyz" and cfg.data.molecule == "double-well":
            self.representation_dim = cfg.data.atom 
        else:
            raise ValueError(f"Transform {cfg.model.transform} not found")
        self.input_dim = self.representation_dim * 2 + 2
            
        model_name = cfg.model.name.lower() 
        if model_name in model_dict.keys():
            model = model_dict[model_name](cfg=cfg, input_dim=self.input_dim)
        else:
            raise ValueError(f"Model {model_name} not found")
        
        self.mu = nn.Linear(self.output_dim, self.representation_dim).to(self.device)
        self.log_var = nn.Linear(self.output_dim, self.representation_dim).to(self.device)
        
        return model
    
    def save_model(self, path, epoch):
        torch.save(self.model.state_dict(), f"{path}/model-{epoch}.pt")
        
    def load_from_checkpoint(self, path):
        self.model.load_state_dict(torch.load(f"{path}/model.pt"))
    
    def forward(self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        step: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        if hasattr(self, "transform"):
            current_state = torch.cat(self.transform.forward(current_state)[:-1], dim=1)
            goal_state = torch.cat(self.transform.forward(goal_state)[:-1], dim=1)
            if self.cfg.model.transform == "ic2":
                current_state_bond = current_state[:, :17]
                current_state_angle = current_state[:, 17:34]
                current_state_torsion = current_state[:, 34:51]
                current_state_fixed_x = current_state[:, 51:66]
                current_state = current_state[:, 34:51]
                # goal_state_bond = goal_state[:, :17]
                # goal_state_angle = goal_state[:, 17:34]
                goal_state = goal_state[:, 34:51]
            elif self.cfg.model.transform == "ic3":
                current_state_bond = current_state[:, :17]
                current_state_angle_torsion = current_state[:, 17:51]
                current_state_fixed_x = current_state[:, 51:66]
                current_state = current_state[:, 17:51]
                goal_state = goal_state[:, 17:51]
            elif self.cfg.model.transform == "ic4":
                current_state_bond = current_state[:, :17]
                current_state_angle = current_state[:, 17:34]
                current_state_torsion = current_state[:, 34:51]
                current_state_fixed_x = current_state[:, 51:66]
                current_state = current_state[:, 34:51]
                # goal_state_bond = goal_state[:, :17]
                # goal_state_angle = goal_state[:, 17:34]
                goal_state = goal_state[:, 34:51]
        
        batch_size = current_state.shape[0]
        # temperature = torch.tensor(temperature).to(current_state.device).repeat(current_state.shape[0], 1)
        
        conditions = torch.cat([
            current_state.reshape(batch_size, -1),
            goal_state.reshape(batch_size, -1),
            step.reshape(batch_size, -1),
            temperature.reshape(batch_size, -1)
        ], dim=1)
        latent = self.model(conditions)
        
        mu = self.mu(latent)
        log_var = self.log_var(latent)
        log_var = self.alpha * log_var
        log_var = torch.clamp(log_var, max=10)
        state_offset = self.reparameterize(mu, log_var)
        
        if hasattr(self, "transform"):
            if self.cfg.model.transform == "ic":
                state_offset = torch.cat(self.transform._inverse(
                    state_offset[:, :17],
                    state_offset[:, 17:34],
                    state_offset[:, 34:51],
                    state_offset[:, 51:66]
                ), dim=1)[:, :-1]
            elif self.cfg.model.transform == "ic2":
                predicted_state_torsion = (current_state_torsion + state_offset) % np.pi
                state_offset = torch.cat(self.transform._inverse(
                    current_state_bond,
                    current_state_angle,
                    predicted_state_torsion,
                    current_state_fixed_x
                ), dim=1)[:, :-1]
            elif self.cfg.model.transform == "ic3":
                state_offset = np.pi * torch.cos(state_offset)
                predicted_state_angle_torsion = (current_state_angle_torsion + state_offset) % (2 *np.pi) - np.pi
                state_offset = torch.cat(self.transform._inverse(
                    current_state_bond,
                    predicted_state_angle_torsion[:, :17],
                    predicted_state_angle_torsion[:, 17:],
                    current_state_fixed_x
                ), dim=1)[:, :-1]
            elif self.cfg.model.transform == "ic4":
                state_offset = np.pi * torch.cos(state_offset)
                predicted_state_torsion = (current_state_torsion + state_offset) % (2 *np.pi) - np.pi
                state_offset = torch.cat(self.transform._inverse(
                    current_state_bond,
                    current_state_angle,
                    predicted_state_torsion,
                    current_state_fixed_x
                ), dim=1)[:, :-1]
            else:
                raise ValueError(f"Transform {self.cfg.model.transform} not found")
        if self.cfg.data.molecule == "alanine":
            state_offset = state_offset.reshape(batch_size, self.atom_num, 3)
        elif self.cfg.data.molecule == "double-well":
            state_offset = state_offset.reshape(batch_size, self.atom_num)
        
        return state_offset, mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var)
        eps = torch.randn_like(std) * self.noise_scale
        
        return mu + eps * std

def load_internal_coordinate_transform():
    z_matrix = np.array([
        [ 0,  1,  4,  6],
        [ 1,  4,  6,  8],
        [ 2,  1,  4,  0],
        [ 3,  1,  4,  0],
        [ 4,  6,  8, 14],
        [ 5,  4,  6,  8],
        [ 7,  6,  8,  4],
        [11, 10,  8,  6],
        [12, 10,  8, 11],
        [13, 10,  8, 11],
        [15, 14,  8, 16],
        [16, 14,  8,  6],
        [17, 16, 14, 15],
        [18, 16, 14,  8],
        [19, 18, 16, 14],
        [20, 18, 16, 19],
        [21, 18, 16, 19]
    ])
    rigid_block = np.array([ 6,  8,  9, 10, 14])
    
    coordinate_transform = bg.RelativeInternalCoordinateTransformation(
        z_matrix=z_matrix,
        fixed_atoms=rigid_block,
        normalize_angles = False,
        eps = 1e-10, 
    )
    
    return coordinate_transform

def load_custom_ic_transform():
    z_matrix = np.array([
        [ 0,  1,  4,  6],
        [ 1,  4,  6,  8],
        [ 2,  1,  4,  0],
        [ 3,  1,  4,  0],
        [ 4,  6,  8, 14],
        [ 5,  4,  6,  8],
        [ 7,  6,  8,  4],
        [11, 10,  8,  6],
        [12, 10,  8, 11],
        [13, 10,  8, 11],
        [15, 14,  8, 16],
        [16, 14,  8,  6],
        [17, 16, 14, 15],
        [18, 16, 14,  8],
        [19, 18, 16, 14],
        [20, 18, 16, 19],
        [21, 18, 16, 19]
    ])
    rigid_block = np.array([ 6,  8,  9, 10, 14])
    
    return