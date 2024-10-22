import torch
import torch.nn as nn

import numpy as np
import bgflow as bg

from . import *


model_dict = {
    "mlp": MLP,
    "egnn": EGNN,
    "sdenet": SDENet,
    "lsde": LSDE,
    "lnsde": LNSDE,
    "cv-mlp": CVMLP,
    "deeplda": DeepLDA
}


class ModelWrapper(nn.Module):
    def __init__(self, cfg, device):
        super(ModelWrapper, self).__init__()
        
        self.cfg = cfg
        self.device = device
        self.atom_num = cfg.data.atom
        self.batch_size = cfg.training.batch_size
        self.noise_scale = cfg.training.noise_scale
        
        self.model = self.load_model(cfg).to(device)
    
    def load_model(self, cfg):
        if cfg.model.transform == "xyz" and cfg.data.molecule == "alanine":
            self.representation_dim = cfg.data.atom * 3
        elif cfg.model.transform == "xyz" and cfg.data.molecule == "double-well":
            self.representation_dim = cfg.data.atom 
        else:
            raise ValueError(f"Transform {cfg.model.transform} not found")
        if cfg.training.repeat:
            self.condition_dim = self.representation_dim * 4
        else:
            self.condition_dim = self.representation_dim * 2 + 2
            
        self.model_name = cfg.model.name.lower() 
        if self.model_name in model_dict.keys():
            model = model_dict[self.model_name](
                cfg=cfg,
                input_dim=self.condition_dim,
                output_dim=self.representation_dim
            )
        else:
            raise ValueError(f"Model {self.model_name} not found")
        
        if self.model_name != "cv-mlp":
            self.mu = nn.Linear(self.representation_dim, self.representation_dim).to(self.device)
            self.log_var = nn.Linear(self.representation_dim, self.representation_dim).to(self.device)
        
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
        batch_size = current_state.shape[0]
        conditions = torch.cat([
            current_state.reshape(batch_size, -1),
            goal_state.reshape(batch_size, -1),
            step.reshape(batch_size, -1),
            temperature.reshape(batch_size, -1)
        ], dim=1)
        latent = self.model(conditions)
        
        if self.model_name in ["sdenet", "lsde", "lnsde"]:
            mu, log_var = latent
            log_var = torch.clamp(log_var, max=10)
            state_offset = self.reparameterize(mu, log_var)
        elif self.model_name == "cv-mlp":
            state_offset = latent
            mu, log_var = 0, 0
        else:
            mu = self.mu(latent)
            log_var = torch.clamp(self.log_var(latent), max=10)
            state_offset = self.reparameterize(mu, log_var)
        
        # Reshape state_offset by molecule type
        if self.cfg.data.molecule == "alanine":
            state_offset = state_offset.reshape(batch_size, self.atom_num, 3)
        elif self.cfg.data.molecule == "double-well":
            state_offset = state_offset.reshape(batch_size, self.atom_num)
        
        return state_offset, mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var)
        eps = torch.randn_like(std) * self.noise_scale
        
        return mu + eps * std
