import torch
import torch.nn as nn

import numpy as np
import bgflow as bg

from mlcolvar.cvs import DeepLDA, AutoEncoderCV, VariationalAutoEncoderCV

from . import *


model_dict = {
    "mlp": MLP,
    "egnn": EGNN,
    "sdenet": SDENet,
    "lsde": LSDE,
    "lnsde": LNSDE,
    "cvmlp": CVMLP,
    "deeplda": DeepLDA,
    "aecv": AutoEncoderCV,
    "vaecv": VariationalAutoEncoderCV,
    "betavae": VariationalAutoEncoderCVBeta
}


class ModelWrapper(nn.Module):
    def __init__(self, cfg, device):
        super(ModelWrapper, self).__init__()
        
        self.cfg = cfg
        self.device = device
        self.model = self.load_model(cfg).to(device)
    
    def load_model(self, cfg):
        if cfg.data.molecule in ["alanine", "chignolin"]:
            self.data_dim = cfg.data.atom * 3
        elif cfg.data.molecule == "double-well":
            self.data_dim = cfg.data.atom 
        else:
            raise ValueError(f"Molecule {cfg.data.molecule} not defined")
        
        self.model_name = cfg.model.name.lower() 
        if self.model_name in ["deeplda", "aecv", "vaecv", "beta-vae"]:
            model = model_dict[self.model_name](**cfg.model.params)
        elif self.model_name in model_dict.keys():
            model = model_dict[self.model_name](self.data_dim, **cfg.model.params)
        else:
            raise ValueError(f"Model {self.model_name} not found")
        
        if self.model_name != "cvmlp":
            self.mu = nn.Linear(self.data_dim, self.data_dim).to(self.device)
            self.log_var = nn.Linear(self.data_dim, self.data_dim).to(self.device)
        
        return model
    
    def save_model(self, path, epoch):
        torch.save(self.model.state_dict(), f"{path}/model-{epoch}.pt")
        
    def load_from_checkpoint(self, path):
        self.model.load_state_dict(torch.load(f"{path}"))
    
    def forward(self,
        current_state: torch.Tensor,
        next_state: torch.Tensor,
        temperature: torch.Tensor,
        goal_state: torch.Tensor = None,
        step: torch.Tensor = None,
    ) -> torch.Tensor:
        # prediction by model
        batch_size = current_state.shape[0]
        if self.model_name in ["cvmlp"]:
            if self.cfg.model.input == "distance":
                current_state = x[:, :-1].reshape(batch_size, -1)
                temperature = x[:, -1].unsqueeze(-1)
                current_state= self.coordinate2distance(current_state)                
            
            current_state_conditions = torch.cat([
                current_state.reshape(batch_size, -1),
                temperature.reshape(batch_size, -1)
            ], dim=1)
            current_state_latent = self.model(current_state_conditions)
            next_state_conditions = torch.cat([
                next_state.reshape(batch_size, -1),
                temperature.reshape(batch_size, -1)
            ], dim=1)
            next_state_latent = self.model(next_state_conditions)
        
        else:
            conditions = torch.cat([
                current_state.reshape(batch_size, -1),
                next_state.reshape(batch_size, -1),
                temperature.reshape(batch_size, -1)
            ], dim=1)
            latent = self.model(conditions)
        
        # Process results
        result_dict = {}
        if self.model_name == "cvmlp":
            result_dict["current_state_rep"] = current_state_latent
            result_dict["next_state_rep"] = next_state_latent
        elif self.model_name in ["sdenet", "lsde", "lnsde"]:
            mu, log_var = latent
            log_var = torch.clamp(log_var, max=10)
            result_dict["mu"] = mu
            result_dict["log_var"] = log_var
            pred = self.reparameterize(mu, log_var)
            result_dict["pred"] = pred
        else:
            mu = self.mu(latent)
            log_var = torch.clamp(self.log_var(latent), max=10)
            result_dict["mu"] = mu
            result_dict["log_var"] = log_var
            pred = self.reparameterize(mu, log_var)
            result_dict["pred"] = pred    
        
        return result_dict
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var)
        eps = torch.randn_like(std) * self.noise_scale
        
        return mu + eps * std
