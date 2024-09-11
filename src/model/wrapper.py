import torch
import torch.nn as nn

from . import *

class ModelWrapper(nn.Module):
    def __init__(self, cfg, device):
        super(ModelWrapper, self).__init__()
        
        self.atom_num = cfg.data.atom
        self.batch_size = cfg.training.batch_size
        self.output_dim = cfg.model.hidden_dim[-1]
        self.model = self.load_model(cfg).to(device)
        self.noise_scale = cfg.training.noise_scale
        
        # self.alpha = torch.nn.Parameter(torch.ones(1)) .to(device)
        self.alpha = 1
        self.mu = nn.Linear(self.output_dim, self.atom_num * 3).to(device)
        self.log_var = nn.Linear(self.output_dim, self.atom_num * 3).to(device)
    
    def load_model(self, cfg):
        model_dict = {
            "mlp": MLP,
        }
        
        model_name = cfg.model.name.lower() 
        if model_name in model_dict.keys():
            model = model_dict[model_name](cfg=cfg)
        else:
            raise ValueError(f"Model {model_name} not found")

        return model
    
    def save_model(self, path, epoch):
        torch.save(self.model.state_dict(), f"{path}/model-{epoch}.pt")
        
    def load_from_checkpoint(self, path):
        self.model.load_state_dict(torch.load(f"{path}/model.pt"))
    
    def forward(self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        step: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        batch_size = current_state.shape[0]
        temperature = torch.tensor(temperature).to(current_state.device).repeat(current_state.shape[0], 1)
        
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
        state_offset = state_offset.reshape(batch_size, self.atom_num, 3)
        
        return state_offset, mu, log_var
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var)
        eps = torch.randn_like(std) * self.noise_scale
        
        return mu + eps * std
