import torch
import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        
        self.input_dim = cfg.data.atom * 3 * 2 + 2
        self.output_dim = cfg.data.atom * 3
        
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, cfg.model.hidden_dim),
            nn.ReLU()
        ])
        for i in range(cfg.model.layers):
            self.layers.append(nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(cfg.model.hidden_dim, self.output_dim))
    
    def forward(self, x, goal_state, step, temperature):
        org_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        goal_state = goal_state.reshape(x.shape[0], -1)
        step = torch.tensor(step).to(x.device).repeat(x.shape[0], 1)
        temperature = torch.tensor(temperature).to(x.device).repeat(x.shape[0], 1)
        
        x = torch.cat([x, goal_state, step, temperature], dim=1)
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.reshape(org_shape)
        return x