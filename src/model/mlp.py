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
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x