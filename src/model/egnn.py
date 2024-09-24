import torch
import torch.nn as nn 
import torch.nn.functional as F

from egnn_pytorch import EGNN as EGGNLayer

class EGNN(nn.Module):
    def __init__(self, cfg, input_dim):
        super(EGNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim
        
        self.encoder = nn.ModuleList([
            nn.Linear(self.input_dim, cfg.model.hidden_dim),
            nn.ReLU()
        ])
        self.layers = nn.ModuleList()
        for i in range(cfg.model.layers):
            self.layers.append(EGGNLayer(dim = cfg.model.hidden_dim))
        self.decoder = nn.ModuleList([
            nn.Linear(cfg.model.hidden_dim, self.output_dim),
            nn.ReLU()
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x