import torch
import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        
        assert cfg.layers == len(cfg.hidden_dim), "The number of layers should be the same as the number of hidden dimensions"
        
        self.input_dim = cfg.input_dim
        self.output_dim = cfg.output_dim
        
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, cfg.hidden_dim[0]),
            nn.ReLU()
        ])
        for i in range(cfg.layers-1):
            self.layers.append(nn.Linear(cfg.hidden_dim[i], cfg.hidden_dim[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(cfg.hidden_dim[-1], self.output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x