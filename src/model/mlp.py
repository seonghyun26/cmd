import torch
import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, cfg, input_dim):
        super(MLP, self).__init__()
        
        assert cfg.model.layers == len(cfg.model.hidden_dim), "The number of layers should be the same as the number of hidden dimensions"
        
        self.input_dim = input_dim
        self.output_dim = cfg.model.hidden_dim[-1]
        
        self.layers = nn.ModuleList([
            nn.Linear(self.input_dim, cfg.model.hidden_dim[0]),
            nn.ReLU()
        ])
        for i in range(cfg.model.layers-1):
            self.layers.append(nn.Linear(cfg.model.hidden_dim[i], cfg.model.hidden_dim[i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(cfg.model.hidden_dim[-1], self.output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x