import torch
import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = cfg.model.residual
        
        # self.encoder = nn.Linear(self.input_dim, cfg.model.hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, cfg.model.hidden_dim),
            nn.ReLU()
        )
        self.layers = nn.ModuleList()
        for i in range(cfg.model.layers):
            self.layers.append(nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim))
            self.layers.append(nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(cfg.model.hidden_dim, self.output_dim),
            nn.ReLU()
        )
        
        if cfg.model.init == "xavier":
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = self.encoder(x)
        
        for idx, layer in enumerate(self.layers):
            if self.residual:
                x_input = x
                x = layer(x)
                x = x + x_input
            else:
                x = layer(x)
        
        x = self.decoder(x)
        # x = nn.ReLU()(x)
        
        return x