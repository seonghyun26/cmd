import torch
import torch.nn as nn 
import torch.nn.functional as F

class SDENet(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(SDENet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = cfg.model.layers + 1
        self.residual = cfg.model.residual
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, cfg.model.hidden_dim),
            nn.ReLU()
        )
        
        # Drift layers
        self.drift_layers = nn.ModuleList()
        for i in range(cfg.model.layers):
            self.drift_layers.append(nn.Sequential(
                nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim),
                nn.ReLU()
            ))
        self.drift_layers.append(nn.Sequential(
            nn.Linear(cfg.model.hidden_dim, self.output_dim),
            nn.ReLU()
        ))

        # Noise layers
        self.noise_layers = nn.ModuleList()
        for i in range(cfg.model.layers):
            self.noise_layers.append(nn.Sequential(
                nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim),
                nn.ReLU()
            ))
        self.noise_layers.append(nn.Sequential(
            nn.Linear(cfg.model.hidden_dim, self.output_dim),
            nn.ReLU()
        ))
    
    def forward(self, x):
        z = self.encoder(x)
        drift = z
        noise = z
        
        for idx in range(self.layer):
            if self.residual and idx != self.layer-1:
                drift = drift + self.drift_layers[idx](drift)
                noise = noise + self.noise_layers[idx](noise)
            else:
                drift = self.drift_layers[idx](drift)
                noise = self.noise_layers[idx](noise)
        
        return drift, noise