import torch
import torch.nn as nn 
import torch.nn.functional as F

class LSDE(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(LSDE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = cfg.model.layers + 1
        self.residual = cfg.model.residual
        
        if cfg.training.repeat:
            self.representation_dim = input_dim // 4
        else:
            self.representation_dim = (input_dim - 2) // 2
        
        
        # Drift layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, cfg.model.hidden_dim),
            nn.ReLU()
        )
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
        self.noise_dim = self.representation_dim if cfg.training.repeat else 1
        self.noise_encoder = nn.Linear(self.noise_dim, cfg.model.hidden_dim)
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
        step = x[:, self.representation_dim * 2 : self.representation_dim * 3]
        noise = self.noise_encoder(step)
        
        for idx in range(self.layer):
            if self.residual and idx != self.layer-1:
                drift = drift + self.drift_layers[idx](drift)
                noise = noise + self.noise_layers[idx](noise)
            else:
                drift = self.drift_layers[idx](drift)
                noise = self.noise_layers[idx](noise)
        
        return drift, noise
    
class LNSDE(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(LNSDE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer = cfg.model.layers + 1
        self.residual = cfg.model.residual
        
        if cfg.training.repeat:
            self.representation_dim = input_dim // 4
        else:
            self.representation_dim = (input_dim - 2) // 2
        
        
        # Drift layers
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, cfg.model.hidden_dim),
            nn.ReLU()
        )
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
        self.noise_dim = self.representation_dim if cfg.training.repeat else 1
        self.noise_encoder = nn.Linear(self.noise_dim, cfg.model.hidden_dim)
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
        current_state = x[:, :self.representation_dim]
        step = x[:, self.representation_dim * 2 : self.representation_dim * 3]
        noise = self.noise_encoder(step)
        
        for idx in range(self.layer):
            if self.residual and idx != self.layer-1:
                drift = drift + self.drift_layers[idx](drift)
                noise = noise + self.noise_layers[idx](noise)
            else:
                drift = self.drift_layers[idx](drift)
                noise = self.noise_layers[idx](noise)
        
        noise = noise * current_state
        
        return drift, noise