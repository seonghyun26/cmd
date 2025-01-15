import torch
import torch.nn as nn 
import torch.nn.functional as F

from ..metric import compute_phi_psi, compute_dihedral_torch


ALANINE_HEAVY_ATOM_IDX = [
    1, 4, 5, 6, 8, 10, 14, 15, 16, 18
]


class MLP(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = cfg.model.residual
        
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
        
        return x
    
class CVMLP(nn.Module):
    def __init__(self, input_dim, data_dim, **kwargs):
        super(CVMLP, self).__init__()

        self.params = kwargs
        self.data_dim = data_dim
        self.input_dim = input_dim
        self.residual = self.params["residual"]
        self.output_dim = self.params["output_dim"]
        self.params["layer_num"] = len(self.params["hidden_dim"])
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.params["hidden_dim"][0]))
        self.layers.append(nn.ReLU())
        for i in range(self.params["layer_num"] - 1):
            self.layers.append(nn.Linear(self.params["hidden_dim"][i], self.params["hidden_dim"][i+1]))
            if self.params["layernorm"] and i == self.params["layer_num"] - 2:
                self.layers.append(nn.LayerNorm(self.params["hidden_dim"][i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.params["hidden_dim"][-1], self.output_dim))
        
        # if self.params["normalized"]:
        #     class CVNormalization(nn.Module):
        #         def forward(self, x):
        #             return F.normalize(x, p=2, dim=1)
        #     self.layers.append(CVNormalization())
    
    def forward(self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
            Args:
                x (batch x representation): state, temperature
            Returns:
                cv (batch x 1): machined learned collective variables for given molecular configuration
        """        
        z = x
        
        for idx, layer in enumerate(self.layers):
            # if self.residual and idx % 2 == 0 and idx > 2:
            #     z_input = z
            #     z = layer(z)
            #     z = z + z_input
            # else:
            #     z = layer(z)
            z = layer(z)
                
        return z

class CVMLPBN(nn.Module):
    def __init__(self, input_dim, data_dim, **kwargs):
        super(CVMLPBN, self).__init__()

        self.params = kwargs
        self.data_dim = data_dim
        self.input_dim = input_dim
        self.output_dim = self.params["output_dim"]
        self.params["layer_num"] = len(self.params["hidden_dim"])
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.params["hidden_dim"][0]))
        self.layers.append(nn.ReLU())
        for i in range(self.params["layer_num"] - 1):
            self.layers.append(nn.Linear(self.params["hidden_dim"][i], self.params["hidden_dim"][i+1]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.params["hidden_dim"][-1], self.output_dim))
        self.layers.append(nn.BatchNorm1d(self.output_dim))
    
    def forward(self,
            x: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            x (batch x representation): state, temperature
        Returns:
            cv (batch x 1): machined learned collective variables for given molecular configuration
        """        
        z = x
        
        for idx, layer in enumerate(self.layers):
            z = layer(z)
        
        return z
