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
    
    
class CVMLP_V1(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(CVMLP_V1, self).__init__()

        # Output dimension is the same as the representation dimension
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.residual = cfg.model.residual
        
        self.encoder = nn.Sequential(
            nn.Linear(4, cfg.model.hidden_dim),
            nn.ReLU()
        )
        
        self.layers = nn.ModuleList()
        for i in range(cfg.model.layers):
            self.layers.append(nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim))
            self.layers.append(nn.ReLU())
        
        self.decoder = nn.Sequential(
            nn.Linear(cfg.model.hidden_dim, 1),
            nn.ReLU()
        )
        
        self.phi_angle = [1, 6, 8, 14]
        self.psi_angle = [6, 8, 14, 16]
    
    def forward(self, x):
        """
        Args:
            x (_type_): current_state, goal_state, step, temperature

        Returns:
            _type_: _description_
        """
        # current_state_cv = compute_phi_psi(self.cfg, current_state.detach().cpu()).to(x.device)
        current_state = x[:, :self.output_dim]
        current_state = current_state.reshape(-1, self.cfg.data.atom, 3)
        current_state.requires_grad = True
        phi_current_state = compute_dihedral_torch(current_state[:, self.phi_angle]).unsqueeze(1)
        psi_current_state = compute_dihedral_torch(current_state[:, self.psi_angle]).unsqueeze(1)
        goal_state = x[:, self.output_dim:self.output_dim * 2]
        goal_state = goal_state.reshape(-1, self.cfg.data.atom, 3)
        phi_goal_state = compute_dihedral_torch(goal_state[:, self.phi_angle]).unsqueeze(1)
        psi_goal_state = compute_dihedral_torch(goal_state[:, self.psi_angle]).unsqueeze(1)
                
        # processed_input = torch.cat([phi, psi, x[:, self.output_dim * 2:]], dim=1)
        phi_force = torch.autograd.grad(outputs=phi_goal_state - phi_current_state, inputs=current_state, grad_outputs=torch.ones_like(phi_goal_state), create_graph=True)[0]
        psi_force = torch.autograd.grad(outputs=psi_goal_state - psi_current_state, inputs=current_state, grad_outputs=torch.ones_like(psi_goal_state), create_graph=True)[0]
        
        # processed_input = torch.cat([phi_current_state, psi_current_state, phi_goal_state, psi_goal_state], dim=1)
        # x = self.encoder(processed_input)
        
        # for idx, layer in enumerate(self.layers):
        #     if self.residual:
        #         x_input = x
        #         x = layer(x)
        #         x = x + x_input
        #     else:
        #         x = layer(x)
        
        # x = self.decoder(x)
        
    
        return 10 * (phi_force + psi_force)
    

class CVMLP(nn.Module):
    def __init__(self, input_dim, data_dim, **kwargs):
        super(CVMLP, self).__init__()

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
    
    def forward(self,
            x: torch.Tensor,
            transformed: bool = False
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

    
    def coordinate2distance(self,
            positions: torch.Tensor
        ) -> torch.Tensor:
        
        num_heavy_atoms = 10
        distances = []
        
        for position in positions:
            heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
            distance = []
            for i in range(num_heavy_atoms):
                for j in range(i + 1, num_heavy_atoms):
                    distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
            distance = torch.stack(distance)
            distances.append(distance)
        distances = torch.stack(distances)
            
        return distances