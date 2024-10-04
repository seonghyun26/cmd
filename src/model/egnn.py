import torch
import torch.nn as nn 
import torch.nn.functional as F

from egnn_pytorch import EGNN as EGGNLayer

ATOM_LIST = ["H", "C", "N", "O"]
ALANINE = [
    0, 1, 0, 0, 1,
    3, 2, 0, 1, 0,
    1, 0, 0, 0, 1,
    3, 2, 0, 1, 0,
    0, 0
]

class EGNN(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(EGNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = cfg.model.hidden_dim
        
        if cfg.data.molecule == "alanine":
            self.atom_list = ALANINE
            self.atom_encoder = nn.Embedding(num_embeddings=4, embedding_dim=cfg.model.hidden_dim - 2)
        else:
            raise ValueError(f"Molecule {cfg.data.molecule} not found")
        
        self.layers = nn.ModuleList()
        for i in range(cfg.model.layers):
            self.layers.append(EGGNLayer(dim = cfg.model.hidden_dim))
    
    def forward(self, x):
        current_coord = x[:, :66].reshape(-1, 22, 3)
        goal_coord = x[:, 66:132].reshape(-1, 22, 3)
        step = x[:, 132].reshape(-1, 1).repeat(1, 22).unsqueeze(2)
        temperature = x[:, 133].reshape(-1, 1).repeat(1, 22).unsqueeze(2)
        atom_feature = self.atom_encoder(torch.tensor(self.atom_list).to(x.device)).repeat(x.shape[0], 1, 1).to(x.device)
        
        atom_feature = torch.cat([atom_feature, step, temperature], dim=2)
        coord = goal_coord - current_coord
        
        for idx, layer in enumerate(self.layers):
            atom_feature, coord = layer(atom_feature, coord)
        
        return coord.reshape(-1, 66)