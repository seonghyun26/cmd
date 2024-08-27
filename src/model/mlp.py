import torch.nn as nn 
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, cfg):
        super(MLP, self).__init__()
        
        self.input_dim = cfg.data.atom * 3 * 2 + 2
        self.output_dim = cfg.data.atom * 3
        
        self.layers = [nn.Linear(input_dim, cfg.model.hidden_dim)]
        for i in range(cfg.model.layers):
            self.layers.append(nn.Linear(cfg.model.hidden_dim, cfg.model.hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(cfg.model.hidden_dim, output_dim))
    
    def forward(self, x, goal_state, step, temperature):
        org_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        temperature = torch.tensor(temperature).to(x.device)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        x = x.reshape(org_shape)
        return x