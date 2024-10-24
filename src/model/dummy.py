import torch.nn as nn 

class Dummy(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(Dummy, self).__init__()
    
    def forward(self, x):
        return x
    