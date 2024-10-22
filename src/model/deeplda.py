import torch
import torch.nn as nn 
import torch.nn.functional as F

class DeepLDA(nn.Module):
    def __init__(self, cfg, input_dim, output_dim):
        super(DeepLDA, self).__init__()

        # DUMMY
    
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
    