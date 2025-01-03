import torch
import torch.nn as nn

import numpy as np

from mlcolvar.cvs import DeepLDA, DeepTDA, DeepTICA, AutoEncoderCV, VariationalAutoEncoderCV

from . import *

model_dict = {
    "mlp": MLP,
    "egnn": EGNN,
    "sdenet": SDENet,
    "lsde": LSDE,
    "lnsde": LNSDE,
    "cvmlp": CVMLP,
    "cvmlp-bn": CVMLPBN,
    "deeplda": DeepLDA,
    "deeptda": DeepTDA,
    "deeptica": DeepTICA,
    "aecv": AutoEncoderCV,
    "vaecv": VariationalAutoEncoderCV,
    "betavae": VariationalAutoEncoderCVBeta
}

COLVAR_METHODS = ["deeplda", "deeptda", "deeptica", "aecv", "vaecv", "beta-vae"]

class ModelWrapper(nn.Module):
    def __init__(self, cfg, device):
        super(ModelWrapper, self).__init__()
        
        self.cfg = cfg
        self.device = device
        self.model = self.load_model(cfg).to(device)
    
    def _set_input_dim(self, cfg, data_dim):
        input_dim = None
        
        if cfg.data.molecule == "alanine":
            if cfg.model.input == "distance":
                input_dim = 45 + 1
            elif cfg.model.input == "coordinate":
                input_dim = data_dim + 1
            else:
                raise ValueError(f"Input type {cfg.model.input} not found")
        
        elif cfg.data.moelcule == "chignolin":
            raise ValueError(f"Molecule {cfg.data.molecule} TBA...")
        
        else:
            raise ValueError(f"Molecule {cfg.data.molecule} not found")
        
        assert input_dim is not None, f"Input dimension not set for {cfg.data.molecule}"
        
        return input_dim
    
    def load_model(self, cfg):
        if cfg.data.molecule in ["alanine", "chignolin"]:
            self.data_dim = cfg.data.atom * 3
        else:
            raise ValueError(f"Molecule {cfg.data.molecule} not defined")
        
        self.model_name = cfg.model.name.lower() 
        if self.model_name == "deeptica":
            model = DeepTICA(
                layers = cfg.model.params.layers,
                n_cvs = cfg.model.params.n_cvs,
                options = {'nn': {'activation': 'shifted_softplus'} }
            )
        elif self.model_name in COLVAR_METHODS:
            model = model_dict[self.model_name](**cfg.model.params)
        elif self.model_name in model_dict.keys():
            model = model_dict[self.model_name](
                input_dim = self._set_input_dim(cfg, self.data_dim),
                data_dim = self.data_dim,
                **cfg.model.params
            )
        else:
            raise ValueError(f"Model {self.model_name} not found")
        
        return model
    
    def save_model(self, path, epoch):
        torch.save(self.model.state_dict(), f"{path}/model-{epoch}.pt")
        
    def load_from_checkpoint(self, path):
        self.model.load_state_dict(torch.load(f"{path}"))
    
    def forward(self,
        current_state: torch.Tensor,
        positive_sample: torch.Tensor,
        negative_sample: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = current_state.shape[0]
        scale = self.cfg.data.scale
        
        # Process input
        current_state_conditions = torch.cat([
            current_state.reshape(batch_size, -1) * scale,
            temperature.reshape(batch_size, -1)
        ], dim=1)
        current_state_representation = self.model(current_state_conditions)
        positive_sample_conditions = torch.cat([
            positive_sample.reshape(batch_size, -1) * scale,
            temperature.reshape(batch_size, -1)
        ], dim=1)
        positive_sample_representation = self.model(positive_sample_conditions)
        negative_sample_conditions = torch.cat([
            negative_sample.reshape(batch_size, -1) * scale,
            temperature.reshape(batch_size, -1)
        ], dim=1)
        negative_sample_representation = self.model(negative_sample_conditions)
        
        # Process results
        result_dict = {}
        result_dict["current_state_rep"] = current_state_representation
        result_dict["positive_sample_rep"] = positive_sample_representation
        result_dict["negative_sample_rep"] = negative_sample_representation

        return result_dict