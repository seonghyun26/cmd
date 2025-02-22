import torch
import torch.nn as nn

from tqdm import tqdm
from collections import OrderedDict

from mlcolvar.core import Normalization
from mlcolvar.cvs import DeepLDA, DeepTDA, DeepTICA
from mlcolvar.cvs import AutoEncoderCV

from . import *
from .spib import SPIB
from ..utils import *

model_dict = {
    "mlp": MLP,
    "clcv": CLCV,
    "cvmlp": CVMLP,
    "cvmlp-test": CVMLPTEST,
    "deeplda": DeepLDA,
    "deeptda": DeepTDA,
    "autoencoder": AutoEncoderCV,
    "timelagged-autoencoder": AutoEncoderCV,
    "vde": VariationalDynamicsEncoder,
    "rmsd": CVMLP,
    "torsion": CVMLP,
    "spib": SPIB,
}

def map_range(x, in_min = -1, in_max = 1):
    out_max = 1
    out_min = -1
    if type(in_min) != torch.Tensor:
        in_min = torch.tensor(in_min, dtype=torch.float32, device=x.device)
    if type(in_max) != torch.Tensor:
        in_max = torch.tensor(in_max, dtype=torch.float32, device=x.device)
    
    normalized_x = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    return normalized_x


class ModelWrapper(nn.Module):
    def __init__(self, cfg, device):
        super(ModelWrapper, self).__init__()
        
        self.cfg = cfg
        self.device = device
        self.model = self._load_model(cfg).to(device)
    
    def _set_input_dim(self, cfg, data_dim):
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
        
        return input_dim
    
    def _load_model(self, cfg):
        self.data_dim = cfg.data.atom * 3
        self.model_name = cfg.model.name.lower() 
        
        if self.model_name == "deeptica":
            model = DeepTICA(
                layers = cfg.model.params.layers,
                n_cvs = cfg.model.params.n_cvs,
                options = dict(cfg.model.params["options"])
            )
        
        elif self.model_name == "deeptda":
            model = DeepTDA(
                n_states = cfg.model.params["n_states"],
                n_cvs = cfg.model.params["n_cvs"],
                target_centers = cfg.model.params["target_centers"],
                target_sigmas = cfg.model.params["target_sigmas"],
                layers = cfg.model.params["layers"],
                options = dict(cfg.model.params["options"])
            )
        
        elif self.model_name in MLCOLVAR_METHODS + ["spib", "vde"]:
            model = model_dict[self.model_name](**cfg.model.params)
        
        elif self.model_name == "clcv":
            model = CLCV(
                cv=cfg.model.cv,
                **cfg.model.params
            )
            
        elif self.model_name == "gnncvtica":
            import mlcolvar.graph as mg
            model = mg.cvs.GraphDeepTICA(
                n_cvs = cfg.model.params["n_cvs"],
                cutoff = cfg.model.params["cutoff"],
                atomic_numbers = cfg.model.params["atomic_number"],
                model_options = dict(cfg.model.params["model_options"]),
            )
        
        elif self.model_name in model_dict.keys():
            model = model_dict[self.model_name](
                input_dim = self._set_input_dim(cfg, self.data_dim),
                data_dim = self.data_dim,
                **cfg.model.params
            )
        
        else:
            raise ValueError(f"Model {self.model_name} not found")
        
        print(model)
        return model
    
    def save_model(self, path, epoch):
        torch.save(self.model.state_dict(), f"{path}/model-{epoch}.pt")
        
    def load_from_checkpoint(self, path):
        if self.model_name == "deeptda":
            loaded_ckpt = torch.load(f"{path}")
            prefix_dict = {
                "norm_in": {},
                "nn": {},
            }
            for key, value in loaded_ckpt.items():
                for prefix in prefix_dict.keys():
                    if key.startswith(prefix):
                        prefix_dict[prefix][key[len(prefix)+1:]] = value
            norm_in = OrderedDict(prefix_dict["norm_in"])
            nn = OrderedDict(prefix_dict["nn"])
            self.model.norm_in.load_state_dict(norm_in)
            self.model.nn.load_state_dict(nn)
        
        elif self.model_name == "spib":
            ckpt_file = torch.load(f"{path}")["state_dict"]
            prefix_dict = {
                "representative_weights.": {},
                "encoder.": {},
                "encoder_mean.": {},
                "encoder_logvar.": {},
                "decoder.": {},
            }
            for key, value in ckpt_file.items():
                for prefix in prefix_dict.keys():
                    if key.startswith(prefix):
                        prefix_dict[prefix][key[len(prefix):]] = value
            self.model.representative_weights.load_state_dict(prefix_dict["representative_weights."])
            self.model.encoder.load_state_dict(prefix_dict["encoder."])
            self.model.encoder_mean.load_state_dict(prefix_dict["encoder_mean."])
            self.model.encoder_logvar.load_state_dict(prefix_dict["encoder_logvar."])
            self.model.decoder.load_state_dict(prefix_dict["decoder."])
        
        else:
            self.model.load_state_dict(torch.load(f"{path}"))
            
        return
    
    def set_normalization(self, mean, std):
        self.normalization = Normalization(
            in_features=mean.shape[0],
            mean=mean,
            range=std
        ).to(self.device)
    
    def forward(self,
        current_state: torch.Tensor,
        positive_sample: torch.Tensor,
        negative_sample: torch.Tensor,
        temperature: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = current_state.shape[0]
        
        # Process input
        current_state_conditions = torch.cat([
            current_state.reshape(batch_size, -1),
            temperature.reshape(batch_size, -1)
        ], dim=1)
        current_state_representation = self.model(current_state_conditions)
        positive_sample_conditions = torch.cat([
            positive_sample.reshape(batch_size, -1),
            temperature.reshape(batch_size, -1)
        ], dim=1)
        positive_sample_representation = self.model(positive_sample_conditions)
        negative_sample_conditions = torch.cat([
            negative_sample.reshape(batch_size, -1),
            temperature.reshape(batch_size, -1)
        ], dim=1)
        negative_sample_representation = self.model(negative_sample_conditions)
        
        # Process results
        result_dict = {}
        result_dict["current_state_rep"] = current_state_representation
        result_dict["positive_sample_rep"] = positive_sample_representation
        result_dict["negative_sample_rep"] = negative_sample_representation
        

        return result_dict
    
    def compute_cv(
        self,
        current_position: torch.Tensor = None,
        temperature: torch.Tensor = None,
        preprocessed_file: str = None,
    ):  
        if self.model_name in CLCV_METHODS:
            if self.cfg.model.input == "distance":
                if preprocessed_file is None:
                    data_num = current_position.shape[0]
                    current_position = coordinate2distance(self.cfg.job.molecule, current_position).reshape(data_num, -1)
                else:
                    current_position = torch.load(preprocessed_file).to(self.device)
                    current_position = self.normalization(current_position)
            
            else:
                raise ValueError(f"Input type {self.cfg.model.input} not found for {self.model_name}")
            
            mlcv = self.model(torch.cat([current_position, temperature], dim=1))
        
        elif self.model_name in ["deeplda", "deeptda", "deeptica"]:
            if preprocessed_file is not None:
                heavy_atom_distance = torch.load(preprocessed_file).to(self.device)
            else:
                data_num = current_position.shape[0]
                heavy_atom_distance = coordinate2distance(self.cfg.job.molecule, current_position).reshape(data_num, -1)
            
            mlcv = self.model(heavy_atom_distance)
            mlcv = map_range(mlcv, self.cfg.job.simulation.cv_min, self.cfg.job.simulation.cv_max)
        
        elif self.model_name == "gnncvtica":
            from torch_geometric.data import Data, Batch
            if preprocessed_file is not None:
                current_position = torch.load(preprocessed_file).to(self.device)
            data_num = current_position.shape[0]
            
            if data_num == 1:
                current_position_data = Data(
                    batch = torch.tensor([0], dtype=torch.int64, device=self.device),
                    edge_index = torch.tensor(ALANINE_HEAVY_ATOM_EDGE_INDEX, dtype=torch.long, device=self.device),
                    shifts = torch.zeros(90, 3, dtype=torch.float32, device=self.device),
                    positions = current_position.reshape(1, -1, 3)[0, ALANINE_HEAVY_ATOM_IDX],
                    node_attrs = torch.tensor(ALANINE_HEAVY_ATOM_ATTRS, dtype=torch.float32, device=self.device),
                )
                mlcv = self.model(current_position_data)
            else:
                data_list = []
                batch_size = 1000
                mlcv_list = []
                for i, position in tqdm(
                    enumerate(current_position),
                    desc="Computing CV values for gnncv",
                ):
                    current_position_data = Data(
                        edge_index = torch.tensor(ALANINE_HEAVY_ATOM_EDGE_INDEX, dtype=torch.long, device=self.device),
                        shifts = torch.zeros(90, 3, dtype=torch.float32, device=self.device),
                        positions = position.reshape(1, -1, 3)[0, ALANINE_HEAVY_ATOM_IDX],
                        node_attrs = torch.tensor(ALANINE_HEAVY_ATOM_ATTRS, dtype=torch.float32, device=self.device),
                    )
                    data_list.append(current_position_data)
                    if (i + 1) % batch_size == 0:
                        batched_data = Batch.from_data_list(data_list)
                        mlcv = self.model(batched_data)
                        mlcv_list.append(mlcv.detach().cpu())
                        del batched_data
                        del current_position_data
                        del data_list
                        data_list = []
                        torch.cuda.empty_cache()
                        import gc
                        gc.collect()
            
                mlcv = torch.cat(mlcv_list, dim=0)
            # mlcv = map_range(mlcv, self.cfg.job.simulation.cv_min, self.cfg.job.simulation.cv_max)
        
        elif self.model_name == "autoencoder":
            if preprocessed_file is not None:
                current_position = torch.load(preprocessed_file).to(self.device)
            data_num = current_position.shape[0]
            current_position = current_position.reshape(data_num, -1, 3)
            backbone_atom_position = current_position[:, ALANINE_BACKBONE_ATOM_IDX].reshape(data_num, -1)
            
            mlcv = self.model(backbone_atom_position)
            mlcv = map_range(mlcv, self.cfg.job.simulation.cv_min, self.cfg.job.simulation.cv_max)
            
        elif self.model_name == "timelagged-autoencoder":
            if preprocessed_file is not None:
                current_position = torch.load(preprocessed_file).to(self.device)
            data_num = current_position.shape[0]
            current_position = current_position.reshape(data_num, -1, 3)
            
            backbone_atom_position = current_position[:, ALANINE_HEAVY_ATOM_IDX].reshape(data_num, -1)
            mlcv = self.model(backbone_atom_position)
            mlcv = map_range(mlcv, self.cfg.job.simulation.cv_min, self.cfg.job.simulation.cv_max)
        
        elif self.model_name == "clcv":
            if preprocessed_file is not None:
                current_position = torch.load(preprocessed_file).to(self.device)
                data_num = current_position.shape[0]
            else:
                data_num = current_position.shape[0]
                heavy_atom_distance = coordinate2distance(self.cfg.job.molecule, current_position).reshape(data_num, -1)
                current_position = heavy_atom_distance.reshape(data_num, -1)
            
            mlcv = self.model(current_position)
        
        elif self.model_name == "spib":
            if preprocessed_file is not None:
                current_position = torch.load(preprocessed_file).to(self.device)
                dihedral_angle = current_position
            else:
                data_num = current_position.shape[0]
                current_position = current_position.reshape(data_num, -1, 3)
                phi = compute_dihedral_torch(current_position[:, ALDP_PHI_ANGLE])
                psi = compute_dihedral_torch(current_position[:, ALDP_PSI_ANGLE])
                theta = compute_dihedral_torch(current_position[:, ALDP_THETA_ANGLE])
                omega = compute_dihedral_torch(current_position[:, ALDP_OMEGA_ANGLE])
                dihedral_angle = torch.stack([phi, psi, theta, omega], dim=1)
            
            mlcv, z_sample, z_mean, z_logvar = self.model(dihedral_angle)
        
        elif self.model_name == "vde":
            if preprocessed_file is not None:
                current_position = torch.load(preprocessed_file).to(self.device)
            else:
                current_position = coordinate2distance(self.cfg.job.molecule, current_position)
            
            mlcv = self.model(current_position)
            mlcv = map_range(mlcv, self.cfg.job.simulation.cv_min, self.cfg.job.simulation.cv_max)

        else:
            raise ValueError(f"Model {self.model_name} not found")


        return mlcv
