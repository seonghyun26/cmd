import wandb
import torch

import mdtraj as md
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from openmm import *
from openmm.app import *
from openmm.unit import *

from .model import MLP
# from .data import *


class MD_Dataset(Dataset):
    def __init__(
        self,
        loaded_traj,
        config,
        args,
        sanity_check=False
    ):
        super(MD_Dataset, self).__init__()
        
        self.molecule = config['molecule']
        self.state = config['state']
        self.temperature = config['temperature']
        self.time = config['time']
        self.force_field = config['force_field']
        self.solvent = config['solvent']
        self.platform = config['platform']
        self.precision = config['precision']
        self.device = "cpu"
        
        data_x_list = []
        data_y_list = []
        data_interval_list = []
        data_goal_list = []
        
        if args.index == "random":
            random_indices = random.sample(range(0, self.time - 1), self.time // args.percent)
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                random_interval = random.sample(range(1, self.time - t), 1)[0]
                goal_state = torch.tensor(loaded_traj[t+random_interval].xyz.squeeze()).to(self.device)
                
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_goal_list.append(goal_state)
                data_interval_list.append(torch.tensor(random_interval).to(self.device).unsqueeze(0))
        else:
            for t in tqdm(
                range((self.time -1) // args.percent),
                desc=f"Loading {args.precent} precent of dataset from initial frame"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_interval_list.append(1)
                
        self.x = torch.stack(data_x_list).to(self.device)
        self.y = torch.stack(data_y_list).to(self.device)
        self.goal = torch.stack(data_goal_list).to(self.device)
        self.delta_time = torch.stack(data_interval_list).to(self.device)
        
        # if sanity_check:
        #     self.sanity_check(loaded_traj)
        
    def sanity_check(self, loaded_traj):
        assert torch.equal(self.x.shape, self.y.shape), f"Shape of x and y not equal"
        
        for t in tqdm(
            range(self.time -1),
            desc="Sanity check"
        ):
            x = self.x[t]
            y = self.y[t]
            x_frame = torch.tensor(loaded_traj[t].xyz.squeeze()).to(self.device)
            y_frame = torch.tensor(loaded_traj[t+1].xyz.squeeze()).to(self.device)
            
            assert torch.equal(x, x_frame), f"Frame {t}, x not equal"
            assert torch.equal(y, y_frame), f"Frame {t+1}, y not equal"        
        
    def __getitem__(self, index):
	    return self.x[index], self.y[index], self.goal[index], self.delta_time[index]
 
    def __len__(self):
	    return self.x.shape[0]

def load_data(cfg):
    data_path = f"/home/shpark/prj-cmd/simulation/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.state}-{cfg.data.index}.pt"
    
    train_dataset = torch.load(f"{data_path}")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=cfg.training.num_workers
    )
    
    return train_loader

        
class ModelWrapper(nn.Module):
    def __init__(self, cfg, device):
        super(ModelWrapper, self).__init__()
        
        self.atom_num = cfg.data.atom
        self.batch_size = cfg.training.batch_size
        self.output_dim = cfg.model.hidden_dim[-1]
        self.model = self.load_model(cfg).to(device)
        
        self.mu = nn.Linear(self.output_dim, self.atom_num * 3).to(device)
        self.var = nn.Linear(self.output_dim, self.atom_num * 3).to(device)
    
    def load_model(self, cfg):
        model_dict = {
            "mlp": MLP,
        }
        
        model_name = cfg.model.name.lower() 
        if model_name in model_dict.keys():
            model = model_dict[model_name](cfg=cfg)
        else:
            raise ValueError(f"Model {model_name} not found")

        return model
    
    def save_model(self, path, epoch):
        torch.save(self.model.state_dict(), f"{path}/model-{epoch}.pt")
        
    def load_from_checkpoint(self, path):
        self.model.load_state_dict(torch.load(f"{path}/model.pt"))
    
    def forward(self,
        current_state: torch.Tensor,
        goal_state: torch.Tensor,
        step: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        batch_size = current_state.shape[0]
        temperature = torch.tensor(temperature).to(current_state.device).repeat(current_state.shape[0], 1)
        
        conditions = torch.cat([
            current_state.reshape(batch_size, -1),
            goal_state.reshape(batch_size, -1),
            step.reshape(batch_size, -1),
            temperature.reshape(batch_size, -1)
        ], dim=1)
        latent = self.model(conditions)
        
        mu = self.mu(latent)
        var = self.var(latent)
        var = torch.clamp(var, max=10)
        state_offset = self.reparameterize(mu, var)
        state_offset = state_offset.reshape(batch_size, self.atom_num, 3)
        
        return state_offset, var
    
    def reparameterize(self, mu, var):
        std = torch.exp(var)
        eps = torch.randn_like(std)
        
        return mu + eps * std

def load_model_wrapper(cfg, device):
    model_wrapper = ModelWrapper(cfg, device)
    optimizer = load_optimizer(cfg, model_wrapper.parameters())
    scheduler = load_scheduler(cfg, optimizer)
    
    return model_wrapper, optimizer, scheduler



def load_optimizer(cfg, model_param):
    optimizer_dict = {
        "Adam": Adam,
        "SGD": SGD,
    }
    
    if cfg.training.optimizer.name in optimizer_dict.keys():
        optimizer = optimizer_dict[cfg.training.optimizer.name](
            model_param,
            **cfg.training.optimizer.params
        )
    else:
        raise ValueError(f"Optimizer {cfg.training.optimizer} not found")
    
    return optimizer


def load_scheduler(cfg, optimizer):
    scheduler_dict = {
        # "LambdaLR": LambdaLR,
        "StepLR": StepLR,
        "MultiStepLR": MultiStepLR,
        "ExponentialLR": ExponentialLR,
        "CosineAnnealingLR": CosineAnnealingLR,
    }
    
    if cfg.training.scheduler.name in scheduler_dict.keys():
        scheduler = scheduler_dict[cfg.training.scheduler.name](
            optimizer,
            **cfg.training.scheduler.params
        )
    else:
        raise ValueError(f"Scheduler {cfg.training.scheduler.name} not found")
    
    return scheduler


def load_loss(cfg):
    loss_name = cfg.training.loss.lower()
    if loss_name == "mse":
        loss = nn.MSELoss(reduction="none" if cfg.training.loss_scale == "step" else "mean")
    else:
        raise ValueError(f"Loss {cfg.training.loss} not found")
    
    return loss


def load_state_file(cfg, state, device):
    state_dir = f"./data/{cfg.job.molecule}/{state}.pdb"
    state = md.load(state_dir).xyz
    state = torch.tensor(state).to(device)
    states = state.repeat(cfg.job.sample_num, 1, 1)
    
    return states


def load_simulation(cfg, pbb_file_path, frame=None):
    # set pdb file with current positions
    pdb = PDBFile(pbb_file_path)
    if frame is not None:
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                pdb.positions[i][j]._value = frame[i][j].item()
        
    
    # Set force field
    force_field = ForceField(*cfg.job.simulation.force_field)
    system = force_field.createSystem(
        pdb.topology,
        nonbondedCutoff=3 * nanometer,
        constraints=HBonds
    )
    integrator = LangevinIntegrator(
        cfg.job.simulation.temperature * kelvin,
        1 / picosecond,
        1 * femtoseconds
    )
    platform = Platform.getPlatformByName(cfg.job.simulation.platform)
    properties = {'Precision': cfg.job.simulation.precision}

    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        platform,
        properties
    )        
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    
    return simulation