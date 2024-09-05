import wandb
import torch

import mdtraj as md
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from openmm import *
from openmm.app import *
from openmm.unit import *

from .model import *
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
        self.latent_dim = cfg.model.encoder.output_dim
        
        # self.encoder = self.load_model(cfg.model.encoder).to(device)
        # self.decoder = self.load_model(cfg.model.decoder).to(device)
        # self.mu = nn.Linear(self.latent_dim, self.latent_dim).to(device)
        # self.logvar = nn.Linear(self.latent_dim, self.latent_dim).to(device)
        self.encoder = self.load_model(cfg.model.encoder)
        self.decoder = self.load_model(cfg.model.decoder)
        self.mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.logvar = nn.Linear(self.latent_dim, self.latent_dim)

    def __parameters__(self):
        return self.encoder.parameters(), self.decoder.parameters()
    
    def num_parameters(self):
        return self.encoder.num_parameters() + self.decoder.num_parameters()
    
    def load_model(self, cfg_model):
        model_dict = {
            "MLP": MLP,
        }
        
        if cfg_model.name in model_dict.keys():
            model = model_dict[cfg_model.name](cfg=cfg_model)
        else:
            raise ValueError(f"Model {cfg_model.name} not found")
        
        if "inii" in cfg_model:
            model.apply(init(cfg_model.init))
        
        return model
    
    def save_model(self, path, epoch):
        torch.save(self.encoder.state_dict(), f"{path}/encoder-{epoch}.pt")
        torch.save(self.decoder.state_dict(), f"{path}/decoder-{epoch}.pt")
        
    def load_from_checkpoint(self, path):
        self.encoder.load_state_dict(torch.load(f"{path}/encoder.pt"))
        self.decoder.load_state_dict(torch.load(f"{path}/decoder.pt"))
    
    def forward(self, next_state, current_state, goal_state, step, temperature):
        # Encode
        x = self.process_data(next_state, current_state, goal_state, step, temperature)
        shape = x.shape
        encoded = self.encoder(x)
        
        # Reparameterize
        mu = self.mu(encoded)
        logvar = self.logvar(encoded)
        # Check if mu and logvar are not nan and inf
        # assert torch.isnan(mu).sum() == 0, f"mu has nan"
        # assert torch.isnan(logvar).sum() == 0, f"logvar has nan"
        # assert torch.isinf(mu).sum() == 0, f"mu has inf"
        # assert torch.isinf(logvar).sum() == 0, f"logvar has inf"
        logvar = torch.clamp(logvar, max=10)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoded = self.decoder(self.process_latent(z, current_state, goal_state, step, temperature))
        decoded = self.process_prediction(decoded, shape)
        
        return encoded, decoded, mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    
    def generate(self, condition):
        # Condition: current_state, goal_state, step, temperature
        sample_num = condition.shape[0]
        gaussian_noise = torch.randn(sample_num, self.latent_dim).to(condition.device)
        generated_values = self.decoder(torch.cat((gaussian_noise, condition), dim=1))
        
        return generated_values
    
    def process_data(self, latent_var, current_state, goal_state, step, temperature):   
        batch_size = latent_var.shape[0]     
        temperature = torch.tensor(temperature).to(current_state.device).repeat(batch_size, 1)
        
        processed_state = torch.cat([
            latent_var.reshape(batch_size, -1), 
            current_state.reshape(batch_size, -1),
            goal_state.reshape(batch_size, -1),
            step.reshape(batch_size, -1),
            temperature
        ], dim=1)
        
        return processed_state
    
    def process_latent(self, latent, current_state, goal_state, step, temperature):
        batch_size = latent.shape[0]
        temperature = torch.tensor(temperature).to(latent.device).repeat(batch_size, 1)
        
        processed_latent = torch.cat((
            latent,
            current_state.reshape(batch_size, -1),
            goal_state.reshape(batch_size, -1),
            step.reshape(batch_size, -1),
            temperature
        ), dim=1)
        
        return processed_latent
    
    def process_prediction(self, prediction, shape):
        processed_prediction = prediction.reshape(
            shape[0],
            self.atom_num,
            3
        )
        return processed_prediction
    
    def train(self):
        self.encoder.train()
        self.decoder.train()
        
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()


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
    if cfg.training.loss == "MSE":
        loss = nn.MSELoss()
    elif cfg.training.loss == "cvae":
        def cvae_loss(x, decoded, mu, logvar):
            recon_loss = nn.MSELoss()(x, decoded)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss, kl_div
        loss = cvae_loss
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