import wandb
import torch

import mdtraj as md
import torch.nn as nn

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split

from openmm import *
from openmm.app import *
from openmm.unit import *

from .data import *
from .loss import *
from .model import ModelWrapper

def load_data(cfg):
    data_path = f"/home/shpark/prj-cmd/simulation/dataset/{cfg.data.molecule}/{cfg.data.temperature}/{cfg.data.state}-{cfg.data.index}.pt"
    
    dataset = torch.load(f"{data_path}")
    if cfg.training.test:
        train_dataset, test_dataset = random_split(dataset, cfg.data.train_test_split)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=cfg.training.shuffle,
            num_workers=cfg.training.num_workers
        )
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=cfg.training.shuffle,
            num_workers=cfg.training.num_workers
        )
    else:
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=cfg.training.batch_size,
            shuffle=cfg.training.shuffle,
            num_workers=cfg.training.num_workers
        )
        test_loader = None
    
    return train_loader, test_loader
       

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
        "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts
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
    loss_name = cfg.training.loss.name.lower()
    loss_func_list = {
        "mse": mse_loss,
        "mse+reg": mse_reg_loss,
        "mse+reg2": mse_reg2_loss,
        "mse+reg3": mse_reg3_loss,
        "mse+reg4": mse_reg4_loss,
    }
    
    if loss_name in loss_func_list.keys():
        loss_func = loss_func_list[loss_name]
    else:
        raise ValueError(f"Loss {loss_name} not found")
        return loss_func


def load_state_file(cfg, state, device):
    state_dir = f"./data/{cfg.job.molecule}/{state}.pdb"
    state = md.load(state_dir).xyz
    state = torch.tensor(state).to(device)
    states = state.repeat(cfg.job.sample_num, 1, 1)
    
    return states


def load_simulation(cfg, pbb_file_path, frame=None):
    # set pdb file with current positions
    pdb = PDBFile(pbb_file_path)
    
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
    # simulation.context.setPositions(pdb.positions)
    # if frame is not None:
    #     for i in range(frame.shape[0]):
    #         for j in range(frame.shape[1]):
    #             pdb.positions[i][j]._value = frame[i][j].item()
    atom_xyz = frame.detach().cpu().numpy()
    atom_list = [Vec3(atom[0], atom[1], atom[2]) for atom in atom_xyz]
    current_state_openmm = Quantity(value=atom_list, unit=nanometer)
    simulation.context.setPositions(current_state_openmm)            
    
    simulation.minimizeEnergy()
    
    return simulation