import os
import torch
import hydra
import wandb
import pdb_numpy

import numpy as np
import mdtraj as md

from tqdm import tqdm
from .load import load_state_file, load_steered_simulation


def generate(cfg, model_wrapper, epoch, device, logger):
    task = cfg.job.name
    atom_num = cfg.data.atom
    sample_num = cfg.job.sample_num
    time_horizon = cfg.job.simulation.time_horizon
    
    if task == "tps":
        inital_state = load_state_file(cfg, cfg.job.start_state, device)
        goal_state = load_state_file(cfg, cfg.job.goal_state, device)
        steered_simulation_list = load_steered_simulation(
            cfg = cfg,
            sample_num = cfg.job.sample_num,
            model_wrapper = model_wrapper,
        )
        position_list = []
        
        try:
            for step in tqdm(
                range(1, time_horizon + 1),
                desc=f"Genearting {sample_num} trajectories for {task}",
            ):
                position = steered_simulation_list.report()
                position = np.array([list(p) for p in position], dtype=np.float32)
                position_list.append(position)
                steered_simulation_list.step(step)
            
            if isinstance(position_list, torch.Tensor):
                trajectory_list = torch.stack(position_list, dim=1)
            elif isinstance(position_list, list):
                trajectory_list = np.stack(position_list, axis=1)
            else:
                raise ValueError(f"Type {type(position_list)} not supported")
            
            if cfg.job.save:
                save_trajectory(cfg, trajectory_list, epoch, logger)
        
        except Exception as e:
            logger.error(f"Error in generating trajectory: {e}")
            trajectory_list = None
    
    elif task == "cv":
        trajectory_list = None
    
    else:
        raise ValueError(f"Task {task} not found")
    
    return trajectory_list
  

def save_trajectory(cfg, trajectory_list, epoch, logger):
    trajectory_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{cfg.job.name}/{epoch}"
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
    
    for idx, trajectory in enumerate(trajectory_list):
        try:
            if isinstance(trajectory, torch.Tensor):
                trajectory = trajectory.detach().cpu().numpy()
            np.save(f"{trajectory_dir}/{idx}.npy", trajectory)
        except Exception as e:
            logger.error(f"Error in saving trajectory: {e}")
    
    logger.info(f"Epoch {epoch}: {trajectory_list.shape[0]} trajectories saved at: {trajectory_dir}")
    
    return
