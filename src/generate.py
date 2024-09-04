import os
import torch
import hydra
import wandb
import pdb_numpy

import numpy as np
import mdtraj as md

from tqdm import tqdm
from .load import load_state_file


def generate(cfg, model_wrapper, device, logger):
    # Load configs for generation
    sample_num = cfg.job.sample_num
    atom_num = cfg.data.atom
    time_horizon = cfg.job.time_horizon
    temperature = cfg.job.temperature
    current_states = load_state_file(cfg, cfg.job.start_state, device)
    state_list = [current_states]
    

    # Set conditions by task
    task = cfg.job.name
    if task == "simulation":
        pass
        # goal_states = load_state_file(cfg, cfg.job.goal_state, device)
    elif task == "tps":
        goal_states = load_state_file(cfg, cfg.job.goal_state, device)
    else:
        raise ValueError(f"Task {task} not found")
    temperature = torch.tensor(temperature).to(current_states.device).repeat(sample_num, 1)
    
    
    # Generate trajectories
    for t in tqdm(
        range(time_horizon),
        desc=f"Genearting {sample_num} trajectories for {task}"
    ):
        try :
            step = torch.tensor(time_horizon - t).to(current_states.device).repeat(sample_num, 1)
            processed_current_states = torch.cat([
                current_states.reshape(sample_num, -1),
                goal_states.reshape(sample_num, -1),
                step,
                temperature
            ], dim=1)
            next_states = model_wrapper.generate(processed_current_states)
            processed_next_states = current_states.reshape(
                sample_num,
                atom_num,
                3
            )
            state_list.append(processed_next_states)
        except Exception as e:
            raise ValueError(f"Error in simulation: {e}")
    
    
    trajectory_list = torch.stack(state_list, dim=1)
    if cfg.job.save:
        save_trajectory(cfg, trajectory_list, logger)
    
    return trajectory_list
    

    
    
def save_trajectory(cfg, trajectory_list, logger):
    trajectory_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{cfg.job.name}/trajectory"
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
    
    for idx, trajectory in enumerate(trajectory_list):
        try:
            np.save(f"{trajectory_dir}/{idx}.npy", trajectory.detach().cpu().numpy())
        except Exception as e:
            logger.error(f"Error in saving trajectory: {e}")
    
    logger.info(f"{trajectory_list.shape[0]} trajectories saved at: {trajectory_dir}")
    
    return
