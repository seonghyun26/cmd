import os
import torch
import hydra
import wandb
import pdb_numpy

import numpy as np
import mdtraj as md

from tqdm import tqdm
from .load import load_state_file


def generate(cfg, model_wrapper, epoch, device, logger):
    # Load configs for generation
    sample_num = cfg.job.sample_num
    atom_num = cfg.data.atom
    time_horizon = cfg.job.time_horizon
    temperature = cfg.job.temperature
    inital_states = load_state_file(cfg, cfg.job.start_state, device)
    inital_states *= 1000.0
    

    # Set conditions by task
    task = cfg.job.name
    if task == "simulation":
        raise NotImplementedError("Simulation task TBA")
    elif task == "tps":
        goal_states = load_state_file(cfg, cfg.job.goal_state, device)
        goal_states *= 1000.0
    else:
        raise ValueError(f"Task {task} not found")
    
    
    # Generate trajectories
    state_list = [inital_states]
    current_states = inital_states
    model_wrapper.eval()
    with torch.no_grad():
        for t in tqdm(
            range(time_horizon),
            desc=f"Epoch {epoch}, genearting {sample_num} trajectories for {task}"
        ):
            step = torch.tensor(time_horizon - t).to(current_states.device).repeat(sample_num, 1)
            states_offset, var = model_wrapper(
                current_state=current_states,
                goal_state=goal_states,
                step=step,
                temperature=temperature
            )
            states_offset = states_offset.reshape(
                sample_num,
                atom_num,
                3
            )
            next_states = current_states + states_offset
            state_list.append(next_states)
            current_states = next_states
    
    
    trajectory_list = torch.stack(state_list, dim=1)
    trajectory_list /= 1000.0
    if cfg.job.save:
        save_trajectory(cfg, trajectory_list, epoch, logger)
    
    return trajectory_list
    

    
    
def save_trajectory(cfg, trajectory_list, epoch, logger):
    trajectory_dir = f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/{cfg.job.name}/{epoch}"
    if not os.path.exists(trajectory_dir):
        os.makedirs(trajectory_dir)
    
    for idx, trajectory in enumerate(trajectory_list):
        try:
            np.save(f"{trajectory_dir}/{idx}.npy", trajectory.detach().cpu().numpy())
        except Exception as e:
            logger.error(f"Error in saving trajectory: {e}")
    
    logger.info(f"{trajectory_list.shape[0]} trajectories saved at: {trajectory_dir}")
    
    return
