import os
import torch
import hydra
import wandb
import pdb_numpy

import numpy as np
import mdtraj as md

from tqdm import tqdm
from .load import load_state_file, load_steered_simulation


def generate_v1(cfg, model_wrapper, epoch, device, logger):
    # Load configs for generation
    atom_num = cfg.data.atom
    sample_num = cfg.job.sample_num
    time_horizon = cfg.job.simulation.time_horizon
    inital_state = load_state_file(cfg, cfg.job.start_state, device)
    

    # Set conditions by task
    task = cfg.job.name
    if task == "simulation":
        raise NotImplementedError("Simulation task TBA")
    elif task == "tps":
        goal_state = load_state_file(cfg, cfg.job.goal_state, device)
    else:
        raise ValueError(f"Task {task} not found")
    
    
    # Generate trajectories
    state_list = [inital_state]
    current_state = inital_state
    model_wrapper.eval()
    with torch.no_grad():
        for t in tqdm(
            range(time_horizon),
            desc=f"Epoch {epoch}, genearting {sample_num} trajectories for {task}",
            leave=False
        ):
            # Generate next state offset
            step = torch.tensor(time_horizon - t).to(current_state.device).repeat(sample_num, 1)
            temperature = torch.tensor(cfg.job.temperature).to(current_state.device).repeat(sample_num, 1)
            if cfg.training.state_representation == "difference":
                goal_representation = goal_state - current_state
            elif cfg.training.state_representation == "original":
                goal_representation = goal_state
            else:
                raise ValueError(f"State representation {cfg.training.state_representation} not found")
            if cfg.training.repeat:
                step = step.repeat(1, current_state.shape[1])
                temperature = temperature.repeat(1, current_state.shape[1])
            state_offset, mu, log_var = model_wrapper(
                current_state=current_state,
                goal_state=goal_representation,
                step=step,
                temperature=temperature,
            )
            
            # Reshape 
            if cfg.data.molecule == "alanine":
                state_offset = state_offset.reshape(
                    sample_num,
                    atom_num,
                    3
                )
            elif cfg.data.molecule == "double-well":
                state_offset = state_offset.reshape(
                    sample_num,
                    atom_num
                )
            else:
                raise ValueError(f"Molecule {cfg.data.molecule} not found")
                
            # Compute next frame
            # simulation_list.step(state_offset)
            # next_state, force = simulation_list.report()
            next_state = current_state + state_offset
            state_list.append(next_state)
            current_state = next_state
            
    
    trajectory_list = torch.stack(state_list, dim=1)
    # trajectory_list /= scale
    
    if cfg.job.save:
        save_trajectory(cfg, trajectory_list, epoch, logger)
    
