import os
import torch
import hydra
import wandb

import numpy as np
import mdtraj as md

from tqdm import tqdm
from .load import load_state


def evaluate(cfg, trajectory_list, logger):
    task = cfg.job.name
    
    if task == "simulation":
        evaluate_sim(cfg, trajectory_list, logger)
    elif task == "tps":
        evaluate_tps(cfg, trajectory_list, logger)
    else:
        raise ValueError(f"Task {task} not found")
    
    return


def evaluate_sim(cfg, trajectory_list, logger):
    # Load ground truth simulation results
    
    # Compare the results
    
    
    return


# Trajectory list shape: (sample_num, time_horizon, atom_num, 3)
def evaluate_tps(cfg, trajectory_list, logger):
    # Load goal state
    goal_state = load_state_file(cfg, cfg.job.goal_state, trajectory_list.device)
    
    # Compute metrics related to TPS
    epd = compute_epd(trajectory_list[:, -1], goal_state)
    
    return



if __name__ == "__main__":
    raise NotImplementedError("TBA")
    # Add parser for evlauation
    
    # Load ckpt file
    
    # evaluate()
    
    #