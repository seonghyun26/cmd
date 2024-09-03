import os
import torch
import hydra
import wandb

import numpy as np
import mdtraj as md

from tqdm import tqdm


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


def evaluate_tps(cfg, trajectory_list, logger):
    # Load goal state
    
    # Compute metrics related to TPS
    
    return