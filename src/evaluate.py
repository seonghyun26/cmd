import os
import torch
import hydra
import wandb

import numpy as np
import mdtraj as md

from tqdm import tqdm
from .load import load_state_file
from .metric import *


def evaluate(cfg, model_wrapper, trajectory_list, logger, epoch, device):
    task = cfg.job.name
    
    if task == "simulation":
        raise ValueError(f"Task {task} not supported")
    elif task == "tps":
        evaluate_tps(cfg, model_wrapper, trajectory_list, logger, epoch, device)
    else:
        raise ValueError(f"Task {task} not found")
    
    return


def evaluate_sim(cfg, trajectory_list, logger):
    # Load ground truth simulation results
    
    # Compare the results
    pass
    
    return


# Trajectory list shape: (sample_num, time_horizon, atom_num, 3)
def evaluate_tps(cfg, model_wrapper, trajectory_list, logger, epoch, device):
    goal_state = load_state_file(cfg, cfg.job.goal_state, device)
    goal_state = goal_state.to("cpu")
    if isinstance(trajectory_list, np.ndarray):
        trajectory_list = torch.tensor(trajectory_list, dtype=torch.float32, device=device)
    trajectory_list = trajectory_list.to("cpu")
    eval_result = {}
    
    if cfg.job.metrics.epd.use:
        logger.info(">> Computing EPD")
        eval_result["eval/epd"] = compute_epd(cfg, trajectory_list, goal_state)
    if cfg.job.metrics.thp.use:
        logger.info(">> Computing THP")
        eval_result["eval/thp"] = compute_thp(cfg, trajectory_list, goal_state)
    if cfg.job.metrics.energy.use:
        logger.info(">> Computing Energy")
        eval_result["eval/max_energy"], eval_result["eval/final_energy"], eval_result["eval/final_energy_err"] = compute_energy(cfg, trajectory_list, goal_state)
    if cfg.job.metrics.ram.use:
        logger.info(">> Plotting Ramachandran plot")
        eval_result["eval/ram"] = compute_ram(cfg, trajectory_list, epoch)
    if cfg.job.metrics.projection.use:
        logger.info(">> Plotting projected CV values")
        eval_result["eval/projection"] = compute_projection(cfg, model_wrapper, epoch)
    
    for key in eval_result.keys():
        logger.info(f"{key}: {eval_result[key]}")
    
    if cfg.logging.wandb:
        wandb.log(eval_result)
    
    return