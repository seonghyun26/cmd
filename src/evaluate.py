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
    
    if task == "tps":
        if trajectory_list is None:
            if cfg.logging.wandb:
                dummy_keys = ["eval/thp", "eval/epd", "eval/rmsd", "eval/max_energy", "eval/final_energy", "eval/final_energy_err", "eval/ram", "eval/transition_path", "eval/projection", "eval/state", "eval/contour"]
                eval_result = {key: None for key in dummy_keys}
                wandb.log(eval_result)
        else:
            evaluate_tps(
                cfg = cfg,
                model_wrapper = model_wrapper, 
                trajectory_list = trajectory_list,
                logger = logger, 
                epoch = epoch,
                device = device
            )
    
    elif task == "cv":
        evaluate_cv(
            cfg = cfg,
            model_wrapper = model_wrapper, 
            logger = logger, 
            epoch = epoch,
            device = device
        )
    
    else:
        raise ValueError(f"Task {task} not found")
    
    return


def evaluate_tps(cfg, model_wrapper, trajectory_list, logger, epoch, device):
    '''
        Trajectory: torch.Tensor (sample_num, time_horizon, atom_num, 3)
    '''
    
    goal_state = load_state_file(cfg, cfg.job.goal_state, device)
    goal_state = goal_state.to(device)
    if isinstance(trajectory_list, np.ndarray):
        trajectory_list = torch.tensor(trajectory_list, dtype=torch.float32, device=device)
    trajectory_list = trajectory_list.to(device)
    eval_result = {}
    
    if cfg.job.metrics.thp.use:
        logger.info(">> Computing THP")
        eval_result["eval/thp"], hit_mask, hit_index = compute_thp(cfg, trajectory_list, goal_state)
        # eval_result["eval/hit_index"] = hit_index.mean()
    if cfg.job.metrics.epd.use:
        logger.info(">> Computing EPD")
        if hit_mask.sum() == 0:
            logger.info("No hit found, skipping epd computation")
            eval_result["eval/epd"], eval_result["eval/rmsd"]  = None, None
        else:
            eval_result["eval/epd"], eval_result["eval/rmsd"]  = compute_epd(cfg, trajectory_list, goal_state, hit_mask, hit_index)
    if cfg.job.metrics.ram.use:
        logger.info(">> Plotting paths")
        eval_result["eval/ram"], eval_result["eval/transition_path"] = compute_ram(cfg, trajectory_list, hit_mask, hit_index, epoch)
    if cfg.job.metrics.projection.use and cfg.model.name not in ["rmsd", "torsion"]:
        logger.info(">> Plotting projected CV values")
        eval_result["eval/projection"], eval_result["eval/state"], eval_result["eval/contour"] = compute_projection(cfg, model_wrapper, epoch)
    if cfg.job.metrics.energy.use:
        if hit_mask.sum() == 0:
            logger.info("No hit found, skipping energy computation")
            eval_result["eval/max_energy"], eval_result["eval/final_energy"], eval_result["eval/final_energy_err"] = None, None, None
        else:
            logger.info(">> Computing Energy")
            eval_result["eval/max_energy"], eval_result["eval/final_energy"], eval_result["eval/final_energy_err"] = compute_energy(cfg, trajectory_list, goal_state, hit_mask, hit_index)
    # if cfg.job.metrics.jacobian.use:
    #     logger.info(">> Computing Jacobian for mlcv against input")
    #     eval_result["eval/jacobian"] = compute_jacobian(cfg, model_wrapper, epoch)
    
    for key in eval_result.keys():
        logger.info(f"{key}: {eval_result[key]}")
    
    if cfg.logging.wandb:
        wandb.log(eval_result)
    
    return


def evaluate_cv(cfg, model_wrapper, logger, epoch, device):
    eval_result = {}
    
    if cfg.job.metrics.projection.use:
        logger.info(">> Plotting projected CV values")
        eval_result["eval/projection"], eval_result["eval/state"], eval_result["eval/contour"] = compute_projection(cfg, model_wrapper, epoch)
        # if cfg.job.metrics.jacobian.use:
    #     logger.info(">> Computing Jacobian for mlcv against input")
    #     eval_result["eval/jacobian"] = compute_jacobian(cfg, model_wrapper, epoch)
    
    for key in eval_result.keys():
        logger.info(f"{key}: {eval_result[key]}")
    
    if cfg.logging.wandb:
        wandb.log(eval_result)
    
    return