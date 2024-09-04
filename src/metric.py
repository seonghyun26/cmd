import os
import torch
import hydra
import wandb
import jax

import jax.numpy as jnp
import numpy as np
import mdtraj as md

from tqdm import tqdm
from .load import load_simulation
from mdtraj.geometry import indices_phi, indices_psi

pairwise_distance = torch.cdist


def compute_epd(cfg, trajectory_list, goal_state):
    atom_num = cfg.data.atom
    unit_scale_factor = 1000
    last_state = trajectory_list[:, -1]
    
    matrix_f_norm = torch.sqrt(torch.square(
        pairwise_distance(last_state, last_state) - pairwise_distance(goal_state, goal_state)
    ).sum((1, 2)))
    epd = matrix_f_norm / (atom_num ** 2) * unit_scale_factor
    
    return epd.mean().item()


def compute_thp(cfg, trajectory_list, goal_state):
    molecule = cfg.job.molecule
    sample_num = cfg.job.sample_num
    last_state = trajectory_list[:, -1]
    
    if molecule == "alanine":
        phi_angle = [1, 6, 8, 14]
        psi_angle = [6, 8, 14, 16]
        
        phi = compute_dihedral(last_state[:, phi_angle])
        psi = compute_dihedral(last_state[:, psi_angle])
        phi_goal = compute_dihedral(goal_state[:, phi_angle])
        psi_goal = compute_dihedral(goal_state[:, psi_angle])
        
        hit = (np.abs(psi - psi_goal) < 0.75) & (np.abs(phi - phi_goal) < 0.75)
    else:
        raise ValueError(f"THP for molecule {molecule} TBA")
    
    return hit.mean() * 100


def compute_dihedral(positions):
    """http://stackoverflow.com/q/20305272/1128289"""
    
    def dihedral(p):
        p = p.numpy()
        b = p[:-1] - p[1:]
        b[0] *= -1
        v = np.array(
            [v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
        
        # Normalize vectors
        v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
        b1 = b[1] / np.linalg.norm(b[1])
        x = np.dot(v[0], v[1])
        m = np.cross(v[0], b1)
        y = np.dot(m, v[1])
        
        return np.arctan2(y, x)
    
    angles = np.array(list(map(dihedral, positions)))
    return angles
    

def compute_energy(cfg, trajectory_list, goal_state):
    sample_num = trajectory_list.shape[0]
    path_length = trajectory_list.shape[1]
    goal_state_file_path = f"data/{cfg.data.molecule}/{cfg.job.goal_state}.pdb"
    goal_simulation = load_simulation(cfg, goal_state_file_path, None)
    goal_state_energy = goal_simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
    
    path_energy_list = []
    for trajectory in tqdm(
        trajectory_list,
        desc=f"Computing energy for {trajectory_list.shape[0]} trajectories"
    ):
        energy_trajectory = potential_energy(cfg, trajectory)
        path_energy_list.append(energy_trajectory)
    path_energy_list = np.array(path_energy_list)
    
    path_maximum_energy = np.max(path_energy_list, axis=1)
    path_final_energy_error = np.array(path_energy_list[:, -1]) - goal_state_energy
    return path_maximum_energy.mean(), path_final_energy_error.mean()


def potential_energy(cfg, trajectory):
    energy_list = []
    
    for frame in trajectory:
        pbb_file_path = f"data/{cfg.data.molecule}/c5.pdb"
        simulation = load_simulation(cfg, pbb_file_path, frame)
        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        energy_list.append(energy._value)
    
    return energy_list


