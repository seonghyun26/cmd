import os
import torch
import hydra
import wandb
import jax

import jax.numpy as jnp
import numpy as np
import mdtraj as md

from tqdm import tqdm
from mdtraj.geometry import indices_phi, indices_psi

from .data import Synthetic
from .utils import *
from .simulation import init_simulation, set_simulation
from .plot import plot_ad_potential, plot_dw_potential, plot_ad_cv

pairwise_distance = torch.cdist


def potential_energy(cfg, trajectory):
    molecule = cfg.job.molecule
    energy_list = []
    
    if molecule == "alanine":
        pbb_file_path = f"{cfg.data.dir}/{cfg.data.molecule}/c5.pdb"
        simulation = init_simulation(cfg, pbb_file_path)
        
        for frame in trajectory:
            try:
                simulation = set_simulation(simulation, frame)
                energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
                energy_list.append(energy._value)
            except Exception as e:
                print(f"Error in computing energy: {e}")
                energy_list.append(10000)
    else: 
        raise ValueError(f"Potential energy for molecule {molecule} TBA")
    
    return energy_list


def compute_thp(
    cfg,
    trajectory_list,
    goal_state
):
    device = trajectory_list.device
    molecule = cfg.job.molecule
    sample_num = cfg.job.sample_num
    cv_bound = cfg.job.metrics.thp.cv_bound
    hit_rate = 0.0
    hit_mask = []
    hit_index = []
    
    if molecule == "alanine":
        psi_goal = compute_dihedral_torch(goal_state[0, ALDP_PSI_ANGLE].reshape(-1, len(ALDP_PSI_ANGLE), 3))
        phi_goal = compute_dihedral_torch(goal_state[0, ALDP_PHI_ANGLE].reshape(-1, len(ALDP_PHI_ANGLE), 3))
        for i in tqdm(
            range(sample_num),
            desc = f"Computing THP for {trajectory_list.shape[0]} trajectories"
        ):
            psi = compute_dihedral_torch(trajectory_list[i, :, ALDP_PSI_ANGLE])
            phi = compute_dihedral_torch(trajectory_list[i, :, ALDP_PHI_ANGLE])
            psi_hit_distance = torch.abs(psi - psi_goal)
            phi_hit_distance = torch.abs(phi - phi_goal)
            cv_distance = torch.sqrt(psi_hit_distance ** 2 + phi_hit_distance ** 2)
            hit_in_path = (psi_hit_distance < cv_bound) & (phi_hit_distance < cv_bound)
            hit_index_in_path = torch.argmin(cv_distance)
            
            if torch.any(hit_in_path):
                hit_rate += 1.0
                hit_mask.append(True)
                hit_index.append(hit_index_in_path)
            else:
                hit_mask.append(False)
                hit_index.append(-1)
                
        hit_rate /= sample_num
        hit_mask = torch.tensor(hit_mask, dtype=torch.bool, device=device)
        hit_index = torch.tensor(hit_index, dtype=torch.int32, device=device)

    elif molecule == "chignolin":
        raise NotImplementedError(f"THP for molecule {molecule} to be implemented")
    
    else:
        raise ValueError(f"THP for molecule {molecule} TBA")
    
    return hit_rate, hit_mask, hit_index


def compute_epd(cfg, trajectory_list, goal_state, hit_mask, hit_index):
    molecule = cfg.job.molecule
    atom_num = cfg.data.atom
    sample_num = trajectory_list.shape[0]
    unit_scale_factor = 1000
    hit_trajectory = trajectory_list[hit_mask]
    hit_path_num = hit_mask.sum().item()
    goal_state = goal_state[hit_mask]
    epd = 0.0
    
    hit_state_list = []
    rmsd = []
    for i in tqdm(
        range(hit_path_num),
        desc = f"Computing EPD, RMSD for {hit_path_num} hitting trajectories"
    ):
        hit_state_list.append(hit_trajectory[i, hit_index[i]])
        rmsd.append(kabsch_rmsd(hit_trajectory[i, hit_index[i]], goal_state[i]))
    
    hit_state_list = torch.stack(hit_state_list)
    matrix_f_norm = torch.sqrt(torch.square(
        pairwise_distance(hit_state_list, hit_state_list) - pairwise_distance(goal_state, goal_state)
    ).sum((1, 2)))
    epd = torch.mean(matrix_f_norm / (atom_num ** 2) * unit_scale_factor)
    rmsd = torch.tensor(rmsd)
    rmsd = torch.mean(rmsd)
    
    return epd, rmsd


def compute_energy(cfg, trajectory_list, goal_state, hit_mask, hit_index):
    molecule = cfg.job.molecule
    sample_num = trajectory_list.shape[0]
    path_length = trajectory_list.shape[1]
    
    try:
        if molecule == "alanine":
            goal_state_file_path = f"data/{cfg.data.molecule}/{cfg.job.goal_state}.pdb"
            goal_simulation = init_simulation(cfg, goal_state_file_path)
            goal_state_energy = goal_simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
            
            path_energy_list = []
            for trajectory in tqdm(
                trajectory_list[hit_mask],
                desc=f"Computing energy for {trajectory_list[hit_mask].shape[0]} hitting trajectories"
            ):
                # if hit_mask[traj_idx]:
                energy_trajectory = potential_energy(cfg, trajectory)
                path_energy_list.append(energy_trajectory)
            path_energy_list = np.array(path_energy_list)
            
            path_maximum_energy = np.max(path_energy_list, axis=1)
            path_final_energy_error = np.array(path_energy_list[:, -1]) - goal_state_energy
        elif molecule == "double-well":
            synthetic = Synthetic()
            path_energy_list = [synthetic.potential(trajectory) for trajectory in trajectory_list]
            path_energy_list = np.array(path_energy_list)
            path_maximum_energy = np.max(path_energy_list, axis=1)
            path_final_energy_error = np.array(path_energy_list[:, -1]) - synthetic.potential(goal_state)
        else: 
            raise ValueError(f"Energy for molecule {molecule} TBA")
    except Exception as e:
        print(f"Error in computing energy: {e}")
        path_maximum_energy = np.ones(sample_num) * 10000
        path_energy_list = np.ones((sample_num, path_length)) * 10000
        path_final_energy_error = np.ones(sample_num) * 10000
    
    return path_maximum_energy.mean(), path_energy_list[:, -1].mean(), path_final_energy_error.mean()

def compute_ram(cfg, trajectory_list, hit_mask, hit_index, epoch):
    molecule = cfg.job.molecule
    hit_path_num = hit_mask.sum().item()
    
    if molecule == "alanine":
        # Load start, goal state and compute phi, psi
        start_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.start_state}.pdb").xyz
        goal_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.goal_state}.pdb").xyz
        start_state = torch.tensor(start_state_xyz)
        goal_state = torch.tensor(goal_state_xyz)
        phi_start = compute_dihedral_torch(start_state[:, ALDP_PHI_ANGLE])
        psi_start = compute_dihedral_torch(start_state[:, ALDP_PSI_ANGLE])
        phi_goal = compute_dihedral_torch(goal_state[:, ALDP_PHI_ANGLE])
        psi_goal = compute_dihedral_torch(goal_state[:, ALDP_PSI_ANGLE])
    
        # Compute phi, psi from trajectory_list
        phi_traj_list = [compute_dihedral_torch(trajectory[:, ALDP_PHI_ANGLE]) for trajectory in trajectory_list]
        psi_traj_list = [compute_dihedral_torch(trajectory[:, ALDP_PSI_ANGLE]) for trajectory in trajectory_list]
        
        ram_plot_img = plot_ad_potential(
            traj_dihedral = (phi_traj_list, psi_traj_list),
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            cv_bound_use = cfg.job.metrics.projection.bound_use,
            cv_bound = cfg.job.metrics.thp.cv_bound,
            epoch = epoch,
            name = "paths"
        )
        
        hit_phi_traj_list = [phi_traj_list[i][:hit_index[i]] for i in range(len(phi_traj_list)) if hit_mask[i]]
        hit_psi_traj_list = [psi_traj_list[i][:hit_index[i]] for i in range(len(psi_traj_list)) if hit_mask[i]]
        transition_path_plot_img = plot_ad_potential(
            traj_dihedral = (hit_phi_traj_list, hit_psi_traj_list),
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            cv_bound_use = cfg.job.metrics.projection.bound_use,
            cv_bound = cfg.job.metrics.thp.cv_bound,
            epoch = epoch,
            name = "transition_paths"
        )
        
    elif molecule == "double-well":
        device = trajectory_list.device
        start_state = torch.tensor([-1.118, 0], dtype=torch.float32).to(device)
        goal_state = torch.tensor([1.118, 0], dtype=torch.float32).to(device)
        
        ram_plot_img = plot_dw_potential(
            traj = trajectory_list,
            start = start_state,
            goal = goal_state,
            epoch = epoch
        )

    elif molecule == "chignolin":
        raise ValueError(f"Projection for molecule {molecule} TBA...")
    
    else:
        raise ValueError(f"Ramachandran plot for molecule {molecule} TBA...")
    
    return wandb.Image(ram_plot_img), wandb.Image(transition_path_plot_img)

def compute_projection(cfg, model_wrapper, epoch):
    molecule = cfg.job.molecule
    device = model_wrapper.device
    if cfg.model.name in CLCV_METHODS:
        cv_dim = cfg.model["params"].output_dim
    else:
        cv_dim = 1
    
    if molecule == "alanine":
        data_dir = f"{cfg.data.dir}/projection/{cfg.job.molecule}/{cfg.job.metrics.projection.version}"
        phis = np.load(f"{data_dir}/phi.npy")
        psis = np.load(f"{data_dir}/psi.npy")
        temperature = torch.tensor(cfg.job.simulation.temperature).repeat(phis.shape[0], 1).to(device)
        
        if cfg.model.input == "distance":
            projection_file = f"{data_dir}/heavy_atom_distance.pt"
        elif cfg.model.input == "coordinate":
            projection_file = f"{data_dir}/coordinate.pt"
        else:
            raise ValueError(f"Input type {cfg.model.input} not found")
        projected_cv = model_wrapper.compute_cv(
            preprocessed_file = projection_file,
            temperature = temperature,
        )
        

        start_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.start_state}.pdb").xyz
        goal_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.goal_state}.pdb").xyz
        start_state = torch.tensor(start_state_xyz)
        goal_state = torch.tensor(goal_state_xyz)
        phi_start = compute_dihedral_torch(start_state[:, ALDP_PHI_ANGLE])
        psi_start = compute_dihedral_torch(start_state[:, ALDP_PSI_ANGLE])
        phi_goal = compute_dihedral_torch(goal_state[:, ALDP_PHI_ANGLE])
        psi_goal = compute_dihedral_torch(goal_state[:, ALDP_PSI_ANGLE])
        
        projection_img = plot_ad_cv(
            phi = phis,
            psi = psis,
            cv = projected_cv.cpu().detach().numpy(),
            cv_dim = cv_dim,
            epoch = epoch,
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            cfg_plot = cfg.job.metrics.projection
        )
    
    elif molecule == "chignolin":
        raise ValueError(f"Projection for molecule {molecule} TBA...")
    
    else:
        raise ValueError(f"Projection for molecule {molecule} not supported")
    
    return wandb.Image(projection_img[0]), wandb.Image(projection_img[1]), wandb.Image(projection_img[2])

def compute_jacobian(cfg, model_wrapper, epoch):
    molecule = cfg.job.molecule
    device = model_wrapper.device
    
    if molecule == "alanine":
        pass
    
    elif molecule == "chignolin":
        raise ValueError(f"Jacobian for molecule {molecule} TBA...")
    
    else:
        raise ValueError(f"Jacobian for molecule {molecule} not supported")
        
    jacobian_img = None
    return wandb.Image(jacobian_img)
