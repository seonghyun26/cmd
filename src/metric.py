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
from .utils import compute_dihedral_torch
from .simulation import init_simulation, set_simulation
from .plot import plot_ad_potential, plot_dw_potential, plot_ad_cv

pairwise_distance = torch.cdist
ALDP_PHI_ANGLE = [4, 6, 8, 14]
ALDP_PSI_ANGLE = [6, 8, 14, 16]



def compute_dihedral(positions):
    """http://stackoverflow.com/q/20305272/1128289"""
    def dihedral(p):
        p = p.numpy()
        b = p[:-1] - p[1:]
        b[0] *= -1
        v = np.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
        
        # Normalize vectors
        v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
        b1 = b[1] / np.linalg.norm(b[1])
        x = np.dot(v[0], v[1])
        m = np.cross(v[0], b1)
        y = np.dot(m, v[1])
        
        return np.arctan2(y, x)
    
    angles = np.array(list(map(dihedral, positions)))
    return angles  

def compute_dihedral_torch(positions):
    """
    Computes the dihedral angle for batches of points P1, P2, P3, P4.
    Args:
        positions: (bacth_size, 4, 3)
    Returns:
        A tensor of shape (batch_size,) containing the dihedral angles in radians.
    """

    P1 = positions[:, 0]
    P2 = positions[:, 1]
    P3 = positions[:, 2]
    P4 = positions[:, 3]
    b1 = P2 - P1
    b2 = P3 - P2
    b3 = P4 - P3
    
    b2_norm = b2 / b2.norm(dim=1, keepdim=True)
    n1 = torch.cross(b1, b2, dim=1)
    n2 = torch.cross(b2, b3, dim=1)
    n1_norm = n1 / n1.norm(dim=1, keepdim=True)
    n2_norm = n2 / n2.norm(dim=1, keepdim=True)
    m1 = torch.cross(n1_norm, b2_norm, dim=1)
    
    # Compute cosine and sine of the angle
    x = (n1_norm * n2_norm).sum(dim=1)
    y = (m1 * n2_norm).sum(dim=1)
    angle = - torch.atan2(y, x)
    
    return angle

def compute_phi_psi(cfg, state):
    if cfg.data.molecule == "alanine":
        phi = np.expand_dims(compute_dihedral(state[:, ALDP_PHI_ANGLE]), axis=1)
        psi = np.expand_dims(compute_dihedral(state[:, ALDP_PSI_ANGLE]), axis=1)
        phi_psi = torch.cat([torch.from_numpy(phi), torch.from_numpy(psi)], dim=1)
    else:
        raise ValueError(f"Phi, Psi for molecule {molecule} TBA...")
    
    return phi_psi

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




def compute_thp(cfg, trajectory_list, goal_state):
    molecule = cfg.job.molecule
    sample_num = cfg.job.sample_num
    cv_bound = cfg.job.metrics.thp.cv_bound
    hit_rate = 0.0
    hit_mask = []
    hit_index = []
    
    if molecule == "alanine":
        psi_goal = compute_dihedral(goal_state[0, ALDP_PSI_ANGLE].reshape(-1, len(ALDP_PSI_ANGLE), 3))
        phi_goal = compute_dihedral(goal_state[0, ALDP_PSI_ANGLE].reshape(-1, len(ALDP_PHI_ANGLE), 3))
        for i in tqdm(
            range(sample_num),
            desc = "Computing THP for trajectories"
        ):
            psi = compute_dihedral(trajectory_list[i, :, ALDP_PSI_ANGLE])
            phi = compute_dihedral(trajectory_list[i, :, ALDP_PHI_ANGLE])
            psi_hit_distance = np.abs(psi - psi_goal)
            phi_hit_distance = np.abs(phi - phi_goal)
            cv_distance = np.sqrt(psi_hit_distance ** 2 + phi_hit_distance ** 2)
            hit_in_path = (psi_hit_distance < cv_bound) & (phi_hit_distance < cv_bound)
            hit_index_in_path = np.argmin(cv_distance)
            if np.any(hit_mask):
                hit_rate += 1.0
                hit_mask.append(True)
                hit_index.append(hit_index_in_path)
            else:
                hit_mask.append(False)
                hit_index.append(-1)
                
        hit_rate /= sample_num
        hit_mask = torch.tensor(hit_mask, dtype=torch.bool, device=trajectory_list.device)
        hit_index = torch.tensor(hit_index, dtype=torch.int32, device=trajectory_list.device)
        
        # phi = compute_dihedral(last_state[:, ALDP_PHI_ANGLE])
        # psi = compute_dihedral(last_state[:, ALDP_PSI_ANGLE])
        # phi_goal = compute_dihedral(goal_state[:, ALDP_PHI_ANGLE])
        # psi_goal = compute_dihedral(goal_state[:, ALDP_PSI_ANGLE])
        # hit = (np.abs(psi - psi_goal) < cv_bound) & (np.abs(phi - phi_goal) < cv_bound)
        # hit_rate = hit.sum() / hit.shape[0]
    elif molecule == "double-well":
        hit = (np.abs(last_state[:, 0] - goal_state[:, 0]) < cv_bound) & (np.abs(last_state[:, 1] - goal_state[:, 1]) < cv_bound)
        hit_rate = torch.all(hit) / hit.shape[0]
    else:
        raise ValueError(f"THP for molecule {molecule} TBA")
    
    return hit_rate, hit_mask, hit_index


def compute_epd(cfg, trajectory_list, goal_state, hit_mask, hit_index):
    molecule = cfg.job.molecule
    atom_num = cfg.data.atom
    unit_scale_factor = 1000
    hit_trajectory = trajectory_list[hit_mask]
    goal_state = goal_state[hit_mask]
    epd = 0.0
    
    # Compute EPD
    if molecule == "alanine":
        for i in range(hit_trajectory.shape[0]):
            hit_state = hit_trajectory[i, hit_index[i]]
            matrix_f_norm = torch.sqrt(torch.square(
                pairwise_distance(hit_state, hit_state) - pairwise_distance(goal_state, goal_state)
            ).sum((1, 2)))
            epd += matrix_f_norm / (atom_num ** 2) * unit_scale_factor
    else:
        raise ValueError(f"EPD for molecule {molecule} TBA")
    epd /= hit_trajectory.shape[0]
    
    # TODO: RMSD computation for alanine
    if molecule == "alanine":
        rmsd = torch.tensor(0)
        pass
    else:
        raise ValueError(f"RMSD for molecule {molecule} TBA")
    
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
                desc=f"Computing energy for {trajectory_list.shape[0]} trajectories"
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
    if molecule == "alanine":
        # Load start, goal state and compute phi, psi
        start_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.start_state}.pdb").xyz
        goal_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.goal_state}.pdb").xyz
        start_state = torch.tensor(start_state_xyz)
        goal_state = torch.tensor(goal_state_xyz)
        phi_start = compute_dihedral(start_state[:, ALDP_PHI_ANGLE])
        psi_start = compute_dihedral(start_state[:, ALDP_PSI_ANGLE])
        phi_goal = compute_dihedral(goal_state[:, ALDP_PHI_ANGLE])
        psi_goal = compute_dihedral(goal_state[:, ALDP_PSI_ANGLE])
    
        # Compute phi, psi from trajectory_list
        phi_traj_list = np.array([compute_dihedral(trajectory[:, ALDP_PHI_ANGLE]) for trajectory in trajectory_list])
        psi_traj_list = np.array([compute_dihedral(trajectory[:, ALDP_PSI_ANGLE]) for trajectory in trajectory_list])
        
        ram_plot_img = plot_ad_potential(
            traj_dihedral = (phi_traj_list, psi_traj_list),
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            cv_bound_use = cfg.job.metrics.projection.bound_use,
            cv_bound = cfg.job.metrics.thp.cv_bound,
            epoch = epoch,
            name = "paths"
        )
        
        transition_path_plot_img = plot_ad_potential(
            traj_dihedral = (phi_traj_list[hit_mask], psi_traj_list[hit_mask]),
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

    else:
        raise ValueError(f"Ramachandran plot for molecule {molecule} TBA...")
    
    return wandb.Image(ram_plot_img), wandb.Image(transition_path_plot_img)

def compute_projection(cfg, model_wrapper, epoch):
    molecule = cfg.job.molecule
    device = model_wrapper.device
    
    if molecule == "alanine":       
        data_dir = f"{cfg.data.dir}/projection"
        if cfg.model.input == "distance":
            heavy_atom_distance = torch.load(f"{data_dir}/alanine_heavy_atom_distance.pt").to(device)
            if cfg.model.name in ["cvmlp", "cvmlp-bn"]:
                temperature = torch.tensor(cfg.job.simulation.temperature).repeat(heavy_atom_distance.shape[0], 1).to(device)
                projected_cv = model_wrapper.model(torch.cat([heavy_atom_distance, temperature], dim=1))
            elif cfg.model.name in ["deeplda", "deeptda", "deeptica", "aecv", "vaecv", "beta-vae"]:
                projected_cv = model_wrapper.model(heavy_atom_distance)
            else:
                raise ValueError(f"Model {cfg.model.name} not found")
        elif cfg.model.input == "coordinate":
            coordinate = torch.load(f"{data_dir}/alanine_coordinate.pt").to(device)
            coordinate = coordinate.reshape(coordinate.shape[0], -1)
            temperature = torch.tensor(cfg.job.simulation.temperature).repeat(coordinate.shape[0], 1).to(device)
            projected_cv = model_wrapper.model(torch.cat([coordinate, temperature], dim=1), transformed=False)
        else:
            raise ValueError(f"Input type {cfg.model.input} not found")
        
        phis = np.load(f"{data_dir}/alanine_phi_list.npy")
        psis = np.load(f"{data_dir}/alanine_psi_list.npy")
        
        start_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.start_state}.pdb").xyz
        goal_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.goal_state}.pdb").xyz
        start_state = torch.tensor(start_state_xyz)
        goal_state = torch.tensor(goal_state_xyz)
        phi_start = compute_dihedral(start_state[:, ALDP_PHI_ANGLE])
        psi_start = compute_dihedral(start_state[:, ALDP_PSI_ANGLE])
        phi_goal = compute_dihedral(goal_state[:, ALDP_PHI_ANGLE])
        psi_goal = compute_dihedral(goal_state[:, ALDP_PSI_ANGLE])
        
        projection_img = plot_ad_cv(
            phi = phis,
            psi = psis,
            cv = projected_cv,
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
