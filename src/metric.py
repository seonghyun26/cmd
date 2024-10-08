import os
import torch
import hydra
import wandb
import jax

import jax.numpy as jnp
import numpy as np
import mdtraj as md

from tqdm import tqdm
from .plot import plot_ad_potential, plot_dw_potential
from .data import Synthetic
from .simulation import init_simulation, set_simulation
from mdtraj.geometry import indices_phi, indices_psi

pairwise_distance = torch.cdist


def compute_epd(cfg, trajectory_list, goal_state):
    molecule = cfg.job.molecule
    atom_num = cfg.data.atom
    unit_scale_factor = 1000
    last_state = trajectory_list[:, -1]
    
    if molecule == "alanine":
        matrix_f_norm = torch.sqrt(torch.square(
            pairwise_distance(last_state, last_state) - pairwise_distance(goal_state, goal_state)
        ).sum((1, 2)))
        epd = matrix_f_norm / (atom_num ** 2) * unit_scale_factor
    elif molecule == "double-well":
        # RMSD for double well
        epd = torch.sqrt(torch.sum(torch.square(last_state - goal_state), dim=1)).mean()
    else:
        raise ValueError(f"EPD for molecule {molecule} TBA")
    
    return epd.mean().item()


def compute_thp(cfg, trajectory_list, goal_state):
    molecule = cfg.job.molecule
    sample_num = cfg.job.sample_num
    last_state = trajectory_list[:, -1]
    cv_bound = cfg.job.thp_cv_bound
    
    if molecule == "alanine":
        phi_angle = [1, 6, 8, 14]
        psi_angle = [6, 8, 14, 16]
        
        phi = compute_dihedral(last_state[:, phi_angle])
        psi = compute_dihedral(last_state[:, psi_angle])
        phi_goal = compute_dihedral(goal_state[:, phi_angle])
        psi_goal = compute_dihedral(goal_state[:, psi_angle])
        
        hit = (np.abs(psi - psi_goal) < cv_bound) & (np.abs(phi - phi_goal) < cv_bound)
        hit_rate = hit.sum() / hit.shape[0]
    elif molecule == "double-well":
        hit = (np.abs(last_state[:, 0] - goal_state[:, 0]) < cv_bound) & (np.abs(last_state[:, 1] - goal_state[:, 1]) < cv_bound)
        hit_rate = torch.all(hit) / hit.shape[0]
    else:
        raise ValueError(f"THP for molecule {molecule} TBA")
    
    return hit_rate


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
    

def compute_phi_psi(cfg, state):
    if cfg.data.molecule == "alanine":
        phi_angle = [1, 6, 8, 14]
        psi_angle = [6, 8, 14, 16]
        phi = compute_dihedral(state[phi_angle])
        psi = compute_dihedral(state[psi_angle])
        phi_psi = torch.stack([phi, psi], dim=1)
    else:
        raise ValueError(f"Phi, Psi for molecule {molecule} TBA...")
    
    return phi_psi


def compute_energy(cfg, trajectory_list, goal_state):
    molecule = cfg.job.molecule
    sample_num = trajectory_list.shape[0]
    path_length = trajectory_list.shape[1]
    
    if molecule == "alanine":
        goal_state_file_path = f"data/{cfg.data.molecule}/{cfg.job.goal_state}.pdb"
        goal_simulation = init_simulation(cfg, goal_state_file_path)
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
    elif molecule == "double-well":
        synthetic = Synthetic()
        path_energy_list = [synthetic.potential(trajectory) for trajectory in trajectory_list]
        path_energy_list = np.array(path_energy_list)
        path_maximum_energy = np.max(path_energy_list, axis=1)
        path_final_energy_error = np.array(path_energy_list[:, -1]) - synthetic.potential(goal_state)
    else: 
        raise ValueError(f"Energy for molecule {molecule} TBA")
    
    return path_maximum_energy.mean(), path_energy_list[:, -1].mean(), path_final_energy_error.mean()


def potential_energy(cfg, trajectory):
    molecule = cfg.job.molecule
    energy_list = []
    
    if molecule == "alanine":
        pbb_file_path = f"data/{cfg.data.molecule}/c5.pdb"
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


def compute_ram(cfg, trajectory_list, epoch):
    molecule = cfg.job.molecule
    if molecule == "alanine":
        landscape_path = f"./data/{cfg.job.molecule}/final_frame.dat"
        ad_potential = AlaninePotential(landscape_path)
        phi_angle = [1, 6, 8, 14]
        psi_angle = [6, 8, 14, 16]
        
        # Load start, goal state and compute phi, psi
        start_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.start_state}.pdb").xyz
        goal_state_xyz = md.load(f"./data/{cfg.job.molecule}/{cfg.job.goal_state}.pdb").xyz
        start_state = torch.tensor(start_state_xyz)
        goal_state = torch.tensor(goal_state_xyz)
        phi_start = compute_dihedral(start_state[:, phi_angle])
        psi_start = compute_dihedral(start_state[:, psi_angle])
        phi_goal = compute_dihedral(goal_state[:, phi_angle])
        psi_goal = compute_dihedral(goal_state[:, psi_angle])
    
        # Compute phi, psi from trajectory_list
        phi_traj_list = np.array([compute_dihedral(trajectory[:, phi_angle]) for trajectory in trajectory_list])
        psi_traj_list = np.array([compute_dihedral(trajectory[:, psi_angle]) for trajectory in trajectory_list])
        
        ram_plot_img = plot_ad_potential(
            potential = ad_potential,
            traj_dihedral = (phi_traj_list, psi_traj_list),
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            epoch = epoch
        )
    elif molecule == "double-well":
        device = trajectory_list.device
        dw_potential_path = f"./data/{cfg.job.molecule}/potential.dat"
        dw_potential = DoubleWellPotential(dw_potential_path)
        start_state = torch.tensor([-1.118, 0], dtype=torch.float32).to(device)
        goal_state = torch.tensor([1.118, 0], dtype=torch.float32).to(device)
        
        ram_plot_img = plot_dw_potential(
            potential = dw_potential,
            traj = trajectory_list,
            start = start_state,
            goal = goal_state,
            epoch = epoch
        )

    else:
        raise ValueError(f"Ramachandran plot for molecule {molecule} TBA...")
    
    return wandb.Image(ram_plot_img)


class AlaninePotential():
    def __init__(self, landscape_path):
        super().__init__()
        self.open_file(landscape_path)

    def open_file(self, landscape_path):
        with open(landscape_path) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor(np.array([x, y]))
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z

    def drift(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp[:, :2].double(), loc.double(), p=2)
        index = distances.argsort(dim=1)[:, :3]

        x = index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        dims = torch.stack([x, y], 2)

        min = dims.argmin(dim=1)
        max = dims.argmax(dim=1)

        min_x = min[:, 0]
        min_y = min[:, 1]
        max_x = max[:, 0]
        max_y = max[:, 1]

        min_x_dim = dims[range(dims.shape[0]), min_x, :]
        min_y_dim = dims[range(dims.shape[0]), min_y, :]
        max_x_dim = dims[range(dims.shape[0]), max_x, :]
        max_y_dim = dims[range(dims.shape[0]), max_y, :]

        min_x_val = self.data[min_x_dim[:, 0], min_x_dim[:, 1]]
        min_y_val = self.data[min_y_dim[:, 0], min_y_dim[:, 1]]
        max_x_val = self.data[max_x_dim[:, 0], max_x_dim[:, 1]]
        max_y_val = self.data[max_y_dim[:, 0], max_y_dim[:, 1]]

        grad = -1 * torch.stack([max_y_val - min_y_val, max_x_val - min_x_val], dim=1)

        return grad
    


class DoubleWellPotential():
    def __init__(self, landscape_path):
        super().__init__()
        self.open_file(landscape_path)

    def open_file(self, landscape_path):
        with open(landscape_path) as f:
            lines = f.readlines()

        spacing = 68
        dims = [spacing, spacing]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // spacing, i % spacing, :] = torch.tensor(np.array([x, y]))
            self.data[i // spacing, i % spacing] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z
    
    
