import torch

from tqdm import tqdm
from .utils import kabsch
from torch.distributions import Normal

from .dynamics import Alanine

class MDSimulation:
    def __init__(self, cfg, sample_num, device):
        self.device = device
        self.molecule = cfg.data.molecule
        self.start_state = cfg.job.start_state
        self.goal_state = cfg.job.goal_state
        self.sample_num = sample_num

        self._set_md(cfg)
        self.md_simulation_list = self._init_md_simulation_list(cfg)
        self.log_prob = Normal(0, self.std).log_prob

        # self.goal_position = self.goal_position - self.goal_position[
        #     :, self.heavy_atoms
        # ].mean(-2, keepdim=True)
        # R, t = kabsch(
        #     self.start_position[:, self.heavy_atoms],
        #     self.goal_position[:, self.heavy_atoms],
        # )
        # self.start_position = torch.matmul(self.start_position, R.transpose(-2, -1)) + t

    def _load_dynamics(self, cfg):
        molecule = cfg.data.molecule
        dynamics = None
        
        if molecule == "alanine":
            dynamics = Alanine(cfg, self.start_state)
        else:
            raise ValueError(f"Molecule {molecule} not found")
        
        assert dynamics is not None, f"Failed to load dynamics for {molecule}"
        
        return dynamics
    
    def _set_md(self, cfg):
        # goal_state_md = getattr(dynamics, self.molecule)(cfg, self.end_state)
        goal_state_md = self._load_dynamics(cfg)
        self.num_particles = cfg.data.atom
        self.heavy_atoms = goal_state_md.heavy_atoms
        self.energy_function = goal_state_md.energy_function
        self.goal_position = torch.tensor(
            goal_state_md.position, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.m = torch.tensor(
            goal_state_md.m,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(-1)
        self.std = torch.tensor(
            goal_state_md.std,
            dtype=torch.float,
            device=self.device,
        )

    def _init_md_simulation_list(self, cfg):
        md_simulation_list = []
        for _ in tqdm(
            range(self.sample_num),
            desc="Initializing MD Simulation",
        ):
            # md = getattr(dynamics, self.molecule.title())(args, self.start_state)
            md_simulation_list.append(self._load_dynamics(cfg))

        self.start_position = torch.tensor(
            md_simulation_list[0].position, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        
        return md_simulation_list

    def step(self, force):
        force = force.detach().cpu().numpy()
        for i in range(self.sample_num):
            self.md_simulation_list[i].step(force[i])

    def report(self):
        position_list, force_list = [], []
        for i in range(self.sample_num):
            position, force = self.md_simulation_list[i].report()
            position_list.append(position)
            force_list.append(force)

        position_list = torch.tensor(position_list, dtype=torch.float, device=self.device)
        force_list = torch.tensor(force_list, dtype=torch.float, device=self.device)
        return position_list, force_list

    def reset(self):
        for i in range(self.sample_num):
            self.md_simulation_list[i].reset()

    def set_position(self, positions):
        for i in range(self.sample_num):
            self.md_simulation_list[i].set_position(positions[i])

    def set_temperature(self, temperature):
        for i in range(self.sample_num):
            self.md_simulation_list[i].set_temperature(temperature)
            
            
class SteeredMDSimulation:
    def __init__(self, cfg, sample_num, device):
        self.device = device
        self.molecule = cfg.data.molecule
        self.start_state = cfg.job.start_state
        self.goal_state = cfg.job.goal_state
        self.sample_num = sample_num

        self._set_md(cfg)
        self.md_simulation_list = self._init_md_simulation_list(cfg)

    def _load_dynamics(self, cfg):
        molecule = cfg.data.molecule
        dynamics = None
        
        if molecule == "alanine":
            dynamics = SteeredAlanine(cfg, self.start_state)
        else:
            raise ValueError(f"Molecule {molecule} not found")
        
        assert dynamics is not None, f"Failed to load dynamics for {molecule}"
        
        return dynamics
    
    def _set_md(self, cfg):
        # goal_state_md = getattr(dynamics, self.molecule)(cfg, self.end_state)
        goal_state_md = self._load_dynamics(cfg)
        self.num_particles = cfg.data.atom
        self.heavy_atoms = goal_state_md.heavy_atoms
        self.energy_function = goal_state_md.energy_function
        self.goal_position = torch.tensor(
            goal_state_md.position, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.m = torch.tensor(
            goal_state_md.m,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(-1)
        self.std = torch.tensor(
            goal_state_md.std,
            dtype=torch.float,
            device=self.device,
        )

    def _init_md_simulation_list(self, cfg):
        md_simulation_list = []
        for _ in tqdm(
            range(self.sample_num),
            desc="Initializing MD Simulation",
        ):
            md_simulation_list.append(self._load_dynamics(cfg))

        self.start_position = torch.tensor(
            md_simulation_list[0].position, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        
        return md_simulation_list

    def step(self, force):
        force = force.detach().cpu().numpy()
        for i in range(self.sample_num):
            self.md_simulation_list[i].step(force[i])

    def report(self):
        position_list, force_list = [], []
        for i in range(self.sample_num):
            position, force = self.md_simulation_list[i].report()
            position_list.append(position)
            force_list.append(force)

        position_list = torch.tensor(position_list, dtype=torch.float, device=self.device)
        force_list = torch.tensor(force_list, dtype=torch.float, device=self.device)
        return position_list, force_list

    def reset(self):
        for i in range(self.sample_num):
            self.md_simulation_list[i].reset()

    def set_position(self, positions):
        for i in range(self.sample_num):
            self.md_simulation_list[i].set_position(positions[i])

    def set_temperature(self, temperature):
        for i in range(self.sample_num):
            self.md_simulation_list[i].set_temperature(temperature)