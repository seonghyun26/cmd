import torch
import openmm.unit as unit

from tqdm import tqdm
from .utils import kabsch
from torch.distributions import Normal

from .dynamics import Alanine, SteeredAlanine

ALANINE_HEAVY_ATOM_IDX = [
    1, 4, 5, 6, 8, 10, 14, 15, 16, 18
]

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
    def __init__(self, cfg, sample_num, model, device):
        self.model = model
        self.device = device
        self.molecule = cfg.data.molecule
        self.start_state = cfg.job.start_state
        self.goal_state = cfg.job.goal_state
        self.sample_num = sample_num
        self.force_type = cfg.job.simulation.force_type

        self._init_md_simulation_list(cfg)

    def _load_dynamics(self, cfg):
        dynamics = None
        molecule = cfg.data.molecule
        if molecule == "alanine":
            dynamics = SteeredAlanine(cfg, self.model, self.device)
        else:
            raise ValueError(f"Molecule {molecule} not found")
        
        assert dynamics is not None, f"Failed to load dynamics for {molecule}"
        
        return dynamics

    def _init_md_simulation_list(self, cfg):
        md_simulation_list = []
        for _ in tqdm(
            range(self.sample_num),
            desc="Initializing MD Simulation",
        ):
            md_simulation_list.append(self._load_dynamics(cfg))
        
        self.md_simulation_list = md_simulation_list

    def step(self, time):
        for i in range(self.sample_num):
            self.md_simulation_list[i].simulation.context.setParameter("time", time)
            self.md_simulation_list[i].step(time * self.md_simulation_list[i].timestep)
            
        # NOTE: parallel computation for model?
        # temperature = torch.tensor(self.md_simulation_list[0].temperature.value_in_unit(unit.kelvin), device=self.device).unsqueeze(0)
        # current_position_list = []
        # currrent_state_list = []
        # for i in range(self.sample_num):
        #     self.md_simulation_list[i].simulation.context.setParameter("time", time)
        #     current_position = torch.tensor(
        #         [list(p) for p in self.md_simulation_list[i].simulation.context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)],
        #         dtype=torch.float32, device = self.device
        #     ).reshape(-1)
        #     current_position.requires_grad = True
        #     current_position_list.append(current_position)
        #     if self.md_simulation_list[i].cfg.model.input == "distance":
        #         current_state = coordinate2distance(self.md_simulation_list[i].cfg.data.molecule, current_position)
        #     else:
        #         current_state = current_position
        #     currrent_state_list.append(torch.cat([current_state, temperature], dim=0))
        # currrent_state_batch = torch.stack(currrent_state_list, dim=0)
        # current_mlcv_batch = self.model(currrent_state_batch)
        # for i in range(self.sample_num):
        #     self.md_simulation_list[i].step(
        #         time * self.md_simulation_list[i].timestep,
        #         current_mlcv_batch[i],
        #         current_position_list[i]
        #         )

    def report(self):
        position_list = []
        for i in range(self.sample_num):
            position = self.md_simulation_list[i].report().value_in_unit(unit.nanometer)
            position_list.append(position)

        return position_list

    def reset(self):
        for i in range(self.sample_num):
            self.md_simulation_list[i].reset()
            
            
def coordinate2distance(molecule, position):
    '''
        Calculates distance between heavy atoms for Deep LDA
        input
            - molecule (str)
            - coordinates (torch.Tensor)
        output
            - distance (torch.Tensor)
    '''
    
    if molecule == "alanine":
        position = position.reshape(-1, 3)
        heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
        num_heavy_atoms = len(heavy_atom_position)
        distance = []
        for i in range(num_heavy_atoms):
            for j in range(i+1, num_heavy_atoms):
                distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
        distance = torch.stack(distance)
    else:
        raise ValueError(f"Heavy atom distance for molecule {molecule} not supported")
    
    return distance