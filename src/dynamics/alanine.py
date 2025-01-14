import os
import torch
import numpy as np

import openmm as mm
import openmm.unit as unit

from openmm import app
from openmmtools.integrators import VVVRIntegrator
# from openmmtorch import TorchForce


from .base import BaseDynamics
from ..utils import kabsch_rmsd
from ..utils import compute_dihedral, compute_dihedral_torch
from ..simulation import load_forcefield, load_system


MLCOLVAR_METHODS = ["deeplda", "deeptda", "deeptica", "aecv", "gnncv"]
ALANINE_HEAVY_ATOM_IDX = [
    1, 4, 5, 6, 8, 10, 14, 15, 16, 18
]

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


class Alanine(BaseDynamics):
    def __init__(self, cfg, state):
        super().__init__(cfg, state)

    def setup(self):
        forcefield = app.ForceField("amber99sbildn.xml")
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        external_force = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")

        # creating the parameters
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        system.addForce(external_force)
        for i in range(len(pdb.positions)):
            external_force.addParticle(i, [0, 0, 0])

        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )

        integrator.setConstraintTolerance(0.00001)

        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)

        return pdb, integrator, simulation, external_force
    
class SteeredAlanine:
    def __init__(self, cfg, model=None, device='cpu'):
        self.cfg = cfg
        self.model = model
        self.device = device
        
        self.k = cfg.job.simulation.k
        self.temperature = cfg.job.simulation.temperature * unit.kelvin
        self.friction = cfg.job.simulation.friction / unit.femtoseconds
        self.timestep = cfg.job.simulation.timestep * unit.femtoseconds
        self.molecule = cfg.job.molecule
        self.time_horizon = cfg.job.simulation.time_horizon
        self.force_type = cfg.job.simulation.force_type

        if self.force_type in MLCOLVAR_METHODS or self.force_type == "cvmlp":
            from mlcolvar.core import Normalization
            self.preprocess = Normalization(in_features=45, mode="mean_std").to(self.device)
        
        # Load pdb files
        start_pdb = app.PDBFile(f"./data/{self.molecule}/{cfg.job.start_state}.pdb")
        goal_pdb = app.PDBFile(f"./data/{self.molecule}/{cfg.job.goal_state}.pdb")
        
        # Load simulation components
        self.forcefield = load_forcefield(cfg, self.molecule)
        self.system = load_system(cfg, self.molecule, start_pdb, self.forcefield)
        self._set_start_position(start_pdb, self.system)
        self._set_goal_position(goal_pdb, self.system)
        
        # Set cv force
        self._set_custom_force()
        
        # Set simulation
        integrator = self._new_integrator()
        integrator.setConstraintTolerance(0.00001)
        self.simulation = app.Simulation(start_pdb.topology, self.system, integrator)
        self.simulation.context.setPositions(self.start_position)
    
    
    def step(self, time):
        self.simulation.context.setParameter("time", time)
        if self.force_type in ["rmsd", "torsion"]:
            pass
        
        elif self.force_type in MLCOLVAR_METHODS:
            # Get position
            current_position = torch.tensor(
                [list(p) for p in self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)],
                dtype=torch.float32, device = self.device
            ).reshape(-1)
            current_position.requires_grad = True
            heavy_atom_distance = self.preprocess(coordinate2distance(self.cfg.job.molecule, current_position))
            current_mlcv = self.model(heavy_atom_distance)
            start_mlcv = self.start_mlcv
            goal_mlcv = self.goal_mlcv             
            
            # Set external forces
            current_target_mlcv = start_mlcv + (goal_mlcv - start_mlcv) * (time / self.time_horizon).value_in_unit(unit.femtosecond)
            mlcv_difference = 0.5 * self.k * torch.linalg.norm(goal_mlcv - current_mlcv, ord=2)
            bias_force = torch.autograd.grad(mlcv_difference, current_position)[0].reshape(-1, 3)
            external_force = self.simulation.system.getForce(5)
            for i in range(bias_force.shape[0]):
                external_force.setParticleParameters(i, i, bias_force[i])
            external_force.updateParametersInContext(self.simulation.context)
        
        elif self.force_type == "cvmlp":
            # Compute CV for states
            temperature = torch.tensor(self.temperature.value_in_unit(unit.kelvin), device=self.device).unsqueeze(0)
            current_mlcv, current_position = self._get_current_mlcv(temperature)
            start_mlcv = self.start_mlcv
            goal_mlcv = self.goal_mlcv
            
            # Set external forces
            current_target_mlcv = start_mlcv + (goal_mlcv - start_mlcv) * (time / self.time_horizon).value_in_unit(unit.femtosecond)
            mlcv_difference = 0.5 * self.k * torch.linalg.norm(current_target_mlcv - current_mlcv, ord=2)
            bias_force = torch.autograd.grad(mlcv_difference, current_position)[0].reshape(-1, 3)
            external_force = self.simulation.system.getForce(5)
            for i in range(bias_force.shape[0]):
                external_force.setParticleParameters(i, i, bias_force[i])
            external_force.updateParametersInContext(self.simulation.context)

        self.simulation.step(1)

    def report(self):
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        return positions

    def reset(self):
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)
    
    def _new_integrator(self):
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        return integrator
    
    def _set_start_position(self, pdb, system):
        # Set start position
        integrator = self._new_integrator()
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        self.start_position = simulation.context.getState(getPositions=True).getPositions()
        temperature = torch.tensor(self.temperature.value_in_unit(unit.kelvin), device=self.device).unsqueeze(0)
        
        # Set start mlcv
        start_position = torch.tensor(
            [list(p) for p in self.start_position.value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(-1)
        start_position.requires_grad = True
        
        if self.force_type in MLCOLVAR_METHODS:
            start_heavy_atom_distance = self.preprocess(coordinate2distance(self.cfg.job.molecule, start_position))
            self.start_mlcv = self.model(start_heavy_atom_distance) 
        elif self.force_type == "cvmlp":
            if self.cfg.model.input == "distance":
                start_heavy_atom_distance = self.preprocess(coordinate2distance(self.cfg.job.molecule, start_position))
                self.start_mlcv = self.model(torch.cat([start_heavy_atom_distance, temperature], dim=0).reshape(1, -1))
            else:
                self.start_mlcv = self.model(torch.cat([start_position, temperature], dim=0).reshape(1, -1))
        elif self.force_type in ["rmsd", "torsion"]:
            pass
        else:
            raise ValueError(f"Force type {self.force_type} not found")
        
    def _set_goal_position(self, pdb, system):
        # Set goal position
        integrator = self._new_integrator()
        simulation = app.Simulation(pdb.topology, system, integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        self.goal_position = simulation.context.getState(getPositions=True).getPositions()
        temperature = torch.tensor(self.temperature.value_in_unit(unit.kelvin), device=self.device).unsqueeze(0)
        
        # Set goal mlcv
        goal_position = torch.tensor(
            [list(p) for p in self.goal_position.value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(-1)
        goal_position.requires_grad = True
        
        if self.force_type in MLCOLVAR_METHODS:
            goal_heavy_atom_distance = self.preprocess(coordinate2distance(self.cfg.job.molecule, goal_position))
            self.goal_mlcv = self.model(goal_heavy_atom_distance) 
        elif self.force_type == "cvmlp":  
            if self.cfg.model.input == "distance":
                goal_heavy_atom_distance = self.preprocess(coordinate2distance(self.cfg.job.molecule, goal_position))
                self.goal_mlcv = self.model(torch.cat([goal_heavy_atom_distance, temperature], dim=0).reshape(1, -1))
            else:
                self.goal_mlcv = self.model(torch.cat([goal_position, temperature], dim=0).reshape(1, -1))
        elif self.force_type in ["rmsd", "torsion"]:
            pass
        else:
            raise ValueError(f"Force type {self.force_type} not found")
        
        
    def _set_custom_force(self):        
        if self.force_type == "torsion":
            ALSP_PSI_ANGLE = [6, 8, 14, 16]
            ALDP_PHI_ANGLE = [4, 6, 8, 14]
            start_position = np.array(
                [list(p) for p in self.start_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )
            goal_position = np.array(
                [list(p) for p in self.goal_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )
            start_psi = compute_dihedral(start_position[ALSP_PSI_ANGLE])
            start_phi = compute_dihedral(start_position[ALDP_PHI_ANGLE])
            goal_psi = compute_dihedral(goal_position[ALSP_PSI_ANGLE])
            goal_phi = compute_dihedral(goal_position[ALDP_PHI_ANGLE])

            # Create CustomTorsionForce for phi and psi angles
            custom_cv_force = mm.CustomTorsionForce(
                "0.5 * k * (theta - (theta_start + (theta_goal - theta_start) * (time / total_time)))^2"
            )
            custom_cv_force.addPerTorsionParameter("theta_start")
            custom_cv_force.addPerTorsionParameter("theta_goal")
            custom_cv_force.addTorsion(*ALSP_PSI_ANGLE, [start_psi, goal_psi])
            custom_cv_force.addTorsion(*ALDP_PHI_ANGLE, [start_phi, goal_phi])
            custom_cv_force.addGlobalParameter("k", self.k)
            custom_cv_force.addGlobalParameter("time", 0)
            custom_cv_force.addGlobalParameter("total_time", self.time_horizon * self.timestep)
            self.system.addForce(custom_cv_force)
        
        elif self.force_type == "rmsd":
            start_position = torch.tensor(
                [list(p) for p in self.start_position.value_in_unit(unit.nanometer)],
                dtype=torch.float32, device = self.device
            ).reshape(-1, 3)
            goal_position = torch.tensor(
                [list(p) for p in self.goal_position.value_in_unit(unit.nanometer)],
                dtype=torch.float32, device = self.device
            ).reshape(-1, 3)

            custom_cv_force = mm.CustomCVForce(
                "0.5 * k * ( rmsd - start_rmsd * (1 - time / total_time) )^2"
            )
            custom_cv_force.addCollectiveVariable("rmsd", mm.RMSDForce(goal_position.cpu().numpy()))
            custom_cv_force.addGlobalParameter("start_rmsd", kabsch_rmsd(start_position, goal_position).squeeze().item())
            custom_cv_force.addGlobalParameter("k", self.k)
            custom_cv_force.addGlobalParameter("time", 0)
            custom_cv_force.addGlobalParameter("total_time", self.time_horizon * self.timestep)
            self.system.addForce(custom_cv_force)
        
        elif self.force_type in MLCOLVAR_METHODS:
            # Get positions
            start_position = torch.tensor(
                [list(p) for p in self.start_position.value_in_unit(unit.nanometer)],
                dtype=torch.float32, device = self.device
            ).reshape(-1)
            goal_position = torch.tensor(
                [list(p) for p in self.goal_position.value_in_unit(unit.nanometer)],
                dtype=torch.float32, device = self.device
            ).reshape(-1)
            start_position.requires_grad = True
            
            # Get leanred collective variables
            # def position2mlcv(cfg, position):
            #     heavy_atom_distance = coordinate2distance(cfg.job.molecule, position)
            #     mlcv = self.model(heavy_atom_distance).cpu().detach().numpy()
            #     return mlcv
            # self.position2mlcv = position2mlcv
            
            # if self.cfg.data.input == "distance":
            #     start_mlcv = self.position2mlcv(self.cfg, start_position)
            #     goal_mlcv = self.position2mlcv(self.cfg, goal_position)
            # elif self.cfg.data.input == "xyz":
            #     start_mlcv = self.model(start_position)
            #     goal_mlcv = self.model(goal_position)
                

            # Create custom force
            # custom_cv_force = mm.CustomCVForce(
            #     "0.5 * k * (mlcv - (start_mlcv + (goal_mlcv - start_mlcv) * (time / total_time)))^2 "
            # )
            # custom_cv_force = mm.CustomCVForce(
            #     "0.5 * k * (mlcv - (start_mlcv + (goal_mlcv - start_mlcv) * (time / total_time)))^2 + 0.001 * dummy"
            # )
            
            # v1: Collective variable
            # custom_cv_force.addCollectiveVariable("mlcv", start_mlcv)
            
            # v2: add by global parameter
            # custom_cv_force.addGlobalParameter("mlcv", start_mlcv)
            
            # v3: add dummy
            # custom_cv_force.addCollectiveVariable("dummy", mm.CustomExternalForce("0"))
            
            # custom_cv_force.addGlobalParameter("start_mlcv", start_mlcv)
            # custom_cv_force.addGlobalParameter("goal_mlcv", goal_mlcv)     
            
            # v4: external force
            external_force = mm.CustomExternalForce(" (fx*x + fy*y + fz*z) ")
            external_force.addGlobalParameter("time", 0)
            external_force.addGlobalParameter("total_time", self.time_horizon * self.timestep)
            external_force.addPerParticleParameter("fx")
            external_force.addPerParticleParameter("fy")
            external_force.addPerParticleParameter("fz")
            for i in range(22):
                external_force.addParticle(i, [0, 0, 0])
            self.system.addForce(external_force)          
               
            # Debugging
            # print(f"Start mlcv: {start_mlcv}")
            # print(f"Goal mlcv: {goal_mlcv}")
            
        elif self.force_type in ["cvmlp"]:
            external_force = mm.CustomExternalForce(" (fx*x + fy*y + fz*z) ")
            # external_force.addGlobalParameter("k", self.k)
            external_force.addGlobalParameter("time", 0)
            external_force.addGlobalParameter("total_time", self.time_horizon * self.timestep)
            external_force.addPerParticleParameter("fx")
            external_force.addPerParticleParameter("fy")
            external_force.addPerParticleParameter("fz")
            for i in range(22):
                external_force.addParticle(i, [0, 0, 0])
            self.system.addForce(external_force)          

        else:
            raise ValueError(f"Force type {self.force_type} not found")
        
        return
    
    def _get_current_mlcv(self, temperature):
        current_position = torch.tensor(
            [list(p) for p in self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(-1)
        current_position.requires_grad = True
        if self.cfg.model.input == "distance":
            heavy_atom_distance = self.preprocess(coordinate2distance(self.cfg.job.molecule, current_position))
            current_mlcv = self.model(torch.cat([heavy_atom_distance, temperature], dim=0).reshape(1, -1))
        else:
            current_mlcv = self.model(torch.cat([current_position, temperature], dim=0).reshape(1, -1))
        
        return current_mlcv, current_position
    
