import os
import torch
import numpy as np

import openmm as mm
import openmm.unit as unit

from openmm import app
from openmmtools.integrators import VVVRIntegrator

from .base import BaseDynamics
from ..utils import *
from ..simulation import load_forcefield, load_system


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
        platform = mm.Platform.getPlatformByName(self.cfg.job.simulation.platform)
        if self.cfg.job.simulation.platform in ["CUDA", "OpenCL"]:
            properties = {'DeviceIndex': '0', 'Precision': self.cfg.job.simulation.precision}
        elif self.cfg.job.simulation.platform == "CPU":
            properties = {}
        else:
            raise ValueError(f"Platform {self.cfg.job.simulation.platform} not found")
        self.simulation = app.Simulation(
            start_pdb.topology,
            self.system,
            integrator,
            platform,
            properties
        )
        simulation.context.setPositions(pdb.positions)

        return pdb, integrator, simulation, external_force
    
class SteeredAlanine:
    def __init__(
        self,
        cfg,
        model_wrapper,
    ):
        self.cfg = cfg
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = model_wrapper.device
        
        self.k = cfg.job.simulation.k
        self.temperature = cfg.job.simulation.temperature * unit.kelvin
        self.friction = cfg.job.simulation.friction / unit.femtoseconds
        self.timestep = cfg.job.simulation.timestep * unit.femtoseconds
        self.molecule = cfg.job.molecule
        self.time_horizon = cfg.job.simulation.time_horizon
        self.force_type = cfg.job.simulation.force_type
        
        # Load pdb files
        start_pdb = app.PDBFile(f"./data/{self.molecule}/{cfg.job.start_state}.pdb")
        goal_pdb = app.PDBFile(f"./data/{self.molecule}/{cfg.job.goal_state}.pdb")
        
        # Load simulation components
        self.forcefield = load_forcefield(cfg, self.molecule)
        self.system = load_system(cfg, self.molecule, start_pdb, self.forcefield)
        self._set_start_position(start_pdb, self.system)
        self._set_goal_position(goal_pdb, self.system)
        self._set_custom_force()
        
        # Set simulation
        integrator = self._new_integrator()
        integrator.setConstraintTolerance(0.00001)
        platform = mm.Platform.getPlatformByName(self.cfg.job.simulation.platform)
        if self.cfg.job.simulation.platform in ["CUDA", "OpenCL"]:
            properties = {'DeviceIndex': '0', 'Precision': self.cfg.job.simulation.precision}
        elif self.cfg.job.simulation.platform == "CPU":
            properties = {}
        else:
            raise ValueError(f"Platform {self.cfg.job.simulation.platform} not found")
        self.simulation = app.Simulation(
            start_pdb.topology,
            self.system,
            integrator,
            platform,
            properties
        )
        self.simulation.context.setPositions(self.start_position)
        self.simulation.minimizeEnergy()
    
    def step(self, time, mlcv=None, position=None):
        # Set & get simulation information
        self.simulation.context.setParameter("time", time)
        temperature = torch.tensor(
            self.temperature.value_in_unit(unit.kelvin),
            device=self.device
        ).reshape(1, 1)
        current_position = torch.tensor(
            [list(p) for p in self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(1, -1)
        current_position.requires_grad = True
        
        # Compute mlcv
        current_mlcv = self.model_wrapper.compute_cv(
            current_position = current_position,
            temperature = temperature
        )
        start_mlcv = self.start_mlcv
        goal_mlcv = self.goal_mlcv
        
        # Set external force
        if self.force_type in ["rmsd", "torsion"]:
            pass
        
        elif self.force_type in MLCOLVAR_METHODS or ["gnncv", "clcv"] or CLCV_METHODS:
            if self.cfg.job.simulation.force_version == "v1":
                mlcv_difference = torch.linalg.norm(goal_mlcv - current_mlcv, ord=2)
            elif self.cfg.job.simulation.force_version == "v2":
                current_target_mlcv = start_mlcv + (goal_mlcv - start_mlcv) * (time / self.time_horizon).value_in_unit(unit.femtosecond)
                mlcv_difference = 0.5 * self.k * torch.linalg.norm(current_target_mlcv - current_mlcv, ord=2)
            elif self.cfg.job.simulation.force_version == "v3":
                current_target_mlcv = start_mlcv + (goal_mlcv - start_mlcv) * (time / self.time_horizon).value_in_unit(unit.femtosecond)
                mlcv_difference = torch.linalg.norm(current_target_mlcv - current_mlcv, ord=2)
            
            bias_force = torch.autograd.grad(mlcv_difference, current_position)[0].reshape(-1, 3)
            external_force = self.simulation.system.getForce(5)
            for i in range(bias_force.shape[0]):
                external_force.setParticleParameters(i, i, bias_force[i])
            external_force.updateParametersInContext(self.simulation.context)
        
        else:
            raise ValueError(f"Force type {self.force_type} not found")
        
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
        platform = mm.Platform.getPlatformByName(self.cfg.job.simulation.platform)
        if self.cfg.job.simulation.platform in ["CUDA", "OpenCL"]:
            properties = {'DeviceIndex': '0', 'Precision': self.cfg.job.simulation.precision}
        elif self.cfg.job.simulation.platform == "CPU":
            properties = {}
        else:
            raise ValueError(f"Platform {self.cfg.job.simulation.platform} not found")
        simulation = app.Simulation(
            pdb.topology,
            system,
            integrator,
            platform,
            properties
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        self.start_position = simulation.context.getState(getPositions=True).getPositions()
        temperature = torch.tensor(self.temperature.value_in_unit(unit.kelvin), device=self.device).reshape(1, 1)
        
        # Set start mlcv
        start_position = torch.tensor(
            [list(p) for p in self.start_position.value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(1, -1)
        start_position.requires_grad = True
        self.start_mlcv = self.model_wrapper.compute_cv(start_position, temperature)
        
        # if self.force_type in MLCOLVAR_METHODS:
        #     start_heavy_atom_distance = self.preprocess(coordinate2distance(self.cfg.job.molecule, start_position))
        #     self.start_mlcv = self.model_wrapper.compute_cv(start_heavy_atom_distance) 
        #     if "output_scale" in self.cfg.model:
        #         self.start_mlcv = self.start_mlcv * self.cfg.model.output_scale
        
        # elif self.force_type == "gnncv":
        #     from torch_geometric.data import Data
        #     start_position_data = Data(
        #         batch = torch.tensor([0], dtype=torch.int64, device=self.device),
        #         node_attrs = torch.tensor(ALANINE_HEAVY_ATOM_ATTRS, dtype=torch.float32, device=self.device),
        #         positions = start_position.reshape(-1, 3)[ALANINE_HEAVY_ATOM_IDX],
        #         edge_index = torch.tensor(ALANINE_HEAVY_ATOM_EDGE_INDEX, dtype=torch.long, device=self.device),
        #         shifts = torch.zeros(90, 3, dtype=torch.float32, device=self.device)
        #     )
        #     self.start_mlcv = self.model_wrapper.compute_cv(start_position_data) 
        
    def _set_goal_position(self, pdb, system):
        # Set goal position
        integrator = self._new_integrator()
        platform = mm.Platform.getPlatformByName(self.cfg.job.simulation.platform)
        if self.cfg.job.simulation.platform in ["CUDA", "OpenCL"]:
            properties = {'DeviceIndex': '0', 'Precision': self.cfg.job.simulation.precision}
        elif self.cfg.job.simulation.platform == "CPU":
            properties = {}
        else:
            raise ValueError(f"Platform {self.cfg.job.simulation.platform} not found")
        simulation = app.Simulation(
            pdb.topology,
            system,
            integrator,
            platform,
            properties
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        self.goal_position = simulation.context.getState(getPositions=True).getPositions()
        temperature = torch.tensor(self.temperature.value_in_unit(unit.kelvin), device=self.device).reshape(1, 1)
        
        # Set goal mlcv
        goal_position = torch.tensor(
            [list(p) for p in self.goal_position.value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(1, -1)
        goal_position.requires_grad = True
        self.goal_mlcv = self.model_wrapper.compute_cv(goal_position, temperature)
          
    def _set_custom_force(self):        
        if self.force_type == "torsion":
            start_position = np.array(
                [list(p) for p in self.start_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )
            goal_position = np.array(
                [list(p) for p in self.goal_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )
            start_psi = compute_dihedral(start_position[ALDP_PSI_ANGLE])
            start_phi = compute_dihedral(start_position[ALDP_PHI_ANGLE])
            goal_psi = compute_dihedral(goal_position[ALDP_PSI_ANGLE])
            goal_phi = compute_dihedral(goal_position[ALDP_PHI_ANGLE])

            # Create CustomTorsionForce for phi and psi angles
            custom_cv_force = mm.CustomTorsionForce(
                "0.5 * k * (theta - (theta_start + (theta_goal - theta_start) * (time / total_time)))^2"
            )
            custom_cv_force.addTorsion(*ALDP_PSI_ANGLE, [start_psi, goal_psi])
            custom_cv_force.addTorsion(*ALDP_PHI_ANGLE, [start_phi, goal_phi])
            custom_cv_force.addPerTorsionParameter("theta_start")
            custom_cv_force.addPerTorsionParameter("theta_goal")
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
        
        elif self.force_type in MLCOLVAR_METHODS or ["gnncv", "clcv"] or CLCV_METHODS:
            if self.cfg.job.simulation.force_version == "v1":
                external_force = mm.CustomExternalForce(" 0.5 * k * (fx*x + fy*y + fz*z) * (time / total_time) ")
                external_force.addGlobalParameter("k", self.k)
            elif self.cfg.job.simulation.force_version == "v2":
                external_force = mm.CustomExternalForce(" (fx*x + fy*y + fz*z) ")
            elif self.cfg.job.simulation.force_version == "v3":
                external_force = mm.CustomExternalForce(" 0.5 * k * (fx*x + fy*y + fz*z) ")
                external_force.addGlobalParameter("k", self.k)
            else:
                raise ValueError(f"Force version {self.cfg.job.simulation.force_version} not found")
            
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
    
