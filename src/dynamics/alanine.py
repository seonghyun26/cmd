import os

import openmm as mm
import openmm.unit as unit

from openmm import app
from openmmtools.integrators import VVVRIntegrator

from .base import BaseDynamics
from ..utils import kabsch

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
    def __init__(self, cfg):
        self.cfg = cfg
        self.k = cfg.job.steered_simulation.k
        self.temperature = cfg.job.steered_simulation.temperature * unit.kelvin
        self.friction = cfg.job.steered_simulation.friction / unit.femtoseconds
        self.timestep = cfg.job.steered_simulation.timestep * unit.femtoseconds
        self.molecule = cfg.job.molecule
        self.time_horizon = cfg.job.steered_simulation.time_horizon
        
        # Load pdb files
        start_pdb = app.PDBFile(f"./data/{self.molecule}/{cfg.job.start_state}.pdb")
        goal_pdb = app.PDBFile(f"./data/{self.molecule}/{cfg.job.goal_state}.pdb")
        
        # Load simulation components
        forcefield = load_forcefield(cfg, self.molecule)
        system = load_system(cfg, self.molecule, start_pdb)
        self._set_start_position(start_pdb, system)
        self._set_goal_position(goal_pdb, system)
        
        # Set cv force
        custom_force = self._set_custom_force(cfg)
        custom_force.addGlobalParameter("k", self.k)
        custom_force.addGlobalParameter("time", 0)
        custom_force.addGlobalParameter("total_time", self.num_steps * self.timestep)
        system.addForce(custom_cv_force)
        
        # Set simulation
        integrator = self._new_integrator()
        integrator.setConstraintTolerance(0.00001)
        self.simulation = app.Simulation(start_pdb.topology, system, integrator)
        self.simulation.context.setPositions(self.start_position)
        
    def _new_integrator():
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        return integrator
    
    def _set_start_position(pdb, system):
        integrator = self._new_integrator()
        simulation = app.Simulation(pdb.topology, system, start_integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        self.start_position = simulation.context.getState(getPositions=True).getPositions()
        
    def _set_goal_position(pdb, system):
        integrator = self._new_integrator()
        simulation = app.Simulation(pdb.topology, system, start_integrator)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        self.goal_position = simulation.context.getState(getPositions=True).getPositions()
        
    def _set_custom_force(cfg):
        force_type = cfg.job.steered_simulation.force_type
        
        if force_type == "torsion":
            angle_1 = [6, 8, 14, 16]
            angle_2 = [1, 6, 8, 14]
            start_position = np.array(
                [list(p) for p in self.position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )

            goal_position = np.array(
                [list(p) for p in self.goal_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )

            start_psi = compute_dihedral(start_position[angle_1])
            start_phi = compute_dihedral(start_position[angle_2])
            goal_psi = compute_dihedral(goal_position[angle_1])
            goal_phi = compute_dihedral(goal_position[angle_2])

            # Create CustomTorsionForce for phi and psi angles
            custom_cv_force = mm.CustomTorsionForce(
                "0.5 * k * (theta - (theta_start + (theta_goal - theta_start) * (time / total_time)))^2"
            )

            # Add per-torsion parameters
            custom_cv_force.addPerTorsionParameter("theta_start")
            custom_cv_force.addPerTorsionParameter("theta_goal")
            custom_cv_force.addTorsion(*angle_1, [start_psi, goal_psi])
            custom_cv_force.addTorsion(*angle_2, [start_phi, goal_phi])
        elif force_type == "rmsd":
            start_position = np.array(
                [list(p) for p in self.start_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )
            goal_position = np.array(
                [list(p) for p in self.goal_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )
            start_rmsd = (
                kabsch(
                    torch.tensor(
                        start_position,
                        dtype=torch.float32,
                    ).unsqueeze(0),
                    torch.tensor(
                        goal_position,
                        dtype=torch.float32,
                    ).unsqueeze(0),
                )
                .squeeze().item()
            )
            rmsd_force = mm.RMSDForce(goal_position)

            custom_cv_force = mm.CustomCVForce(
                "0.5 * k * (rmsd - start_rmsd * (1 - time / total_time))^2"
            )
            custom_cv_force.addCollectiveVariable("rmsd", rmsd_force)
            custom_cv_force.addGlobalParameter("start_rmsd", start_rmsd)
        elif force_type == "deepcv":
            pass
        else:
            raise ValueError(f"Force type {force_type} not found")
            
        return custom_cv_force
        
def load_forcefield(cfg, molecule):
    if molecule == "alanine":
        forcefield = app.ForceField(*cfg.job.steered_simulation.force_field)
    elif molecule == "chignolin":
        path = os.path.join(
            os.getcwd(),
            "openmmforcefields/openmmforcefields/ffxml/amber/protein.ff14SBonlysc.xml",
        )
        forcefield = app.ForceField(*cfg.job.steered_simulation.force_field)
    else:
        raise ValueError(f"Molecule {molecule} not found")
        
    return forcefield

def load_system(cfg, molecule, pdb):
    if molecule == "alanine":
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
    elif molecule == "chignolin": 
        system = forcefield.createSystem(
            pdb.topology,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
    else:
        raise ValueError(f"Molecule {molecule} not found")  
    
    return system