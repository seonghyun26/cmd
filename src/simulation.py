import numpy as np

from openmm import app
from openmm import *
from openmm.app import *
from openmm.unit import *
from openmmtools.integrators import VVVRIntegrator

import openmm as mm
import openmm.unit as unit


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

def load_system(cfg, molecule, pdb, forcefield):
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


def init_simulation(cfg, pbb_file_path, frame=None):
    pdb = PDBFile(pbb_file_path)
    
    # Set force field
    # force_field = ForceField(*cfg.job.simulation.force_field)
    # system = force_field.createSystem(
    #     pdb.topology,
    #     nonbondedCutoff=3 * nanometer,
    #     constraints=HBonds
    # )
    force_field = load_forcefield(cfg, cfg.job.molecule)
    system = load_system(cfg, cfg.job.molecule, pdb, force_field)
    integrator = LangevinIntegrator(
        cfg.job.simulation.temperature * kelvin,
        1 / picosecond,
        1 * femtoseconds
    )
    platform = Platform.getPlatformByName(cfg.job.simulation.platform)
    properties = {'Precision': cfg.job.simulation.precision}

    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        platform,
        properties
    )        
    
    simulation.context.setPositions(pdb.positions)   
    simulation.minimizeEnergy()
    
    return simulation

def set_simulation(simulation, frame):
    if frame is not None:
        atom_xyz = frame.detach().cpu().numpy()
        atom_list = [Vec3(atom[0], atom[1], atom[2]) for atom in atom_xyz]
        current_state_openmm = Quantity(value=atom_list, unit=nanometer)
        simulation.context.setPositions(current_state_openmm)
    else:
        raise ValueError("Frame is None")
    
    # simulation.context.setVelocitiesToTemperature(simulation.integrator.getTemperature())
    simulation.context.setVelocities(Quantity(value=np.zeros(frame.shape), unit=nanometer/picosecond))
    
    
    return simulation