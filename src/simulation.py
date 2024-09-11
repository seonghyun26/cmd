from openmm import *
from openmm.app import *
from openmm.unit import *

import numpy as np


def init_simulation(cfg, pbb_file_path, frame=None):
    # set pdb file with current positions
    pdb = PDBFile(pbb_file_path)
    
    # Set force field
    force_field = ForceField(*cfg.job.simulation.force_field)
    system = force_field.createSystem(
        pdb.topology,
        nonbondedCutoff=3 * nanometer,
        constraints=HBonds
    )
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
    
    # NOTE: Check if this is right
    # simulation.context.setVelocitiesToTemperature(simulation.integrator.getTemperature())
    simulation.context.setVelocities(Quantity(value=np.zeros(frame.shape), unit=nanometer/picosecond))
    
    
    return simulation