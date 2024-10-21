import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import joblib
import mdtraj as md
import pyemma.coordinates as coor

import openmm as mm
from openmm import app
import openmm.unit as unit
from openmmtools.integrators import VVVRIntegrator

parser = argparse.ArgumentParser()

# System Config
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--molecule", default="alanine", type=str)
parser.add_argument("--save_dir", default="assets/paths/steer/alanine/", type=str)

# Sampling Config
parser.add_argument("--k", default=0, type=float)
parser.add_argument("--cv", default="rmsd", type=str)
parser.add_argument("--start_state", default="c5", type=str)
parser.add_argument("--end_state", default="c7ax", type=str)
parser.add_argument("--num_steps", default=1000, type=int)
parser.add_argument("--timestep", default=1, type=float)
parser.add_argument("--num_samples", default=64, type=int)
parser.add_argument("--temperature", default=300, type=float)
parser.add_argument("--friction", default=0.001, type=float)
parser.add_argument("--ground_truth", action="store_true")

args = parser.parse_args()


def kabsch(P, Q):

    # Compute centroids
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)

    # Center the points
    p = P - centroid_P
    q = Q - centroid_Q

    # Compute the covariance matrix
    H = torch.matmul(p.transpose(-2, -1), q)

    # SVD
    U, S, Vt = torch.linalg.svd(H)

    # Validate right-handed coordinate system
    d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))  # B

    Vt[d < 0.0, -1] *= -1.0

    # Optimal rotation and translation
    R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    t = centroid_Q - torch.matmul(centroid_P, R.transpose(-2, -1))

    # Calculate RMSD
    P = torch.matmul(P, R.transpose(-2, -1)) + t
    rmsd = (P - Q).square().sum(-1).mean(-1).sqrt()
    return rmsd


def compute_dihedral(position):
    v = position[:-1] - position[1:]
    v0 = -v[0]
    v1 = v[2]
    v2 = v[1]

    s0 = np.sum(v0 * v2, axis=-1, keepdims=True) / np.sum(
        v2 * v2, axis=-1, keepdims=True
    )
    s1 = np.sum(v1 * v2, axis=-1, keepdims=True) / np.sum(
        v2 * v2, axis=-1, keepdims=True
    )

    v0 = v0 - s0 * v2
    v1 = v1 - s1 * v2

    v0 = v0 / np.linalg.norm(v0, axis=-1, keepdims=True)
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

    x = np.sum(v0 * v1, axis=-1)
    v3 = np.cross(v0, v2, axis=-1)
    y = np.sum(v3 * v1, axis=-1)
    return np.arctan2(y, x)


class SteeredDynamics:
    def __init__(self, args):
        self.k = args.k
        self.temperature = args.temperature * unit.kelvin
        self.friction = args.friction / unit.femtoseconds
        self.timestep = args.timestep * unit.femtoseconds
        self.molecule = args.molecule

        # Load molecule files
        pdb = app.PDBFile(f"./data/{args.molecule}/{args.start_state}.pdb")
        end_pdb = app.PDBFile(f"./data/{args.molecule}/{args.end_state}.pdb")
        
        # Set force field and system
        if self.molecule == "alanine":
            forcefield = app.ForceField("amber99sbildn.xml", "tip3p.xml")

            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.PME,
                constraints=app.HBonds,
                ewaldErrorTolerance=0.0005,
            )
        elif self.molecule == "chignolin":
            path = os.path.join(
                os.getcwd(),
                "openmmforcefields/openmmforcefields/ffxml/amber/protein.ff14SBonlysc.xml",
            )
            forcefield = app.ForceField(path, "implicit/gbn2.xml")

            system = forcefield.createSystem(
                pdb.topology,
                constraints=app.HBonds,
                ewaldErrorTolerance=0.0005,
            )
        else:
            raise ValueError(f"Molecule {self.molecule} not found")

        # Set integrator
        start_integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        start_integrator.setConstraintTolerance(0.00001)

        
        # Get start and target_position with energy minimization
        self.start_simulation = app.Simulation(pdb.topology, system, start_integrator)
        self.start_simulation.context.setPositions(pdb.positions)
        self.start_simulation.minimizeEnergy()
        self.position = self.start_simulation.context.getState(
            getPositions=True
        ).getPositions()
        end_integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        end_integrator.setConstraintTolerance(0.00001)
        self.end_simulation = app.Simulation(pdb.topology, system, end_integrator)
        self.end_simulation.context.setPositions(end_pdb.positions)
        self.end_simulation.minimizeEnergy()
        target_position = self.end_simulation.context.getState(
            getPositions=True
        ).getPositions()


        # Set custom force
        if args.cv == "torsion":
            angle_1 = [6, 8, 14, 16]
            angle_2 = [1, 6, 8, 14]
            start_position = np.array(
                [list(p) for p in self.position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )

            target_position = np.array(
                [list(p) for p in target_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )

            start_psi = compute_dihedral(start_position[angle_1])
            start_phi = compute_dihedral(start_position[angle_2])

            target_psi = compute_dihedral(target_position[angle_1])
            target_phi = compute_dihedral(target_position[angle_2])

            # Create CustomTorsionForce for phi and psi angles
            custom_cv_force = mm.CustomTorsionForce(
                "0.5 * k * (theta - (theta_start + (theta_target - theta_start) * (time / total_time)))^2"
            )

            # Add per-torsion parameters
            custom_cv_force.addPerTorsionParameter("theta_start")
            custom_cv_force.addPerTorsionParameter("theta_target")

            custom_cv_force.addTorsion(*angle_1, [start_psi, target_psi])
            custom_cv_force.addTorsion(*angle_2, [start_phi, target_phi])

        elif args.cv == "rmsd":
            start_position = np.array(
                [list(p) for p in self.position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )

            target_position = np.array(
                [list(p) for p in target_position.value_in_unit(unit.nanometer)],
                dtype=np.float32,
            )
            start_rmsd = (
                kabsch(
                    torch.tensor(
                        start_position,
                        dtype=torch.float32,
                    ).unsqueeze(0),
                    torch.tensor(
                        target_position,
                        dtype=torch.float32,
                    ).unsqueeze(0),
                )
                .squeeze()
                .item()
            )
            rmsd_force = mm.RMSDForce(target_position)

            custom_cv_force = mm.CustomCVForce(
                "0.5 * k * (rmsd-start_rmsd*(1- time / total_time))^2"
            )
            custom_cv_force.addCollectiveVariable("rmsd", rmsd_force)
            custom_cv_force.addGlobalParameter("start_rmsd", start_rmsd)

        
        # Add custom cv foce to the system
        custom_cv_force.addGlobalParameter("k", self.k)
        custom_cv_force.addGlobalParameter("time", 0)
        custom_cv_force.addGlobalParameter("total_time", args.num_steps * args.timestep)
        system.addForce(custom_cv_force)

        # Set integrator for simulation
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        self.simulation = app.Simulation(pdb.topology, system, integrator)
        self.simulation.context.setPositions(self.position)

    def step(self, time):
        self.simulation.context.setParameter("time", time)
        self.simulation.step(1)

    def report(self):
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        return positions

    def reset(self):
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)


if __name__ == "__main__":
    for name in ["positions"]:
        if not os.path.exists(f"{args.save_dir}/{name}"):
            os.makedirs(f"{args.save_dir}/{name}")

    mds = SteeredDynamics(args)

    if not args.ground_truth:
        for i in tqdm(range(args.num_samples), desc="Sampling"):
            positions = []
            for step in range(1, args.num_steps + 1):
                position = mds.report().value_in_unit(unit.nanometer)
                # change to numpy array with float32
                position = np.array([list(p) for p in position], dtype=np.float32)
                positions.append(position)
                mds.step(step * args.timestep)
            mds.reset()
            np.save(f"{args.save_dir}/positions/{i}.npy", positions)
