#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Add centroid to benzene residues
#

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from mdtraj.reporters import DCDReporter

import utils


# REad file
prmtop = AmberPrmtopFile('system.prmtop')
inpcrd = AmberInpcrdFile('system.inpcrd')

# Configuration system
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=10 * angstrom, constraints=HBonds, hydrogenMass=3 * amu)

# Add harmonic constraints
harmonic_force_id, atom_idxs = utils.add_harmonic_constraints(prmtop, inpcrd, system, "protein and not element H", 2.5)
print('Number of particles constrainted: %d' % len(atom_idxs))

# NPT
properties = {"Precision": "mixed"}
platform = Platform.getPlatformByName('OpenCL')
system.addForce(MonteCarloBarostat(1 * bar, 300 * kelvin))
integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 4 * femtoseconds)
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(inpcrd.positions)

# Energy minimization
simulation.minimizeEnergy()

# MD simulations - production
simulation.reporters.append(DCDReporter('system.dcd', 250))
simulation.reporters.append(CheckpointReporter('system.chk', 2500))
simulation.reporters.append(StateDataReporter('openmm.log', 250, step=True, time=True, 
                                              potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                              temperature=True, volume=True, density=True, speed=True))

simulation.step(25000000)
