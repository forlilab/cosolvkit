#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Add centroid to benzene residues
#

from openmm.app import *
from openmm import *
from mdtraj.reporters import DCDReporter

import utils


# REad file
prmtop = AmberPrmtopFile('system.prmtop')
inpcrd = AmberInpcrdFile('system.inpcrd')

# Configuration system
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=12 * unit.angstrom, constraints=HBonds, hydrogenMass=1.5 * unit.amu)

# Add harmonic constraints
atom_idxs = utils.add_harmonic_restraints(prmtop, inpcrd, system, "protein and not element H", 2.5)
print('Number of particles constrainted: %d' % len(atom_idxs))

# NPT
properties = {"Precision": "mixed"}
platform = Platform.getPlatformByName('OpenCL')
system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtoseconds)
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
