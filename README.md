[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

# CoSolvKit
The python package for creating cosolvent box

## Prerequisites

You need, at a minimum (requirements):
* python 3
* numpy 
* scipy
* RDKit
* ambertools
* parmed
* MDAnalysis
* griddataformats
* MDTraj
* OpenMM

## Installation
I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this:
```bash
$ conda create -n cosolvkit python=3.7
$ conda activate cosolvkit
$ conda install -c conda-forge numpy scipy mkl \
rdkit ambertools parmed mdanalysis griddataformats openmm mdtraj
```

Finally, we can install the `CoSolvKit` package
```bash
$ git clone https://github.com/jeeberhardt/cosolvkit
$ cd cosolvkit
$ pip install .
```

## Quick tutorial

1. **Preparation**
```python
from cosolvkit import CoSolventBox

cosolv = CoSolventBox(concentration=1.0, cutoff=12, box='cubic') # 1 M concentration
cosolv.add_receptor("protein.pdb")
cosolv.add_cosolvent(name='benzene', smiles='c1ccccc1', resname="BEN")
cosolv.add_cosolvent(name='methanol', smiles='CO', resname="MEH")
cosolv.add_cosolvent(name='propane', smiles='CCC', resname="PRP")
cosolv.add_cosolvent(name='imidazole', smiles='C1=CN=CN1')
cosolv.add_cosolvent(name='acetamide', smiles='CC(=O)NC', resname="ACM")
cosolv.build()
cosolv.export_pdb(filename="cosolv_system.pdb")
cosolv.write_tleap_input(filename="tleap.cmd", prmtop_filename="cosolv_system.prmtop",
                         inpcrd_filename="cosolv_system.inpcrd")
```

2. **Run tleap to create Amber topology and coordinates files**
```bash
tleap -s -f tleap.cmd # Here you can specify disulfide bridges, salt concentration, etc...
```

3. **Run MD simulations**
```
See Amber/OpenMM input files in the data/amber_protocol and data/openmm_protocol directory.
```

4. **Analysis**
```python
from MDAnalysis import Universe

from cosolvkit import Analysis

u = Universe("cosolvent_system.prmtop", ["traj_1.nc", "traj_2.nc"])
# Volume occupied by the water molecules, obtained during the preparation
volume = 423700.936 # A**3
temperature = 300. # K

a = Analysis(u.select_atoms("(resname BEN or resname PRP)"), verbose=True)
a.run()
a.atomic_grid_free_energy(volume, temperature)
a.export_density("map_density_hydrophobe.dx")
a.export_atomic_grid_free_energy("map_agfe_hydrophobe.dx")

a = Analysis(u.select_atoms("(resname MEH or resname ACM) and name O*"), verbose=True)
a.run()
a.atomic_grid_free_energy(volume, temperature)
a.export_density("map_density_O.dx")
a.export_atomic_grid_free_energy("map_agfe_O.dx")
```

## Add cosolvent molecules to pre-existing waterbox

You already have your system ready and it contains a super fancy lipid membrane built with [`packmol-memgen`](https://github.com/callumjd/AMBER-Membrane_protein_tutorial)? Well, no worry you can still add cosolvent molecules to it!

**Disclaimer**: You will have issue with systems prepared with CHARMM-GUI. The conversion step to the amber format using `charmmlipid2amber.py` does not produce a readable file by `tleap` (at least on my side...).

```python
from cosolvkit import CoSolventBox

cosolv = CoSolventBox(concentration=1.0, use_existing_waterbox=True) # 0.1 M concentration
cosolv.add_receptor("bilayer_protein.pdb")
cosolv.add_cosolvent(name='benzene', smiles='c1ccccc1', resname="BEN")
cosolv.build()
cosolv.export_pdb(filename='cosolv_system.pdb')
cosolv.write_tleap_input(filename='tleap.cmd', prmtop_filename='cosolv_system.prmtop',
                         inpcrd_filename='cosolv_system.inpcrd')
```

## Add centroid-repulsive potential with OpenMM

To overcome aggregation of small hydrophobic molecules at high concentration (1 M), a repulsive interaction energy between fragments can be added, insuring a faster sampling. This repulsive potential is applied only to the selected fragments, without perturbing the interactions between fragments and the protein. The repulsive potential is implemented by adding a virtual site (massless particle) at the geometric center of each fragment, and the energy is described using a Lennard-Jones potential (epsilon = -0.01 kcal/mol and sigma = 12 Angstrom).

Luckily for us, OpenMM is flexible enough to make the addition of this repulsive potential between fragments effortless (for you). The addition of centroids in fragments and the repulsive potential to the `System` holds in one line using the `add_repulsive_centroid_force` function. Thus making the integration very easy in existing OpenMM protocols. In this example, a mixture of benzene (`BEN`) and propane (`PRP`) was generated at approximately 1 M in a small box of 40 x 40 x 40 Angstrom (see `data` directory). The MD simulation will be run in NPT condition at 300 K during 100 ps using periodic boundary conditions.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from openmm.app import *
from openmm import *
from mdtraj.reporters import DCDReporter

from cosolvkit import utils


# Read file
prmtop = AmberPrmtopFile('cosolv_ben_prp_system.prmtop')
inpcrd = AmberInpcrdFile('cosolv_ben_prp_system.inpcrd')

# Configuration system
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=12 * unit.angstrom, constraints=HBonds, hydrogenMass=3 * unit.amu)

# This is where the magic is happening!
# Add harmonic constraints on protein if present
#atom_idxs = utils.add_harmonic_restraints(prmtop, inpcrd, system, "protein and not element H", 2.5)
#print('Number of particles constrainted: %d' % len(atom_idxs))

# Add centroids and repulsive forces
n_particles, virtual_site_idxs, repulsive_force_id = utils.add_repulsive_centroid_force(prmtop, inpcrd, system, residue_names=["BEN", "PRP"])
print("Number of particles before adding centroids: %d" % n_particles)
print('Number of centroids added: %d' % len(virtual_site_idxs))
# The magic ends here.

# NPT
properties = {"Precision": "mixed"}
platform = Platform.getPlatformByName('OpenCL')
system.addForce(MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))
integrator = LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 4 * unit.femtoseconds)
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(inpcrd.positions)

# Energy minimization
simulation.minimizeEnergy()

# MD simulations - equilibration(10 ps)
simulation.step(2500)

# MD simulations - production (200 ps, of course it has to be much more!)
# Write every atoms except centroids
simulation.reporters.append(DCDReporter('cosolv_repulsive.dcd', 250, atomSubset=range(n_particles)))
simulation.reporters.append(CheckpointReporter('cosolv_repulsive.chk', 2500))
simulation.reporters.append(StateDataReporter("openmm.log" 250, step=True, time=True, 
                                              potentialEnergy=True, kineticEnergy=True, 
                                              totalEnergy=True, temperature=True, volume=True, 
                                              density=True, speed=True))
simulation.step(25000)
```

## List of cosolvent molecules
Non-exhaustive list of suggested cosolvents (molecule_name, SMILES string and resname):
* Benzene 1ccccc1 BEN
* Methanol CO MEH
* Propane CCC PRP
* Imidazole C1=CN=CN1 IMI
* Acetamide CC(=O)NC ACM
* Methylammonium C[NH3+] MAM
* Acetate CC(=O)[O-] ACT
* Formamide C(=O)N FOM
* Acetaldehyde CC=O ACD

## Citations
* Ustach, Vincent D., et al. "Optimization and Evaluation of Site-Identification by Ligand Competitive Saturation (SILCS) as a Tool for Target-Based Ligand Optimization." Journal of chemical information and modeling 59.6 (2019): 3018-3035.
* Schott-Verdugo, S., & Gohlke, H. (2019). PACKMOL-memgen: a simple-to-use, generalized workflow for membrane-protein–lipid-bilayer system building. Journal of chemical information and modeling, 59(6), 2522-2528.
