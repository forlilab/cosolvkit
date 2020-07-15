[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![PyPI version fury.io](https://img.shields.io/badge/version-0.1-green.svg)](https://pypi.python.org/pypi/ansicolortags/)

# CoSolvKit
The python package for creating cosolvent box

## Prerequisites

You need, at a minimum (requirements):
* python 3
* RDKit
* numpy 
* scipy
* ambertools
* parmed
* MDAnalysis
* MDTraj
* openbabel
* OpenMM (for centroid repulsive potentials)

## Installation
I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this:
```bash
$ conda create -n cosolvkit python=3.6
$ conda activate cosolvkit
$ conda install -c conda-forge -c ambermd -c omnia numpy scipy mkl openbabel \
rdkit ambertools parmed mdanalysis openmm mdtraj
```

Finally, we can install the `CoSolvKit` package
```bash
$ git clone https://github.com/jeeberhardt/cosolvkit
$ cd cosolvkit
$ python setup.py build install
```

## Quick tutorial

1. **Preparation**
```python
from cosolvkit import CoSolventBox

cosolv = CoSolventBox(concentration=0.25, cutoff=12, box='cubic') # 0.25 M concentration
cosolv.add_receptor("protein.pdb")
cosolv.add_cosolvent(name='benzene', smiles='c1ccccc1')
cosolv.add_cosolvent(name='methanol', smiles='CO', resname="MEH")
cosolv.add_cosolvent(name='propane', smiles='CCC', resname="PRP")
cosolv.add_cosolvent(name='imidazole', smiles='C1=CN=CN1')
cosolv.add_cosolvent(name='acetamide', smiles='CC(=O)NC', resname="ACM")
cosolv.build()
cosolv.export(prefix="cosolv")
```

2. **Run MD simulations**
```
See Amber input files in the data/amber_protocol directory.
```

3. **Analysis**
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
a.agfe.export("map_agfe_hydrophobe.dx")

a = Analysis(u.select_atoms("(resname MEH or resname ACM) and name O*"), verbose=True)
a.run()
a.atomic_grid_free_energy(volume, temperature)
a.agfe.export("map_agfe_O.dx")
```

## Add centroid-repulsive potential with OpenMM

To overcome aggregation of small hydrophobic molecules at high concentration (1 M), a repulsive interaction energy between fragments can be added, insuring a faster sampling. This repulsive potential is applied only to the selected fragments, without perturbing the interactions between fragments and the protein. The repulsive potential is implemented by adding a virtual site (massless particle) at the geometric center of each fragment, and the energy is described using a Lennard-Jones potential (epsilon = -0.01 kcal/mol and sigma = 12 Angstrom).

Luckily for us, OpenMM is flexible enough to make the addition of this repulsive potential between fragments effortless (for you). The addition of centroids in fragments and the repulsive potential to the `System` holds in one line using the `add_repulsive_centroid_force` function. Thus making the integration very easy in existing OpenMM protocols. In this example, a mixture of benzene (`BEN`) and propane (`PRP`) was generated at approximately 1 M in a small box of 40 x 40 x 40 Angstrom (see `data` directory). The MD simulation will be run in NPT condition at 300 K during 100 ps using periodic boundary conditions.

```python
from sys import stdout

from mdtraj.reporters import NetCDFReporter
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *

from cosolvkit import utils


# Read file
prmtop = AmberPrmtopFile('cosolv_ben_prp_system.prmtop')
inpcrd = AmberInpcrdFile('cosolv_ben_prp_system.inpcrd')

# Configuration system
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=12 * angstrom, constraints=HBonds)

# This is where the magic is happening!
# Add centroids and repulsive forces between benzene and propane fragments
n_particles, _, force_id = utils.add_repulsive_centroid_force(prmtop, inpcrd, system, ["BEN", "PRP"])
# The magic ends here.

# NPT
properties = {"Precision": "mixed"}
platform = Platform.getPlatformByName('CUDA')
system.addForce(MonteCarloBarostat(1 * bar, 300 * kelvin))
integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(inpcrd.positions)

# Energy minimization
simulation.minimizeEnergy()

# MD simulations - equilibration(10 ps)
simulation.step(5000)

# MD simulations - production (100 ps, of course it has to be much more!)
# Write every atoms except centroids
simulation.reporters.append(NetCDFReporter('cosolv_repulsive.nc', 100, 
                                           atomSubset=range(n_particles)))
simulation.reporters.append(StateDataReporter(stdout, 500, step=True, time=True, 
                                              potentialEnergy=True, kineticEnergy=True, 
                                              totalEnergy=True, temperature=True, volume=True, 
                                              density=True, speed=True))
simulation.step(50000)
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
