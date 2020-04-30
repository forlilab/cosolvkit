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
* openbabel

## Installation
I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this:
```bash
$ conda create -n cosolvkit python=3.6
$ conda activate cosolvkit
$ conda install -c conda-forge -c ambermd numpy scipy mkl openbabel rdkit ambertools parmed mdanalysis
```

Finally, we can install the `CoSolvKit` package
```bash
$ git clone https://github.com/jeeberhardt/cosolvkit
$ cd cosolvkit
$ python setup.py build install
```

## Quick tutorial

### Preparation
```python
from cosolvkit import CoSolventBox

# Preparation
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

### Analysis
```python
from MDAnalysis import Universe

from cosolvkit import Analysis

# Analysis
u = Universe("cosolvent_system.prmtop", ["traj_1.nc", "traj_2.nc"])
# Volume occupied by the water molecules, obtained during the preparation
volume = 423700.936 # A**3
temperature = 300. # K

a = Analysis(u.select_atoms("(resname BEN or resname PRP)"), verbose=True)
a.run()
a.grid_free_energy(volume, temperature)
a.gfe.export("map_gfe_hydrophobe.dx")

a = Analysis(u.select_atoms("(resname MEH or resname ACM) and name O*"), verbose=True)
a.run()
a.grid_free_energy(volume, temperature)
a.gfe.export("map_gfe_O.dx")
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