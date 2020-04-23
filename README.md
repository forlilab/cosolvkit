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

```python
from MDAnalysis import Universe

from cosolvkit import CoSolventBox
from cosolvkit import Analysis

# Preparation
cosolv = CoSolventBox(concentration=0.15, cutoff=12, box='cubic') # 0.15 M concentration
cosolv.add_receptor("protein.pdb")
cosolv.add_cosolvent(name='benzene', smiles='c1ccccc1')
cosolv.add_cosolvent(name='methanol', smiles='CO', resname="MEH")
cosolv.add_cosolvent(name='propane', smiles='CCC', resname="PRP")
cosolv.add_cosolvent(name='imidazole', smiles='C1=CN=CN1')
cosolv.add_cosolvent(name='acetamide', smiles='CC(=O)NC', resname="ACT")
cosolv.build()
cosolv.export(prefix="cosolv")

# Analysis
u = Universe("cosolvent_system.prmtop", ["traj_1.nc", "traj_2.nc"])

a = Analysis(u.select_atoms("(resname BEN or resname PRP)"), verbose=True)
a.run()
a.density.export("map_hydrophobe.dx")

a = Analysis(u.select_atoms("(resname IMI)"), verbose=True)
a.run()
a.density.export("map_imidazole.dx")

a = Analysis(u.select_atoms("(resname MEH or resname ACT) and name O*"), verbose=True)
a.run()
a.density.export("map_O.dx")

a = Analysis(u.select_atoms("(resname ACT) and name N*"), verbose=True)
a.run()
a.density.export("map_N.dx")
```