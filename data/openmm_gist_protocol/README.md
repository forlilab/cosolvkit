# Quick tutorial for GIST with OpenMM

### Receptor preparation

Preparation using AmberTools20 and `wk_prepare_receptor.py` script
```bash
$ wk_prepare_receptor.py -i protein.pdb -o protein_prepared --dry --nohyd --pdb
```

### Water box preparation
```python
from cosolvkit import CosolventBox

# Will add a padding of 5 A in each dimension to the 20x20x20 box, plus 12 A of padding for the water 
cosolv = CosolventBox(concentration=0, cutoff=12, box="orthorombic", center=[10, 0, -12], box_size=[20, 20, 20])
cosolv.add_receptor("protein_prepared.pdb") # Otherwise `protein_clean.pdb` if fails
cosolv.build()
cosolv.export(prefix="gist")
```

### Run MD simulation using OpenMM
1. Copy `gist_system.prmtop`, `gist_system.inpcrd` and `gist_system.pdb` to the cluster (ex: Garibaldi).
2. Run a MD simulation of 100 ns (copy also `utils.py`, located in the `cosolvkit` directory, to the cluster. We will need it to apply the harmonic constraints on the system.)

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from mdtraj.reporters import DCDReporter

import utils


# REad file
prmtop = AmberPrmtopFile('gist_system.prmtop') # Change here if necessary
inpcrd = AmberInpcrdFile('gist_system.inpcrd') # Change here if necessary

# Configuration system
system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=12 * angstrom, constraints=HBonds, hydrogenMass=3 * amu)

# Add harmonic constraints of 2.5 kcal/mol/A**2
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
simulation.reporters.append(DCDReporter('gist_system.dcd', 250))
simulation.reporters.append(CheckpointReporter('gist_system.chk', 2500))
simulation.reporters.append(StateDataReporter('openmm.log', 250, step=True, time=True, 
                                              potentialEnergy=True, kineticEnergy=True, totalEnergy=True, 
                                              temperature=True, volume=True, density=True, speed=True))

# Run for 100 ns
simulation.step(25000000)
```

3. Align the trajectory on the reference `gist_system.pdb`
```bash 
parm gist_system.prmtop
parm gist_system.prmtop [ref]
reference gist_system.pdb parm [ref]
trajin gist_system.dcd
autoimage
center :1-80 mass origin # Change to select only the protein, look into gist_system.pdb for last protein resid
image origin center familiar
rms reference :1-80@CA,C,O,N norotate # Same here
trajout gist_system_aligned.dcd
go
```

4. Run Grid Inhomogeneous Solvation Theory (GIST)
```
parm gist_system.prmtop
trajin gist_system_aligned.dcd 70000 100000 # Take only the last 30 ns
gist gridspacn 0.5 gridcntr 10 0 -12 griddim 60 60 60 # Box of 30x30x30 A
go
quit
```

### Gaussian smoothing of the GIST maps using `waterkit`
Hydration sites are identified based on the oxygen density map (gO) in an iterative way, by selecting the voxel with the highest density, then the second highest and so on, whil keeping a minimum distance of 2.5 A between them. Energies for each of those identified hydration sites are then computed by adding all the surrounding voxels and Gaussian weighted by their distances from the hydration site.

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
from gridData import Grid
from waterkit.analysis import blur_map

gO = Grid("gist-gO.dx")
esw = Grid('gist-Esw-dens.dx')
eww = Grid('gist-Eww-dens.dx')
dg = (esw + 2 * eww)

# This part might fail to produce a grid precisely centered on what you asked for...
# The `autodock_format` flag is supposed to produce a grid with an odd number of points
map_smooth = blur_map(dg, radius=1.4, gridsixe=0.375, center=[10, 0, -12], box_size=[20, 20, 20], autodock_format=True)
map_smooth.export("gist-dG-dens_smoothed.dx")
```
