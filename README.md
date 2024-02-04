[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

# CosolvKit
The python package for creating cosolvent system

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
* PDBFixer
* espaloma
* pymol

## Installation
I highly recommand you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this:
```bash
$ conda create -n cosolvkit -c conda-forge python=3 numpy scipy mkl \
rdkit ambertools parmed mdanalysis griddataformats openmm pdbfixer \
mdtraj espaloma pymol-open-source
$ conda activate cosolvkit
```

Finally, we can install the `CosolvKit` package
```bash
$ git clone https://github.com/forlilab/cosolvkit
$ cd cosolvkit
$ pip install -e .
```

## Quick tutorial

1. **Preparation**
Cosolvent System definition
```python
import openmm.unit as openmmunit
from cosolvkit.cosolvent_system import CosolventSystem
# If starting from PDB file path
cosolv = CosolventSystem.from_filename(cosolvents, forcefields, simulation_format, receptor_path, radius=radius)

# If starting from a pdb string or without receptor
cosolv = CosolventSystem(cosolvents, forcefields, simulation_format, receptor_path, radius=radius)

# If creating a cosolvent box without receptor
cosolv = CosolventSystem(cosolvents, forcefields, simulation_format, None, radius=10*openmmunit.angstrom)
```
Cosolvent System creation
```python
# If using water as solvent
cosolv.build(neutralize=True)

# If using different solvent i.e. methanol
cosolv.build(solvent_smiles="CH3OH")
```

Saving Cosolvent System according to the simulation_format
```python
cosolv.save_topology(cosolv.modeller.topology, 
                     cosolv.modeller.positions,
                     cosolv.system,
                     simulation_format,
                     cosolv.forcefield,
                     output_path)
```

3. **Run MD simulations**
If you don't want to setup your own simulation, we provide a standard simulation protocol using ```OpenMM```
```pyhton
from cosolvkit.simulation import run_simulation

print("Running MD simulation")
start = time.time()
# Depending on the simulation format you would pass either a topology and positions file or a pdb and system file
run_simulation(
                simulation_format = simulation_format,
                topology = None,
                positions = None,
                pdb = 'system.pdb',
                system = 'system.xml',
                warming_steps = 100000,
                simulation_steps = 6250000, # 25ns
                results_path = results_path, # This should be the name of system being simulated
                seed=None
    )
print(f"Simulation finished after {(time.time() - start)/60:.2f} min.")
```

4. **Analysis**
```python
from cosolvkit.analysis import Report
"""
Report class:
    log_file: is the statistics.csv or whatever log_file produced during the simulation.
        At least Volume, Temperature and Pot_e should be reported on this log file.
    traj_file: trajectory file
    top_file: topology file
    cosolvents_file: json file describing the cosolvents

generate_report_():
    out_path: where to save the results. 3 folders will be created:
        - report
            - autocorrelation
            - rdf
    analysis_selection_string: selection string of cosolvents you want to analyse. This
        follows MDAnalysis selection strings style. If no selection string, one density file
        for each cosolvent will be created.

generate_pymol_report()
    selection_string: important residues to select and show in the PyMol session.
"""
report = Report(log_file, traj_file, top_file, cosolvents_file)
report.generate_report(out_path=out_path, analysis_selection_string="")
report.generate_pymol_reports(report.topology, 
                              report.trajectory, 
                              density_file=report.density_file, 
                              selection_string='', 
                              out_path=out_path)
```

## Add cosolvent molecules to pre-existing waterbox

You already have your system ready and it contains a super fancy lipid membrane built with [`packmol-memgen`](https://github.com/callumjd/AMBER-Membrane_protein_tutorial)? Well, no worry you can still add cosolvent molecules to it!

**Disclaimer**: You will have issue with systems prepared with CHARMM-GUI. The conversion step to the amber format using `charmmlipid2amber.py` does not produce a readable file by `tleap` (at least on my side...).

```python
from cosolvkit import CoSolventBox

cosolv = CoSolventBox(use_existing_waterbox=True)
cosolv.add_receptor("bilayer_protein.pdb")
cosolv.add_cosolvent(name='benzene', concentration=1.0, smiles='c1ccccc1')
cosolv.build()
cosolv.export_pdb(filename='cosolv_system.pdb')
cosolv.prepare_system_for_amber(filename='tleap.cmd', prmtop_filename='cosolv_system.prmtop',
                                inpcrd_filename='cosolv_system.inpcrd')
```

## Keep existing water molecules

You already placed water molecules at some very strategic positions around a ligand, for example, and you want to keep them. That's also easy to do!

```python
from cosolvkit import CoSolventBox

cosolv = CoSolventBox(box='orthorombic', cutoff=12, keep_existing_water=True)
cosolv.add_receptor("complex_protein_ligand.pdb")
cosolv.build()
cosolv.export_pdb(filename='system.pdb')
cosolv.prepare_system_for_amber(filename='tleap.cmd', prmtop_filename='cosolv_system.prmtop',
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
