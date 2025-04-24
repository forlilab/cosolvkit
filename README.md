[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) 
[![License: LGPL v2.1](https://img.shields.io/badge/License-LGPL_v2.1-green.svg)](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html) 
[![Documentation Status](https://readthedocs.org/projects/cosolvkit/badge/?version=latest)](https://cosolvkit.readthedocs.io/en/latest/?badge=latest)
![Conda Version](https://img.shields.io/conda/vn/conda-forge/cosolvkit)
![Conda Platform](https://img.shields.io/conda/pn/conda-forge/cosolvkit)
![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/cosolvkit)
[![PyPI - Version](https://img.shields.io/pypi/v/cosolvkit)](https://pypi.org/project/cosolvkit/0.4.7/) 
[![Powered by RDKit](https://img.shields.io/badge/Powered%20by-RDKit-3838ff.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQBAMAAADt3eJSAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAFVBMVEXc3NwUFP8UPP9kZP+MjP+0tP////9ZXZotAAAAAXRSTlMAQObYZgAAAAFiS0dEBmFmuH0AAAAHdElNRQfmAwsPGi+MyC9RAAAAQElEQVQI12NgQABGQUEBMENISUkRLKBsbGwEEhIyBgJFsICLC0iIUdnExcUZwnANQWfApKCK4doRBsKtQFgKAQC5Ww1JEHSEkAAAACV0RVh0ZGF0ZTpjcmVhdGUAMjAyMi0wMy0xMVQxNToyNjo0NyswMDowMDzr2J4AAAAldEVYdGRhdGU6bW9kaWZ5ADIwMjItMDMtMTFUMTU6MjY6NDcrMDA6MDBNtmAiAAAAAElFTkSuQmCC)](https://www.rdkit.org/)
      
    

# CosolvKit
The python package for creating cosolvent system.  

The original paper is freely accessible at the link (https://pubs.acs.org/doi/10.1021/acs.jcim.4c01398).

## Documentation
The installation instructions, documentation and tutorials can be found on http://cosolvkit.readthedocs.io/.

## Installation
`Cosolvkit` package is available via `conda` and can be installed:
```bash
$ conda install --channel conda-forge::cosolvkit
```

or via `mamba`:
```bash
$ mamba install -c conda-forge cosolvkit
```
**_*Please note that Apple M1 chips are not supported by some of CosolvKit's dependencies.
we recommend macOS users of Apple Silicon install the x86_64 version of MiniForge and run CosolvKit through Rosetta._**


Finally, we can install the `CosolvKit` package via `pip`:
```bash
$ conda create -n cosolvkit -c conda-forge -f environment.yml
$ conda activate cosolvkit
$ pip instal cosolvkit
```

or directly download and install the source code from git: 
I highly recommend you to install the Anaconda distribution (https://www.continuum.io/downloads) if you want a clean python environnment with nearly all the prerequisites already installed. To install everything properly, you just have to do this (for faster installation, use `mamba` or `micromamba` instead of `conda`):

```bash
$ git clone https://github.com/forlilab/cosolvkit
$ cd cosolvkit
$ conda create -n cosolvkit -c conda-forge -f environment.yml
$ conda activate cosolvkit
$ pip install -e .
```

## Example
To run an example of CosolvKit system creation run the command `create_cosolvent_system -c cosolvkit/data/config.json`.
By default this example creates the cosolvent system without running any MD simualtion or post processing analysis.
If you want it to run the simulation as well modify the field `run_md: true` in the `config.json`.
The results can be found in `cosolvkit/data/results`.

**_*Please note that the `run_md: true` and `add_repulsive: true` and their sub-parameters are only available when the `md_format: "openmm"`._**

## Quick tutorial

The script `create_cosolvent_system.py` provide all the necessary tools to build a cosolvent system and optionally run an MD simulation with standard setup.
The main entry point of the script is the file `config.json` where all the necessary flags and command line options are specified.

| Argument                | Type  | Description                                           | Default value   | OPENMM | AMBER | GROMACS | CHARMM | 
|:------------------------|:------|:------------------------------------------------------|:----------------|:-------|:------|:--------|:-------|
|cosolvents               | string |Path to the json file containing the cosolvents to add to the system. | no default | | 
|forcefields              | string | Path to the json file containing the forcefields to use. | no default | | 
|md_format                | string | Format to use for the MD simulations and topology files. Supported formats: [OPENMM, AMBER, GROMACS, CHARMM] | no default | |
|protein_path             | string | If receptor is `true` this should be the path to the protein structure. | no default | |
|clean_protein            | boolean | Flag indicating if cleaning the protein with `PDBFixer` | TRUE | |
|keep_heterogens          | boolean | Flag indicating if keeping the heterogen atoms while cleaning the protein. Waters will be always kept. | FALSE | |
|variants                 | dictionary | Dictionary of residues for which a variant is requested (different protonation state) in the form {"chain_id:res_id":"protonation_state"}, `None` for the rest of the residues. | empty dictionary | |
|add_repulsive            | boolean | Flag indicating if adding repulsive forces between certain residues or not. | FALSE | ✅ | ❌ | ❌ | ❌ |
|repulsive_residues       | list | List of residues for which applying the repulsive forces. | empty list | ✅ | ❌ | ❌ | ❌ |
|repulsive_epsilon        | float | Depth of the potential well in kcal/mol | 0.01 kcal/mol | ✅ | ❌ | ❌ | ❌ |
|repulsive_sigma          | float | Inter-particle distance in Angstrom | 4.0 Angstrom | ✅ | ❌ | ❌ | ❌ |
|solvent_smiles           | string | Smiles string of the solvent to use. | H2O | |
|solvent_copies           | integer | If specified, the box won't be filled up with solvent, but will have the exact number of solvent molecules specified. | no default | |
|membrane                 | boolean | Flag indicating if the system has membranes or not. | FALSE | |
|lipid_type               | string | If membrane is TRUE specify the lipid to use. Supported lipids: ["POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"] | "POPC" | |
|lipid_patch_path         | string | If the lipid required is not in the available, it is possible to pass a pre-equilibrated patch of the lipid of interest. | no default | |
|memb_cosolvent_placement      | string | Which side of the membrane to place the cosolvents. Available options: [both -> no preference, outside, inside] | both | |
|waters_to_keep           | list | List of indices of waters of interest in a membrane system. | no default | |
|box_size                   | float | If no receptor, the box_size is necessary to set the size of the simulation box. | no default | |
|output_dir               | string | Path to where save the results. | no default | |
|run_cosolvent_system     | boolean | Flag indicating if running creating the system or not. | TRUE | |
|run_md                   | boolean | Flag indicating if running the md simulation after creating the system or not. | FALSE | ✅ | ❌ | ❌ | ❌ |

1. **Preparation**
```bash
$ create_cosolvent_system -c config.json
```

3. **Run MD simulations**
If you don't want to setup your own simulation, we provide a standard simulation protocol using `OpenMM`

```python
from cosolvkit.simulation import run_simulation

print("Running MD simulation")
start = time.time()
run_simulation(
                pdb_fname = 'system.pdb',
                system_fname = 'system.xml',
                temperature = 300,
                time_step = 0.04,
                simulation_steps = 25000000, 
                results_path = 'results',
                )
print(f"Simulation finished after {(time.time() - start)/60:.2f} min.")
```

4. **Analysis**

#### 4.1 centering, imaging, and aligning a trajectory
To generate meaningful cosolvent densities for visualization, the trajectory
must be centered and aligned on the region of interest. Centering is
placing a set of atoms at the center of the simulation box
(without rotating the box),
imaging is placing all atoms inside the box if they traveled beyond
the periodic boundary conditions, and aligning is rotating and translating
the system so that a selection of atoms overlaps with some reference positions.

Usually, trajectories are aligned on a macromolecule such as a protein, but
parts of macromolecules that are flexible and move during the simulation can
still cause densities to smear. If such flexible parts are of interest, it is
a good idea to align the trajectories on each flexible part independently.
If the region of interest is a specific location of a large or flexible protein,
it is best to align using the vicinity of the region of interest, rather than
the whole protein.

One option to align and image trajectories is `cpptraj`. It should be installed
automatically by the installation instructions above. First we create an input
file for `cpptraj`, which we will call `process.cpptraj`:
```
parm system.pdb
trajin trajectory.dcd
center :@CA
autoimage
reference system.pdb [myref]
rms ref [myref] @CA out protein.rmsd
trajout trajectory_aligned.xtc
```
There are two important selections in this input file that are system specific and need to be edited manually, the one for centering
the trajectory after `center` command, and the one for aligning after `rms`.
See the [documentation for defining selections](https://amberhub.chpc.utah.edu/atom-mask-selection-syntax/).
To run it, `system.pdb` needs to be on the working directory:
```
cpptraj -i process.cpptraj
```
It will write `trajectory_aligned.xtc`. This trajectory should inspected to make sure the
region of interest is not moving or wrapping around the periodic boundaries.
First, load `system.pdb` into Pymol, and then type the following into
Pymol's command line: `load_traj trajectory_aligned.xtc, system`.

An example of another program that can image and center trajectories is
MDAnalysis. For imaging, see its documentation about
[wrapping and unwrapping](https://docs.mdanalysis.org/stable/documentation_pages/transformations/wrap.html).

#### 4.2 the actual analysis
```python
from cosolvkit.analysis import Report
"""
"""
    report = Report(statistics_file, traj_file_aligned, top_file, 
                    cosolvent_names=['BEN'], out_path=out_path)
    
    report.generate_report(equilibration=True, 
                           rdf=True,
                           rmsf=True, 
                           avg_selection='protein',
                           align_selection='protein and name CA',
                           )
    report.generate_density_maps(
                                 cosolvent_names=['BEN'],
                                 gridsize=0.375,
                                 temperature=300,
                                 use_atomtypes=True, 
                                 atomtypes_definitions='../cosolvkit/data/dacar_atomtypes.json', 
                                 )    
    report.generate_pymol_session(density_files=out_path)
```

## Add centroid-repulsive potential

To overcome aggregation of small hydrophobic molecules at high concentration (1 M), a repulsive interaction energy between fragments can be added, insuring a faster sampling. This repulsive potential is applied only to the selected fragments, without perturbing the interactions between fragments and the protein. The repulsive potential is implemented by adding a virtual site (massless particle) at the geometric center of each fragment, and the energy is described using a Lennard-Jones potential (epsilon = -0.01 kcal/mol and sigma = 12 Angstrom).

Luckily for us, OpenMM is flexible enough to make the addition of this repulsive potential between fragments effortless (for you). The addition of centroids in fragments and the repulsive potential to the `System` holds in one line using the `add_repulsive_centroid_force` function. Thus making the integration very easy in existing OpenMM protocols. In this example, we are adding repulsive forces between `BEN` and `PRP` molecules.

```python
from cosolvkit.cosolvent_system import CosolventSystem

cosolv = CosolventSystem(cosolvents, forcefields, simulation_format, receptor_path, radius=radius)
# build the system in water
cosolv.build(neutralize=True)
cosolv.add_repulsive_forces(resiude_names=["BEN", "PRP"])
```

## List of cosolvent molecules
Non-exhaustive list of suggested cosolvents (molecule_name, SMILES string and resname):
* Benzene C1=CC=CC=C1 BEN
* Methanol CO MEH
* Propane CCC PRP
* Imidazole C1=CN=CN1 IMI
* Acetamide CC(=O)NC ACM
* Methylammonium C[NH3+] MAM
* Acetate CC(=O)[O-] ACT
* Formamide C(=O)N FOM
* Acetaldehyde CC=O ACD


## Config files
An example of the following configuration files can be found in the data folder:
* config.json
* cosolvents.json
* forcefields.json
