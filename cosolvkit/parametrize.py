import json
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from openff.toolkit import Molecule
import parmed


available_engines = ["Amber", "Gromacs", "CHARMM"]

def parametrize_system(cosolvent_pdb, cosolvents, simulation_engine="Amber"):
    """
        Parametrizes the PDB cosolvent file generated with the chosen FF.
    """
    with open("forcefields.json") as fi:
        ffs = json.load(fi)

    forcefield = ForceField(ffs)
    forcefield = _parametrize_cosolvents(cosolvents, forcefield, "espaloma")

    system = forcefield.createSystem(cosolvent_pdb.topology, 
                                     nonbondedMethod=PME, 
                                     nonbondedCutoff=12 * angstrom,
                                     constraints=HBonds,
                                     hydrogenMass=3 * amu)
    _save_MD_inputs(system, simulation_engine)
    return 

def _save_MD_inputs(system, simulation_engine):
    if simulation_engine == "Amber":
        pass
    elif simulation_engine == "Gromacs":
        pass
    elif simulation_engine == "CHARMM":
        pass
    else:
        print("The specified simulation engine is not supported!")
        print(f"Available simulation engines:\n\t{available_engines}")
    return 

def _parametrize_cosolvents(cosolvents, forcefield, small_molecule_ff="espaloma"):
    molecules = list()
    for cosolvent in cosolvents:
        molecules.append(Molecule.from_smiles(cosolvent))
    if small_molecule_ff == "espaloma":
        small_ff = EspalomaTemplateGenerator(molecules=molecules)
    elif small_molecule_ff == "gaff":
        small_ff = GAFFTemplateGenerator(molecules=molecules)
    else:
        small_ff = SMIRNOFFTemplateGenerator(molecules=molecules)
    forcefield.registerTemplateGenerator(small_ff.generator)
    return forcefield