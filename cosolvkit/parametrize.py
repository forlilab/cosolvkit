import json
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from openff.toolkit import Molecule
from cosolvkit.utils import fix_pdb
from cosolvkit.cosolventbox import _add_cosolvent_as_concentrations
import parmed


available_engines = ["Amber", "Gromacs", "CHARMM"]

def parametrize_system(cosolvent_pdb, cosolvents, forcefields, simulation_engine="Amber", outpath=None):
    """
        Parametrizes the PDB cosolvent file generated with the chosen FF.
    """
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    with open(forcefields) as fi:
        ffs = json.load(fi)
    
    with open(cosolvents) as fi:
        cosolvs = json.load(fi)

    print("Parametrizing with forcefield...")
    forcefield = ForceField(*ffs[simulation_engine])
    small_molecule_ff = _parametrize_cosolvents(cosolvs, "espaloma")
    forcefield.registerTemplateGenerator(small_molecule_ff.generator)

    print("Cleaning PDB and setting up the box...")
    fix_pdb(cosolvent_pdb) 
    suffix, extension = cosolvent_pdb.split('.')
    pdb = PDBFile(f"{suffix}_clean.{extension}")
    modeller = Modeller(pdb.topology, pdb.positions)
    chains = list([x.id for x in modeller.getTopology().chains()])
    # Set box dimensions
    # positions = modeller.positions
    # topology = modeller.getTopology()
    # minRange = Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
    # maxRange = Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
    # modeller.topology.setUnitCellDimensions(maxRange - minRange)
    modeller.addSolvent(forcefield, boxShape='cube', padding=12*angstrom, neutralize=True)
    add_cosolvents(modeller, cosolvents, chains)
    print("Building the system...")
    system = forcefield.createSystem(modeller.topology, 
                                     nonbondedMethod=PME, 
                                     nonbondedCutoff=10 * angstrom,
                                     constraints=HBonds,
                                     hydrogenMass=1.5 * amu)
    print("Saving topology files...")
    _save_topology(modeller.topology, modeller.positions, system, simulation_engine, outpath)
    return 

def _save_topology(topology, positions, system, simulation_engine, outpath):
    parmed_structure = parmed.openmm.load_topology(topology, system, positions)

    if simulation_engine == "Amber":
        # Add dummy bond type for None ones so that parmed doesn't trip
        bond_type = parmed.BondType(1.0, 1.0, list=parmed_structure.bond_types)
        parmed_structure.bond_types.append(bond_type)
        for bond in parmed_structure.bonds:
            if bond.type is None:
                bond.type = bond_type

        parmed_structure.save(f'{outpath}/system.prmtop', overwrite=True)
        parmed_structure.save(f'{outpath}/system.inpcrd', overwrite=True)

    elif simulation_engine == "Gromacs":
        parmed_structure.save(f'{outpath}/system.top', overwrite=True)
        parmed_structure.save(f'{outpath}/system.gro', overwrite=True)

    elif simulation_engine == "CHARMM":
        parmed_structure.save(f'{outpath}/system.psf', overwrite=True)

    else:
        print("The specified simulation engine is not supported!")
        print(f"Available simulation engines:\n\t{available_engines}")
    return 

def _parametrize_cosolvents(cosolvents, small_molecule_ff="espaloma"):
    molecules = list()
    for cosolvent in cosolvents:
        try:
            molecules.append(Molecule.from_smiles(cosolvent["smiles"]))
        except Exception as e:
            print(e)
            print(cosolvent)
    if small_molecule_ff == "espaloma":
        small_ff = EspalomaTemplateGenerator(molecules=molecules, forcefield='espaloma-0.3.2')
    elif small_molecule_ff == "gaff":
        small_ff = GAFFTemplateGenerator(molecules=molecules)
    else:
        small_ff = SMIRNOFFTemplateGenerator(molecules=molecules)
    # forcefield.registerTemplateGenerator(small_ff.generator)
    return small_ff

def add_cosolvents(modeller, cosolvents, chain_ids):
    box_center, box_origin, box_size = get_box_data(modeller.getTopology(), padding=12*angstrom)
    waters = []
    receptor = []

    for atom_idx, atom in enumerate(modeller.getTopology().atoms()):
        if atom.residue.name == "HOH":
            waters.append(modeller.positions[atom_idx].value_in_unit(angstrom))
        elif atom.residue.chain_id in chain_ids:
            receptor.append(modeller.positions[atom_idx].value_in_unit(angstrom))
    
    # Once we identify water atoms and receptor atoms we can select all the waters
    # whose distance from the receptor is higher than the cutoff and select randomly
    # the waters to replace with the cosolvent unitll the desired concentration is met
    # 
    # To consider the different volume occupied by the water and the cosolvkit. Check how 
    # Jerome addressed that problem.
    
    cosolvent_positions = _add_cosolvent_as_concentrations(wat_xyzs=np.array(waters), 
                                                           cosolvents=cosolvents, 
                                                           box_origin=box_origin, 
                                                           box_size=box_size, 
                                                           target_concentrations=None, 
                                                           receptor_xyzs=np.array(receptor))[1]
    return


def get_box_data(topology, padding):
    periodic_vectors = topology.getPeriodicBoxVectors().value_in_unit(angstrom)
    xmin = np.min(periodic_vectors[:, 0]) - padding
    xmax = np.max(periodic_vectors[:, 0]) + padding
    ymin = np.min(periodic_vectors[:, 1]) - padding
    ymax = np.max(periodic_vectors[:, 1]) + padding
    zmin = np.min(periodic_vectors[:, 2]) - padding
    zmax = np.max(periodic_vectors[:, 2]) + padding
    lmax = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
    size = np.ceil(np.array([lmax, lmax, lmax])).astype(int)

    center = np.mean([[xmin, ymin, zmin], [xmax, ymax, zmax]], axis=0)
    origin = center - (size / 2)
    return center, origin, size 
