import json
import os
import sys
import io
import logging
from collections import defaultdict
from typing import Union, Tuple
import numpy as np
from scipy import spatial
from scipy.stats import qmc
import math 
from itertools import product
import parmed
from openmm import Vec3, unit, XmlSerializer, System, CustomNonbondedForce, NonbondedForce, OpenMMException
import openmm.app as app
import openmm.unit as openmmunit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from openmmforcefields.generators.template_generators import SmallMoleculeTemplateGenerator
from cosolvkit.utils import fix_pdb, MutuallyExclusiveParametersError, MD_FORMAT_EXTENSIONS
from openff.units.openmm import to_openmm


proteinResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR', 'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL']
rnaResidues = ['A', 'G', 'C', 'U', 'I']
dnaResidues = ['DA', 'DG', 'DC', 'DT', 'DI']

class CosolventMolecule(object):
    def __init__(self, name: str, smiles: str=None, mol_save_dir: str = None, mol_filename: str=None, resname: str=None, copies: int=None, concentration: float=None):
        """Creates a Cosolvent object.

        :param name: name of the molecule
        :type name: str
        :param smiles: SMILES string of the molecule in the chose protonation state, defaults to None
        :type smiles: str, optional
        :param mol_save_dir: directory to save MOL file
        :type mol_save_dir: str, optional 
        :param mol_filename: MOL/SDF filename of the molecule, defaults to None
        :type mol_filename: str, optional
        :param resname: 3-letters residue name of the molecule. If None, the first 3 uppercase letters of the name will be used, defaults to None
        :type resname: str, optional
        :param copies: number of copies of cosolvent the user wants to place, defaults to None
        :type copies: int, optional
        :param concentration: if the number of copies is unknown the user can specify the concentration in Molar units, defaults to None
        :type concentration: float, optional
        """
        if resname is None:
            self.resname = name[:3].upper()
        else:
            self.resname = resname

        # assert not self.resname in AMBER_SUPPORTED_RESNAMES, f"Error: the residue name {self.resname} is already defined in AMBER."
        assert (smiles is not None or mol_filename is not None), "A smiles or a mol/sdf filename has to be defined."
        if mol_filename is not None:
            assert os.path.exists(mol_filename), f"The mol/sdf file {mol_filename} does not exist."

        self.mol = None
        self.atom_names = None
        self.name = name
        self.charge = None
        self.positions = None
        self.pdb_conect = None
        self.mol_save_dir = mol_save_dir 
        self.smiles = smiles
        self.copies = copies
        self.concentration = concentration

        # We use mol file as input for antechamber instead of mol2
        # because of multiple issues I had in the past with that format...
        if mol_filename is None:
            # RDKit parameters for the ETKDGv3 conformational search
            params = AllChem.ETKDGv3()
            params.enforceChirality = True
            params.useSmallRingTorsions = True
            params.useMacrocycleTorsions = True
            params.forceTransAmides = True
            params.useExpTorsionAnglePrefs = True
            params.useBasicKnowledge = True
            # Not a mistake, it's 2 not 3
            # https://www.rdkit.org/docs/RDKit_Book.html#parameters-controlling-conformer-generation
            params.ETVersion = 2

            # Setup rdkit molecule
            mol = Chem.MolFromSmiles(smiles)
            mol.SetProp("_Name", str(name))
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, params=params)
            AllChem.MMFFOptimizeMolecule(mol)

            if mol_save_dir is None:
                mol_filename = '%s.mol' % self.name
            else:
                os.makedirs(self.mol_save_dir, exist_ok=True)
                mol_filename = '%s/%s.mol' % (self.mol_save_dir, self.name)
            with Chem.SDWriter(mol_filename) as w:
                w.write(mol)
        else:
            mol = Chem.MolFromMolFile(mol_filename, removeHs=False)
            mol.SetProp("_Name", name)

        self.mol = mol
        self.charge = Chem.rdmolops.GetFormalCharge(mol)
        self.atom_names = self._generate_atom_names_from_mol(mol)
        # We make sure the cosolvent is centered, especially if it was provided by the user
        # And honestly, never trust the user!
        positions = mol.GetConformer().GetPositions()
        positions = positions - np.mean(positions, axis=0)
        self.positions = positions
        self.mol_filename = mol_filename
        self.pdb_conect = self._get_pdb_conect(mol)
        return
    
    
    def _generate_atom_names_from_mol(self, rdkit_mol: rdkit.Chem.rdchem.Mol) -> list:
        """Generates atom names from an RDKit molecule.

        :param rdkit_mol: RDKit molecule
        :type rdkit_mol: rdkit.Chem.rdchem.Mol
        :return: list of atom names.
        :rtype: list
        """
        atom_names = []
        counter = {}

        for atom in rdkit_mol.GetAtoms():
            atom_name = atom.GetSymbol()

            if atom_name in counter:
                counter[atom_name] += 1
            else:
                counter[atom_name] = 1

            atom_name += str(counter[atom_name])
            atom_names.append(atom_name)

        return atom_names


    def _get_pdb_conect(self, rdkit_mol: rdkit.Chem.rdchem.Mol) -> list:
        """Generates bonds definition (unused)

        :param rdkit_mol: RDKit molecule
        :type rdkit_mol: rdkit.Chem.rdchem.Mol
        :return: list of bonds
        :rtype: list
        """
        conect = []

        pdb_string = Chem.MolToPDBBlock(rdkit_mol)
        pdb_lines = pdb_string.split('\n')

        for i, line in enumerate(pdb_lines):
            if 'CONECT' in line:
                conect.append(np.array([int(n) for n in line.split()[1:]]))

        return conect

class CosolventSystem(object):
    def __init__(self, 
                 cosolvents: dict,
                 forcefields: dict,
                 ligands: dict,
                 simulation_format: str, 
                 modeller: app.Modeller,  
                 padding: openmmunit.Quantity, 
                 box_size: openmmunit.Quantity = None):
        """Create cosolvent system.

        :param cosolvents: dictionary of cosolvent molecules
        :type cosolvents: dict
        :param forcefields: dictionary of forcefields to use
        :type forcefields: dict
        :param ligands: dictionary of ligands to use
        :type ligands: dict
        :param simulation_format: MD format that want to be used for the simulation. Supported formats: Amber, Gromacs, CHARMM, openMM 
        :type simulation_format: str
        :param modeller: openmm modeller created from topology and positions.
        :type modeller: openmm.app.Modeller
        :param padding: specifies the padding used to create the simulation box, defaults to 10 * openmmunit.angstrom
        :type padding: openmm.unit.Quantity, optional
        :param box_size: Specifies the size to create the box without receptor, defaults to None
        :type box_size: openmm.unit.Quantity, optional
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Private
        self._available_formats = ["AMBER", "GROMACS", "CHARMM", "OPENMM"]
        self._cosolvent_positions = defaultdict(list)
        self._box = None
        self._periodic_box_vectors = None
        self._box_volume = None

        # Public
        self.protein_radius = 3.5 * openmmunit.angstrom
        self.cosolvents_radius = 2.5 * openmmunit.angstrom
        self.modeller = None
        self.system = None
        self.modeller = None
        self.cosolvents = dict()
        self.box_size = box_size
        self.small_molecule_forcefield = forcefields["small_molecules"][0]
        padding = padding * openmmunit.angstrom
        
        assert (simulation_format.upper() in self._available_formats), f"Error! The simulation format supplied is not supported! Available simulation engines:\n\t{self._available_formats}"
        
        self.ligands = ligands

        # Creating cosolvent molecules
        for c in cosolvents:
            cosolvent = CosolventMolecule(**c)
            cosolvent_xyz = cosolvent.positions*openmmunit.angstrom
            cosolvent_xyz = cosolvent_xyz.value_in_unit(openmmunit.nanometer)
            self.cosolvents[cosolvent] = cosolvent_xyz

        self.modeller = modeller
        
        if self.box_size is not None:
            assert (isinstance(box_size, openmmunit.Quantity)) and (box_size.unit == openmmunit.angstrom), \
                "Error! If no receptor is passed, the box_size parameter has to be set and it needs to be in angstrom openmm.unit"
            self.vectors, self.box, self.lowerBound, self.upperBound = self._build_box(None, padding, box_size=box_size)
            self.receptor = False
        else:
            self.receptor = True
            self.vectors, self.box, self.lowerBound, self.upperBound = self._build_box(self.modeller.positions, padding, box_size=None)
        
        # Setting up the box - This has to be done before building the system with
        # the cosolvent molecules.
        self.modeller.topology.setPeriodicBoxVectors(self.vectors)
        self._periodic_box_vectors = self.modeller.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)
        vX, vY, vZ = self.modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.nanometer)
        self.box_volume = vX * vY * vZ
        self.logger.info("Parameterizing system components with forcefields")
        self.forcefield = self._parametrize_system(forcefields, simulation_format, self.cosolvents)
        return
    
#region Public
    def build(self,
              solvent_smiles: str="H2O", 
              n_solvent_molecules: int=None,
              neutralize: bool=True,
              iteratively_adjust_copies: bool=False):
        """This function adds the cosolvents specified in the CosolvSystem
        and solvates with the desired solvent. If n_solvent_molecules is not passed
        the function will try to fill the box with the desired solvent to a certain extent.
        Please note that the solvation with solvents different from water may highly impact
        the execution time.

        :param solvent_smiles: smiles string defining the desired solvent to use, defaults to "H2O"
        :type solvent_smiles: str, optional
        :param n_solvent_molecules: number of molecules of solvent to add, defaults to None
        :type n_solvent_molecules: int, optional
        :param neutralize: if True, the system charge will be neutralized by OpenMM, defaults to True
        :type neutralize: bool, optional
        :param iteratively_adjust_copies: if True, the number of copies of each cosolvent will iteratively be reduced until a valid starting configuration is found
        :type iteratively_adjust_copies: bool, optional 
        """
        self.logger.info("Checking volumes..")
        volume_not_occupied_by_cosolvent = self.fitting_checks()
        assert volume_not_occupied_by_cosolvent is not None, "The requested volume for the cosolvents exceeds the available volume! Please try increasing the padding or box_size."
        receptor_positions = self.modeller.positions.value_in_unit(openmmunit.nanometer)
        if iteratively_adjust_copies:
            cosolv_xyzs = self.add_cosolvents_adaptive(self.cosolvents, self.vectors, self.lowerBound, self.upperBound, receptor_positions)
        else:
            cosolv_xyzs = self.add_cosolvents(self.cosolvents, self.vectors, self.lowerBound, self.upperBound, receptor_positions)
        self.modeller = self._setup_new_topology(cosolv_xyzs, self.modeller.topology, self.modeller.positions)
        
        # Parametrize ligands
        if self.ligands is not None:
            self._parametrize_ligands(self.ligands)

        if solvent_smiles == "H2O":
            if n_solvent_molecules is None: self.modeller.addSolvent(self.forcefield, neutralize=neutralize)
            else: self.modeller.addSolvent(self.forcefield, numAdded=n_solvent_molecules, neutralize=neutralize)
            self.logger.info(f"Waters added: {self._get_n_waters()}")
        elif solvent_smiles is not None:
            c = {"name": "solvent",
                 "smiles": solvent_smiles}
            solvent_mol = CosolventMolecule(**c)
            cosolv_xyz = solvent_mol.positions*openmmunit.angstrom
            if n_solvent_molecules is not None:
                solvent_mol.copies = n_solvent_molecules
            else:
                one_mol_vol = self.calculate_mol_volume(cosolv_xyz)
                solvent_mol.copies = int(math.floor((volume_not_occupied_by_cosolvent/one_mol_vol)+0.5)*.6)
            d_mol = {solvent_mol: cosolv_xyz.value_in_unit(openmmunit.nanometer)}
            # need to register the custom solvent if not present already
            self.forcefield.registerTemplateGenerator(self._parametrize_cosolvents(d_mol).generator)
            self.logger.info(f"Placing {solvent_mol.copies}")
            if iteratively_adjust_copies: 
                solv_xyz = self.add_cosolvents_adaptive(d_mol, self.vectors, self.lowerBound, self.upperBound, self.modeller.positions)
            else:
                solv_xyz = self.add_cosolvents(d_mol, self.vectors, self.lowerBound, self.upperBound, self.modeller.positions)
            self.modeller = self._setup_new_topology(solv_xyz, self.modeller.topology, self.modeller.positions)
            
        self.system = self._create_system(self.forcefield, self.modeller.topology)
        return
    
    def add_repulsive_forces(self, residues_names: list, epsilon: float=0.01, sigma: float=4.0):
        """This function adds a LJ repulsive potential between the specified molecules.

        :param residues_names: list of residue names
        :type residues_names: list
        :param epsilon: depth of the potential well in kcal/mol, defaults to 0.01
        :type epsilon: float, optional
        :param sigma: inter-particle distance in Angstrom, defaults to 4.0
        :type sigma: float, optional
        """            
        epsilon = np.sqrt(epsilon * epsilon) * openmmunit.kilocalories_per_mole
        sigma = sigma * openmmunit.angstrom

        forces = { force.__class__.__name__ : force for force in self.system.getForces()}
        nb_force = forces['NonbondedForce']
        cutoff_distance = nb_force.getCutoffDistance()
        energy_expression = "4*epsilon * (sigma / r)^12;" #Only the repulsive term of the LJ potential
        energy_expression += f"epsilon = {epsilon.value_in_unit_system(openmmunit.md_unit_system)};"
        energy_expression += f"sigma = {sigma.value_in_unit_system(openmmunit.md_unit_system)};"
        repulsive_force = CustomNonbondedForce(energy_expression)
        repulsive_force.addPerParticleParameter("sigma")
        repulsive_force.addPerParticleParameter("epsilon")
        repulsive_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
        repulsive_force.setCutoffDistance(cutoff_distance)
        repulsive_force.setUseLongRangeCorrection(False)
        repulsive_force.setUseSwitchingFunction(True)
        repulsive_force.setSwitchingDistance(cutoff_distance - 0.1 * openmmunit.nanometer)

        target_indices = defaultdict(list)
        for i, atom in enumerate(self.modeller.getTopology().atoms()):
            if not atom.residue.name in residues_names:
                charge, sigma, epsilon = nb_force.getParticleParameters(i)
            else:
                target_indices[atom.residue.id].append(i) 
            repulsive_force.addParticle([sigma, epsilon])
        
        for index in range(nb_force.getNumExceptions()):
            idx, jdx, c, s, eps = nb_force.getExceptionParameters(index)
            repulsive_force.addExclusion(idx, jdx)
        
        for res in target_indices.keys():
            indexes = set()
            for x in target_indices.keys():
                if x!= res:
                    for y in target_indices[x]:
                        indexes.add(y)
            repulsive_force.addInteractionGroup(set(target_indices[res]), indexes)
        self.system.addForce(repulsive_force)

        return

    def save_pdb(self, topology: app.Topology, positions: list, out_path: str):
        """Saves the specified topology and position to the out_path file.

        :param topology: topology used
        :type topology: openmm.app.Topology
        :param positions: list of 3D coords
        :type positions: list
        :param out_path: path to where to save the dile
        :type out_path: str
        """
        app.PDBFile.writeFile(
            topology,
            positions,
            open(out_path, "w"),
            keepIds=True
        )
        return

    def save_system(self, out_path: str, system: System):
        """Saves the openmm system to the desired out path.

        :param out_path: path where to save the System
        :type out_path: str
        :param system: system to save
        :type system: openmm.System
        """
        with open(f"{out_path}/system{MD_FORMAT_EXTENSIONS['OPENMM']['system']}", "w") as fo:
            fo.write(XmlSerializer.serialize(system))
        return
    
    def load_system(self, system_path: str) -> System:
        """Loads the specified openmm system.

        :param system_path: path to the system file.
        :type system_path: str
        :return: a system instance.
        :rtype: openmm.System
        """
        with open(system_path) as fi:
            system = XmlSerializer.deserialize(fi.read())
        return system

    def save_topology(self, topology: app.Topology, positions: list, system: System, simulation_format: str, forcefield: app.ForceField, out_path: str):
        """Save the topology files necessary for MD simulations according to the simulation engine specified.

        :param topology: openmm topology
        :type topology: openmm.app.Topology
        :param positions: list of 3D coordinates
        :type positions: list
        :param system: openmm system
        :type system: openmm.System
        :param simulation_format: name of the simulation engine
        :type simulation_format: str
        :param forcefield: openmm forcefield
        :type forcefield: openmm.app.ForceField
        :param out_path: output path to where to save the topology files
        :type out_path: str
        """
        new_system = forcefield.createSystem(topology,
                                             nonbondedMethod=app.PME,
                                             nonbondedCutoff=10*openmmunit.angstrom,
                                             removeCMMotion=False,
                                             rigidWater=False,
                                             hydrogenMass=3.0*openmmunit.amu)
        
        parmed_structure = parmed.openmm.topsystem.load_topology(topology, new_system, positions)   
        
        simulation_format = simulation_format.upper()
        parmed_formats = {"AMBER": "amber", "GROMACS": "gro", "CHARMM": "charmm"}
        top_extension = MD_FORMAT_EXTENSIONS[simulation_format]['topology']
        pos_extension = MD_FORMAT_EXTENSIONS[simulation_format]['position']
        if simulation_format != "OPENMM":
            parmed_structure.save(f'{out_path}/system{top_extension}', overwrite=True, format=parmed_formats[simulation_format])
            parmed_structure.save(f'{out_path}/system{pos_extension}', overwrite=True)
        elif simulation_format == "OPENMM":
            self.save_system(out_path, system)
            self.save_pdb(topology, positions, f"{out_path}/system{pos_extension}")
            parmed_structure.save(f'{out_path}/system{top_extension}', overwrite=True)
        else:
            self.logger.info("The specified simulation engine is not supported!")
            self.logger.info(f"Available simulation engines:\n\t{self._available_formats}")
        return

    def reduce_copies(self, factor_reduction: float):
        """Reduces the number of copies of each cosolvent by a constant factor 

        :param factor_reduction: value between 0-1 representing reduction factor to apply to number of cosolvent copies
        :type factor_reduction: float
        """
        for cosolvent in self.cosolvents:
            proposed_copies = int(cosolvent.copies*factor_reduction)
            if proposed_copies >= 1:
                cosolvent.copies = proposed_copies
        return 

#endregion
    
#region Private
#region Misc
    def _copies_from_concentration(self, water_volume: float):
        """Computes the number of copies of cosolvent necessary to reach the desired concentration

        :param water_volume: volume available to be filled with cosolvents
        :type water_volume: float
        """
        for cosolvent in self.cosolvents:
            if cosolvent.concentration is not None:
                cosolvent.copies = int(math.floor((((cosolvent.concentration*openmmunit.molar)*(water_volume*openmmunit.liters))*openmmunit.AVOGADRO_CONSTANT_NA) + 0.5))
        return
    
    def _get_n_waters(self) -> int:
        """Returns the number of waters in the system.

        :return: number of waters in the system
        :rtype: int
        """
        res = [r.name for r in self.modeller.topology.residues()]
        return res.count('HOH')
 
    def _setup_new_topology(self, cosolvents_positions: dict, receptor_topology: app.Topology = None, receptor_positions:list = None) -> app.Modeller:
        """Returns a new modeller with the topolgy with the new molecules specified

        :param cosolvents_positions: keys are cosolvent molecules and values are lists of position of the new molecules to add
        :type cosolvents_positions: dict
        :param receptor_topology: old topology to which add the new molecules, defaults to None
        :type receptor_topology: openmm.app.Topology, optional
        :param receptor_positions: old positions to which add the new molecules, defaults to None
        :type receptor_positions: list, optional
        :return: new modeller containing combined topology and positions
        :rtype: openmm.app.Modeller
        """
        new_mod = None
        last_res_id = 0
        
        # Add the receptor first if present
        if receptor_topology is not None and receptor_positions is not None and len(receptor_positions) > 0: 
            new_mod = app.Modeller(receptor_topology, receptor_positions)
            last_res_id = int(list(receptor_topology.residues())[-1].index)+1

        # Add cosolvent molecules
        molecules = []
        molecules_positions = []
        cosolvent_names = []
        for cosolvent in cosolvents_positions:
            for i in range(len(cosolvents_positions[cosolvent])):
                cosolvent_names.append(cosolvent.resname)
                mol = Molecule.from_smiles(cosolvent.smiles, name=cosolvent.resname)
                mol.generate_conformers(n_conformers=1)
                molecules.append(mol)
                [molecules_positions.append(x) for x in cosolvents_positions[cosolvent][i]]

        molecules_positions = np.array(molecules_positions)*openmmunit.nanometer
        new_top_openff = Topology.from_molecules(molecules)
        new_top = self._to_openmm_topology(new_top_openff, starting_id=last_res_id)       
        residues = list(new_top.residues())
        for i in range(len(cosolvent_names)):
            residues[i].name = cosolvent_names[i]
          
        if new_mod is None:
            new_mod = app.Modeller(new_top, molecules_positions)
        else:
            new_mod.add(new_top, molecules_positions)
            
        new_mod.topology.setPeriodicBoxVectors(self._periodic_box_vectors)

        return new_mod
    
    def _to_openmm_topology(self, off_topology: Topology, starting_id: int) -> app.Topology:
        """Converts an openff topology to openmm without specifying a different chain for each residue.

        :param off_topology: Openff Topology
        :type off_topology: openff.Topology
        :param starting_id: starting index
        :type starting_id: int
        :raises RuntimeError: if something goes wrong
        :return: openmm topology
        :rtype: openmm.app.Topology
        """
        from openff.toolkit.topology.molecule import Bond

        omm_topology = app.Topology()

        off_topology._ensure_unique_atom_names("residues")

        # Go through atoms in OpenFF to preserve the order.
        omm_atoms = []

        last_chain = None
        cnt = 0
        # For each atom in each molecule, determine which chain/residue it should be a part of
        for molecule in off_topology.molecules:
            # No chain or residue can span more than one OFF molecule, so reset these to None for the first
            # atom in each molecule.
            last_residue = None
            for atom in molecule.atoms:
                
                atom_residue_name = molecule.name

                # If the residue number is undefined, assume a default of "0"
                if "residue_number" in atom.metadata:
                    atom_residue_number = atom.metadata["residue_number"]
                else:
                    atom_residue_number = str(starting_id+cnt)

                # If the insertion code  is undefined, assume a default of " "
                if "insertion_code" in atom.metadata:
                    atom_insertion_code = atom.metadata["insertion_code"]
                else:
                    atom_insertion_code = " "

                # If the chain ID is undefined, assume a default of "X"
                if "chain_id" in atom.metadata:
                    atom_chain_id = atom.metadata["chain_id"]
                else:
                    atom_chain_id = "X"

                # Determine whether this atom should be part of the last atom's chain, or if it
                # should start a new chain
                if last_chain is None:
                    chain = omm_topology.addChain(atom_chain_id)
                elif last_chain.id == atom_chain_id:
                    chain = last_chain
                else:
                    chain = omm_topology.addChain(atom_chain_id)
                # Determine whether this atom should be a part of the last atom's residue, or if it
                # should start a new residue
                if last_residue is None:
                    residue = omm_topology.addResidue(
                        atom_residue_name,
                        chain,
                        id=atom_residue_number,
                        insertionCode=atom_insertion_code,
                    )
                elif (
                    (last_residue.name == atom_residue_name)
                    and (int(last_residue.id) == int(atom_residue_number))
                    and (last_residue.insertionCode == atom_insertion_code)
                    and (chain.id == last_chain.id)
                ):
                    residue = last_residue
                else:
                    residue = omm_topology.addResidue(
                        atom_residue_name,
                        chain,
                        id=atom_residue_number,
                        insertionCode=atom_insertion_code,
                    )

                # Add atom.
                element = app.Element.getByAtomicNumber(atom.atomic_number)
                omm_atom = omm_topology.addAtom(atom.name, element, residue)

                # Make sure that OpenFF and OpenMM Topology atoms have the same indices.
                assert off_topology.atom_index(atom) == int(omm_atom.id) - 1
                omm_atoms.append(omm_atom)

                last_chain = chain
                last_residue = residue
            
            cnt += 1 
            # Add all bonds.
            bond_types = {1: app.Single, 2: app.Double, 3: app.Triple}
            for bond in molecule.bonds:
                atom1, atom2 = bond.atoms
                atom1_idx, atom2_idx = off_topology.atom_index(
                    atom1
                ), off_topology.atom_index(atom2)
                if isinstance(bond, Bond):
                    if bond.is_aromatic:
                        bond_type = app.Aromatic
                    else:
                        bond_type = bond_types[bond.bond_order]
                    bond_order = bond.bond_order
                elif isinstance(bond, _SimpleBond):
                    bond_type = None
                    bond_order = None
                else:
                    raise RuntimeError(
                        "Unexpected bond type found while iterating over Topology.bonds."
                        f"Found {type(bond)}, allowed are Bond and _SimpleBond."
                    )

                omm_topology.addBond(
                    omm_atoms[atom1_idx],
                    omm_atoms[atom2_idx],
                    type=bond_type,
                    order=bond_order,
                )

        if off_topology.box_vectors is not None:
            from openff.units.openmm import to_openmm

            omm_topology.setPeriodicBoxVectors(to_openmm(off_topology.box_vectors))
        return omm_topology

 
    def _create_system(self, forcefield: app.forcefield, topology: app.Topology) -> System:
        """Returns system created from the Forcefield and the Topology.

        :param forcefield: forcefield(s) used to build the system
        :type forcefield: openmm.app.forcefield
        :param topology: topology used to build the system
        :type topology: openmm.app.Topology
        :return: the new system
        :rtype: openmm.System
        """
        system = forcefield.createSystem(topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=10*openmmunit.angstrom,
                                         switchDistance=9*openmmunit.angstrom,
                                         removeCMMotion=True,
                                         constraints=app.HBonds,
                                         hydrogenMass=3.0*openmmunit.amu)
        return system 
#endregion
#region FillTheVoid
    def add_cosolvents(self, 
                       cosolvents: dict, 
                       vectors: tuple[Vec3, Vec3, Vec3], 
                       lowerBound: openmmunit.Quantity | Vec3, 
                       upperBound: openmmunit.Quantity | Vec3,
                       receptor_positions: list,
                       max_attempts_per_mol: int = 10) -> dict:
        """This function adds the desired number of cosolvent molecules using the halton sequence
        to generate random uniformly distributed points inside the grid where to place the cosolvent molecules.
        At first, if a receptor/protein is present the halton sequence points that would clash with the protein
        are pruned. We note that each molecule is attempted to be inserted max_attempts_per_mol times, and 
        if this condition is not satisfied, then the program terminates (as we were unable to add the desired 
        number of cosolvent molecules). 

        :param cosolvents: keys are cosolvent molecules and values are 3D coordinates of the molecule
        :type cosolvents: dict
        :param vectors: vectors defining the simulation box
        :type vectors: Tuple[openmm.Vec3, openmm.Vec3, openmm.Vec3]
        :param lowerBound: lower bound of the simulation box
        :type lowerBound: Union[openmm.unit.Quantity, Vec3]
        :param upperBound: upper bound of the simulation box
        :type upperBound: [openmm.unit.Quantity, Vec3]
        :param receptor_positions: list of 3D coordinates of the receptor
        :type receptor_positions: list
        :param max_attempts_per_mol: the maximum number of times we attempt to insert a single molecule
        :type max_attempts_per_mol: int
        :return: keys are cosolvent molecules and values are 3D coordinates of the newly added cosolvent molecules
        :rtype: dict
        """
        edge_cutoff = 2.5*openmmunit.angstrom
        prot_kdtree = None
        # This is used to update the kdtree of the placed cosolvents
        placed_atoms_positions = []
        if receptor_positions is not None and len(receptor_positions) > 0:
            prot_kdtree = spatial.cKDTree(receptor_positions)
        cosolv_xyzs = defaultdict(list)
        
        sampler = qmc.Halton(d=3)
        points = sampler.random(5000000)
        points= qmc.scale(points, [lowerBound[0], 
                                   lowerBound[1], 
                                   lowerBound[2]], 
                                  [upperBound[0], 
                                   upperBound[1], 
                                   upperBound[2]])
        used_halton_ids = list()
        if prot_kdtree is not None:
            banned_ids = prot_kdtree.query_ball_point(points, self.protein_radius.value_in_unit(openmmunit.nanometer))
            used_halton_ids = list(np.unique(np.hstack(banned_ids)).astype(int))
        used_halton_ids = self.delete_edges_points(points, lowerBound, vectors, edge_cutoff.value_in_unit(openmmunit.nanometer), used_halton_ids)
        valid_ids = np.array(range(0, len(points)))
        valid_ids = np.delete(valid_ids, used_halton_ids)
        num_mol_insertion_attempts = 0 

        for cosolvent in cosolvents:
            self.logger.info(f"Attempting to place {cosolvent.copies} copies of {cosolvent.name}")
            c_xyz = cosolvents[cosolvent]
            while len(cosolv_xyzs[cosolvent]) < cosolvent.copies:
            # for replicate in range(cosolvent.copies):
                # counter = replicate
                if len(placed_atoms_positions) < 1:
                    counter = np.random.choice(len(points))
                    xyz = points[counter]
                    cosolv_xyz = c_xyz + xyz
                    if self.check_coordinates_to_add(cosolv_xyz, None, prot_kdtree):
                        [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
                        cosolv_xyzs[cosolvent].append(cosolv_xyz*openmmunit.nanometer)
                        used_halton_ids.append(counter)
                        kdtree = spatial.cKDTree(placed_atoms_positions)
                else:
                    kdtree = spatial.cKDTree(placed_atoms_positions)
                    cosolv_xyz, valid_ids, num_trials = self.accept_reject(c_xyz, points, kdtree, valid_ids, lowerBound, vectors, prot_kdtree)
                    mol_num = len(cosolv_xyzs[cosolvent])+1
                    num_mol_insertion_attempts += 1 

                    if isinstance(cosolv_xyz, int):
                        self.logger.info("Could not place cosolvent molecule %d!" % mol_num)
                        if num_mol_insertion_attempts < max_attempts_per_mol: 
                            self.logger.warning("Attempting again...")
                        else:
                            self.logger.error("Unable to insert cosolvent molecule %d after %d attempts" % (mol_num, max_attempts_per_mol))
                            sys.exit(1)
                    else:
                        cosolv_xyzs[cosolvent].append(cosolv_xyz*openmmunit.nanometer)
                        [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
                        num_mol_insertion_attempts = 0
        self.logger.info("The following cosolvents were added:")
        for cosolvent in cosolv_xyzs:
            self.logger.info(f"{cosolvent.name}: {len(cosolv_xyzs[cosolvent])}")
        return cosolv_xyzs


    def add_cosolvents_adaptive(self, 
                       cosolvents: dict, 
                       vectors: tuple[Vec3, Vec3, Vec3], 
                       lowerBound: openmmunit.Quantity | Vec3, 
                       upperBound: openmmunit.Quantity | Vec3,
                       receptor_positions: list,
                       max_autoadjust_attempts: int = 10,
                       copies_factor_reduction: float = 0.9,
                       max_num_trials: int = 2500) -> dict:
        """This function attempts to add the desired number of cosolvent molecules using the halton sequence
        to generate random uniformly distributed points inside the grid where to place the cosolvent molecules.
        At first, if a receptor/protein is present the halton sequence points that would clash with the protein
        are pruned. Concentrations are iteratively reduced if initial conditions do not result in a valid 
        starting configuration. 

        :param cosolvents: keys are cosolvent molecules and values are 3D coordinates of the molecule
        :type cosolvents: dict
        :param vectors: vectors defining the simulation box
        :type vectors: tuple[openmm.Vec3, openmm.Vec3, openmm.Vec3]
        :param lowerBound: lower bound of the simulation box
        :type lowerBound: openmm.unit.Quantity | Vec3
        :param upperBound: upper bound of the simulation box
        :type upperBound: openmm.unit.Quantity | Vec3
        :param receptor_positions: list of 3D coordinates of the receptor
        :type receptor_positions: list
        :param max_autoadjust_attempts: the maximum number of times we attempt to retry to add cosolvents after adjusting molecule copy numbers  
        :type max_autoadjust_attempts: int
        :param copies_factor_reduction: the multiplicative factor by which we reduce molecule copy numbers (i.e n*copies_factor_reduction)
        :type copies_factor_reduction: float
        :param max_num_trials: the maximum number of halton moves to make
        :param max_num_trials: int 
        :return: keys are cosolvent molecules and values are 3D coordinates of the newly added cosolvent molecules
        :rtype: dict
        """
        edge_cutoff = 2.5*openmmunit.angstrom
        prot_kdtree = None
        if receptor_positions is not None and len(receptor_positions) > 0:
            prot_kdtree = spatial.cKDTree(receptor_positions)
        
        sampler = qmc.Halton(d=3)
        points = sampler.random(5000000)
        points= qmc.scale(points, [lowerBound[0], 
                                   lowerBound[1], 
                                   lowerBound[2]], 
                                  [upperBound[0], 
                                   upperBound[1], 
                                   upperBound[2]])
        used_halton_ids = list()
        if prot_kdtree is not None:
            banned_ids = prot_kdtree.query_ball_point(points, self.protein_radius.value_in_unit(openmmunit.nanometer))
            used_halton_ids = list(np.unique(np.hstack(banned_ids)).astype(int))
        used_halton_ids = self.delete_edges_points(points, lowerBound, vectors, edge_cutoff.value_in_unit(openmmunit.nanometer), used_halton_ids)
        valid_ids = np.array(range(0, len(points)))
        valid_ids = np.delete(valid_ids, used_halton_ids)


        num_autoadjust_attempts = 0
        
        while num_autoadjust_attempts < max_autoadjust_attempts:
            self.logger.info("*****************************************************************")
            cosolv_xyzs = defaultdict(list)
            # This is used to update the kdtree of the placed cosolvents
            placed_atoms_positions = []
            terminate_early = False 
            for cosolvent in cosolvents:
                if terminate_early:
                    break  
                self.logger.info(f"Attempting to place {cosolvent.copies} copies of {cosolvent.name}")
                c_xyz = cosolvents[cosolvent]
                while len(cosolv_xyzs[cosolvent]) < cosolvent.copies and not(terminate_early):
                    if len(placed_atoms_positions) < 1:
                        counter = np.random.choice(len(points))
                        xyz = points[counter]
                        cosolv_xyz = c_xyz + xyz
                        if self.check_coordinates_to_add(cosolv_xyz, None, prot_kdtree):
                            [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
                            cosolv_xyzs[cosolvent].append(cosolv_xyz*openmmunit.nanometer)
                            used_halton_ids.append(counter)
                            kdtree = spatial.cKDTree(placed_atoms_positions)
                    else:
                        kdtree = spatial.cKDTree(placed_atoms_positions)
                        cosolv_xyz, valid_ids, num_trials = self.accept_reject(c_xyz, points, kdtree, valid_ids, lowerBound, vectors, prot_kdtree, max_num_trials)
                        mol_num = len(cosolv_xyzs[cosolvent])+1

                        if isinstance(cosolv_xyz, int):
                            self.logger.warning("Could not place cosolvent molecule %d!" % mol_num)
                            terminate_early = True 
                            num_autoadjust_attempts += 1
                            self.logger.warning("Reducing number of cosolvent copies by factor of %.2f" % copies_factor_reduction) 
                            self.reduce_copies(copies_factor_reduction) 
                        else:
                            self.logger.info("Placed cosolvent molecule %d after %d trials" % (mol_num, num_trials))
                            cosolv_xyzs[cosolvent].append(cosolv_xyz*openmmunit.nanometer)
                            [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
                if terminate_early:
                    self.logger.warning("Attempting to place cosolvents again with reduced number of copies")
                else:
                    self.logger.info(f"Successfully placed {cosolvent.copies} copies of {cosolvent.name}!")
            if not(terminate_early):
                break #all cosolvents have been added  

        if num_autoadjust_attempts == max_autoadjust_attempts:
            self.logger.error("Could not place cosolvents after %d rounds of copies reduction" % max_autoadjust_attempts)
            sys.exit(1)
            
        self.logger.info("Successfully added the following cosolvents:")
        for cosolvent in cosolv_xyzs:
            self.logger.info(f"{cosolvent.name}: {len(cosolv_xyzs[cosolvent])}")
        return cosolv_xyzs
    
    def delete_edges_points(self, xyzs, lowerBound, upperBound, distance, used_halton_ids):
        xyzs = np.atleast_2d(xyzs)
        x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

        xmin, xmax = lowerBound[0], upperBound[0][0]
        ymin, ymax = lowerBound[1], upperBound[1][1]
        zmin, zmax = lowerBound[2], upperBound[2][2]

        x_close = np.logical_or(np.abs(xmin - x) <= distance, np.abs(xmax - x) <= distance)
        y_close = np.logical_or(np.abs(ymin - y) <= distance, np.abs(ymax - y) <= distance)
        z_close = np.logical_or(np.abs(zmin - z) <= distance, np.abs(zmax - z) <= distance)
        close_to = np.unique(np.argwhere(np.any((x_close, y_close, z_close), axis=0)).flatten())

        return used_halton_ids+list(close_to)

    def check_coordinates_to_add(self, new_coords: np.ndarray, cosolvent_kdtree: spatial.cKDTree, protein_kdtree: spatial.cKDTree) -> bool:
        """Checks that the new coordinates don't clash with the receptor (if present) and/or other cosolvent molecules

        :param new_coords: coordinates of the new molecule of shape (n, 3)
        :type new_coords: np.ndarray
        :param cosolvent_kdtree: tree of the cosolvent molecules present in the box
        :type cosolvent_kdtree: spatial.cKDTree
        :param protein_kdtree: tree of the receptor's coordinates
        :type protein_kdtree: spatial.cKDTree
        :return: True if there are no clashes, False otherwise
        :rtype: bool
        """
        cosolvent_clashes = False
        protein_clashes = False
        check_clashes = cosolvent_kdtree is not None or protein_kdtree is not None
        if cosolvent_kdtree is not None:
            cosolvent_clashes = any(cosolvent_kdtree.query_ball_point(new_coords, self.cosolvents_radius.value_in_unit(openmmunit.nanometer)))
        if protein_kdtree is not None:
            protein_clashes = any(protein_kdtree.query_ball_point(new_coords, self.protein_radius.value_in_unit(openmmunit.nanometer)))
        if check_clashes:
            if not protein_clashes and not cosolvent_clashes:
                return True
            else: return False
        else:
            return self.is_in_box(new_coords, self.lowerBound, self.vectors)

    def accept_reject(self, 
                      xyz: np.ndarray, 
                      halton: list, 
                      kdtree: spatial.cKDTree, 
                      valid_ids: list, 
                      lowerBound: openmmunit.Quantity | Vec3, 
                      upperBound: openmmunit.Quantity | Vec3, 
                      protein_kdtree: spatial.cKDTree,
                      max_num_trials: int = 10000) -> tuple[np.ndarray, list]:
        """Accepts or reject the halton move. A random halton point is selected and checked, if accepted
        the cosolvent is placed there, otherwise a local search is performed in the neighbors of the point 
        (1 tile). If the local search produces no clashes the new position is accepted, otherwise a new 
        random halton point is selected and the old one is marked as not good. The algorithm stops
        when a move is accepted or 10000 of num_trialss are done and no move is accepted.

        :param xyz: 3D coordinates of the cosolvent molecule
        :type xyz: np.ndarray
        :param halton: halton sequence
        :type halton: list
        :param kdtree: tree of the cosolvent molecules positions already placed in the box
        :type kdtree: spatial.cKDTree
        :param valid_ids: valid halton indices
        :type valid_ids: list
        :param lowerBound: lower bound of the box
        :type lowerBound: Union[openmm.unit.Quantity, Vec3]
        :param upperBound: upper bound of the box
        :type upperBound: Union[openmm.unit.Quantity, Vec3]
        :param protein_kdtree: tree of the protein's positions
        :type protein_kdtree: spatial.cKDTree
        :param max_num_trials: the maximum number of halton moves to make
        :param max_num_trials: int 
        :return: accepted coordinates for the cosolvent and the used halton ids
        :rtype: Tuple[np.ndarray, list]
        """        
        num_trials = 0
        accepted = False
        coords_to_return = 0
        moves = self.local_search()
        while not accepted and num_trials < max_num_trials:
            halton_idx = np.random.choice(len(valid_ids))
            rotated_xyz = self.generate_rotation(xyz)
            cosolv_xyz = rotated_xyz + halton[halton_idx]
            valid_ids = np.delete(valid_ids, halton_idx)
            if self.check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
                accepted = True
                coords_to_return = cosolv_xyz
            else:
                num_trials += 1
                for move in moves:
                    move = move*openmmunit.angstrom
                    rotated_xyz = self.generate_rotation(xyz)
                    cosolv_xyz = rotated_xyz + halton[halton_idx] + move.value_in_unit(openmmunit.nanometer)
                    if self.is_in_box(cosolv_xyz, lowerBound, upperBound):
                        if self.check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
                            accepted = True
                            coords_to_return = cosolv_xyz
                            break
                    num_trials += 1
        return coords_to_return, valid_ids, num_trials 

    def is_in_box(self, 
                  xyzs: np.ndarray, 
                  lowerBound: Union[openmmunit.Quantity, Vec3], 
                  upperBound: Union[openmmunit.Quantity, Vec3]) -> bool:
        """Checks if the coordinates are in the box or not

        :param xyzs: coordinates to check
        :type xyzs: np.ndarray
        :param lowerBound: lower bound of the box
        :type lowerBound: Union[openmmunit.Quantity, Vec3]
        :param upperBound: upper bound of the box
        :type upperBound: Union[openmmunit.Quantity, Vec3]
        :return: True if all the coordinates are in the box, Flase otherwise
        :rtype: bool
        """
        # cutoff = (1.5*openmmunit.angstrom).value_in_unit(openmmunit.nanometer)
        xyzs = np.atleast_2d(xyzs)
        x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

        xmin, xmax = lowerBound[0], upperBound[0][0]
        ymin, ymax = lowerBound[1], upperBound[1][1]
        zmin, zmax = lowerBound[2], upperBound[2][2]

        x_in = np.logical_and(xmin <= x, x <= xmax)
        y_in = np.logical_and(ymin <= y, y <= ymax)
        z_in = np.logical_and(zmin <= z, z <= zmax)
        all_in = np.all((x_in, y_in, z_in), axis=0)

        return np.all(all_in)

    def local_search(self) -> list:
        """Return all the possible moves in the 1 tile neighbors

        :return: combinations
        :rtype: list
        """
        step = 1
        moves = filter(lambda point: not all(axis ==0 for axis in point), list(product([-step, 0, step], repeat=3)))
        return moves

    def generate_rotation(self, coords: np.ndarray) -> np.ndarray:
        """Rotate a list of 3D [x,y,z] vectors about corresponding random uniformly
            distributed quaternion [w, x, y, z]

        :param coords: list of [x, y, z] cartesian vector coordinates
        :type coords: np.ndarray
        :return: rotated coordinates
        :rtype: np.ndarray
        """
        rand = np.random.rand(3)
        r1 = np.sqrt(1.0 - rand[0])
        r2 = np.sqrt(rand[0])
        pi2 = math.pi * 2.0
        t1 = pi2 * rand[1]
        t2 = pi2 * rand[2]
        qrot = np.array([np.cos(t2) * r2,
                        np.sin(t1) * r1,
                        np.cos(t1) * r1,
                        np.sin(t2) * r2])
        rotation = spatial.transform.Rotation.from_quat(qrot)
        return rotation.apply(coords)
    
#region SizeChecks
    def calculate_mol_volume(self, mol_positions: np.ndarray) -> float:
        """Calculates the volume occupied by the 3D coordinates provided based
        on voxelization.

        :param mol_positions: 3D coordinates of the molecule
        :type mol_positions: np.ndarray
        :return: volume occupied in nm**3
        :rtype: float
        """
        padding = 3.5*openmmunit.angstrom
        offset = 1.5*openmmunit.angstrom
        mesh_step = 0.3*openmmunit.angstrom
        padding = padding.value_in_unit(openmmunit.nanometer)
        offset = offset.value_in_unit(openmmunit.nanometer)
        mesh_step = mesh_step.value_in_unit(openmmunit.nanometer)
        if isinstance(mol_positions, openmmunit.Quantity):
            mol_positions = mol_positions.value_in_unit(openmmunit.nanometer)
        minRange = np.array([min((pos[i] for pos in mol_positions)) for i in range(3)])
        maxRange = np.array([max((pos[i] for pos in mol_positions)) for i in range(3)])
        x = np.arange(minRange[0]-padding, maxRange[0]+padding, mesh_step)
        y = np.arange(minRange[1]-padding, maxRange[1]+padding, mesh_step)
        z = np.arange(minRange[2]-padding, maxRange[2]+padding, mesh_step)
        X, Y, Z = np.meshgrid(x, y, z)
        center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        kdtree = spatial.cKDTree(center_xyzs)
        query = kdtree.query_ball_point(mol_positions, offset)
        points = np.unique(np.hstack(query)).astype(int)
        return round(len(points)*mesh_step**3, 2)

    def fitting_checks(self) -> Union[float, None]:
        """Checks if the required cosolvents can fit in the box and 
        do not exceed the 50% of the available fillable volume 
        (volume not occupied by the receptor, if present).

        :return: available volume if the cosolvents can fit, None otherwise
        :rtype: Union[float, None]
        """

        self.logger.info(f"Volume of the box: {self.box_volume:.2f} nm³")

        prot_volume = 0
        if self.receptor:
            prot_volume = self.calculate_mol_volume(self.modeller.positions)
            self.logger.info(f"Volume of the protein: {prot_volume:.2f} nm³")
        empty_volume = self.cubic_nanometers_to_liters(self.box_volume - prot_volume)
        self._copies_from_concentration(empty_volume)
        cosolvs_volume = defaultdict(float)
        for cosolvent in self.cosolvents:
            cosolvs_volume[cosolvent] = self.calculate_mol_volume(self.cosolvents[cosolvent])*cosolvent.copies
        volume_occupied_by_cosolvent = round(sum(cosolvs_volume.values()), 3)
        empty_available_volume = round(self.liters_to_cubic_nanometers(empty_volume)/2., 3)
        self.logger.info(f"Volume requested for cosolvents: {volume_occupied_by_cosolvent:.2f} nm³")
        self.logger.info(f"Volume available for cosolvents: {empty_available_volume} nm³")
        if volume_occupied_by_cosolvent > empty_available_volume:
            return None
        return empty_available_volume

    def liters_to_cubic_nanometers(self, liters: Union[float, openmmunit.Quantity]) -> float:
        """Converts liters in cubic nanometers

        :param liters: volume to convert
        :type liters: Union[float, openmm.unit.Quantity]
        :return: converted volume
        :rtype: float
        """
        if isinstance(liters, openmmunit.Quantity):
            liters = liters.value_in_unit(openmmunit.liters)
        value = liters * 1e+24
        return value

    def cubic_nanometers_to_liters(self, vol: float) -> float:
        """Converts cubic nanometers in liters

        :param vol: volume to convert
        :type vol: float
        :return: converted volume
        :rtype: float
        """
        value = vol * 1e-24
        return value
#endregion
#endregion                

#region ForceFieldParametrization
    def _parametrize_system(self, forcefields: dict, engine: str, cosolvents: dict) -> app.ForceField:
        """Parametrize the system with the specified forcefields

        :param forcefields: dictionary of the forcefields to use (from forcefields.json)
        :type forcefields: dict
        :param engine: name of the simulation engine to use
        :type engine: str
        :param cosolvents: cosolvent moleucles (from cosolvents.json)
        :type cosolvents: dict
        :return: forcefield object
        :rtype: openmm.app.ForceField
        """
        engine = engine.upper()
        forcefield = app.ForceField(*forcefields[engine])
        sm_ff = forcefields["small_molecules"][0]
        small_molecule_ff = self._parametrize_cosolvents(cosolvents, small_molecule_ff=sm_ff)
        forcefield.registerTemplateGenerator(small_molecule_ff.generator)
        return forcefield

    def _parametrize_cosolvents(self, cosolvents: dict, small_molecule_ff="espaloma") -> SmallMoleculeTemplateGenerator:
        """Parametrizes cosolvent molecules according to the forcefiled specified.

        :param cosolvents: cosolvents specified
        :type cosolvents: dict
        :param small_molecule_ff: name of the forcefield to use, defaults to "espaloma"
        :type small_molecule_ff: str, optional
        :return: forcefield object for the small molecules
        :rtype: SmallMoleculeTemplateGenerator
        """
        molecules = list()
        for cosolvent in cosolvents:
            try:
                molecules.append(Molecule.from_smiles(cosolvent.smiles, name=cosolvent.name))
            except Exception as e:
                self.logger.info(e)
                self.logger.info(cosolvent)
        if small_molecule_ff == "espaloma":
            small_ff = EspalomaTemplateGenerator(molecules=molecules, forcefield='espaloma-0.3.2', template_generator_kwargs={"reference_forcefield": "openff_unconstrained-2.1.0", "charge_method": "nn"})
        elif small_molecule_ff == "gaff":
            small_ff = GAFFTemplateGenerator(molecules=molecules, forcefield='gaff-2.11')
        else:
            small_ff = SMIRNOFFTemplateGenerator(molecules=molecules)
        return small_ff
           
    def _parametrize_ligands(self, ligands: dict) -> None:
        """Parametrizes ligands according to the forcefiled specified.

        :param ligands: ligands specified
        :type ligands: dict
        :param small_molecule_ff: name of the forcefield to use, defaults to "espaloma"
        :type small_molecule_ff: str, optional
        :return: forcefield object for the small molecules
        :rtype: SmallMoleculeTemplateGenerator
        """
        for ligname, lig_path in ligands.items():
            try:
                rdkit_mol = SDMolSupplier(lig_path)[0]
                ligand = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)
                
                if self.small_molecule_forcefield == "espaloma":
                    template_generator = EspalomaTemplateGenerator(
                        molecules=ligand, forcefield="espaloma-0.3.2"
                    )
                elif self.small_molecule_forcefield == "SMIRNOFF":
                    template_generator = SMIRNOFFTemplateGenerator(
                        molecules=ligand, forcefield="openff-1.2.0"
                    )
                elif self.small_molecule_forcefield == "GAFF":
                    template_generator = GAFFTemplateGenerator(
                        molecules=ligand, forcefield="gaff-2.11"
                    )

                # add the template generator to the ff
                self.forcefield.registerTemplateGenerator(template_generator.generator)

                # make an OpenFF Topology of the ligand
                ligand_off_topology = Topology.from_molecules(molecules=[ligand])

                # convert it to an OpenMM Topology
                ligand_topology = ligand_off_topology.to_openmm()

                # get the positions of the ligand
                ligand_positions = to_openmm(ligand.conformers[0])
                
                for res in ligand_topology.residues():
                    res.name = ligname
                self.modeller.add(ligand_topology, ligand_positions)

            except Exception as e:
                self.logger.error(f'Something went wrong parameterizing {ligname} with {self.small_molecule_forcefield} forcefield\n{e}')
                sys.exit(1)
        return None
#endregion
    
#region SimulationBox
    def _build_box(self, 
                   positions: np.ndarray, 
                   padding: openmmunit.Quantity, 
                   box_size: openmmunit.Quantity = None) -> Tuple[Tuple[Vec3, Vec3, Vec3], 
                                                                Vec3, 
                                                                Union[openmmunit.Quantity, Vec3],
                                                                Union[openmmunit.Quantity, Vec3]]:
        """Builds the simulation box. If a receptor is passed it is used alongside with the padding
        parameter to build the box automatically, otherwise a radius has to be passed. If no receptor
        the box is centered on the point [0, 0, 0].

        :param positions: coordinates of the receptor if present
        :type positions: np.ndarray
        :param padding: padding to be used
        :type padding: openmm.unit.Quantity
        :param box_size: box_size specified if no receptor is passed, defaults to None
        :type box_size: openmm.unit.Quantity, optional
        :return: The first element returned is a tuple containing the three vectors describing the simulation box.
                The second element is the box itself.
                Third and fourth elements are the lower and upper bound of the simulation box.
        :rtype: tuple[tuple[Vec3, Vec3, Vec3], Vec3, Union[openmmunit.Quantity, Vec3], Union[openmmunit.Quantity, Vec3]]
        """
        padding = padding.value_in_unit(openmmunit.nanometer)
        if positions is not None:
            positions = positions.value_in_unit(openmmunit.nanometer)
            minRange = Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
            maxRange = Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
            center = 0.5*(minRange+maxRange)
            radius = max(unit.norm(center-pos) for pos in positions)
            width = max(2*radius+padding, 2*padding)
        else:
            center = Vec3(0, 0, 0)
            radius = box_size.value_in_unit(openmmunit.nanometer)
            maxRange = Vec3(radius, radius, radius)
            minRange = Vec3(-radius, -radius, -radius)
            width = radius

        vectors = (Vec3(width, 0, 0), Vec3(0, width, 0), Vec3(0, 0, width))
        box = Vec3(vectors[0][0], vectors[1][1], vectors[2][2])
        origin = center - (np.ceil(np.array((width, width, width))))
        self._box_origin = origin
        self._box_size = np.ceil(np.array([maxRange[0]-minRange[0],
                                           maxRange[1]-minRange[1],
                                           maxRange[2]-minRange[2]])).astype(int)
        lowerBound = center-box/2
        upperBound = center+box/2
        return vectors, box, lowerBound, upperBound
#endregion
#endregion
    
class CosolventMembraneSystem(CosolventSystem):  
    def __init__(self, 
                 cosolvents: str,
                 forcefields: str,
                 ligands: dict,
                 simulation_format: str, 
                 modeller: app.Modeller,  
                 padding: openmmunit.Quantity = 10 * openmmunit.angstrom, 
                 box_size: openmmunit.Quantity = None,
                 lipid_type: str=None,
                 lipid_patch_path: str=None):
        """Creates a CosolventMembraneSystem.

        :param cosolvents: path to the cosolvents.json file
        :type cosolvents: str
        :param forcefields: path to the forcefields.json file
        :type forcefields: str
        :param simulation_format: MD format that want to be used for the simulation
        :type simulation_format: str
        :param modeller: Modeller containing topology and positions information
        :type modeller: openmm.app.Modeller
        :param padding: specify the padding to be used to create the simulation box, defaults to 12*openmmunit.angstrom
        :type padding: openmm.unit.Quantity, optional
        :param box_size: specifies the size to create the box without receptor, defaults to None
        :type box_size: openmm.unit.Quantity, optional
        :param lipid_type: lipid type to use to build the membrane system, defaults to None. Supported types: ["POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"]. 
                                        Mutually exclusive with <lipid_patch_path>.
        :type lipid_type: str, optional
        :param lipid_patch_path: if lipid type is None the path to a pre-equilibrated patch of custom lipids membrane can be passed, defaults to None. Mutually exclusive with <lipid_type>
        :type lipid_patch_path: str, optional
        :raises MutuallyExclusiveParametersError: custom Exception
        """
        super().__init__(cosolvents=cosolvents,
                         forcefields=forcefields,
                         ligands=ligands,
                         simulation_format=simulation_format,
                         modeller=modeller,
                         padding=padding,
                         box_size=box_size)

        self.protein_raidus = 1.5 * openmmunit.angstrom
        self.cosolvents_radius = 2.5 * openmmunit.angstrom           
        self.lipid_type = lipid_type
        self.lipid_patch = None
        
        self._available_lipids = ["POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"]
        self._cosolvent_placement = None
         
        if self.lipid_type is not None and lipid_patch_path is None:
            assert self.lipid_type in self._available_lipids, self.logger.info(f"Error! The specified lipid is not supported! Please choose between the following lipid types:\n\t{self._available_lipids}")
        elif lipid_patch_path is not None and self.lipid_type is None:
            self.lipid_patch = app.PDBFile(lipid_patch_path)
        else:
            self.logger.error("Error! <lipid_type> and <lipid_patch_path> are mutually exclusive parameters. Please pass just one of them.")
            raise MutuallyExclusiveParametersError("Error! <lipid_type> and <lipid_patch_path> are mutually exclusive parameters. Please pass just one of them.")
    
    def add_membrane(self, cosolvent_placement: str='both', neutralize: bool=True, waters_to_keep: list=None):
        """Create a membrane system

        :param cosolvent_placement: determines on what side of the membrane will the cosolvents be placed, defaults to both.
                                    * inside: inside the membrane
                                    * outside: outside the membrane
                                    * both: everywhere
        :type cosolvent_placement: str, optional
        :param neutralize: if neutralize the system when solvating the membrane, defaults to True
        :type neutralize: bool, optional
        :param waters_to_keep: a list of the indices of key waters that should not be deleted, defaults to None
        :type waters_to_keep: list, optional
        :raises SystemError: if OpenMM is not able to relax the system after adding the membrane a SystemError is raised
        """
        waters_residue_names = ["HOH", "WAT"]
        # OpenMM default
        padding = 1 * openmmunit.nanometer
        self._cosolvent_placement = cosolvent_placement
        if self._cosolvent_placement == 'both': self.logger.info("No preference on what side of the membrane to place the cosolvents")
        elif self._cosolvent_placement == 'outside': self.logger.info("Placing cosolvent molecules outside of the membrane")
        elif self._cosolvent_placement == 'inside': self.logger.info("Placing cosolvent molecules inside the membrane")
        else: 
            self.logger.error("Error! Available options for <cosolvent_placement> are ['both' -> no preference, 'outside' -> outside, 'inside' -> inside]")
            raise SystemError
        try:
            if self.lipid_type is not None:
                self.modeller.addMembrane(forcefield=self.forcefield,
                                        lipidType=self.lipid_type,
                                        neutralize=neutralize,
                                        minimumPadding=padding)
            elif self.lipid_patch is not None:
                self.modeller.addMembrane(forcefield=self.forcefield,
                                        lipidType=self.lipid_patch,
                                        neutralize=neutralize,
                                        minimumPadding=padding)
            if waters_to_keep is not None:
                waters_to_delete = [atom for atom in self.modeller.topology.atoms() if atom.residue.index not in waters_to_keep and atom.residue.name in waters_residue_names]
            else:
                waters_to_delete = [atom for atom in self.modeller.topology.atoms() if atom.residue.name in waters_residue_names]
            self.modeller.delete(waters_to_delete)
        except OpenMMException as e:
            self.logger.error("Something went wrong during the relaxation of the membrane.\nProbably a problem related to particle's coordinates.")
            sys.exit(1)
        self.logger.info("Membrane system built.")
        positions = self.modeller.positions.value_in_unit(openmmunit.nanometer)
        minRange = Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
        maxRange = Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
        center = 0.5*(minRange+maxRange)
        radius = max(unit.norm(center-pos) for pos in positions)
        width = 2*radius
        self.vectors = self.modeller.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)
        box = Vec3(self.vectors[0][0], self.vectors[1][1], self.vectors[2][2])
        origin = center - (np.ceil(np.array((width, width, width))))
        self._box_origin = origin
        self._box_size = np.ceil(np.array([maxRange[0]-minRange[0],
                                           maxRange[1]-minRange[1],
                                           maxRange[2]-minRange[2]])).astype(int)
        self.lowerBound = center-box/2
        self.upperBound = center+box/2
        self._periodic_box_vectors = self.modeller.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)
        return

    def build(self, neutralize: bool=True, iteratively_adjust_copies: bool=False):
        """Adds the cosolvent molecules to the system

        :param neutralize: if neutralize the system during solvation, defaults to True
        :type neutralize: bool, optional
        :param iteratively_adjust_copies: if True, the number of copies of each cosolvent will iteratively be reduced until a valid starting configuration is found
        :type iteratively_adjust_copies: bool, optional 
        """
        if self._cosolvent_placement != 'both':
            lipid_positions = list()
            atoms = list(self.modeller.topology.atoms())
            positions = self.modeller.positions.value_in_unit(openmmunit.nanometer)
            for i in range(len(atoms)):
                if atoms[i].residue.name not in proteinResidues and atoms[i].residue.name not in dnaResidues and atoms[i].residue.name not in rnaResidues:
                    lipid_positions.append(positions[i])
            minRange = min((pos[2] for pos in lipid_positions))
            maxRange = max((pos[2] for pos in lipid_positions))
            if self._cosolvent_placement == 'inside':
                upperBound = Vec3(self.upperBound[0], self.upperBound[1], minRange)
                lowerBound = self.lowerBound
            else:
                upperBound = self.upperBound
                lowerBound = Vec3(self.lowerBound[0], self.lowerBound[1],maxRange)
        else:
            upperBound = self.upperBound
            lowerBound = self.lowerBound
        self.logger.info("Checking volumes...")
        volume_not_occupied_by_cosolvent = self.fitting_checks()
        assert volume_not_occupied_by_cosolvent is not None, "The requested volume for the cosolvents exceeds the available volume! Please try increasing the padding or box_size."
        receptor_positions = self.modeller.positions.value_in_unit(openmmunit.nanometer)
        if iteratively_adjust_copies:
            cosolv_xyzs = self.add_cosolvents_adaptive(self.cosolvents, self.vectors, lowerBound, upperBound, receptor_positions)
        else:
            cosolv_xyzs = self.add_cosolvents(self.cosolvents, self.vectors, lowerBound, upperBound, receptor_positions)
        self.modeller = self._setup_new_topology(cosolv_xyzs, self.modeller.topology, self.modeller.positions)
        self.modeller.addSolvent(forcefield=self.forcefield, neutralize=neutralize)
        self.system = self._create_system(self.forcefield, self.modeller.topology)
        return
        
