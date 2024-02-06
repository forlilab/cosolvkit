import json
import os
import sys
import io
from collections import defaultdict
import numpy as np
from scipy import spatial
from scipy.stats import qmc
import math 
from itertools import product
import parmed
from openmm import Vec3, unit, XmlSerializer, System, CustomNonbondedForce, NonbondedForce, OpenMMException
import openmm.app as app
import openmm.unit as openmmunit
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from openmmforcefields.generators.template_generators import SmallMoleculeTemplateGenerator
from cosolvkit.utils import fix_pdb, MutuallyExclusiveParametersError


proteinResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR', 'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL']
rnaResidues = ['A', 'G', 'C', 'U', 'I']
dnaResidues = ['DA', 'DG', 'DC', 'DT', 'DI']

class CosolventMolecule(object):
    def __init__(self, name, smiles=None, mol_filename=None, resname=None, copies=None, concentration=None):
        """Create a Cosolvent object.

        Parameters
        ----------
        name : str
            Name of the molecule.
        smiles : str, default=None
            SMILES string of the molecule in the chosen protonation state.
        mol_filename : str, default=None
            MOL/SDF filename of the molecule in the chosen protonation state.
        resname : str, default=None
            3-letters residue name of the molecule. If None, the first 3 
            letters of the name will be used as uppercase.

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
        self.mol_filename = None
        self.pdb_conect = None
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
            mol.SetProp("_Name", name)
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, params=params)
            AllChem.MMFFOptimizeMolecule(mol)

            mol_filename = '%s.mol' % self.name
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
    
    
    def _generate_atom_names_from_mol(self, rdkit_mol):
        """Generate atom names from an RDKit molecule.

        Parameters
        ----------
        rdkit_mol : rdkit.Chem.rdchem.Mol
            RDKit molecule.

        Returns
        -------
        list
            List of atom names.
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


    def _get_pdb_conect(self, rdkit_mol):
        conect = []

        pdb_string = Chem.MolToPDBBlock(rdkit_mol)
        pdb_lines = pdb_string.split('\n')

        for i, line in enumerate(pdb_lines):
            if 'CONECT' in line:
                conect.append(np.array([int(n) for n in line.split()[1:]]))

        return conect

class CosolventSystem(object):
    def __init__(self, 
                 cosolvents: str,
                 forcefields: str,
                 simulation_format: str, 
                 receptor: str = None,  
                 padding: openmmunit.Quantity = 12*openmmunit.angstrom, 
                 radius: openmmunit.Quantity = None,
                 clean_protein: bool=False):
        """
            Create cosolvent system.
            By default it accepts a pdb string for the receptor, otherwise can call
            the from_filename method and pass a pdb file path.

            Args:
                cosolvents : str
                    Path to the cosolvents.json file
                forcefields : str
                    Path to the forcefields.json file
                simulation_format : str
                    MD format that want to be used for the simulation.
                    Supported formats: Amber, Gromacs, CHARMM or openMM
                receptor : None | str
                    PDB string of the protein. 
                    By default is None to allow cosolvent
                    simulations without receptor
                padding : openmm.unit.Quantity
                    Specifies the padding used to create the simulation box 
                    if no receptor is provided. Default to 12 Angstrom
                radius : openmm.unit.Quantity
                    Specifies the radius to create the box without receptor.
                    Default is None
                clean_protein : bool
                    Determines if the protein will be cleaned and prepared
                    with PDBFixer or not.
                    Default is False
        """ 
        
        # Private
        self._available_formats = ["AMBER", "GROMACS", "CHARMM", "OPENMM"]
        self._cosolvent_positions = defaultdict(list)
        self._box = None
        self._periodic_box_vectors = None
        self._box_volume = None
        self._padding = padding

        # Public
        self.protein_radius = 3.5 * openmmunit.angstrom
        self.cosolvents_radius = 2.5*openmmunit.angstrom
        self.modeller = None
        self.system = None
        self.receptor = receptor
        self.cosolvents = dict()
        
        assert (simulation_format.upper() in self._available_formats), f"Error! The simulation format supplied is not supported! Available simulation engines:\n\t{self._available_formats}"

        # Creating the cosolvent molecules from json file
        with open(cosolvents) as fi:
            cosolvents_dict = json.load(fi)
        for c in cosolvents_dict:
            cosolvent = CosolventMolecule(**c)
            cosolvent_xyz = cosolvent.positions*openmmunit.angstrom
            cosolvent_xyz = cosolvent_xyz.value_in_unit(openmmunit.nanometer)
            self.cosolvents[cosolvent] = cosolvent_xyz

        if receptor is not None:
            print("Cleaning protein")
            if clean_protein:
                top, pos = fix_pdb(receptor)
            else:
                pdbfile = app.PDBFile(receptor)
                top, pos = pdbfile.topology, pdbfile.positions
            self.modeller = app.Modeller(top, pos)
        
        if self.receptor is None:
            assert radius is not None, "Error! If no receptor is passed, the radius parameter has to be set and it needs to be in angstrom openmm.unit"
            assert (isinstance(radius, openmmunit.Quantity)) and (radius.unit == openmmunit.angstrom), \
                "Error! If no receptor is passed, the radius parameter has to be set and it needs to be in angstrom openmm.unit"
            self.vectors, self.box, self.lowerBound, self.upperBound = self._build_box(None, padding, radius=radius)
            self.modeller = app.Modeller(app.Topology(), None)
        else:
            self.vectors, self.box, self.lowerBound, self.upperBound = self._build_box(self.modeller.positions, padding, radius=None)
        
        # Setting up the box - This has to be done before building the system with
        # the cosolvent molecules.
        self.modeller.topology.setPeriodicBoxVectors(self.vectors)
        self._periodic_box_vectors = self.modeller.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)
        print(self.vectors, self._periodic_box_vectors)
        vX, vY, vZ = self.modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.nanometer)
        self.box_volume = vX * vY * vZ
        print("Parametrizing system with forcefields")
        self.forcefield = self._parametrize_system(forcefields, simulation_format, self.cosolvents)
        if not clean_protein:
            self.modeller.addHydrogens(self.forcefield)
        print(f"Box Volume: {self.box_volume} nm**3")
        print(f"\t{self.box_volume*1000} A^3")
        return
    
    @classmethod
    def from_filename(cls, 
                      cosolvents: str,
                      forcefields: str,
                      simulation_format: str, 
                      receptor: str,  
                      padding: openmmunit.Quantity = 12*openmmunit.angstrom,
                      clean_protein: bool=False):
        """
            Create a CosolventSystem with receptor from the pdb file path.

            Args:
                    cosolvents : str
                        Path to the cosolvents.json file
                    forcefields : str
                        Path to the forcefields.json file
                    simulation_format : str
                        MD format that want to be used for the simulation.
                        Supported formats: Amber, Gromacs, CHARMM or openMM
                    receptor : None | str
                        PDB string of the protein. 
                        By default is None to allow cosolvent
                        simulations without receptor
                    padding : openmm.unit.Quantity
                        Specifies the padding used to create the simulation box 
                        if no receptor is provided. Default to 12 Angstrom
                    radius : openmm.unit.Quantity
                        Specifies the radius to create the box without receptor.
                        Default is None
                    clean_protein : bool
                        Determines if the protein will be cleaned and prepared
                        with PDBFixer or not.
                        Default is False
        """
        with open(receptor) as fi:
            pdb_string = fi.read()
        return cls(cosolvents, forcefields, simulation_format, io.StringIO(pdb_string), padding, None, clean_protein)
    
#region Public
    def build(self,
              solvent_smiles: str="H2O", 
              n_solvent_molecules: int=None,
              neutralize: bool=False):
        """This function adds thd cosolvents specified in the CosolvSystem
        and solvates with the desired solvent. If n_solvent_molecules is not passed
        the function will try to fill the box with the desired solvent to a certain extent.
        Please note that the solvation with solvents different from water may highly impact
        the execution time.

        Args:
            solvent_smiles (str, optional): smiles string defining the desired solvent to use. Defaults to "H2O".
            n_solvent_molecules (int, optional): number of mulecules of solvent to add. Defaults to None.
            neutralize (bool, optional): if True, the system charge will be neutralized by OpenMM. Defaults to False.
        """
        volume_not_occupied_by_cosolvent = self.fitting_checks()
        assert volume_not_occupied_by_cosolvent is not None, "The requested volume for the cosolvents exceeds the available volume! Please try increasing the box padding or radius."
        receptor_positions = self.modeller.positions.value_in_unit(openmmunit.nanometer)
        cosolv_xyzs = self.add_cosolvents(self.cosolvents, self.vectors, self.lowerBound, self.upperBound, receptor_positions)
        self.modeller = self._setup_new_topology(cosolv_xyzs, self.modeller.topology, self.modeller.positions)
        if solvent_smiles == "H2O":
            if n_solvent_molecules is None: self.modeller.addSolvent(self.forcefield, neutralize=neutralize)
            else: self.modeller.addSolvent(self.forcefield, numAdded=n_solvent_molecules, neutralize=neutralize)
            print(f"Waters added: {self._get_n_waters()}")
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
            print(f"Placing {solvent_mol.copies}")
            solv_xyz = self.add_cosolvents(d_mol, self.vectors, self.lowerBound, self.upperBound, self.modeller.positions)
            solv_xyz = self.add_cosolvents(d_mol, self.vectors, self.lowerBound, self.upperBound, self.modeller.positions)
            self.modeller = self._setup_new_topology(solv_xyz, self.modeller.topology, self.modeller.positions)
            
        self.system = self._create_system(self.forcefield, self.modeller.topology)
        return
    
    def add_repulsive_forces(self, residues_names: list, epsilon: float=-0.01, sigma: float=12.0):
        """
            This function adds a LJ repulsive potential between the specified molecules.

            Args:
                residues_names (list): list of residue names.
                epsilon (float): depth of the potential well in kcal/mol (default: -0.01 kcal/mol)
                sigma (float): inter-particle distance in Angstrom (default: 12 A)      
        """
        epsilon = np.sqrt(epsilon * epsilon) * openmmunit.kilocalories_per_mole
        sigma = sigma * openmmunit.angstrom

        forces = { force.__class__.__name__ : force for force in self.system.getForces()}
        nb_force = forces['NonbondedForce']
        cutoff_distance = nb_force.getCutoffDistance()
        energy_expression = "4*epsilon * ((sigma / r)^12 * (sigma / r)^6);"
        energy_expression += f"epsilon = {epsilon.value_in_unit_system(openmmunit.md_unit_system)};"
        energy_expression += f"sigma = {sigma.value_in_unit_system(openmmunit.md_unit_system)};"
        repulsive_force = CustomNonbondedForce(energy_expression)
        repulsive_force.addPerParticleParameter("sigma")
        repulsive_force.addPerParticleParameter("epsilon")
        repulsive_force.setNonbondedMethod(NonbondedForce.CutoffPeriodic)
        repulsive_force.setCutoffDistance(cutoff_distance)
        repulsive_force.setUseLongRangeCorrection(False)
        repulsive_force.setUseSwitchingFunction(True)
        repulsive_force.setSwitchingDistance(cutoff_distance - 1.0*unit.angstroms)

        target_indices = []
        for i, atom in enumerate(self.modeller.getTopology().atoms()):
            if not atom.residue.name in residues_names:
                charge, sigma, epsilon = nb_force.getParticleParameters(i)
            else:
                target_indices.append(i)
            repulsive_force.addParticle([sigma, epsilon])
        
        for index in range(nb_force.getNumExceptions()):
            idx, jdx, c, s, eps = nb_force.getExceptionParameters(index)
            repulsive_force.addExclusion(idx, jdx)

        repulsive_force.addInteractionGroup(target_indices, target_indices)
        self.system.addForce(repulsive_force)
        return

    def save_pdb(self, topology: app.Topology, positions: list, out_path: str):
        """Saves the specified topology and position to the out_path file.

        Args:
            topology (app.Topology): topology used
            positions (list): list of 3D coords
            out_path (str): path to where to save the file
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

        Args:
            out_path (str): path where to save the System
            system (System): system to be saved
        """
        with open(f"{out_path}/system.xml", "w") as fo:
            fo.write(XmlSerializer.serialize(system))
        return
    
    def load_system(self, system_path: str) -> System:
        """Loads the desired system.

        Args:
            system_path (str): path to the system file

        Returns:
            System: system
        """
        with open(system_path) as fi:
            system = XmlSerializer.deserialize(fi.read())
        return system

    def save_topology(self, topology: app.Topology, positions: list, system: System, simulation_format: str, forcefield: app.ForceField, out_path: str):
        """Save the topology files necessary for MD simulations according to the simulation engine specified.

        Args:
            topology (app.Topology): openmm topology 
            positions (list): list of 3D coordinates of the topology
            system (System): openmm system
            simulation_format (str): name of the simulation engine
            forcefield (app.Forcefield): openmm forcefield
            out_path (str): output path to where to save the topology files
        """
        new_system = forcefield.createSystem(topology,
                                             nonbondedMethod=app.PME,
                                             nonbondedCutoff=10*openmmunit.angstrom,
                                             removeCMMotion=False,
                                             rigidWater=False,
                                             hydrogenMass=1.5*openmmunit.amu)
        
        parmed_structure = parmed.openmm.load_topology(topology, new_system, positions)

        simulation_format = simulation_format.upper()
        if simulation_format == "AMBER":
            parmed_structure.save(f'{out_path}/system.prmtop', overwrite=True)
            parmed_structure.save(f'{out_path}/system.inpcrd', overwrite=True)

        elif simulation_format == "GROMACS":
            parmed_structure.save(f'{out_path}/system.top', overwrite=True)
            parmed_structure.save(f'{out_path}/system.gro', overwrite=True)

        elif simulation_format == "CHARMM":
            parmed_structure.save(f'{out_path}/system.psf', overwrite=True)
            parmed_structure.save(f'{out_path}/system.crd', overwrite=True)
            
        elif simulation_format == "OPENMM":
            self.save_system(out_path, system)
            self.save_pdb(topology, positions, f"{out_path}/system.pdb")
            parmed_structure.save(f'{out_path}/system.prmtop', overwrite=True)
        else:
            print("The specified simulation engine is not supported!")
            print(f"Available simulation engines:\n\t{self._available_formats}")
        return
#endregion
    
#region Private
#region Misc
    def _copies_from_concentration(self, water_volume: float):
        """Computes the number of copies of cosolvent necessary to reach the desired concentration

        Args:
            water_volume (float): volume available to be filled with cosolvents.
        """
        for cosolvent in self.cosolvents:
            if cosolvent.concentration is not None:
                cosolvent.copies = int(math.floor((((cosolvent.concentration*openmmunit.molar)*(water_volume*openmmunit.liters))*openmmunit.AVOGADRO_CONSTANT_NA) + 0.5))
        return

    
    def _get_n_waters(self) -> int:
        """Returns the number of waters in the system

        Returns:
            int: number of waters in the system
        """
        res = [r.name for r in self.modeller.topology.residues()]
        return res.count('HOH')
    
    def _setup_new_topology(self, cosolvents_positions: dict, receptor_topology: app.Topology = None, receptor_positions:list = None) -> app.Modeller:
        """Returns a new modeller with the topolgy with the new molecules specified

        Args:
            cosolvents_positions (dict): keys are cosolvent molecules and values are lists of position of the new molecules to add
            receptor_topology (app.Topology, optional): old topology to which add the new molecules. Defaults to None.
            receptor_positions (list, optional): old positions to which add the new molecules. Defaults to None.

        Returns:
            app.Modeller: new modeller containing combined topology and positions
        """
        # Adding the cosolvent molecules
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

        molecules_positions = np.array(molecules_positions)
        new_top = Topology.from_molecules(molecules).to_openmm()
        residues = list(new_top.residues())
        for i in range(len(residues)):
            residues[i].name = cosolvent_names[i]
        new_mod = app.Modeller(new_top, molecules_positions)
        if receptor_topology is not None and receptor_positions is not None and len(receptor_positions) > 0: 
            new_mod.add(receptor_topology, receptor_positions)
        new_mod.topology.setPeriodicBoxVectors(self._periodic_box_vectors)
        return new_mod
    
    def _create_system(self, forcefield: app.forcefield, topology: app.Topology) -> System:
        """Returns system created from the Forcefield and the Topology.

        Args:
            forcefield (app.forcefield): Forcefield(s) used to build the system
            topology (app.Topology): Topology used to build the system 

        Returns:
            System: created system
        """
        print("Creating system")
        system = forcefield.createSystem(topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=10*openmmunit.angstrom,
                                         constraints=app.HBonds,
                                         hydrogenMass=1.5*openmmunit.amu)
        return system 
#endregion
#region FillTheVoid
    def add_cosolvents(self, 
                       cosolvents: dict, 
                       vectors: tuple[Vec3, Vec3, Vec3], 
                       lowerBound: openmmunit.Quantity | Vec3, 
                       upperBound: openmmunit.Quantity | Vec3,
                       receptor_positions: list) -> dict:
        """This function adds the desired number of cosolvent molecules using the halton sequence
        to generate random uniformly distributed points inside the grid where to place the cosolvent molecules.
        At first, if a receptor/protein is present the halton sequence points that would clash with the protein
        are pruned.

        Args:
            cosolvents (dict): keys are cosolvent molecules and values are 3D coordinates of the molecule
            vectors (tuple[Vec3, Vec3, Vec3]): vectors defining the simulation box
            lowerBound (openmmunit.Quantity | Vec3): lower bound of the simulation box
            upperBound (openmmunit.Quantity | Vec3): upper bound of the simulation box
            receptor_positions (list): 3D coordinates of the receptor

        Returns:
            dict: keys are cosolvent molecules and values are 3D coordinates of the newly added cosolvents molecules
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
        for cosolvent in cosolvents:
            print(f"Placing {cosolvent.copies} copies of {cosolvent.name}")
            c_xyz = cosolvents[cosolvent]
            for replicate in range(cosolvent.copies):
                counter = replicate
                if len(placed_atoms_positions) < 1:
                    xyz = points[counter]
                    cosolv_xyz = c_xyz + xyz
                    if self.check_coordinates_to_add(cosolv_xyz, None, prot_kdtree):
                        [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
                        cosolv_xyzs[cosolvent].append(cosolv_xyz*openmmunit.nanometer)
                        used_halton_ids.append(counter)
                        kdtree = spatial.cKDTree(placed_atoms_positions)
                else:
                    kdtree = spatial.cKDTree(placed_atoms_positions)
                    cosolv_xyz, valid_ids = self.accept_reject(c_xyz, points, kdtree, valid_ids, lowerBound, vectors, prot_kdtree)

                    if isinstance(cosolv_xyz, int):
                        print("Could not place the cosolvent molecule!")
                    else:
                        cosolv_xyzs[cosolvent].append(cosolv_xyz*openmmunit.nanometer)
                        [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
            print("Done!")
        print("Added cosolvents:")
        for cosolvent in cosolv_xyzs:
            print(f"{cosolvent.name}: {len(cosolv_xyzs[cosolvent])}")
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

        Args:
            new_coords (np.ndarray): coordinates of the new molecule of shape (n, 3)
            cosolvent_kdtree (spatial.cKDTree): binary tree of the cosolvent molecules present in the box
            protein_kdtree (spatial.cKDTree): binary tree of the receptor's coordinates

        Returns:
            bool: True if there are no clashes False otherwise
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
                      protein_kdtree: spatial.cKDTree) -> tuple[np.ndarray, list]:
        """Accepts or reject the halton move. A random halton point is selected and checked, if accepted
        the cosolvent is placed there, otherwise a local search is performed in the neighbors of the point 
        (1 tile). If the local search produces no clashes the new position is accepted, otherwise a new 
        random halton point is selected and the old one is marked as not good. The algorithm stops
        when a move is accepted or 1000000 of trials are done and no move is accepted.

        Args:
            xyz (np.ndarray): 3D coordinates of the cosolvent molecule
            halton (list): halton sequence
            kdtree (spatial.cKDTree): binary tree of the cosolvent molecules positions already placed in the box
            valid_ids (list): valid halton indices
            lowerBound (openmmunit.Quantity | Vec3): lower bound of the box
            upperBound (openmmunit.Quantity | Vec3): upper bound of the box
            protein_kdtree (spatial.cKDTree): binary tree of the protein's positions

        Returns:
            tuple[np.ndarray, list]: accepted coordinates for the cosolvent and the used halton ids
        """
        
        trial = 0
        accepted = False
        coords_to_return = 0
        moves = self.local_search()
        while not accepted and trial < 1000000:
            halton_idx = np.random.choice(len(valid_ids))
            rotated_xyz = self.generate_rotation(xyz)
            cosolv_xyz = rotated_xyz + halton[halton_idx]
            valid_ids = np.delete(valid_ids, halton_idx)
            if self.check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
                accepted = True
                coords_to_return = cosolv_xyz
            else:
                trial += 1
                for move in moves:
                    move = move*openmmunit.angstrom
                    rotated_xyz = self.generate_rotation(xyz)
                    cosolv_xyz = rotated_xyz + halton[halton_idx] + move.value_in_unit(openmmunit.nanometer)
                    if self.is_in_box(cosolv_xyz, lowerBound, upperBound):
                        if self.check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
                            accepted = True
                            coords_to_return = cosolv_xyz
                            break
                    trial += 1
        return coords_to_return, valid_ids

    def is_in_box(self, 
                  xyzs: np.ndarray, 
                  lowerBound: openmmunit.Quantity | Vec3, 
                  upperBound: openmmunit.Quantity | Vec3) -> bool:
        """Checks if the coordinates are in the box or not

        Args:
            xyzs (np.ndarray): coordinates to check
            lowerBound (openmmunit.Quantity | Vec3): lower bound of the box
            upperBound (openmmunit.Quantity | Vec3): upper bound of the box

        Returns:
            bool: True if all the coordinates are in the box, False otherwise
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

        Returns:
            list: combinations
        """
        step = 1
        moves = filter(lambda point: not all(axis ==0 for axis in point), list(product([-step, 0, step], repeat=3)))
        return moves

    def generate_rotation(self, coords: np.ndarray) -> np.ndarray:
        """ Rotate a list of 3D [x,y,z] vectors about corresponding random uniformly
            distributed quaternion [w, x, y, z]
        Args:
            coords (np.ndarray) with shape [n,3]: list of [x,y,z] cartesian vector coordinates
        Returns:
            np.ndarray: rotated coordinates
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

        Args:
            mol_positions (np.ndarray): 3D coordinates

        Returns:
            float: volume occupied in nm**3
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

    def fitting_checks(self) -> float | None:
        """Checks if the required cosolvents can fit in the box and 
        do not exceed the 50% of the available fillable volume 
        (volume not occupied by the receptor, if present).

        Returns:
            float | None: available volume if the cosolvents can fit, None otherwise
        """
        prot_volume = 0
        if self.receptor:
            prot_volume = self.calculate_mol_volume(self.modeller.positions)
            prot_volume = prot_volume
            print(f"Volume protein: {prot_volume*1000} A^3")
        empty_volume = self.cubic_nanometers_to_liters(self.box_volume - prot_volume)
        self._copies_from_concentration(empty_volume)
        cosolvs_volume = defaultdict(float)
        for cosolvent in self.cosolvents:
            cosolvs_volume[cosolvent] = self.calculate_mol_volume(self.cosolvents[cosolvent])*cosolvent.copies
        volume_occupied_by_cosolvent = round(sum(cosolvs_volume.values()), 3)
        empty_available_volume = round(self.liters_to_cubic_nanometers(empty_volume)/2., 3)
        print(f"Volume requested for cosolvents: {volume_occupied_by_cosolvent} nm**3")
        print(f"Volume available for cosolvents: {empty_available_volume} nm**3")
        if volume_occupied_by_cosolvent > empty_available_volume:
            return None
        return empty_available_volume

    def liters_to_cubic_nanometers(self, liters: float | openmmunit.Quantity) -> float:
        """Converts liters in cubic nanometers

        Args:
            liters (float | openmmunit.Quantity): volume to convert

        Returns:
            float: converted volume 
        """
        if isinstance(liters, openmmunit.Quantity):
            liters = liters.value_in_unit(openmmunit.liters)
        value = liters * 1e+24
        return value

    def cubic_nanometers_to_liters(self, vol: float) -> float:
        """Converts cubic nanometers in liters

        Args:
            vol (float): volume to convert

        Returns:
            float: converted volume
        """
        value = vol * 1e-24
        return value
#endregion
#endregion                

#region ForceFieldParametrization
    def _parametrize_system(self, forcefields: str, engine: str, cosolvents: dict) -> app.ForceField:
        """Parametrize the system with the specified forcefields

        Args:
            forcefields (str): path to the json file containing the forcefields to use
            engine (str): name of the simulation engine
            cosolvents (dict): cosolvent molecules

        Returns:
            app.ForceField: forcefield obj
        """
        with open(forcefields) as fi:
            ffs = json.load(fi)
        engine = engine.upper()
        forcefield = app.ForceField(*ffs[engine])
        sm_ff = ffs["small_molecules"][0]
        small_molecule_ff = self._parametrize_cosolvents(cosolvents, small_molecule_ff=sm_ff)
        forcefield.registerTemplateGenerator(small_molecule_ff.generator)
        return forcefield

    def _parametrize_cosolvents(self, cosolvents: dict, small_molecule_ff="espaloma") -> SmallMoleculeTemplateGenerator:
        """Parametrizes cosolvent molecules according to the forcefiled specified.

        Args:
            cosolvents (dict): cosolvents specified
            small_molecule_ff (str, optional): name of the forcefield to use. Defaults to "espaloma".

        Returns:
            SmallMoleculeTemplateGenerator: forcefiled obj
        """
        molecules = list()
        for cosolvent in cosolvents:
            try:
                molecules.append(Molecule.from_smiles(cosolvent.smiles, name=cosolvent.name))
            except Exception as e:
                print(e)
                print(cosolvent)
        if small_molecule_ff == "espaloma":
            small_ff = EspalomaTemplateGenerator(molecules=molecules, forcefield='espaloma-0.3.2')
        elif small_molecule_ff == "gaff":
            small_ff = GAFFTemplateGenerator(molecules=molecules)
        else:
            small_ff = SMIRNOFFTemplateGenerator(molecules=molecules)
        return small_ff
#endregion
    
#region SimulationBox
    def _build_box(self, 
                   positions: np.ndarray, 
                   padding: openmmunit.Quantity, 
                   radius: openmmunit.Quantity = None) -> tuple[tuple[Vec3, Vec3, Vec3], 
                                                                Vec3, 
                                                                openmmunit.Quantity | Vec3,
                                                                openmmunit.Quantity | Vec3]:
        """Builds the simulation box. If a receptor is passed it is used alongside with the padding
        parameter to build the box automatically, otherwise a radius has to be passed. If no receptor
        the box is centered on the point [0, 0, 0].

        Args:
            positions (np.ndarray): coordinates of the receptor if present
            padding (openmmunit.Quantity): padding to be used
            radius (openmmunit.Quantity, optional): radius specified if no receptor is passed. Defaults to None.

        Returns:
            tuple[tuple[Vec3, Vec3, Vec3], Vec3, openmmunit.Quantity | Vec3, openmmunit.Quantity | Vec3]: 
                The first element returned is a tuple containing the three vectors describing the simulation box.
                The second element is the box itself.
                Third and fourth elements are the lower and upper bound of the simulation box. 
        """
        padding = padding.value_in_unit(openmmunit.nanometer)
        if positions is not None:
            positions = positions.value_in_unit(openmmunit.nanometer)
            minRange = Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
            maxRange = Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
            center = 0.5*(minRange+maxRange)
            radius = max(unit.norm(center-pos) for pos in positions)
        else:
            center = Vec3(0, 0, 0)
            radius = radius.value_in_unit(openmmunit.nanometer)
            maxRange = Vec3(radius, radius, radius)
            minRange = Vec3(-radius, -radius, -radius)
        width = max(2*radius+padding, 2*padding)
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
                 simulation_format: str, 
                 receptor: str = None,  
                 padding: openmmunit.Quantity = 12*openmmunit.angstrom, 
                 radius: openmmunit.Quantity = None,
                 clean_protein: bool=False,
                 lipid_type: str=None,
                 lipid_patch_path: str=None):
        """
        Creates a CosolventMembraneSystem.

        Args:
            cosolvents (str): Path to the cosolvents.json file
            forcefields (str): Path to the forcefields.json file
            simulation_format (str): MD format that want to be used for the simulation.
                                     Supported formats: Amber, Gromacs, CHARMM or openMM
            receptor (str, optional): PDB string of the protein. 
                                      By default is None to allow cosolvent
                                      simulations without receptor. Defaults to None.
            padding (openmmunit.Quantity, optional): Specifies the padding used to create the simulation box 
                                                     if no receptor is provided. Default to 12 Angstrom. 
                                                     Defaults to 12*openmmunit.angstrom.
            radius (openmmunit.Quantity, optional): Specifies the radius to create the box without receptor.
                                                    Defaults to None.
            clean_protein (bool, optional): Determines if the protein will be cleaned and prepared with PDBFixer or not. 
                                            Defaults to False.
            lipid_type (str, optional): Lipid type to use to build the membrane system, 
                                        supported types: ["POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"]. 
                                        Mutually exclusive with <lipid_patch_path>.
                                        Defaults to None.
            lipid_patch_path (str, optional): If lipid type is None the path to a pre-equilibrated patch of custom
                                              lipids membrane can be passed. Mutually exclusive with <lipid_type>.
                                              Defaults to None.
        """
        
        super().__init__(cosolvents=cosolvents,
                         forcefields=forcefields,
                         simulation_format=simulation_format,
                         receptor=receptor,
                         padding=padding,
                         radius=radius,
                         clean_protein=clean_protein)
        
        self.protein_raidus = 1.5 * openmmunit.angstrom
        self.cosolvents_radius = 2.5*openmmunit.angstrom           
        self.lipid_type = lipid_type
        self.lipid_patch = None
        
        self._available_lipids = ["POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"]
        self._cosolvent_placement = None
         
        if self.lipid_type is not None and lipid_patch_path is None:
            assert self.lipid_type in self._available_lipids, print(f"Error! The specified lipid is not supported! Please choose between the following lipid types:\n\t{self._available_lipids}")
        elif lipid_patch_path is not None and self.lipid_type is None:
            self.lipid_patch = app.PDBFile(lipid_patch_path)
        else:
            raise MutuallyExclusiveParametersError("Error! <lipid_type> and <lipid_patch_path> are mutually exclusive parameters. Please pass just one of them.")
    
    @classmethod
    def from_filename(cls, 
                      cosolvents: str,
                      forcefields: str,
                      simulation_format: str, 
                      receptor: str,  
                      padding: openmmunit.Quantity = 12*openmmunit.angstrom,
                      clean_protein: bool=False,
                      lipid_type: str=None,
                      lipid_patch_path: str=None):
        """
        Create a CosolventMembraneSystem with receptor from the pdb file path.

        Args:
            cosolvents (str): Path to the cosolvents.json file
            forcefields (str): Path to the forcefields.json file
            simulation_format (str): MD format that want to be used for the simulation.
                                     Supported formats: Amber, Gromacs, CHARMM or openMM
            receptor (str, optional): PDB string of the protein. 
                                      By default is None to allow cosolvent
                                      simulations without receptor. Defaults to None.
            padding (openmmunit.Quantity, optional): Specifies the padding used to create the simulation box 
                                                     if no receptor is provided. Default to 12 Angstrom. 
                                                     Defaults to 12*openmmunit.angstrom.
            clean_protein (bool, optional): Determines if the protein will be cleaned and prepared with PDBFixer or not. 
                                            Defaults to False.
            lipid_type (str, optional): Lipid type to use to build the membrane system, 
                                        supported types: ["POPC", "POPE", "DLPC", "DLPE", "DMPC", "DOPC", "DPPC"]. 
                                        Mutually exclusive with <lipid_patch_path>.
                                        Defaults to None.
            lipid_patch_path (str, optional): If lipid type is None the path to a pre-equilibrated patch of custom
                                              lipids membrane can be passed. Mutually exclusive with <lipid_type>.
                                              Defaults to None.
        """
        with open(receptor) as fi:
            pdb_string = fi.read()
        return cls(cosolvents, 
                   forcefields, 
                   simulation_format, 
                   io.StringIO(pdb_string), 
                   padding, 
                   None, 
                   clean_protein, 
                   lipid_type, 
                   lipid_patch_path)
    
    def add_membrane(self, cosolvent_placement: int=0, neutralize: bool=True, waters_to_keep: list=[]):
        """Create the membrane system.

        Args:
            cosolvent_placement (int): Determines on what side of the membrane will the cosolvents be placed.
                                       * -1: Inside the membrane
                                       * +1: Outside the membrane
                                       *  0: Everywhere 
                                       Defaults to 0.
            neutralize (bool, optional): If neutralize the system when solvating the membrane. Defaults to True.
            waters_to_keep (list, optional): A list of the indices of key waters that should not be deleted. 
                                             Defaults to [].

        Raises:
            SystemError: If OpenMM is not able to relax the system after adding the membrane a SystemError is raised.
        """
        waters_residue_names = ["HOH", "WAT"]
        # OpenMM default
        padding = 1 * openmmunit.nanometer
        self._cosolvent_placement = cosolvent_placement
        if self._cosolvent_placement == 0: print("No preference of what side of the membrane to place the cosolvents")
        elif self._cosolvent_placement == 1: print("Placing cosolvent molecules outside of the membrane")
        elif self._cosolvent_placement == -1: print("Placing cosolvent molecules inside the membrane")
        else: 
            print("Error! Available options for <cosolvent_placement> are [0 -> no preference, 1 -> outside, -1 -> inside]")
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
            waters_to_delete = [atom for atom in self.modeller.topology.atoms() if atom.residue.index not in waters_to_keep and atom.residue.name in waters_residue_names]
            self.modeller.delete(waters_to_delete)
        except OpenMMException as e:
            print("Something went wrong during the relaxation of the membrane.\nProbably a problem related to particle's coordinates.")
            sys.exit(1)
        print("Membrane system built.")
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

    def build(self, neutralize: bool=True):
        """
        Adds the cosolvent molecules to the system.


        Args:
            neutralize (bool, optional): If neutralize the system during solvation. Defaults to True.
        """
        if self._cosolvent_placement != 0:
            lipid_positions = list()
            atoms = list(self.modeller.topology.atoms())
            positions = self.modeller.positions.value_in_unit(openmmunit.nanometer)
            for i in range(len(atoms)):
                if atoms[i].residue.name not in proteinResidues and atoms[i].residue.name not in dnaResidues and atoms[i].residue.name not in rnaResidues:
                    lipid_positions.append(positions[i])
            minRange = min((pos[2] for pos in lipid_positions))
            maxRange = max((pos[2] for pos in lipid_positions))
            if self._cosolvent_placement == -1:
                upperBound = Vec3(self.upperBound[0], self.upperBound[1], minRange)
                lowerBound = self.lowerBound
            else:
                upperBound = self.upperBound
                lowerBound = Vec3(self.lowerBound[0], self.lowerBound[1],maxRange)
        else:
            upperBound = self.upperBound
            lowerBound = self.lowerBound
        print("Checking volumes...")
        volume_not_occupied_by_cosolvent = self.fitting_checks()
        assert volume_not_occupied_by_cosolvent is not None, "The requested volume for the cosolvents exceeds the available volume! Please try increasing the box padding or radius."
        receptor_positions = self.modeller.positions.value_in_unit(openmmunit.nanometer)
        cosolv_xyzs = self.add_cosolvents(self.cosolvents, self.vectors, lowerBound, upperBound, receptor_positions, True)
        self.modeller = self._setup_new_topology(cosolv_xyzs, self.modeller.topology, self.modeller.positions)
        self.modeller.addSolvent(forcefield=self.forcefield, neutralize=neutralize)
            
        self.system = self._create_system(self.forcefield, self.modeller.topology)
        return
        
