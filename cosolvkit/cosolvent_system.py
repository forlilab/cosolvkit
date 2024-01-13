import json
import os
from collections import defaultdict
import numpy as np
from scipy import spatial
from scipy.stats import qmc
import math 
from itertools import product
import parmed
from openmm import Vec3, unit, XmlSerializer, System
import openmm.app as app
import openmm.unit as openmmunit
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from cosolvkit.utils import fix_pdb

class CoSolvent:

    def __init__(self, name, smiles=None, mol_filename=None, resname=None, copies=None, concentration=None):
        """Create a CoSolvent object.

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
        # self.pdb_conect = _get_pdb_conect(mol)
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

class CosolventSystem:
    # Need to add concentrations
    def __init__(self, 
                 cosolvents: str,
                 forcefields: str,
                 simulation_engine: str, 
                 receptor: str = None, 
                 hydrate: bool = True,
                 n_waters: int = None, 
                 padding: openmmunit.Quantity = 12*openmmunit.angstrom, 
                 radius: openmmunit.Quantity = None ):
        """
            Create cosolvent system.

            Parameters
            ----------
            cosolvents : str
                Path to the cosolvents.json file
            forcefields : str
                Path to the forcefields.json file
            simulation_engine : str
                Engine that want to be used for the simulation.
                Supported engines: Amber, Gromacs, CHARMM
            receptor : None | str
                Path to the pdb file of the receptor. 
                By default is None to allow cosolvent
                simulations without receptor
            hydrate : bool
                If True, waters will be added by openmm
            n_waters : None | int
                If hydrate is True and n_waters is specified,
                openmm will solvate the system with exactly the number
                of waters specified. By default is None
            padding : openmm.unit.Quantity
                Specifies the padding used to create the simulation box 
                if no receptor is provided. Default to 12 Angstrom
            radius : openmm.unit.Quantity
                Specifies the radius to create the box without receptor.
                Default is None
        """ 
        
        # Private
        self._available_engines = ["AMBER", "GROMACS", "CHARMM"]
        self._cosolvent_positions = defaultdict(list)
        self._box = None
        self._periodic_box_vectors = None
        self._box_volume = None
        self._hydrate = hydrate
        self._padding = padding
        self.n_waters = n_waters

        # Public
        self.modeller = None
        self.system = None
        self.receptor = receptor
        self.cosolvents = dict()
        
        assert (simulation_engine.upper() in self._available_engines), "Error! The simulation engine supplied is not supported!"

        # Creating the cosolvent molecules from json file
        with open(cosolvents) as fi:
            cosolvents_dict = json.load(fi)
        for c in cosolvents_dict:
            cosolvent = CoSolvent(**c)
            cosolvent_xyz = cosolvent.positions*openmmunit.angstrom
            cosolvent_xyz = cosolvent_xyz.value_in_unit(openmmunit.nanometer)
            self.cosolvents[cosolvent] = cosolvent_xyz

        if receptor is not None:
            print("Cleaning protein")
            top, pos = fix_pdb(receptor)
            self.modeller = app.Modeller(top, pos)
            if self.modeller.getTopology() is not None:
                self.modeller.deleteWater()
        
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
        vX, vY, vZ = self.modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.nanometer)
        self.box_volume = vX * vY * vZ
        print("Parametrizing system with forcefields")
        self.forcefield = self._parametrize_system(forcefields, simulation_engine, self.cosolvents)
        # print(f"Number of possible added waters with mesh approach: {self._number_of_possible_added_waters_mesh()}")
        # print("Adding cosolvents and hydrating")
        # self.modeller.addSolvent(self.forcefield)
        # self._added_waters = self._get_n_waters()
        # self._receptor_xyzs, self._water_xyzs, self._wat_res_mapping = self._process_positions(self.modeller)
        print(f"Box Volume: {self.box_volume} nm**3")
        # print(f"Number of waters added: {self._added_waters}")
        return
    
    ############################# PUBLIC
    def build(self, solvent_smiles="H2O", n_solvent_molecules=None):
        volume_not_occupied_by_cosolvent = self.fitting_checks()
        assert volume_not_occupied_by_cosolvent is not None, "The requested volume for the cosolvents exceeds the available volume! Please try increasing the box padding or radius."
        cosolv_xyzs = self.add_cosolvents(self.cosolvents, self.vectors, self.lowerBound, self.upperBound, self.modeller.positions)
        self.modeller = self._setup_new_topology(cosolv_xyzs, self.modeller.topology, self.modeller.positions)
        if solvent_smiles == "H2O":
            if n_solvent_molecules is None: self.modeller.addSolvent(self.forcefield, neutralize=False)
            else: self.modeller.addSolvent(self.forcefield, numAdded=n_solvent_molecules, neutralize=False)
        elif solvent_smiles is not None:
            c = {"name": "solvent",
                 "smiles": solvent_smiles}
            solvent_mol = CoSolvent(**c)
            cosolv_xyz = solvent_mol.positions*openmmunit.angstrom
            if n_solvent_molecules is not None:
                solvent_mol.copies = n_solvent_molecules
            else:
                one_mol_vol = self.calculate_mol_volume(cosolv_xyz)
                solvent_mol.copies = int(math.floor((volume_not_occupied_by_cosolvent/one_mol_vol)+0.5)*.6)
            print(f"Placing {solvent_mol.copies}")
            solv_xyz = self.add_cosolvents({solvent_mol: cosolv_xyz.value_in_unit(openmmunit.nanometer)}, self.vectors, self.lowerBound, self.upperBound, self.modeller.positions)
            self.modeller = self._setup_new_topology(solv_xyz, self.modeller.topology, self.modeller.positions)
        # # cosolv_xyz = self._add_cosolvents(self.cosolvents)
        # self._add_cosolvents_fill_the_void(self.cosolvents,
        #                                    False)
        self.system = self._create_system(self.forcefield, self.modeller.topology)
        return
    
    def save_pdb(self, topology, positions, out_path):
        app.PDBFile.writeFile(
            topology,
            positions,
            open(out_path, "w"),
            keepIds=True
        )
        return

    def save_system(self, out_path: str, system: System):
        with open(f"{out_path}/system.xml", "w") as fo:
            fo.write(XmlSerializer.serialize(system))
        return
    
    def load_system(self, system_path: str):
        with open(system_path) as fi:
            system = XmlSerializer.deserialize(fi.read())
        return system

    def save_topology(self, topology, positions, system, simulation_engine, out_path):
        parmed_structure = parmed.openmm.load_topology(topology, system, positions)

        simulation_engine = simulation_engine.upper()
        if simulation_engine == "AMBER":
            # Add dummy bond type for None ones so that parmed doesn't trip
            bond_type = parmed.BondType(1.0, 1.0, list=parmed_structure.bond_types)
            parmed_structure.bond_types.append(bond_type)
            for bond in parmed_structure.bonds:
                if bond.type is None:
                    bond.type = bond_type

            parmed_structure.save(f'{out_path}/system.prmtop', overwrite=True)
            parmed_structure.save(f'{out_path}/system.inpcrd', overwrite=True)

        elif simulation_engine == "GROMACS":
            parmed_structure.save(f'{out_path}/system.top', overwrite=True)
            parmed_structure.save(f'{out_path}/system.gro', overwrite=True)

        elif simulation_engine == "CHARMM":
            parmed_structure.save(f'{out_path}/system.psf', overwrite=True)

        else:
            print("The specified simulation engine is not supported!")
            print(f"Available simulation engines:\n\t{self._available_engines}")
        return
    
    ############################# PRIVATE
    def _copies_from_concentration(self, water_volume):
        for cosolvent in self.cosolvents:
            if cosolvent.concentration is not None:
                cosolvent.copies = int(math.floor((((cosolvent.concentration*openmmunit.molar)*(water_volume*openmmunit.liters))*openmmunit.AVOGADRO_CONSTANT_NA) + 0.5))
        return

    
    def _get_n_waters(self):
        res = [r.name for r in self.modeller.topology.residues()]
        return res.count('HOH')

    ##################### FIND AND REPLACE APPROACH #####################
    def _process_positions(self, modeller):
        _xyz = [(x, y, z) for x, y, z in modeller.positions.value_in_unit(openmmunit.nanometer)]
        xyz = np.array(_xyz, dtype=float)
        wat_res_mapping = {}
        wat_xyz = []
        protein_xyz = []
        for res in modeller.topology.residues():
            if res.name == "HOH":
                for atom in res.atoms():
                    wat_res_mapping[atom.index] = res
                    wat_xyz.append(xyz[atom.index])
            else:
                for atom in res.atoms():
                    protein_xyz.append(xyz[atom.index])
        return np.array(protein_xyz, dtype=float), np.array(wat_xyz, dtype=float), wat_res_mapping
    
    def _water_close_to_edge(self, wat_xyzs, distance, box_origin, box_size, offset=0):
        """Is it too close from the edge?
        """
        wat_xyzs = np.atleast_2d(wat_xyzs)
        x, y, z = wat_xyzs[:, 0], wat_xyzs[:, 1], wat_xyzs[:, 2]

        xmin, xmax = box_origin[0], box_origin[0] + box_size[0]
        ymin, ymax = box_origin[1], box_origin[1] + box_size[1]
        zmin, zmax = box_origin[2], box_origin[2] + box_size[2]
        distance = distance.value_in_unit(openmmunit.nanometer)
        x_close = np.logical_or(np.abs(xmin - x) <= distance, np.abs(xmax - x) <= distance)
        y_close = np.logical_or(np.abs(ymin - y) <= distance, np.abs(ymax - y) <= distance)
        z_close = np.logical_or(np.abs(zmin - z) <= distance, np.abs(zmax - z) <= distance)
        close_to = np.any((x_close, y_close, z_close), axis=0)

        for i in range(0, wat_xyzs.shape[0]-2, 3):
            close_to[[i, i + 1, i + 2]] = [np.all(close_to[[i, i + 1, i + 2]])] * 3
        indices = np.asarray(close_to==True).nonzero()[0]
        return indices+offset
    
    def _water_close_to_receptor(self, wat_xyzs, receptor_xyzs, distance=3., offset=0):
        kdtree = spatial.cKDTree(wat_xyzs)

        ids = kdtree.query_ball_point(receptor_xyzs, distance.value_in_unit(openmmunit.nanometer))
        ids = np.unique(np.hstack(ids)).astype(int)
        close_to = np.zeros(len(wat_xyzs), bool)
        close_to[ids] = True

        for i in range(0, wat_xyzs.shape[0]-2, 3):
            close_to[[i, i+1, i+2]] = [np.all(close_to[[i, i+1, i+2]])]*3
        indices = np.asarray(close_to==True).nonzero()[0]
        return indices+offset
    
    
    def _add_cosolvents(self, cosolvents):
        banned_ids = list()
        current_number_copies = {}
        final_number_copies = {}
        cosolv_xyzs = defaultdict(list)
        distance_from_cosolvent = 3.5 * openmmunit.angstrom
        offset = list(self._wat_res_mapping.keys())[0]
        water_close_to_edge = self._water_close_to_edge(self._water_xyzs, 
                                                        3.*openmmunit.angstrom, 
                                                        self._box_origin, 
                                                        self._box_size,
                                                        offset=offset)
        banned_ids.append(water_close_to_edge)
        wat_xyzs = self._water_xyzs

        if len(self._receptor_xyzs) > 0:
            water_close_to_protein = self._water_close_to_receptor(wat_xyzs, 
                                                                   self._receptor_xyzs, 
                                                                   distance=4.5*openmmunit.angstrom, 
                                                                   offset=offset)
            banned_ids.append(water_close_to_protein)
        for cosolvent in cosolvents:
            # if cosolvent.concentration is not None:
            #     current_number_copies[cosolvent.name] = 0
            #     n_copies = (self._added_waters * (cosolvent.concentration * openmmunit.molar)) / water_concentration
            #     final_number_copies[cosolvent.name] = int(floor(n_copies + 0.5))
            # else:
                current_number_copies[cosolvent.name] = 0
                final_number_copies[cosolvent.name] = cosolvent.copies
                
        placement_order = []
        for cosolv_name, n in final_number_copies.items():
            placement_order += [cosolv_name] * n
        np.random.shuffle(placement_order)

        oxy_ids = wat_xyzs[::3]
        valid_ids = np.array((range(0, oxy_ids.shape[0])))
        banned_ids = [x for xs in banned_ids for x in xs]
        waters_to_delete = []

        for cosolv_name in placement_order:
            kdtree = spatial.cKDTree(wat_xyzs)

            c = [cosolvent for cosolvent in cosolvents if cosolvent.name == cosolv_name][0]
            if current_number_copies[cosolv_name] < final_number_copies[cosolv_name]:
                wat_id = np.random.choice(valid_ids[~np.isin(valid_ids, np.array(banned_ids))])
                wat_xyz = oxy_ids[wat_id]

                # Translate fragment on the top of the selected water molecule
                cosolv_xyz = self.cosolvents[c] + wat_xyz
                np.append(wat_xyzs, cosolv_xyz).reshape(wat_xyzs.shape[0]+cosolv_xyz.shape[0], 3)
                # Add fragment to list
                cosolv_xyzs[c].append(cosolv_xyz)

                # Get the ids of all the closest water atoms
                to_be_removed = kdtree.query_ball_point(cosolv_xyz, distance_from_cosolvent.value_in_unit(openmmunit.nanometer))
                if any(to_be_removed) > 0:
                    # Keep the unique ids
                    to_be_removed = np.unique(np.hstack(to_be_removed)).astype(int)
                    to_be_removed = to_be_removed+offset
                    # Get the ids of the water oxygen atoms
                    for i in to_be_removed:
                        if i not in banned_ids: banned_ids.append(i)
                        if i not in waters_to_delete: waters_to_delete.append(i)
                    # [banned_ids.append(i) for i in to_be_removed if i not in banned_ids]
                    # [waters_to_delete.append(i) for i in to_be_removed if i not in waters_to_delete]
                current_number_copies[cosolv_name] += 1
        # Delete waters
        waters_removed = set([self._wat_res_mapping[x] for x in waters_to_delete])
        print(f"Removed {len(waters_removed)} waters for the cosolvents!")
        self.modeller.delete(waters_removed)
        return cosolv_xyzs
    ##################### FIND AND REPLACE APPROACH ####################

    ##################### FILL THE VOID APPROACH ####################
    def add_cosolvents(self, cosolvents, vectors, lowerBound, upperBound, receptor_positions):
        protein_radius = 3.5*openmmunit.angstrom
        prot_kdtree = None
        placed_atoms_positions = []
        if receptor_positions is not None and len(receptor_positions) > 0:
            prot_kdtree = spatial.cKDTree(receptor_positions)
        cosolv_xyzs = defaultdict(list)
        sampler = qmc.Halton(d=3)
        points = sampler.random(1000000)
        points= qmc.scale(points, [lowerBound[0], lowerBound[0], lowerBound[0]], [upperBound[0], upperBound[1], upperBound[2]])
        used_halton_ids = list()
        if prot_kdtree is not None:
            banned_ids = prot_kdtree.query_ball_point(points, protein_radius.value_in_unit(openmmunit.nanometer))
            used_halton_ids = list(np.unique(np.hstack(banned_ids)).astype(int))
        for cosolvent in cosolvents:
            print(f"Placing {cosolvent.name}")
            c_xyz = cosolvents[cosolvent]
            for replicate in range(cosolvent.copies):
                counter = replicate
                if len(placed_atoms_positions) < 1:
                    xyz = points[counter]
                    cosolv_xyz = c_xyz + xyz
                    [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
                    cosolv_xyzs[cosolvent].append(cosolv_xyz)
                    used_halton_ids.append(counter)
                    kdtree = spatial.cKDTree(placed_atoms_positions)
                else:
                    kdtree = spatial.cKDTree(placed_atoms_positions)
                    cosolv_xyz, used_halton_ids = self.accept_reject(c_xyz, points, kdtree, used_halton_ids, lowerBound, vectors, prot_kdtree)
                    if isinstance(cosolv_xyz, int):
                        print("Could not place the cosolvent molecule!")
                    else:
                        cosolv_xyzs[cosolvent].append(cosolv_xyz)
                        [placed_atoms_positions.append(pos) for pos in cosolv_xyz]
            print("Done!")
        print("Added cosolvents:")
        for cosolvent in cosolv_xyzs:
            print(f"{cosolvent.name}: {len(cosolv_xyzs[cosolvent])}")
        return cosolv_xyzs

    def check_coordinates_to_add(self, new_coords, cosolvent_kdtree, protein_kdtree):
        protein_radius = 3.5*openmmunit.angstrom
        cosolv_radius = 2.5*openmmunit.angstrom
        if protein_kdtree is not None and not any(protein_kdtree.query_ball_point(new_coords, protein_radius.value_in_unit(openmmunit.nanometer))):
            if cosolvent_kdtree is not None:
                if not any(cosolvent_kdtree.query_ball_point(new_coords, cosolv_radius.value_in_unit(openmmunit.nanometer))):
                    return True
                else: return False
            else:
                return True
        elif protein_kdtree is None and cosolvent_kdtree is not None:
            if not any(cosolvent_kdtree.query_ball_point(new_coords, cosolv_radius.value_in_unit(openmmunit.nanometer))):
                return True
            else: 
                return False
        else:
            return False

    def accept_reject(self, xyz, halton, kdtree, used, lowerBound, upperBound, protein_kdtree):
        trial = 0
        accepted = False
        coords_to_return = 0
        moves = self.local_search()
        valid_ids = np.array(range(0, len(halton)))
        while not accepted and trial < 1000000:
            halton_idx = np.random.choice(valid_ids[~np.isin(valid_ids, np.array(used))])
            rotated_xyz = self.generate_rotation(xyz)
            cosolv_xyz = rotated_xyz + halton[halton_idx]
            if self.check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
                used.append(halton_idx)
                accepted = True
                coords_to_return = cosolv_xyz
            else:
                trial += 1
                for move in moves:
                    rotated_xyz = self.generate_rotation(xyz)
                    cosolv_xyz = rotated_xyz + halton_idx + move
                    if self.is_in_box(cosolv_xyz, lowerBound, upperBound):
                        if self.check_coordinates_to_add(cosolv_xyz, kdtree, protein_kdtree):
                            accepted = True
                            used.append(halton_idx)
                            coords_to_return = cosolv_xyz
                            break
                    trial += 1
        return coords_to_return, used

    def is_in_box(self, xyzs, lowerBound, upperBound):
        """Is in the box or not?
        """
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

    def local_search(self):
        step = 1
        moves = filter(lambda point: not all(axis ==0 for axis in point), list(product([-step, 0, step], repeat=3)))
        return moves

    def generate_rotation(self, coords):
        """
            Rotate a list of 3D [x,y,z] vectors about corresponding random uniformly
            distributed quaternion [w, x, y, z]
        
            Parameters
            ----------
            coords : numpy.ndarray with shape [n,3]
                list of [x,y,z] cartesian vector coordinates
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
    
    ##################### SANITY CHECKS ####################
    def calculate_mol_volume(self, mol_positions):
        """Computes volume occupied by the receptor in nm**3"""
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

    def fitting_checks(self):
        prot_volume = 0
        if self.receptor:
            prot_volume = self.calculate_mol_volume(self.modeller.positions)
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

    def liters_to_cubic_nanometers(self, liters):
        if isinstance(liters, openmmunit.Quantity):
            liters = liters.value_in_unit(openmmunit.liters)
        value = liters * 1e+24
        return value

    def cubic_nanometers_to_liters(self, vol):
        value = vol * 1e-24
        return value
    ##################### SANITY CHECKS ####################

    ##################### FILL THE VOID APPROACH ####################
                
    def _setup_new_topology(self, cosolvents_positions, receptor_topology=None, receptor_positions=None):
        # Adding the cosolvent molecules
        molecules = []
        molecules_positions = []
        for cosolvent in cosolvents_positions:
            for i in range(len(cosolvents_positions[cosolvent])):
                molecules.append(Molecule.from_smiles(cosolvent.smiles, name=cosolvent.name))
                [molecules_positions.append(x) for x in cosolvents_positions[cosolvent][i]]

        molecules_positions = np.array(molecules_positions)
        new_top = Topology.from_molecules(molecules)
        new_mod = app.Modeller(new_top.to_openmm(), molecules_positions)
        if receptor_topology is not None and receptor_positions is not None and len(receptor_positions) > 0: 
            new_mod.add(receptor_topology, receptor_positions)
        new_mod.topology.setPeriodicBoxVectors(self._periodic_box_vectors)
        return new_mod

    ##################### FORCEFIELD PARAMETRIZATION ####################
    def _parametrize_system(self, forcefields: str, engine: str, cosolvents: dict):
        with open(forcefields) as fi:
            ffs = json.load(fi)
        engine = engine.upper()
        forcefield = app.ForceField(*ffs[engine])
        sm_ff = ffs["small_molecules"][0]
        small_molecule_ff = self._parametrize_cosolvents(cosolvents, small_molecule_ff=sm_ff)
        forcefield.registerTemplateGenerator(small_molecule_ff.generator)
        return forcefield

    def _parametrize_cosolvents(self, cosolvents, small_molecule_ff="espaloma"):
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
    ##################### FORCEFIELD PARAMETRIZATION ####################

    def _create_system(self, forcefield, topology):
        print("Creating system")
        system = forcefield.createSystem(topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=10*openmmunit.angstrom,
                                         constraints=app.HBonds,
                                         hydrogenMass=1.5*openmmunit.amu)
        return system 
    
    ##################### BOX ####################
    def _build_box(self, positions, padding, radius=None):
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
        self._box_origin = center - (np.ceil(np.array((vectors[0][0], vectors[1][1], vectors[2][2]))))
        self._box_size = np.ceil(np.array([maxRange[0]-minRange[0],
                                           maxRange[1]-minRange[1],
                                           maxRange[2]-minRange[2]])).astype(int)
        lowerBound = center-box/2
        upperBound = center+box/2
        return vectors, box, lowerBound, upperBound 
    
    # def _build_mesh(self, positions):
    #     # For water this is translated in 30.9
    #     padding = 1.3*openmmunit.angstrom
    #     offset = 1.5*openmmunit.angstrom
    #     mesh_step = 0.3*openmmunit.angstrom
    #     minRange = np.array([min((pos[i] for pos in positions)) for i in range(3)])
    #     maxRange = np.array([max((pos[i] for pos in positions)) for i in range(3)])
    #     # padding = padding.value_in_unit(openmmunit.nanometer)
    #     # mesh_step = mesh_step.value_in_unit(openmmunit.nanometer)
    #     x = np.arange(minRange[0], maxRange[0]+padding, mesh_step)
    #     y = np.arange(minRange[1], maxRange[1]+padding, mesh_step)
    #     z = np.arange(minRange[2], maxRange[2]+padding, mesh_step)
    #     X, Y, Z = np.meshgrid(x, y, z)
    #     center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    #     kdtree = spatial.cKDTree(center_xyzs)
    #     # offset = offset.value_in_unit(openmmunit.nanometer)
    #     query = kdtree.query_ball_point(positions, offset)
    #     query = np.unique(np.hstack(query)).astype(int)
    #     close_to = np.zeros(len(center_xyzs), bool)
    #     close_to[query] = True
    #     return round(np.count_nonzero(close_to)*mesh_step, 2)