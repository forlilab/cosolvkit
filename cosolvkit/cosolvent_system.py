import json
import os
from collections import defaultdict
import numpy as np
from scipy import spatial
from math import floor
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
            self._box = self._build_box(None, padding, radius=radius)
            self.modeller = app.Modeller(app.Topology(), None)
        else:
            self._box = self._build_box(self.modeller.positions, padding, radius=None)
        
        # Setting up the box - This has to be done before building the system with
        # the cosolvent molecules.
        self.modeller.topology.setPeriodicBoxVectors(self._box[0])
        self._periodic_box_vectors = self.modeller.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)
        vX, vY, vZ = self.modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.angstrom)
        self._box_volume = vX * vY * vZ
        print("Parametrizing system with forcefields")
        self.forcefield = self._parametrize_system(forcefields, simulation_engine, self.cosolvents)
        # print(f"Number of possible added waters with mesh approach: {self._number_of_possible_added_waters_mesh()}")
        print("Adding cosolvents and hydrating")
        # self.modeller.addSolvent(self.forcefield)
        # self._added_waters = self._get_n_waters()
        # self._receptor_xyzs, self._water_xyzs, self._wat_res_mapping = self._process_positions(self.modeller)
        # print(f"Box Volume: {self._box_volume} A**3")
        # print(f"Number of waters added: {self._added_waters}")
        return
    
    ############################# PUBLIC
    def build(self):
        # cosolv_xyz = self._add_cosolvents(self.cosolvents)
        self._add_cosolvents_fill_the_void(self.cosolvents,
                                           False)
        # self.system = self._create_system(self.forcefield, self.modeller.topology)
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
    def _check_cosolvent_occupancy(self):
        water_concentration = 55.4 * openmmunit.molar
        total_volume = 0
        cosolvs_volume = defaultdict(float)
        for cosolvent in self.cosolvents:
            if cosolvent.concentration is not None:
                n_copies = (self._added_waters * (cosolvent.concentration * openmmunit.molar)) / water_concentration
                cosolvent.copies = int(floor(n_copies + 0.5))
            cosolvs_volume[cosolvent] = self._build_mesh(self.cosolvents[cosolvent]) * cosolvent.copies
            total_volume += cosolvs_volume[cosolvent]
        wat = CoSolvent("wat", "O")
        cosolvs_volume["HOH"] = self._build_mesh(wat.positions) * self._added_waters
        # cosolvs_volume["HOH"] = 30.9 * self._added_waters
        assert total_volume <= cosolvs_volume["HOH"]/2.0, \
            "Error! The volume occupied by the requested cosolvent molecules exceeds the volume limit of 50% of the solvent\n" + \
            f"Volume requested for cosolvents: {total_volume}\n" + \
            f"Volume available for cosolvents: {cosolvs_volume['HOH']/2.}\n" + \
            f"Total Volume available: {cosolvs_volume['HOH']}"
        
        print(f"Volume requested for cosolvents: {total_volume}")
        print(f"Volume available for cosolvents: {cosolvs_volume['HOH']/2.}")
        print(f"Total Volume available: {cosolvs_volume['HOH']}")
        return cosolvs_volume

    
    def _get_n_waters(self):
        res = [r.name for r in self.modeller.topology.residues()]
        return res.count('HOH')
    
    def _number_of_possible_added_waters_mesh(self):
        # wat = CoSolvent("wat", "O")
        # wat_xyzs = wat.positions*openmmunit.angstrom
        # sizeX, sizeY, sizeZ = wat_xyzs.max(axis=0) - wat_xyzs.min(axis=0)
        watref_dims = [18.856, 18.856, 18.856]*openmmunit.angstrom
        watref_dims = watref_dims.value_in_unit(openmmunit.angstrom)
        vX, vY, vZ = self.modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.angstrom)
        positions = self.modeller.positions.value_in_unit(openmmunit.angstrom)
        if len(positions) > 0:
            center = [(max((pos[i] for pos in positions))+min((pos[i] for pos in positions)))/2 for i in range(3)]
        else:
            center = Vec3(0, 0, 0)
        origin = center - (np.ceil(np.array([vX, vY, vZ])).astype(int)/2)
        xmin, xmax = origin[0], origin[0] + vX
        ymin, ymax = origin[1], origin[1] + vY
        zmin, zmax = origin[2], origin[2] + vZ

        # sizeX = sizeX.value_in_unit(openmmunit.nanometer)
        # sizeY = sizeY.value_in_unit(openmmunit.nanometer)
        # sizeZ = sizeZ.value_in_unit(openmmunit.nanometer)
        # x = np.arange(xmin, xmax, sizeX) + (sizeX/2.)
        # y = np.arange(ymin, ymax, sizeY) + (sizeY/2.)
        # z = np.arange(zmin, zmax, sizeZ) + (sizeZ/2.)
        x = np.arange(xmin, xmax, watref_dims[0]) + (watref_dims[0]/2.)
        y = np.arange(ymin, ymax, watref_dims[1]) + (watref_dims[1]/2.)
        z = np.arange(zmin, zmax, watref_dims[2]) + (watref_dims[2]/2.)

        X, Y, Z = np.meshgrid(x, y, z)
        center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        print(len(center_xyzs))
        return center_xyzs

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
    
    def _add_cosolvents_fill_the_void(self, cosolvents, hydrate=True):
        raw_positions = list()
        cosolvent_copies = dict()
        cosolvent_kdtree = None
        kdtree = None
        cosolv_xyzs = defaultdict(list)
        if self.receptor:
            kdtree = spatial.cKDTree(self.modeller.positions.value_in_unit(openmmunit.nanometer))
        for cosolvent in cosolvents:
            cosolvent_copies[cosolvent] = 0
            cosolvent_xyz = cosolvents[cosolvent]
            vecs = self.modeller.topology.getPeriodicBoxVectors().value_in_unit(openmmunit.nanometer)
            x = np.arange(0, vecs[0][0], 1)
            y = np.arange(0, vecs[1][1], 1)
            z = np.arange(0, vecs[2][2], 1)
            X, Y, Z = np.meshgrid(x, y, z)
            center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
            # sizeX, sizeY, sizeZ = cosolvent_xyz.max(axis=0) - cosolvent_xyz.min(axis=0)
            # center_xyzs = self._build_mesh(self.modeller, sizeX, sizeY, sizeZ, cutoff=self._cosolvents_cutoff)
            # Randomize the cosolvent molecules placement
            np.random.shuffle(center_xyzs)
            if len(cosolv_xyzs) > 0:
                cosolvent_kdtree = spatial.cKDTree(raw_positions)
            for i in range(len(center_xyzs)):
                if cosolvent_copies[cosolvent] < cosolvent.copies:
                    new_coords = cosolvent_xyz + center_xyzs[i]
                    if self._check_coordinates_to_add(new_coords,
                                                      cosolvent_kdtree,
                                                      kdtree):
                            cosolv_xyzs[cosolvent].append(new_coords)
                            cosolvent_copies[cosolvent] += 1
                            [raw_positions.append(pos) for pos in new_coords]
        self.modeller = self._setup_new_topology(cosolv_xyzs, self.modeller.topology, self.modeller.positions)
        if hydrate:
            if self.n_waters is not None:
                self.modeller.addSolvent(self.forcefield, numAdded=self.n_waters)
            else: self.modeller.addSolvent(self.forcefield)
        return self.modeller
    
    def _check_coordinates_to_add(self, new_coords, cosolvent_kdtree, kdtree):
            cosolvents_cutoff = 3.5*openmmunit.angstrom
            receptor_cutoff = 4.5*openmmunit.angstrom
            if kdtree is not None and not any(kdtree.query_ball_point(new_coords, receptor_cutoff.value_in_unit(openmmunit.nanometer))):
                if cosolvent_kdtree is not None:
                    if not any(cosolvent_kdtree.query_ball_point(new_coords, cosolvents_cutoff.value_in_unit(openmmunit.nanometer))):
                        return True
                    else: return False
                else:
                    return True
            else:
                if cosolvent_kdtree is not None:
                    if not any(cosolvent_kdtree.query_ball_point(new_coords, cosolvents_cutoff.value_in_unit(openmmunit.nanometer))):
                        return True
                    else: return False
                else:
                    return True

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
    
    def _create_system(self, forcefield, topology):
        print("Creating system")
        system = forcefield.createSystem(topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=10*openmmunit.angstrom,
                                         constraints=app.HBonds,
                                         hydrogenMass=1.5*openmmunit.amu)
        return system 
    
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
        return vectors, box
    
    # def _build_mesh(self, modeller, sizeX, sizeY, sizeZ, cutoff, water=False):
    #     vX, vY, vZ = modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.nanometer)
    #     positions = modeller.positions.value_in_unit(openmmunit.nanometer)
    #     if len(positions) > 0:
    #         center = [(max((pos[i] for pos in positions))+min((pos[i] for pos in positions)))/2 for i in range(3)]
    #     else:
    #         center = Vec3(0, 0, 0)
    #     origin = center - (np.ceil(np.array([vX, vY, vZ])).astype(int)/2)
    #     xmin, xmax = origin[0], origin[0] + vX
    #     ymin, ymax = origin[1], origin[1] + vY
    #     zmin, zmax = origin[2], origin[2] + vZ

    #     cutoff = cutoff.value_in_unit(openmmunit.nanometer)
    #     if not water:
    #         x = np.arange(xmin, xmax, sizeX+cutoff) + cutoff
    #         y = np.arange(ymin, ymax, sizeY+cutoff) + cutoff
    #         z = np.arange(zmin, zmax, sizeZ+cutoff) + cutoff
    #     else:
    #         sizeX = sizeX.value_in_unit(openmmunit.nanometer)
    #         sizeY = sizeY.value_in_unit(openmmunit.nanometer)
    #         sizeZ = sizeZ.value_in_unit(openmmunit.nanometer)
    #         x = np.arange(xmin, xmax, sizeX) + (sizeX/2.)
    #         y = np.arange(ymin, ymax, sizeY) + (sizeY/2.)
    #         z = np.arange(zmin, zmax, sizeZ) + (sizeZ/2.)

    #     X, Y, Z = np.meshgrid(x, y, z)
    #     center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    #     return center_xyzs
    
    def _build_mesh(self, positions):
        # For water this is translated in 30.9
        padding = 1.3*openmmunit.angstrom
        offset = 1.5*openmmunit.angstrom
        mesh_step = 0.3*openmmunit.angstrom
        minRange = np.array([min((pos[i] for pos in positions)) for i in range(3)])
        maxRange = np.array([max((pos[i] for pos in positions)) for i in range(3)])
        # padding = padding.value_in_unit(openmmunit.nanometer)
        # mesh_step = mesh_step.value_in_unit(openmmunit.nanometer)
        x = np.arange(minRange[0], maxRange[0]+padding, mesh_step)
        y = np.arange(minRange[1], maxRange[1]+padding, mesh_step)
        z = np.arange(minRange[2], maxRange[2]+padding, mesh_step)
        X, Y, Z = np.meshgrid(x, y, z)
        center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        kdtree = spatial.cKDTree(center_xyzs)
        # offset = offset.value_in_unit(openmmunit.nanometer)
        query = kdtree.query_ball_point(positions, offset)
        query = np.unique(np.hstack(query)).astype(int)
        close_to = np.zeros(len(center_xyzs), bool)
        close_to[query] = True
        return round(np.count_nonzero(close_to)*mesh_step, 2)



        
