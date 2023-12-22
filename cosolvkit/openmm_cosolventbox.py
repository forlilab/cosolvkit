import json
import numpy as np
from copy import deepcopy
from math import ceil, floor, sqrt
from collections import defaultdict
import openmm.app as app
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from openmm import *
import openmm.unit as openmmunit
from openff.toolkit import Molecule, Topology
import parmed
from scipy import spatial
from rdkit import Chem
from rdkit.Chem import AllChem
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
        self._kdtree = None
        self._receptor_cutoff = 4.5*openmmunit.angstrom
        self._cosolvents_cutoff = 3.5*openmmunit.angstrom
        self._edge_cutoff = 3.0*openmmunit.angstrom
        
        # We refer to the following paper for the volume of 1 water molecule
        # https://doi.org/10.1063/1.1676585
        # https://web.ics.purdue.edu/~gchopra/class/public/readings/Molecular_Simulation_I_Lecture4/Rahman_Stillinger_JCP_71_Water_Dynamics.pdf
        self._water_dims = [18.856*openmmunit.angstrom, 18.856*openmmunit.angstrom, 18.856*openmmunit.angstrom]
        self._water_molecule_volume = (18.856 * 18.856 * 18.856) / 216

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
            self.cosolvents[cosolvent] = cosolvent_xyz.value_in_unit(openmmunit.nanometer)

        if receptor is not None:
            print("Cleaning protein")
            top, pos = fix_pdb(receptor)
            self.modeller = app.Modeller(top, pos)
            self._kdtree = spatial.cKDTree(self.modeller.positions.value_in_unit(openmmunit.nanometer))
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
        if self.modeller.positions is not None:
            self.cells = _CellList(self.modeller.positions, 2.5, self._periodic_box_vectors, True)
        else:
            self.cells = None
        print("Adding cosolvents and hydrating")
        self.modeller.addSolvent(self.forcefield)
        # self.modeller = self.build(self.cosolvents, self._cosolvent_positions, self._hydrate, self.forcefield)
        # self.system = self.create_system(self.forcefield, self.modeller.topology)
        self._added_waters, self._waters_positions, self._receptor_positions = self._get_waters(self.modeller.positions)
        print(f"Box Volume: {self._box_volume} A**3")
        print(f"Number of waters added: {self._added_waters}")
        return
    
    def _get_waters(self, positions):
        cnt = 0
        water_pos = {}
        residues = list(self.modeller.topology.residues())
        for i in range(len(residues)):
            res = residues[i]
            if res.name == "HOH":
                cnt+=1
        return cnt, 
    
    def _add_cosolvents(self, cosolvents):
        for cosolvent in cosolvents:
            cosolvent_xyz = cosolvents[cosolvent]
            sizeX, sizeY, sizeZ = cosolvent_xyz.max(axis=0) - cosolvent_xyz.min(axis=0)

    def _parametrize_system(self, forcefields: str, engine: str, cosolvents: dict):
        with open(forcefields) as fi:
            ffs = json.load(fi)
        engine = engine.upper()
        forcefield = app.ForceField(*ffs[engine])
        sm_ff = ffs["small_molecules"]
        small_molecule_ff = self._parametrize_cosolvents(cosolvents, small_molecule_ff=sm_ff)
        forcefield.registerTemplateGenerator(small_molecule_ff.generator)
        return forcefield

    def _parametrize_cosolvents(self, cosolvents, small_molecule_ff="espaloma"):
        molecules = list()
        for cosolvent in cosolvents:
            try:
                molecules.append(Molecule.from_smiles(cosolvent.smiles))
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

    def build(self, cosolvents, cosolvent_positions, hydrate=True, forcefield=None):
        raw_positions = list()
        cosolvent_copies = dict()
        cosolvent_kdtree = None
        for cosolvent in cosolvents:
            # print(cosolvent.name)
            cosolvent_copies[cosolvent] = 0
            cosolvent_xyz = cosolvents[cosolvent]
            sizeX, sizeY, sizeZ = cosolvent_xyz.max(axis=0) - cosolvent_xyz.min(axis=0)
            center_xyzs = self._build_mesh(self.modeller, sizeX, sizeY, sizeZ, cutoff=self._cosolvents_cutoff)
            # Randomize the cosolvent molecules placement
            np.random.shuffle(center_xyzs)
            if len(cosolvent_positions) > 0:
                cosolvent_kdtree = spatial.cKDTree(raw_positions)
            for i in range(len(center_xyzs)):
                if cosolvent_copies[cosolvent] < cosolvent.copies:
                    new_coords = cosolvent_xyz + center_xyzs[i]
                    if self._check_coordinates_to_add(new_coords,
                                                      cosolvent_kdtree,
                                                      self._kdtree):
                            cosolvent_positions[cosolvent].append(new_coords)
                            cosolvent_copies[cosolvent] += 1
                            [raw_positions.append(pos) for pos in new_coords]
        modeller = self._setup_new_topology(cosolvent_positions, self.modeller.topology, self.modeller.positions)
        if hydrate:
            if self.n_waters is not None:
                modeller.addSolvent(forcefield, numAdded=self.n_waters)
            else: modeller.addSolvent(forcefield)
        return modeller
    
    def _build(self):

        pass
    
    def create_system(self, forcefield, topology):
        print("Creating system")
        system = forcefield.createSystem(topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=10*openmmunit.angstrom,
                                         constraints=app.HBonds,
                                         hydrogenMass=1.5*openmmunit.amu)
        return system
    
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
    
    def _check_coordinates_to_add(self, new_coords, cosolvent_kdtree, kdtree):
        if kdtree is not None and not any(kdtree.query_ball_point(new_coords, self._receptor_cutoff.value_in_unit(openmmunit.nanometer))):
            if cosolvent_kdtree is not None:
                if not any(cosolvent_kdtree.query_ball_point(new_coords, self._cosolvents_cutoff.value_in_unit(openmmunit.nanometer))):
                    return True
                else: return False
            else:
                return True
        else:
            if cosolvent_kdtree is not None:
                if not any(cosolvent_kdtree.query_ball_point(new_coords, self._cosolvents_cutoff.value_in_unit(openmmunit.nanometer))):
                    return True
                else: return False
            else:
                return True
    
    def _setup_new_topology(self, cosolvents_positions, receptor_topology=None, receptor_positions=None):
        # Adding the cosolvent molecules
        molecules = []
        molecules_positions = []
        for cosolvent in cosolvents_positions:
            for i in range(len(cosolvents_positions[cosolvent])):
                molecules.append(Molecule.from_smiles(cosolvent.smiles))
                [molecules_positions.append(x) for x in cosolvents_positions[cosolvent][i]]

        molecules_positions = np.array(molecules_positions)
        new_top = Topology.from_molecules(molecules)
        new_mod = app.Modeller(new_top.to_openmm(), molecules_positions)
        if receptor_topology is not None and receptor_positions is not None: 
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
            radius = radius.value_in_unit(openmmunit.nanometer)
        width = max(2*radius+padding, 2*padding)
        vectors = (Vec3(width, 0, 0), Vec3(0, width, 0), Vec3(0, 0, width))
        box = Vec3(vectors[0][0], vectors[1][1], vectors[2][2])
        return vectors, box
    
    def _build_mesh(self, modeller, sizeX, sizeY, sizeZ, cutoff, water=False):
        vX, vY, vZ = modeller.topology.getUnitCellDimensions().value_in_unit(openmmunit.nanometer)
        positions = modeller.positions.value_in_unit(openmmunit.nanometer)
        if len(positions) > 0:
            center = [(max((pos[i] for pos in positions))+min((pos[i] for pos in positions)))/2 for i in range(3)]
        else:
            center = Vec3(0, 0, 0)
        origin = center - (np.ceil(np.array([vX, vY, vZ])).astype(int)/2)
        xmin, xmax = origin[0], origin[0] + vX
        ymin, ymax = origin[1], origin[1] + vY
        zmin, zmax = origin[2], origin[2] + vZ

        cutoff = cutoff.value_in_unit(openmmunit.nanometer)
        if not water:
            x = np.arange(xmin, xmax, sizeX+cutoff) + cutoff
            y = np.arange(ymin, ymax, sizeY+cutoff) + cutoff
            z = np.arange(zmin, zmax, sizeZ+cutoff) + cutoff
        else:
            sizeX = sizeX.value_in_unit(openmmunit.nanometer)
            sizeY = sizeY.value_in_unit(openmmunit.nanometer)
            sizeZ = sizeZ.value_in_unit(openmmunit.nanometer)
            x = np.arange(xmin, xmax, sizeX) + (sizeX/2.)
            y = np.arange(ymin, ymax, sizeY) + (sizeY/2.)
            z = np.arange(zmin, zmax, sizeZ) + (sizeZ/2.)

        X, Y, Z = np.meshgrid(x, y, z)
        center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
        
        # if water:
        #     wat_xyzs = []
        #     if positions is not None:
        #         for i in center_xyzs:
        #             wat_xyzs.append(i)
        #         too_close_receptor = self._water_close_to_receptor(np.vstack(wat_xyzs), positions, distance=2.5*openmmunit.angstrom)
        #         return center_xyzs[~too_close_receptor]
        return center_xyzs
    
    def _water_close_to_receptor(self, wat_xyzs, poisitions, distance):
        distance = distance.value_in_unit(openmmunit.nanometer)
        kdtree = spatial.cKDTree(wat_xyzs)
        ids = kdtree.query_ball_point(poisitions, distance)
        ids = np.unique(np.hstack(ids)).astype(int)
        close_to = np.zeros(len(wat_xyzs), bool)
        close_to[ids] = True
        for i in range(0, wat_xyzs.shape[0], 3):
            close_to[[i, i+1, i+2]] = [np.all(close_to[[i, i+1, i+2]])] * 3
        return close_to






class _CellList(object):
    """This class organizes atom positions into cells, so the neighbors of a point can be quickly retrieved"""

    def __init__(self, positions, maxCutoff, vectors, periodic):
        self.positions = deepcopy(positions)
        self.cells = {}
        self.numCells = tuple((max(1, int(floor(vectors[i][i]/maxCutoff))) for i in range(3)))
        self.cellSize = tuple((vectors[i][i]/self.numCells[i] for i in range(3)))
        self.vectors = vectors
        self.periodic = periodic
        invBox = Vec3(1.0/vectors[0][0], 1.0/vectors[1][1], 1.0/vectors[2][2])
        for i in range(len(self.positions)):
            pos = self.positions[i]
            if periodic:
                pos = pos - floor(pos[2]*invBox[2])*vectors[2]
                pos -= floor(pos[1]*invBox[1])*vectors[1]
                pos -= floor(pos[0]*invBox[0])*vectors[0]
                self.positions[i] = pos
            cell = self.cellForPosition(pos)
            if cell in self.cells:
                self.cells[cell].append(i)
            else:
                self.cells[cell] = [i]

    def cellForPosition(self, pos):
        if self.periodic:
            invBox = Vec3(1.0/self.vectors[0][0], 1.0/self.vectors[1][1], 1.0/self.vectors[2][2])
            pos = pos-floor(pos[2]*invBox[2])*self.vectors[2]
            pos -= floor(pos[1]*invBox[1])*self.vectors[1]
            pos -= floor(pos[0]*invBox[0])*self.vectors[0]
        return tuple((int(floor(pos[j]/self.cellSize[j]))%self.numCells[j] for j in range(3)))

    def neighbors(self, pos):
        processedCells = set()
        offsets = (-1, 0, 1)
        for i in offsets:
            for j in offsets:
                for k in offsets:
                    cell = self.cellForPosition(Vec3(pos[0]+i*self.cellSize[0], pos[1]+j*self.cellSize[1], pos[2]+k*self.cellSize[2]))
                    if cell in self.cells and cell not in processedCells:
                        processedCells.add(cell)
                        for atom in self.cells[cell]:
                            yield atom