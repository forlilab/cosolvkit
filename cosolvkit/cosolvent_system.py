import openmm.app as app
import openmm.unit as openmmunit
import openmm
import json
from copy import deepcopy
from collections import defaultdict
from math import ceil, floor, sqrt
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, Topology
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from cosolvkit.utils import fix_pdb

class _CellList(object):
    """This class organizes atom positions into cells, so the neighbors of a point can be quickly retrieved"""

    def __init__(self, positions, maxCutoff, vectors, periodic):
        self.positions = deepcopy(positions)
        self.cells = {}
        self.numCells = tuple((max(1, int(floor(vectors[i][i]/maxCutoff))) for i in range(3)))
        self.cellSize = tuple((vectors[i][i]/self.numCells[i] for i in range(3)))
        self.vectors = vectors
        self.periodic = periodic
        invBox = openmm.Vec3(1.0/vectors[0][0], 1.0/vectors[1][1], 1.0/vectors[2][2])
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
            invBox = openmm.Vec3(1.0/self.vectors[0][0], 1.0/self.vectors[1][1], 1.0/self.vectors[2][2])
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
                    cell = self.cellForPosition(openmm.Vec3(pos[0]+i*self.cellSize[0], pos[1]+j*self.cellSize[1], pos[2]+k*self.cellSize[2]))
                    if cell in self.cells and cell not in processedCells:
                        processedCells.add(cell)
                        for atom in self.cells[cell]:
                            yield atom


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

class CosolventSystem(app.Modeller):
    def __init__(self,
                 cosolvents_path: str, 
                 forcefields_path: str,
                 simulation_engine: str,
                 receptor_path: str = None,
                 n_waters: int = None,
                 padding: openmmunit.Quantity = 12*openmmunit.angstrom,
                 radius: openmmunit.Quantity = None):
        
        if receptor_path is not None:
            self.topology, self.positions = fix_pdb(receptor_path)
        else:
            self.topology, self.positions = app.Topology(), list()
        super(app.Modeller, self).__init__(self.topology, self.positions)

        self._available_engines = ["AMBER", "GROMACS", "CHARMM"]
        self._receptor_cutoff = 4.5*openmmunit.angstrom
        self._cosolvent_cutoff = 3.5*openmmunit.angstrom
        self._edge_cutoff = 3.0*openmmunit.angstrom
        self._box = None
        self._periodic_box_vecotrs = None
        self._box_volume = None
        
        self.n_waters = n_waters
        self.modeller = None
        self.system = None
        self.cosolvents = defaultdict(list)

        assert (simulation_engine.upper() in self._available_engines), "Error! The specified simulation engine is not supported."
        with open(cosolvents_path) as fi:
            cosolvents_dict = json.load(fi)
        for c in cosolvents_dict:
            cosolvent = CoSolvent(**c)
            cosolvent_xyz = cosolvent.positions*openmmunit.angstrom
            self.cosolvents[cosolvent] = cosolvent_xyz.value_in_unit(openmmunit.nanometer)
        
        self._periodic_vectors, self._box = self._build_simulation_box(self.positions, padding, radius)
        self.topology.setPeriodicBoxVectors(self._periodic_box_vecotrs)

        self._cosolvents_periodic_vectors = self._build_box_for_cosolvent(self.cosolvents, self._cosolvent_cutoff)
        self._protein_periodic_vectors = self._build_box_for_protein(self.positions, self._receptor_cutoff)
        self._protein_cells = None
        if len(self.positions) > 0:
            self._protein_cells = _CellList(self.positions, self._receptor_cutoff, self._protein_periodic_vectors, True)
        self.periodic_distance_fn = app.internal.compiled.periodicDistance(self._protein_periodic_vectors)

        # Forcefields
        self.forcefield = self._parametrize_system(forcefields_path, simulation_engine, self.cosolvents)

        # Hydrate
        self.modeller = app.Modeller(self.topology, self.positions)
        self.modeller.addSolvent(self.forcefield)

        # Now need to find the replaceable waters (distant enough from the receptor)
        

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


    def _build_simulation_box(self, positions, padding, radius=None):
        padding = padding.value_in_unit(openmmunit.nanometer)
        if positions is not None:
            positions = positions.value_in_unit(openmmunit.nanometer)
            minRange = openmm.Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
            maxRange = openmm.Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
            center = 0.5*(minRange+maxRange)
            radius = max(openmm.unit.norm(center-pos) for pos in positions)
        else:
            radius = radius.value_in_unit(openmmunit.nanometer)
        width = max(2*radius+padding, 2*padding)
        vectors = (openmm.Vec3(width, 0, 0), 
                   openmm.Vec3(0, width, 0), 
                   openmm.Vec3(0, 0, width))
        box = openmm.Vec3(vectors[0][0], vectors[1][1], vectors[2][2])
        return vectors.value_in_unit(openmmunit.nanometer), box

    def _build_box_for_cosolvent(self, cosolvents, cutoff):
        cutoff = cutoff.value_in_unit(openmmunit.nanometer)
        cosolvents_vectors = defaultdict(tuple[openmm.Vec3, openmm.Vec3, openmm.Vec3])
        for cosolvent in cosolvents:
            positions = cosolvent.positions * openmmunit.angstrom
            positions = positions.value_in_unit(openmmunit.nanometer)
            minRange = openmm.Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
            maxRange = openmm.Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
            center = 0.5*(minRange+maxRange)
            radius = max(openmm.unit.norm(center-pos) for pos in positions)
            width = max(2*radius+cutoff, 2*cutoff)
            vectors = (openmm.Vec3(width, 0, 0), 
                    openmm.Vec3(0, width, 0), 
                    openmm.Vec3(0, 0, width))
            cosolvents_vectors[cosolvent] = vectors.value_in_unit(openmmunit.nanometer)
        return cosolvents_vectors

    def _build_box_for_protein(self, protein_positions, cutoff):
        cutoff = cutoff.value_in_unit(openmmunit.nanometer)
        positions = protein_positions.value_in_unit(openmmunit.nanometer)
        minRange = openmm.Vec3(*(min((pos[i] for pos in positions)) for i in range(3)))
        maxRange = openmm.Vec3(*(max((pos[i] for pos in positions)) for i in range(3)))
        center = 0.5*(minRange+maxRange)
        radius = max(openmm.unit.norm(center-pos) for pos in positions)
        width = max(2*radius+cutoff, 2*cutoff)
        vectors = (openmm.Vec3(width, 0, 0), 
                   openmm.Vec3(0, width, 0), 
                   openmm.Vec3(0, 0, width))
        return vectors

    def _test_distance(self, cellsList):
        pass