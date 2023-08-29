#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to create the Cosolvent
#

import os
import shutil
import sys
from io import StringIO

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pdb4amber.residue import AMBER_SUPPORTED_RESNAMES

from . import utils


TLEAP_TEMPLATE = """
source leaprc.protein.ff14SB
source leaprc.DNA.bsc1
source leaprc.water.tip3p
source leaprc.%(gaff_version)s
loadamberparams out.frcmod
%(resname)s = loadmol2 out.mol2
check %(resname)s
saveoff %(resname)s out.lib
quit
"""


def _generate_atom_names_from_mol(rdkit_mol):
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


def _transfer_coordinates_from_pdb_to_mol2(pdb_filename, mol2_filename, new_mol2_filename=None):
    """ Transfer coordinates from pdb to mol2 filename. 
    I neither trust RDKit or OpenBabel for doing that...
    """
    i = 0
    pdb_coordinates = []
    output_mol2 = ""
    coordinate_flag = False

    # Get all the coordinates from the pdb
    with open(pdb_filename) as f:
        lines = f.readlines()

        for line in lines:
            x = line[31:39].strip()
            y = line[39:47].strip()
            z = line[47:54].strip()
            pdb_coordinates.append([x, y, z])

    pdb_coordinates = np.array(pdb_coordinates, dtype=float)

    # We read the mol2 file and modify each atom line
    with open(mol2_filename) as f:
        lines = f.readlines()

        for line in lines:
            # Stop reading coordinates
            # '@<TRIPOS>SUBSTRUCTURE' in case there is only one atom...
            # But who would do this?!
            if '@<TRIPOS>BOND' in line or '@<TRIPOS>SUBSTRUCTURE' in line:
                coordinate_flag = False

            if coordinate_flag:
                x, y, z = pdb_coordinates[i]
                new_line = line[0:17] + "%10.4f %10.4f %10.4f " % (x, y, z) + line[50:]
                output_mol2 += new_line
                i += 1
            else:
                output_mol2 += line

            # It's time to read the coordinates
            if '@<TRIPOS>ATOM' in line:
                coordinate_flag = True

    # Write the new mol2 file
    if new_mol2_filename is None:
        with open(mol2_filename, 'w') as w:
            w.write(output_mol2)
    else:
        with open(new_mol2_filename, 'w') as w:
            w.write(output_mol2)


def _run_antechamber(mol_filename, molecule_name, resname, charge=0, charge_method="bcc", gaff_version="gaff2"):
    """Run antechamber.
    """
    original_mol_filename = os.path.abspath(mol_filename)
    local_mol_filename = os.path.basename(mol_filename)
    cwd_dir = os.path.dirname(original_mol_filename)
    output_mol2_filename = cwd_dir + os.path.sep + '%s.mol2' % molecule_name
    output_frcmod_filename = cwd_dir + os.path.sep + '%s.frcmod' % molecule_name
    output_lib_filename = cwd_dir + os.path.sep + '%s.lib' % molecule_name

    with utils.temporary_directory(prefix=molecule_name, dir='.') as tmp_dir:
        shutil.copy2(original_mol_filename, local_mol_filename)

        # Run Antechamber
        # We output a mol2 (and not a mol file) because the mol2 contains the GAFF atom types (instead of SYBYL)
        cmd = "antechamber -i %s -fi mdl -o out.mol2 -fo mol2 -s 2 -at %s -c %s -nc %d -rn %s"
        cmd = cmd % (local_mol_filename, gaff_version, charge_method, charge, resname)
        ante_outputs, ante_errors = utils.execute_command(cmd)

        # Run parmchk2 for the additional force field file
        cmd = 'parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s %s' 
        cmd = cmd % (gaff_version)
        parm_outputs, parm_errors = utils.execute_command(cmd)

        # Run tleap for the library file
        with open('tleap.cmd', 'w') as w:
            w.write(TLEAP_TEMPLATE % {'gaff_version': gaff_version, 
                                      'molecule_name': molecule_name,
                                      'resname': resname
                                      })
        cmd = 'tleap -s -f tleap.cmd'
        tleap_outputs, tleap_errors = utils.execute_command(cmd)

        try:
            # The final mol2 file from antechamber does not contain
            # the optimized geometry from sqm. Why?!
            _transfer_coordinates_from_pdb_to_mol2('sqm.pdb', 'out.mol2')
        except FileNotFoundError:
            # Antechamber is the primary source of error due to weird atomic valence (mol2 BONDS or charge)
            print("ERROR: Parametrization of %s failed. Check atomic valence, bonds and charge." % molecule_name)
            print("ANTECHAMBER ERROR LOG: ")
            print(ante_errors)
            print("PARMCHK2 ERROR LOG: ")
            print(parm_errors)
            print("TLEAP ERROR LOG: ")
            print(tleap_errors)
            sys.exit(1)

        # Copy back all we need
        shutil.copy('out.mol2', output_mol2_filename)
        shutil.copy('out.frcmod', output_frcmod_filename)
        shutil.copy('out.lib', output_lib_filename)

    return output_mol2_filename, output_frcmod_filename, output_lib_filename


class CoSolvent:

    def __init__(self, name, smiles=None, mol_filename=None, resname=None):
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

        assert not self.resname in AMBER_SUPPORTED_RESNAMES, f"Error: the residue name {self.resname} is already defined in AMBER."
        assert (smiles is not None or mol_filename is not None), "A smiles or a mol/sdf filename has to be defined."
        if mol_filename is not None:
            assert os.path.exists(mol_filename), f"The mol/sdf file {mol_filename} does not exist."

        self.mol = None
        self.atom_names = None
        self.name = name
        self.charge = None
        self.positions = None
        self.mol_filename = None

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
        self.atom_names = _generate_atom_names_from_mol(mol)
        # We make sure the cosolvent is centered, especially if it was provided by the user
        # And honestly, never trust the user!
        positions = mol.GetConformer().GetPositions()
        positions = positions - np.mean(positions, axis=0)
        self.positions = positions
        self.mol_filename = mol_filename

    def parametrize(self, charge_method="bcc", gaff_version="gaff2"):
        """Run antechamber for the small parametrization.

        Parameters
        ----------
        charge_method : str, default="bcc"
            Charge method to use for antechamber.
        gaff_version : str, default="gaff2"
            GAFF version to use for antechamber.

        Returns
        -------
        mol2_filename : str
            MOL2 filename of the molecule.
        frcmod_filename : str
            FRCMOD filename of the molecule.
        lib_filename : str
            LIB filename of the molecule.

        """
        mol2_filename, frcmod_filename, lib_filename = _run_antechamber(self.mol_filename, self.name, self.resname, self.charge, charge_method, gaff_version)

        return mol2_filename, frcmod_filename, lib_filename
