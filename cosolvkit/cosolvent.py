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


def _atom_names_from_lib(lib_filename):
    """Get atom names from a lib file.
    
    Parameters
    ----------
    lib_filename : str
        Path to the lib file.

    Returns
    -------
    list
        List of atom names.
    """
    atom_names = []
    read_flag = False

    with open(lib_filename) as f:
        lines = f.readlines()

        for line in lines:
            if '.unit.atomspertinfo' in line:
                read_flag = False
                break

            if read_flag:
                sline = line.split()
                atom_names.append(sline[0].replace('"', ''))

            if '.unit.atoms' in line:
                read_flag = True

    return atom_names


def _read_coordinates_from_mol(mol_filename):
    mol = Chem.MolFromMolFile(mol_filename, removeHs=False)
    positions = mol.GetConformer().GetPositions()
    positions = np.array(positions)

    return positions


def _read_coordinates_from_mol2(mol2_filename):
    coordinate_flag = False
    positions = []

    # We read the mol2 file and modify each atom line
    with open(mol2_filename) as f:
        lines = f.readlines()

        for line in lines:
            # Stop reading coordinates
            # '@<TRIPOS>SUBSTRUCTURE' in case there is only one atom...
            # But who would do this?!
            if '@<TRIPOS>BOND' in line or '@<TRIPOS>SUBSTRUCTURE' in line:
                break

            if coordinate_flag:
                sline = line.split()
                positions.append([sline[2], sline[3], sline[4]])

            # It's time to read the coordinates
            if '@<TRIPOS>ATOM' in line:
                coordinate_flag = True

    positions = np.array(positions).astype(float)

    return positions


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
        print(cmd)
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

    def __init__(self, name, smiles=None, mol_filename=None, lib_filename=None, frcmod_filename=None, charge=0, resname=None):
        """Create a CoSolvent object.

        Parameters
        ----------
        name : str
            Name of the molecule.
        smiles : str, default=None
            SMILES string of the molecule.
        mol_filename : str, default=None
            MOL/SDF filename of the molecule.
        lib_filename : str, default=None
            AMBER library filename of the molecule.
        frcmod_filename : str, default=None
            AMBER frcmod filename of the molecule.
        charge : int, default=0
            Charge of the molecule.
        resname : str, default=None
            Residue name of the molecule.

        """
        if resname is None:
            self.resname = name[:3].upper()
        else:
            self.resname = resname

        assert not self.resname in AMBER_SUPPORTED_RESNAMES, \
               print("Error: the residue name %s is already defined in AMBER." % self.resname)
        assert (smiles is not None or mol_filename is not None), print("A smiles or a mol/sdf filename has to be defined.")
        assert (lib_filename is None and frcmod_filename is None) or \
               (lib_filename is not None and frcmod_filename is not None), \
               print("Both lib and frcmod files have to defined, or none of them.")

        self.atom_names = None
        self.name = name
        self.charge = charge
        self.positions = None
        self._frcmod_filename = None
        self._lib_filename = None

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

            mol_filename = '%s.mol' % self.name
            # Setup rdkit molecule
            RDmol = Chem.MolFromSmiles(smiles)
            RDmol.SetProp("_Name", name)
            RDMol = Chem.AddHs(RDmol)
            AllChem.EmbedMolecule(RDMol, params=params)
            AllChem.MMFFOptimizeMolecule(RDMol)
            with Chem.SDWriter(mol_filename) as w:
                w.write(RDMol)

        if lib_filename is None and frcmod_filename is None:
            mol2_filename, frcmod_filename, lib_filename = self.parametrize(mol_filename, self.name, self.resname, self.charge)
            positions = _read_coordinates_from_mol2(mol2_filename)
        else:
            positions = _read_coordinates_from_mol(mol_filename)

        # We make sure the cosolvent is centered, especially if it was provided by the user
        # And honestly, never trust the user!
        positions = positions - np.mean(positions, axis=0)
        self.positions = positions

        self.atom_names = _atom_names_from_lib(lib_filename)

        self._lib_filename = lib_filename
        self._frcmod_filename = frcmod_filename

    def parametrize(self, mol_filename, molecule_name, resname, charge, charge_method="bcc", gaff_version="gaff2"):
        """Run antechamber for the small parametrization.

        Parameters
        ----------
        mol_filename : str
            MOL/SDF filename of the molecule.
        molecule_name : str
            Name of the molecule.
        resname : str
            Residue name of the molecule.
        charge : int
            Charge of the molecule.
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
        mol2_filename, frcmod_filename, lib_filename = _run_antechamber(mol_filename, molecule_name, resname,
                                                                        charge, charge_method, gaff_version)

        return mol2_filename, frcmod_filename, lib_filename
