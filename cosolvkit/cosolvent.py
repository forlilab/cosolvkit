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
from openbabel import openbabel as ob
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


def _atom_types_from_mol2(mol2_filename):
    """Get atom names using OpenBabel. 
    Because RDKit does not know how to read a mol2 file...
    """
    atom_names = []
    read_flag = False
    
    with open(mol2_filename) as f:
        lines = f.readlines()

        for line in lines:
            if '@<TRIPOS>BOND' in line:
                read_flag = False
                break

            if read_flag:
                sline = line.split()
                atom_names.append(sline[1])

            if '@<TRIPOS>ATOM' in line:
                read_flag = True

    return atom_names


def _write_mol2_from_RDMol(fname, RDMol):
    """Write mol2 file
    """
    write_flag = True

    sio = StringIO()
    w = Chem.SDWriter(sio)
    w.write(RDMol)
    w.flush()
    mol_string = sio.getvalue()

    OBMol = ob.OBMol()
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("mol", "mol2")
    obConversion.ReadString(OBMol, mol_string)
    mol2_string = obConversion.WriteString(OBMol)

    # We have to ignore the UNITY_ATOM_ATTR info
    # because it does not work with charges fragmentd (acetate, methylammonium..)
    with open(fname, "w") as w:
        for line in mol2_string.splitlines():
            if '@<TRIPOS>UNITY_ATOM_ATTR' in line:
                write_flag = False

            if '@<TRIPOS>BOND' in line:
                write_flag = True
                w.write(line + '\n')

            if write_flag:
                w.write(line + '\n')


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

    positions = np.array(positions).astype(np.float)

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

    pdb_coordinates = np.array(pdb_coordinates, dtype=np.float)

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


def _run_antechamber(mol2_filename, molecule_name, resname, charge=0, charge_method="bcc", gaff_version="gaff2"):
    """Run antechamber.
    """
    original_mol2_filename = os.path.abspath(mol2_filename)
    local_mol2_filename = os.path.basename(mol2_filename)
    cwd_dir = os.path.dirname(original_mol2_filename)
    output_frcmod_filename = cwd_dir + os.path.sep + '%s.frcmod' % molecule_name
    output_lib_filename = cwd_dir + os.path.sep + '%s.lib' % molecule_name

    with utils.temporary_directory(prefix=molecule_name, dir='.') as tmp_dir:
        shutil.copy2(original_mol2_filename, local_mol2_filename)

        # Run Antechamber
        cmd = "antechamber -i %s -fi mol2 -o out.mol2 -fo mol2 -s 2 -at %s -c %s -nc %d -rn %s"
        cmd = cmd % (local_mol2_filename, gaff_version, charge_method, charge, resname)
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
        shutil.copy('out.mol2', original_mol2_filename)
        shutil.copy('out.frcmod', output_frcmod_filename)
        shutil.copy('out.lib', output_lib_filename)

    return original_mol2_filename, output_frcmod_filename, output_lib_filename


class CoSolvent:

    def __init__(self, name, smiles=None, mol2_filename=None, lib_filename=None, frcmod_filename=None, charge=0, resname=None):
        if resname is None:
            self.resname = name[:3].upper()
        else:
            self.resname = resname

        assert not self.resname in AMBER_SUPPORTED_RESNAMES, \
               print("Error: the residue name %s is already defined in AMBER." % self.resname)
        assert (smiles is not None or mol2_filename is not None), print("A smiles or a mol2 filename has to be defined.")
        assert (lib_filename is None and frcmod_filename is None) or \
               (lib_filename is not None and frcmod_filename is not None), \
               print("Both lib and frcmod files have to defined, or none of them.")

        self.atom_names = None
        self.name = name
        self.charge = charge
        self._mol2_filename = None
        self._frcmod_filename = None
        self._lib_filename = None
        self._RDMol = None

        if mol2_filename is None:
            mol2_filename = '%s.mol2' % self.name
            # Setup rdkit molecule
            RDmol = Chem.MolFromSmiles(smiles)
            RDmol.SetProp("_Name", name)
            self._RDMol = Chem.AddHs(RDmol)
            AllChem.EmbedMolecule(self._RDMol)
            _write_mol2_from_RDMol(mol2_filename, self._RDMol)

        if lib_filename is None and frcmod_filename is None:
            mol2_filename, frcmod_filename, lib_filename = self.parametrize(mol2_filename, self.name, self.resname, self.charge)

        positions = _read_coordinates_from_mol2(mol2_filename)
        # We make sure the cosolvent is centered, especially if it was provided by the user
        # And honestly, never trust the user!
        positions = positions - np.mean(positions, axis=0)
        self.positions = positions

        self.atom_names = _atom_types_from_mol2(mol2_filename)

        self._mol2_filename = mol2_filename
        self._lib_filename = lib_filename
        self._frcmod_filename = frcmod_filename

    def parametrize(self, mol2_filename, molecule_name, resname, charge, charge_method="bcc", gaff_version="gaff2"):
        """Run antechamber for the small parametrization.
        """
        mol2_filename, frcmod_filename, lib_filename = _run_antechamber(mol2_filename, molecule_name, resname, 
                                                                        charge, charge_method, gaff_version)

        return mol2_filename, frcmod_filename, lib_filename
