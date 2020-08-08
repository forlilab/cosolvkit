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
        outputs, errors = utils.execute_command(cmd)

        # Run parmchk2 for the additional force field file
        cmd = 'parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s %s' 
        cmd = cmd % (gaff_version)
        outputs, errors = utils.execute_command(cmd)

        # Run tleap for the library file
        with open('tleap.cmd', 'w') as w:
            w.write(TLEAP_TEMPLATE % {'gaff_version': gaff_version, 
                                      'molecule_name': molecule_name,
                                      'resname': resname
                                      })
        cmd = 'tleap -s -f tleap.cmd'
        outputs, errors = utils.execute_command(cmd)

        # The final mol2 file from antechamber does not contain
        # the optimized geometry from sqm. Why?!
        _transfer_coordinates_from_pdb_to_mol2('sqm.pdb', 'out.mol2')

        # Copy back all we need
        shutil.copy('out.mol2', original_mol2_filename)
        shutil.copy('out.frcmod', output_frcmod_filename)
        shutil.copy('out.lib', output_lib_filename)

    return original_mol2_filename, output_frcmod_filename, output_lib_filename


class CoSolvent:

    def __init__(self, name, smiles, charge=0, resname=None):
        self._name = name
        self._charge = charge
        if resname is None:
            self.resname = name[:3].upper()
        else:
            self.resname = resname

        assert not self.resname in AMBER_SUPPORTED_RESNAMES, print("Error: the residue name %s is already defined in AMBER." % self.resname)

        self.atom_names = None
        self._mol2_filename = None
        self._frcmod_filename = None
        self._lib_filename = None

        # Setup rdkit molecule
        RDmol = Chem.MolFromSmiles(smiles)
        RDmol.SetProp("_Name", name)
        self._RDmol = Chem.AddHs(RDmol)
        AllChem.EmbedMolecule(self._RDmol)
        # Get some properties
        self.positions = np.array([c.GetPositions() for c in self._RDmol.GetConformers()][0])
        self.volume = AllChem.ComputeMolVolume(self._RDmol)

        self.parametrize()

    def write_mol2(self, fname):
        """Write mol2 file
        """
        write_flag = True

        sio = StringIO()
        w = Chem.SDWriter(sio)
        w.write(self._RDmol)
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

    def atom_types_from_mol2(self, mol2_filename):
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

        self.atom_names = atom_names

    def parametrize(self, charge_method="bcc", gaff_version="gaff2"):
        """Run antechamber for the small parametrization.
        """
        mol2_filename = '%s.mol2' % self._name
        self.write_mol2(mol2_filename)

        mol2_filename, frcmod_filename, lib_filename = _run_antechamber(mol2_filename, self._name, self.resname, 
                                                                        self._charge, charge_method, gaff_version)

        self.atom_types_from_mol2(mol2_filename)

        self._mol2_filename = mol2_filename
        self._frcmod_filename = frcmod_filename
        self._lib_filename = lib_filename
