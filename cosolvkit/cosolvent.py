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

from . import utils


TLEAP_TEMPLATE = """
source leaprc.protein.ff14SB
source leaprc.DNA.bsc1
source leaprc.water.tip3p
source leaprc.%(gaff_version)s
loadamberparams out.frcmod
%(residue_name)s = loadmol2 out.mol2
check %(residue_name)s
saveoff %(residue_name)s out.lib
quit
"""


AMBER_SUPPORTED_RESNAMES = ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 
                            'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 
                            'TYR', 'VAL', 'HID', 'HIE', 'HIN', 'HIP', 'CYX', 'ASH', 'GLH', 
                            'LYH', 'ACE', 'NME', 'GL4', 'AS4', 'C', 'G', 'U', 'A', 'DC', 'DG', 
                            'DT', 'DA', 'OHE', 'C5', 'G5', 'U5', 'A5', 'C3', 'G3', 'U3', 'A3', 
                            'DC5', 'DG5', 'DT5', 'DA5', 'DC3', 'DG3', 'DT3', 'DA3', 'WAT', 
                            'HOH', 'AG', 'AL', 'Ag', 'BA', 'BR', 'Be', 'CA', 'CD', 'CE', 'CL', 
                            'CO', 'CR', 'CS', 'CU', 'CU1', 'Ce', 'Cl-', 'Cr', 'Dy', 'EU', 'EU3', 
                            'Er', 'F', 'FE', 'FE2', 'GD3', 'HE+', 'HG', 'HZ+', 'Hf', 'IN', 'IOD', 
                            'K', 'K+', 'LA', 'LI', 'LU', 'MG', 'MN', 'NA', 'NH4', 'NI', 'Na+', 
                            'Nd', 'PB', 'PD', 'PR', 'PT', 'Pu', 'RB', 'Ra', 'SM', 'SR', 'Sm', 
                            'Sn', 'TB', 'TL', 'Th', 'Tl', 'Tm', 'U4+', 'V2+', 'Y', 'YB2', 'ZN', 'Zr')



def _run_antechamber(mol2_filename, molecule_name, residue_name, charge=0, charge_method="bcc", gaff_version="gaff2"):
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
        cmd = "antechamber -i %s -fi mol2 -o out.mol2 -fo mol2 -s 2 -at gaff -c %s -nc %d -rn %s"
        cmd = cmd % (local_mol2_filename, charge_method, charge, residue_name)
        outputs, errors = utils.execute_command(cmd)

        # Run parmchk2 for the additional force field file
        cmd = 'parmchk2 -i out.mol2 -f mol2 -o out.frcmod -s %s' 
        cmd = cmd % (gaff_version)
        outputs, errors = utils.execute_command(cmd)

        # Run tleap for the library file
        with open('tleap.cmd', 'w') as w:
            w.write(TLEAP_TEMPLATE % {'gaff_version': gaff_version, 
                                      'molecule_name': molecule_name,
                                      'residue_name': residue_name
                                      })
        cmd = 'tleap -s -f tleap.cmd'
        outputs, errors = utils.execute_command(cmd)

        # Copy back all we need
        shutil.copy('out.mol2', original_mol2_filename)
        shutil.copy('out.frcmod', output_frcmod_filename)
        shutil.copy('out.lib', output_lib_filename)

    return original_mol2_filename, output_frcmod_filename, output_lib_filename


class CoSolvent:

    def __init__(self, name, smiles, charge=0, residue_name=None):
        self._name = name
        self._charge = charge
        if residue_name is None:
            self.residue_name = name[:3].upper()
        else:
            self.residue_name = residue_name

        assert not self.residue_name in AMBER_SUPPORTED_RESNAMES, print("Error: the residue name %s is already defined in AMBER." % self.residue_name)

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
        sio = StringIO()
        w = Chem.SDWriter(sio)
        w.write(self._RDmol)
        w.flush()
        mol_string = sio.getvalue()

        OBMol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("mol", "mol2")
        obConversion.ReadString(OBMol, mol_string)
        obConversion.WriteFile(OBMol, fname)

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

        mol2_filename, frcmod_filename, lib_filename = _run_antechamber(mol2_filename, self._name, self.residue_name, 
                                                                        self._charge, charge_method, gaff_version)

        self.atom_types_from_mol2(mol2_filename)

        self._mol2_filename = mol2_filename
        self._frcmod_filename = frcmod_filename
        self._lib_filename = lib_filename
