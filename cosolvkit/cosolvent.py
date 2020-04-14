#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to create the Cosolvent
#

from io import StringIO

import numpy as np
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem


#def run_antechamber(molecule_name, mol2_filename, charge=0, charge_method="am1bcc", gaff_version="gaff2"):



class CoSolvent:

    def __init__(self, name, smiles, charge=0):
        self._name = name
        self._charge = charge
        # Setup rdkit molecule
        RDmol = Chem.MolFromSmiles(smiles)
        RDmol.SetProp("_Name", name)
        self._RDmol = Chem.AddHs(RDmol)
        AllChem.EmbedMolecule(self._RDmol)
        # Get some properties
        self.positions = np.array([c.GetPositions() for c in self._RDmol.GetConformers()][0])
        self.symbols = np.array([a.GetSymbol() for a in self._RDmol.GetAtoms()])
        self.volume = AllChem.ComputeMolVolume(self._RDmol)

        self.parametrize()

    def write_mol2_file(self, fname):
        sio = StringIO()
        w = Chem.SDWriter(sio)
        w.write(self._RDmol)
        w.flush()
        mol_string = sio.getvalue()
        print(mol_string)

        OBMol = ob.OBMol()
        obConversion = ob.OBConversion()
        obConversion.SetInAndOutFormats("mol", "mol2")
        obConversion.ReadString(OBMol, mol_string)
        obConversion.WriteFile(OBMol, fname)

    def parametrize(self, charge_method="am1bcc", gaff_version="gaff2"):
        self.write_mol2_file('%s.mol2' % self._name)
