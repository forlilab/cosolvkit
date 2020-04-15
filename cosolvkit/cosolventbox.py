#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to create the CoSolvent box
#

import itertools
import os

import numpy as np
from scipy import spatial

from .cosolvent import CoSolvent
from . import utils


def _positions_from_pdb_file(pdb_filename):
    """Get the atomic coordinates from the pdb file
    """
    positions = []

    with open(pdb_filename) as f:
        lines = f.readlines()

        for line in lines:
            if "ATOM" in line or "HETATM" in line:
                positions.append([np.float(line[30:38]), np.float(line[38:47]), np.float(line[47:55])])

    positions = np.array(positions)

    return positions


def _is_close_to_edge(xyz, distance, box_dimension):
    """Is it too close from the edge?
    """
    xyz = np.atleast_2d(xyz)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    xmin, xmax = box_dimension[0]
    ymin, ymax = box_dimension[1]
    zmin, zmax = box_dimension[2]

    x_close = np.logical_or(np.abs(xmin - x) <= distance, np.abs(xmax - x) <= distance)
    y_close = np.logical_or(np.abs(ymin - y) <= distance, np.abs(ymax - y) <= distance)
    z_close = np.logical_or(np.abs(zmin - z) <= distance, np.abs(zmax - z) <= distance)
    close_to = np.any((x_close, y_close, z_close), axis=0)

    return close_to


def _water_is_in_box(xyz, box_dimension):
    """Check if the water is in the box or not.
    """
    xyz = np.atleast_2d(xyz)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    xmin, xmax = box_dimension[0]
    ymin, ymax = box_dimension[1]
    zmin, zmax = box_dimension[2]

    x_in = np.logical_and(xmin - 1. <= x, x <= xmax + 1.)
    y_in = np.logical_and(ymin - 1. <= y, y <= ymax + 1.)
    z_in = np.logical_and(zmin - 1. <= z, z <= zmax + 1.)
    all_in = np.all((x_in, y_in, z_in), axis=0)

    for i in range(0, xyz.shape[0], 3):
        all_in[[i, i + 1, i + 2]] = [np.all(all_in[[i, i + 1, i + 2]])] * 3

    return all_in


def _create_waterbox(box_dimension, receptor_xyzs=None, watref_xyzs=None, watref_dims=None):
    """Create the water box.
    """
    wat_xyzs = []

    watref_xyzs = np.atleast_2d(watref_xyzs)

    xmin, xmax = box_dimension[0]
    ymin, ymax = box_dimension[1]
    zmin, zmax = box_dimension[2]

    x = np.arange(xmin, xmax, watref_dims[0]) + (watref_dims[0] / 2.)
    y = np.arange(ymin, ymax, watref_dims[1]) + (watref_dims[1] / 2.)
    z = np.arange(zmin, zmax, watref_dims[2]) + (watref_dims[2] / 2.)

    X, Y, Z = np.meshgrid(x, y, z)
    center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

    for center_xyz in center_xyzs:
        wat_xyzs.append(watref_xyzs + center_xyz)
    wat_xyzs = np.vstack(wat_xyzs)

    # we cut everything that goes outside the box
    wat_ids = _water_is_in_box(wat_xyzs, box_dimension)
    wat_xyzs = wat_xyzs[wat_ids]

    # Remove water molecules that overlap with the receptor
    if receptor_xyzs is not None:
        kdtree = spatial.cKDTree(wat_xyzs)
        # Get the ids of all the closest water atoms
        to_be_removed = kdtree.query_ball_point(receptor_xyzs, r=1.4, p=2)
        # Keep the unique ids
        to_be_removed = np.unique(np.hstack(to_be_removed))
        # Get the ids of the water oxygen atoms
        to_be_removed = np.unique(to_be_removed - (to_be_removed % 3.))
        # Complete with the ids of the hydrogen atoms
        to_be_removed = [[r, r + 1, r + 2]  for r in to_be_removed]
        to_be_removed = np.hstack(to_be_removed).astype(int)

        # Remove those water molecules
        mask = np.ones(len(wat_xyzs), np.bool)
        mask[to_be_removed] = 0
        wat_xyzs = wat_xyzs[mask]

    return wat_xyzs


def _add_cosolvent(wat_xyzs, cosolvents, box_dimension, receptor_xyzs=None, final_concentration=5.):
    """Add cosolvent to the water box.
    """
    i = 1
    water_vol = 20.088000000000005
    frag_vol = 0.
    concentration = 0.

    cosolv_xyzs = {name: [] for name in cosolvents}
    cosolv_names = cosolvents.keys()

    if receptor_xyzs is not None:
        kdtree_receptor = spatial.cKDTree(receptor_xyzs)

    for cosolv_name in itertools.cycle(cosolv_names):
        # Update kdtree
        kdtree = spatial.cKDTree(wat_xyzs)

        # Choose a random water molecule
        wat_o = wat_xyzs[::3]
        wat_id = np.random.choice(range(0, wat_o.shape[0]))
        wat_xyz = wat_o[wat_id]

        # Check if it is not too close from the edges or the protein
        # Not the best efficient way...
        too_close_edge = _is_close_to_edge(wat_xyz, 2., box_dimension)[0]

        if receptor_xyzs is not None:
            too_close_protein = kdtree_receptor.query_ball_point([wat_xyz], r=3, p=2)
            too_close_protein = np.unique(np.hstack(too_close_protein))

        if not too_close_edge and too_close_protein.size == 0:
            # Translate fragment on the top of the selected water molecule
            cosolv_xyz = cosolvents[cosolv_name].positions + wat_xyz

            # Add fragment to list
            cosolv_xyzs[cosolv_name].append(cosolv_xyz)
            frag_vol += cosolvents[cosolv_name].volume

            # Get the ids of all the closest water atoms
            to_be_removed = kdtree.query_ball_point(cosolv_xyz, r=1.5, p=2)
            # Keep the unique ids
            to_be_removed = np.unique(np.hstack(to_be_removed))
            # Get the ids of the water oxygen atoms
            to_be_removed = np.unique(to_be_removed - (to_be_removed % 3.))
            # Complete with the ids of the hydrogen atoms
            to_be_removed = [[r, r + 1, r + 2]  for r in to_be_removed]
            to_be_removed = np.hstack(to_be_removed).astype(int)

            # Remove those water molecules
            mask = np.ones(len(wat_xyzs), np.bool)
            mask[to_be_removed] = 0
            wat_xyzs = wat_xyzs[mask]

            # Compute v/v concentration
            n_water = wat_xyzs.shape[0]
            wat_vol = n_water * water_vol
            concentration = (frag_vol / wat_vol) * 100.

            if concentration >= final_concentration:
                break

            i += 1
    
    return wat_xyzs, cosolv_xyzs, concentration


class CoSolventBox:

    def __init__(self, concentration=10, cutoff=12, box="cubic", dimensions=None, pH=7.):
        """Initialize the cosolvent box
        """
        assert box in ["cubic", "orthorombic"], "Error: the water box can be only cubic or orthorombic."

        self._concentration = concentration
        self._cutoff = cutoff
        self._pH = pH
        self._box = box
        if dimensions is not None:
            assert np.ravel(dimensions).size == 3, "Error: dimensions should contain only (a, b, c)."
            self._dimensions = np.array([[0, dimensions[0]], 
                                         [0, dimensions[1]], 
                                         [0, dimensions[2]]]).astype(np.float)
        else:
            self._dimensions = None

        # Read the reference water box
        d = utils.path_module("cosolvkit")
        waterbox_filename = os.path.join(d, "data/waterbox.pdb")
        self._watref_xyzs = _positions_from_pdb_file(waterbox_filename)
        self._watref_dims = [18.856, 18.856, 18.856]

        self._receptor_xyzs = None
        self._cosolvents = {}
        self._wat_xysz = None
        self._cosolv_xyzs = None

    def add_receptor(self, receptor_filename):
        """Add receptor
        """
        self._receptor_filename = receptor_filename
        self._receptor_xyzs = _positions_from_pdb_file(receptor_filename)

        if self._dimensions is None:
            xmin = np.min(self._receptor_xyzs[:,0]) - self._cutoff
            xmax = np.max(self._receptor_xyzs[:,0]) + self._cutoff
            ymin = np.min(self._receptor_xyzs[:,1]) - self._cutoff
            ymax = np.max(self._receptor_xyzs[:,1]) + self._cutoff
            zmin = np.min(self._receptor_xyzs[:,2]) - self._cutoff
            zmax = np.max(self._receptor_xyzs[:,2]) + self._cutoff

            if self._box == "orthorombic":
                self._dimensions = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
            else:
                lmin = np.min([xmin, ymin, zmin])
                lmax = np.max([xmax, ymax, zmax])
                self._dimensions = np.array([[lmin, lmax], [lmin, lmax], [lmin, lmax]])

        # Center box around the receptor
        receptor_center = np.mean(self._receptor_xyzs, axis=0)
        box_center = np.mean(self._dimensions, axis=1)
        self._dimensions -= box_center[:,None]
        self._dimensions += receptor_center[:,None]
        
    def add_cosolvent(self, name, smiles, charge=0, resname=None):
        """Add cosolvent and parametrize it
        """
        c = CoSolvent(name, smiles, charge, resname)
        self._cosolvents[name] = c
    
    def build(self):
        """Build the cosolvent box
        """
        if self._dimensions is not None:
            self._wat_xyzs = _create_waterbox(self._dimensions, self._receptor_xyzs,
                                              self._watref_xyzs, self._watref_dims)

            if self._cosolvents:
                wat_xyzs, cosolv_xyzs, conc = _add_cosolvent(self._wat_xyzs, self._cosolvents,
                                                             self._dimensions, self._receptor_xyzs,
                                                             self._concentration)

                self._wat_xyzs = wat_xyzs
                self._cosolv_xyzs = cosolv_xyzs
                self._final_concentration = conc

                print("------------------------------------")
                print("Final concentration: %5.3f %%" % conc)
                print("Water (WAT): %3d" % (self._wat_xyzs.shape[0] / 3))
                for cosolv_name in self._cosolvents:
                    print("%s (%s): %3d" % (cosolv_name.capitalize(), 
                                            self._cosolvents[cosolv_name].resname,
                                            len(self._cosolv_xyzs[cosolv_name])))
                print("------------------------------------")

        else:
            print("Error: box dimensions was not defined.")

    def export(self, prefix=None):
        """Export pdb file for tleap
        """
        n_atom = 0
        n_residue = 1
        n_atom_water = 1
        prmtop_filename = "system.prmtop"
        inpcrd_filename = "system.inpcrd"
        pdb_filename = "system.pdb"
        water_atom_names = ["O", "H1", "H2"] * int(self._wat_xyzs.shape[0] / 3)
        # We get ride of the segid, otherwise the number of atoms cannot exceed 9.999
        template = "%-6s%5d  %-3s%1s%3s %5d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f              \n"

        if prefix is not None:
            prmtop_filename = prefix + "_" + prmtop_filename
            inpcrd_filename = prefix + "_" + inpcrd_filename
            pdb_filename = prefix + "_" + pdb_filename

        # Create system
        with open("system.pdb", 'w') as w:
            # Write receptor first
            with open(self._receptor_filename) as f:
                lines = f.readlines()

                for line in lines:
                    if "ATOM" in line or "HETATM" in line:
                        w.write(line)
                        n_atom += 1
                    elif "TER" in line:
                        w.write(line)

            # Write cosolvent molecules
            if self._cosolv_xyzs is not None:
                for name in self._cosolvents:
                    cosolv_xyzs = self._cosolv_xyzs[name]
                    resname = self._cosolvents[name].resname
                    atom_names = self._cosolvents[name].atom_names

                    for residue_xyzs in cosolv_xyzs:
                        for atom_xyz, atom_name in zip(residue_xyzs, atom_names):
                            x, y, z = atom_xyz
                            w.write(template % ("ATOM", n_atom, atom_name, " ", resname, n_residue, " ", x, y, z, 0., 0.))
                            n_atom += 1
                        n_residue += 1

                    w.write("TER\n")

            # And water molecules at the end
            for wat_xyz, atom_name in zip(self._wat_xyzs, water_atom_names):
                x, y, z = wat_xyz
                w.write(template % ("ATOM", n_atom, atom_name, " ", 'WAT', n_residue, " ", x, y, z, 0., 0.))

                if n_atom_water % 3 == 0:
                    n_residue += 1

                n_atom_water += 1
                n_atom += 1

            w.write("TER\n")
            w.write("END\n")

        # Create tleap template
        TLEAP_TEMPLATE = ("source leaprc.protein.ff14SB\n"
                          "source leaprc.DNA.bsc1\n"
                          "source leaprc.water.tip3p\n"
                          "source leaprc.gaff2\n"
                          )

        if self._cosolvents is not None:
            for name in self._cosolvents:
                frcmod_filename = os.path.basename(self._cosolvents[name]._frcmod_filename)
                lib_filename = os.path.basename(self._cosolvents[name]._lib_filename)

                TLEAP_TEMPLATE += "loadamberparams %s\n" % frcmod_filename
                TLEAP_TEMPLATE += "loadoff %s\n" % lib_filename

        TLEAP_TEMPLATE += "set default nocenter off\n"
        TLEAP_TEMPLATE += "m = loadpdb system.pdb\n"
        TLEAP_TEMPLATE += "charge m\n"
        TLEAP_TEMPLATE += "addIonsRand m Cl- 0\n"
        TLEAP_TEMPLATE += "addIonsRand m Na+ 0\n"
        TLEAP_TEMPLATE += "check m\n"
        TLEAP_TEMPLATE += "setBox m \"vdw\"\n"
        TLEAP_TEMPLATE += "saveamberparm m %s %s\n" % (prmtop_filename, inpcrd_filename)
        TLEAP_TEMPLATE += "savepdb m %s\n" % (pdb_filename)
        TLEAP_TEMPLATE += "quit\n"

        with open("tleap.cmd", 'w') as w:
            w.write(TLEAP_TEMPLATE)

        cmd = 'tleap -f tleap.cmd'
        outputs, errors = utils.execute_command(cmd)
