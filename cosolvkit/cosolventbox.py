#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to create the CoSolvent box
#

import itertools
import os
from operator import itemgetter

import numpy as np
from scipy import spatial

from .cosolvent import CoSolvent
from . import utils


AVOGADRO_CONSTANT_NA = 6.02214179e+23


def _read_pdb(pdb_filename, ignore_hydrogen=False):
    data = []
    dtype = [("name", "U4"), ("resname", "U3"), ("resid", "i4"), ('chain', 'U1'), ("xyz", "f4", (3))]

    with open(pdb_filename) as f:
        lines = f.readlines()

        for line in lines:
            if "ATOM" in line or "HETATM" in line:
                name = line[12:16].strip()

                if (not ignore_hydrogen and name[0] == 'H') or name[0] != "H":
                    data.append((line[12:16].strip(), 
                                 line[17:20].strip(), 
                                 int(line[22:26]), 
                                 line[21:22].strip(),
                                 [np.float(line[30:38]), np.float(line[38:47]), np.float(line[47:55])]))

    data = np.array(data, dtype=dtype)

    return data


def _is_water_close_to_edge(wat_xyzs, distance, box_origin, box_size):
    """Is it too close from the edge?
    """
    wat_xyzs = np.atleast_2d(wat_xyzs)
    x, y, z = wat_xyzs[:, 0], wat_xyzs[:, 1], wat_xyzs[:, 2]

    xmin, xmax = box_origin[0], box_origin[0] + box_size[0]
    ymin, ymax = box_origin[1], box_origin[1] + box_size[1]
    zmin, zmax = box_origin[2], box_origin[2] + box_size[2]

    x_close = np.logical_or(np.abs(xmin - x) <= distance, np.abs(xmax - x) <= distance)
    y_close = np.logical_or(np.abs(ymin - y) <= distance, np.abs(ymax - y) <= distance)
    z_close = np.logical_or(np.abs(zmin - z) <= distance, np.abs(zmax - z) <= distance)
    close_to = np.any((x_close, y_close, z_close), axis=0)

    for i in range(0, wat_xyzs.shape[0], 3):
        close_to[[i, i + 1, i + 2]] = [np.all(close_to[[i, i + 1, i + 2]])] * 3

    return close_to


def _is_water_close_from_receptor(wat_xyzs, receptor_xyzs, distance=3.):
    """Is too close from the receptor?
    """
    kdtree = spatial.cKDTree(wat_xyzs)

    ids = kdtree.query_ball_point(receptor_xyzs, r=distance, p=2)
    # Keep the unique ids
    ids = np.unique(np.hstack(ids)).astype(np.int)

    close_to = np.zeros(len(wat_xyzs), np.bool)
    close_to[ids] = True

    for i in range(0, wat_xyzs.shape[0], 3):
        close_to[[i, i + 1, i + 2]] = [np.all(close_to[[i, i + 1, i + 2]])] * 3

    return close_to


def _is_in_box(xyzs, box_origin, box_size, box_buffer=0):
    """Is in the box or not?
    """
    xyzs = np.atleast_2d(xyzs)
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    xmin, xmax = box_origin[0] - box_buffer, box_origin[0] + box_size[0] + box_buffer
    ymin, ymax = box_origin[1] - box_buffer, box_origin[1] + box_size[1] + box_buffer
    zmin, zmax = box_origin[2] - box_buffer, box_origin[2] + box_size[2] + box_buffer

    x_in = np.logical_and(xmin <= x, x <= xmax)
    y_in = np.logical_and(ymin <= y, y <= ymax)
    z_in = np.logical_and(zmin <= z, z <= zmax)
    all_in = np.all((x_in, y_in, z_in), axis=0)

    return all_in

    
def _water_is_in_box(wat_xyzs, box_origin, box_size):
    """Check if the water is in the box or not.
    """
    all_in = _is_in_box(wat_xyzs, box_origin, box_size, 1.)

    for i in range(0, wat_xyzs.shape[0], 3):
        all_in[[i, i + 1, i + 2]] = [np.all(all_in[[i, i + 1, i + 2]])] * 3

    return all_in


def _protein_in_box(receptor_data, box_origin, box_size, truncate_buffer=5., ignore_peptide_size=4):
    data = []
    protein_terminus = {}
    peptides_terminus = {}

    all_in = _is_in_box(receptor_data['xyz'], box_origin, box_size + (2 * truncate_buffer))
    residues_in_box = np.unique(receptor_data[["resid", "chain"]][all_in], axis=0)
    chain_ids = np.unique(residues_in_box["chain"])

    for chain_id in chain_ids:
        resids_to_keep = []
        peptides_terminus[chain_id] = []

        resids = residues_in_box[residues_in_box["chain"] == chain_id]["resid"]
        # we need that information to know where to put the charged patchs
        protein_terminus[chain_id] = (resids[0], resids[-1])

        # Get continuous peptide             
        for k, g in itertools.groupby(enumerate(resids), lambda x: x[0] - x[1]):
            peptide_terminus = [None, None]
            peptide_resids = list(map(itemgetter(1), g))

            if not len(peptide_resids) < ignore_peptide_size:
                resids_to_keep.extend(peptide_resids)

                # We need that information to know where to put the neutral patchs
                if peptide_resids[0] != protein_terminus[chain_id][0]:
                    peptide_terminus[0] = peptide_resids[0]
                if peptide_resids[-1] != protein_terminus[chain_id][1]:
                    peptide_terminus[1] = peptide_resids[-1]

                peptides_terminus[chain_id].append(peptide_terminus)
            else:
                print("Warning: peptide %s will be ignored (minimal size allowed: %d)." % (peptide_resids, ignore_peptide_size))

        selected_residues = receptor_data[receptor_data["chain"] == chain_id]
        selected_residues = selected_residues[np.isin(selected_residues["resid"], resids_to_keep)]
        data.append(selected_residues)

    data = np.concatenate(data)

    return data, protein_terminus, peptides_terminus


def _create_waterbox(box_origin, box_size, receptor_xyzs=None, watref_xyzs=None, watref_dims=None):
    """Create the water box.
    """
    wat_xyzs = []

    watref_xyzs = np.atleast_2d(watref_xyzs)

    xmin, xmax = box_origin[0], box_origin[0] + box_size[0]
    ymin, ymax = box_origin[1], box_origin[1] + box_size[1]
    zmin, zmax = box_origin[2], box_origin[2] + box_size[2]

    x = np.arange(xmin, xmax, watref_dims[0]) + (watref_dims[0] / 2.)
    y = np.arange(ymin, ymax, watref_dims[1]) + (watref_dims[1] / 2.)
    z = np.arange(zmin, zmax, watref_dims[2]) + (watref_dims[2] / 2.)

    X, Y, Z = np.meshgrid(x, y, z)
    center_xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)

    for center_xyz in center_xyzs:
        wat_xyzs.append(watref_xyzs + center_xyz)
    wat_xyzs = np.vstack(wat_xyzs)

    # we cut everything that goes outside the box
    wat_ids = _water_is_in_box(wat_xyzs, box_origin, box_size)
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


def _volume_water(n_water):
    """ Compute volume of the box based on the number
    of water molecules. The volume of one water molecule is based
    on the reference water box.
    """
    return ((18.856 * 18.856 * 18.856) / 216) * n_water


def _volume_protein(n_water, gridsize):
    vol_box = np.prod(gridsize)
    vol_water = _volume_water(n_water)

    #assert vol_water <= vol_box, "The volume of water (%f) is superior than the whole box (%f)." % (vol_water, vol_box)

    return vol_box - vol_water


def _add_cosolvent(wat_xyzs, cosolvents, box_origin, box_size, volume, receptor_xyzs=None, concentration=0.25):
    """Add cosolvent to the water box.
    """
    cosolv_xyzs = {name: [] for name in cosolvents}
    cosolv_names = cosolvents.keys()

    # Get water molecules that are too close from the edges of the box
    too_close_edge = _is_water_close_to_edge(wat_xyzs, 3., box_origin, box_size)
    # Get water molecules that are too close from the receptor
    if receptor_xyzs is not None:
        too_close_protein = _is_water_close_from_receptor(wat_xyzs, receptor_xyzs, 3.)
        # Combine both
        to_keep = np.any((too_close_edge, too_close_protein), axis=0)
    else:
        to_keep = too_close_edge

    # Put aside water edges box and close to the protein
    to_keep_wat_xyzs = wat_xyzs[to_keep]
    # We will work on the ones in between
    wat_xyzs = wat_xyzs[~to_keep]

    for i, cosolv_name in enumerate(itertools.cycle(cosolv_names)):
        # Update kdtree
        kdtree = spatial.cKDTree(wat_xyzs)

        # Choose a random water molecule
        wat_o = wat_xyzs[::3]
        wat_id = np.random.choice(range(0, wat_o.shape[0]))
        wat_xyz = wat_o[wat_id]

        # Translate fragment on the top of the selected water molecule
        cosolv_xyz = cosolvents[cosolv_name].positions + wat_xyz

        # Add fragment to list
        cosolv_xyzs[cosolv_name].append(cosolv_xyz)

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

        # We compute the concentration only after 
        # placing the same number of cosolvent molecules
        if (i + 1) % len(cosolvents) == 0:
            n_water = (to_keep_wat_xyzs.shape[0] + wat_xyzs.shape[0]) / 3
            # 1 cosolvent molecule per 55 waters correspond
            # to a concentration of 1 M
            final_concentration = 55. / (n_water / (i + 1))

            if final_concentration >= concentration:
                break

    # Add back water molecules we put aside at the beginning
    wat_xyzs = np.vstack((to_keep_wat_xyzs, wat_xyzs))

    return wat_xyzs, cosolv_xyzs, final_concentration


class CoSolventBox:

    def __init__(self, concentration=0.25, cutoff=12, box="cubic", center=None, gridsize=None, 
                 truncate=False, truncate_buffer=5., min_size_peptide=4):
        """Initialize the cosolvent box
        """
        assert box in ["cubic", "orthorombic"], "Error: the water box can be only cubic or orthorombic."

        self._concentration = concentration
        self._cutoff = cutoff
        self._truncate = truncate
        self._truncate_buffer = truncate_buffer
        self._min_size_peptide = min_size_peptide
        self._box = box

        if center is not None and gridsize is not None:
            gridsize = np.array(gridsize)

            # Check center
            assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
            # Check gridsize
            assert np.ravel(gridsize).size == 3, "Error: grid size should contain only (a, b, c)."
            assert (gridsize > 0).all(), "Error: grid size cannot contain negative numbers."

            self._center = center
            # It's easier to work with integers for grid size
            self._gridsize = np.ceil(gridsize).astype(np.int)
            self._origin = self._center - (self._gridsize  / 2.)
        elif (center is not None and gridsize is None) or (center is None and gridsize is not None):
            print("Error: cannot define the size of the grid without defining its center. Et vice et versa !")
            sys.exit(1)
        else:
            self._center = None
            self._gridsize = None
            self._origin = None

        # Read the reference water box
        d = utils.path_module("cosolvkit")
        waterbox_filename = os.path.join(d, "data/waterbox.pdb")
        self._watref_xyzs = _read_pdb(waterbox_filename)['xyz']
        self._watref_dims = [18.856, 18.856, 18.856]

        self._receptor_data = None
        self._protein_terminus = None
        self._peptides_terminus = None
        self._cosolvents = {}
        self._wat_xysz = None
        self._cosolv_xyzs = None

    def add_receptor(self, receptor_filename):
        """Add receptor
        """
        self._receptor_filename = receptor_filename
        self._receptor_data = _read_pdb(receptor_filename, ignore_hydrogen=True)

        if self._center is None and self._gridsize is None:
            xmin = np.min(self._receptor_data['xyz'][:,0]) - self._cutoff
            xmax = np.max(self._receptor_data['xyz'][:,0]) + self._cutoff
            ymin = np.min(self._receptor_data['xyz'][:,1]) - self._cutoff
            ymax = np.max(self._receptor_data['xyz'][:,1]) + self._cutoff
            zmin = np.min(self._receptor_data['xyz'][:,2]) - self._cutoff
            zmax = np.max(self._receptor_data['xyz'][:,2]) + self._cutoff

            if self._box == "orthorombic":
                self._gridsize = np.ceil(np.array([xmax - xmin, ymax - ymin, zmax - zmin])).astype(np.int)
            else:
                lmax = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
                self._gridsize = np.ceil(np.array([lmax, lmax, lmax])).astype(np.int)

            self._center = np.mean(self._receptor_data['xyz'], axis=0)
            self._origin = self._center - (self._gridsize / 2)

        if self._truncate:
            results = _protein_in_box(self._receptor_data, self._origin, self._gridsize, 
                                      self._truncate_buffer, self._min_size_peptide)
            self._receptor_data, self._protein_terminus, self._peptides_terminus = results

            xmin = np.min(self._receptor_data['xyz'][:,0]) - self._cutoff
            xmax = np.max(self._receptor_data['xyz'][:,0]) + self._cutoff
            ymin = np.min(self._receptor_data['xyz'][:,1]) - self._cutoff
            ymax = np.max(self._receptor_data['xyz'][:,1]) + self._cutoff
            zmin = np.min(self._receptor_data['xyz'][:,2]) - self._cutoff
            zmax = np.max(self._receptor_data['xyz'][:,2]) + self._cutoff

            self._gridsize = np.ceil(np.array([xmax - xmin, ymax - ymin, zmax - zmin])).astype(np.int)
            self._center = np.mean(self._receptor_data['xyz'], axis=0)
            self._origin = self._center - (self._gridsize  / 2.)
        
    def add_cosolvent(self, name, smiles, charge=0, resname=None):
        """Add cosolvent and parametrize it
        """
        c = CoSolvent(name, smiles, charge, resname)
        self._cosolvents[name] = c
    
    def build(self):
        """Build the cosolvent box
        """
        if self._origin is not None:
            if self._receptor_data is not None:
                receptor_xyzs = self._receptor_data['xyz']
            else:
                receptor_xyzs = None

            self._wat_xyzs = _create_waterbox(self._origin, self._gridsize, receptor_xyzs,
                                              self._watref_xyzs, self._watref_dims)

            n_water = np.int(self._wat_xyzs.shape[0] / 3)
            self._volume = _volume_water(n_water)
            volume_protein = _volume_protein(n_water, self._gridsize)

            print("------------------------------------")
            print("Volume box: %10.4f A**3" % self._volume)
            if self._receptor_data is not None:
                print("Volume protein (box - water): %10.4f A**3" % volume_protein)
            print("Water (before cosolvent): %d" % n_water)
            print("Box type: %s" % self._box)
            print("Box center: %8.3f %8.3f %8.3f" % (self._center[0], self._center[1], self._center[2]))
            print("Box dimensions: x %d y %d z %d (A)" % (self._gridsize[0], self._gridsize[1], self._gridsize[2]))

            if self._cosolvents:
                wat_xyzs, cosolv_xyzs, conc = _add_cosolvent(self._wat_xyzs, self._cosolvents,
                                                             self._origin, self._gridsize, self._volume,
                                                             receptor_xyzs, self._concentration)

                self._wat_xyzs = wat_xyzs
                self._cosolv_xyzs = cosolv_xyzs
                n_water = np.int(self._wat_xyzs.shape[0] / 3)

                print("")
                print("Target concentration (M): %5.3f" % self._concentration)
                print("Final concentration (M): %5.3f" % conc)
                print("Water (WAT): %3d" % (n_water))
                for cosolv_name in self._cosolvents:
                    print("%s (%s): %3d" % (cosolv_name.capitalize(), 
                                            self._cosolvents[cosolv_name].resname,
                                            len(self._cosolv_xyzs[cosolv_name])))

        else:
            print("Error: box dimensions was not defined.")
            sys.exit(1)

        print("------------------------------------")

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
        template = "%-6s%5d %-4s%1s%3s %5d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f              \n"

        if self._receptor_data is not None:
            resid_peptide = self._receptor_data[0]['resid']
        else:
            resid_peptide = 0

        if prefix is not None:
            prmtop_filename = prefix + "_" + prmtop_filename
            inpcrd_filename = prefix + "_" + inpcrd_filename
            pdb_filename = prefix + "_" + pdb_filename

        # Create system
        with open("system.pdb", 'w') as w:
            # Write protein first
            if self._receptor_data is not None:
                for i, atom in enumerate(self._receptor_data):
                    x, y, z = atom['xyz']

                    # Special case when the atom types is 4 caracters long
                    if len(atom['name']) <= 3:
                        name = ' ' + atom['name']
                    else:
                        name = atom['name']

                    w.write(template % ("ATOM", i + 1, name, " ", atom['resname'], atom['resid'], 
                                        atom['chain'], x, y, z, 0., 0.))

                    try:
                        # We are looking for gap in the sequence, maybe due to the truncation
                        if (atom['resid'] != self._receptor_data[i + 1]['resid']) and \
                           (atom['resid'] + 1 != self._receptor_data[i + 1]['resid']):
                            w.write("TER\n")

                            if (atom['resid'] - resid_peptide + 1) < 14:
                                print("Warning: Isolated short peptide of length %d, resid %d to %d." % \
                                      (atom['resid'] - resid_peptide + 1, resid_peptide, atom['resid']))
                                resid_peptide = self._receptor_data[i + 1]['resid']
                    except:
                        continue

                w.write("TER\n")

            # Write cosolvent molecules
            if self._cosolv_xyzs is not None:
                for name in self._cosolvents:
                    cosolv_xyzs = self._cosolv_xyzs[name]
                    resname = self._cosolvents[name].resname
                    atom_names = self._cosolvents[name].atom_names

                    for residue_xyzs in cosolv_xyzs:
                        for atom_xyz, atom_name in zip(residue_xyzs, atom_names):
                            x, y, z = atom_xyz
                            w.write(template % ("ATOM", n_atom, atom_name, " ", resname, n_residue, 
                                                " ", x, y, z, 0., 0.))
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
        TLEAP_TEMPLATE = ("source leaprc.protein.ff19SB\n"
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

        TLEAP_TEMPLATE += "set default nocenter on\n"
        TLEAP_TEMPLATE += "m = loadpdb system.pdb\n"
        TLEAP_TEMPLATE += "charge m\n"
        TLEAP_TEMPLATE += "addIonsRand m Cl- 0\n"
        TLEAP_TEMPLATE += "addIonsRand m Na+ 0\n"
        TLEAP_TEMPLATE += "check m\n"
        TLEAP_TEMPLATE += "set m box {%d %d %d}\n" % (self._gridsize[0], self._gridsize[1], self._gridsize[2])
        TLEAP_TEMPLATE += "saveamberparm m %s %s\n" % (prmtop_filename, inpcrd_filename)
        TLEAP_TEMPLATE += "savepdb m %s\n" % (pdb_filename)
        TLEAP_TEMPLATE += "quit\n"

        with open("tleap.cmd", 'w') as w:
            w.write(TLEAP_TEMPLATE)

        cmd = 'tleap -f tleap.cmd'
        outputs, errors = utils.execute_command(cmd)
