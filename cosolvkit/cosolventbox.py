#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to create the CoSolvent box
#

import itertools
import os
import sys
from operator import itemgetter

import numpy as np
from scipy import spatial

from .cosolvent import CoSolvent
from . import utils


AVOGADRO_CONSTANT_NA = 6.02214179e+23


def _read_pdb(pdb_filename):
    data = []
    dtype = [("name", "U4"), ("resname", "U3"), ("resid", "i4"), ('chain', 'U1'), ("xyz", "f4", (3)), ('is_hydrogen', '?'), ('is_ter', "?")]

    with open(pdb_filename) as f:
        lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith("ATOM") or line.startswith("HETATM"):
                name = line[12:16].strip()

                resname = line[17:20].strip()
                resid = int(line[22:26])
                chain = line[21:22].strip()
                xyz = [np.float(line[30:38]), np.float(line[38:47]), np.float(line[47:55])]
                is_hydrogen = True if name[0] == 'H' else False
                is_ter = True if lines[i + 1].startswith('TER') else False
 
                data.append((name, resname, resid, chain, xyz, is_hydrogen, is_ter))

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

    ids = kdtree.query_ball_point(receptor_xyzs, distance)
    # Keep the unique ids
    ids = np.unique(np.hstack(ids)).astype(np.int)

    close_to = np.zeros(len(wat_xyzs), np.bool)
    close_to[ids] = True

    for i in range(0, wat_xyzs.shape[0], 3):
        close_to[[i, i + 1, i + 2]] = [np.all(close_to[[i, i + 1, i + 2]])] * 3

    return close_to


def _is_in_box(xyzs, box_origin, box_size):
    """Is in the box or not?
    """
    xyzs = np.atleast_2d(xyzs)
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]

    xmin, xmax = box_origin[0], box_origin[0] + box_size[0]
    ymin, ymax = box_origin[1], box_origin[1] + box_size[1]
    zmin, zmax = box_origin[2], box_origin[2] + box_size[2]

    x_in = np.logical_and(xmin <= x, x <= xmax)
    y_in = np.logical_and(ymin <= y, y <= ymax)
    z_in = np.logical_and(zmin <= z, z <= zmax)
    all_in = np.all((x_in, y_in, z_in), axis=0)

    return all_in

    
def _water_is_in_box(wat_xyzs, box_origin, box_size):
    """Check if the water is in the box or not.
    """
    all_in = _is_in_box(wat_xyzs, box_origin, box_size)

    for i in range(0, wat_xyzs.shape[0], 3):
        all_in[[i, i + 1, i + 2]] = [np.all(all_in[[i, i + 1, i + 2]])] * 3

    return all_in


def _atoms_in_box(atom_data, box_origin=None, box_size=None, ignore_peptide_size=4):
    atoms_in_box = []

    if box_origin is not None and box_size is not None:
        all_in = _is_in_box(atom_data['xyz'], box_origin, box_size)
    else:
        all_in = [True] * atom_data.shape[0]

    residues_in_box = np.unique(atom_data[["resid", "chain"]][all_in], axis=0)

    # We want to keep the same chain order as it appears in the pdb file
    _, idx = np.unique(residues_in_box["chain"], return_index=True)
    chain_ids = residues_in_box["chain"][np.sort(idx)]

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
                warning_msg = "Warning: peptide %s will be ignored (minimal size allowed: %d)."
                print(warning_msg % (peptide_resids, ignore_peptide_size))

        # Tag to True residues that are in box based on the chain id and the resid
        atoms_in_box.append((atom_data["chain"] == chain_id) & np.isin(atom_data["resid"], resids_to_keep))

    atoms_in_box = np.any(atoms_in_box, axis=0)
    atom_in_box_ids = np.where(atoms_in_box == True)[0]

    return atom_in_box_ids, protein_terminus, peptides_terminus


def _create_waterbox(box_origin, box_size, receptor_xyzs=None, watref_xyzs=None, watref_dims=None):
    """Create the water box.
    """
    distance_from_receptor = 2.5
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

    # Cut everything that goes outside the box
    in_box = _water_is_in_box(wat_xyzs, box_origin, box_size)
    wat_xyzs = wat_xyzs[in_box]
    
    # Remove water molecules that are too close from the receptor
    if receptor_xyzs is not None:
        too_close_protein = _is_water_close_from_receptor(wat_xyzs, receptor_xyzs, distance_from_receptor)
        wat_xyzs = wat_xyzs[~too_close_protein]

    return wat_xyzs


def _volume_water(n_water):
    """ Compute volume of the box based on the number
    of water molecules. The volume of one water molecule is based
    on the reference water box.
    """
    return ((18.856 * 18.856 * 18.856) / 216) * n_water


def _volume_protein(n_water, box_size):
    vol_box = np.prod(box_size)
    vol_water = _volume_water(n_water)

    #assert vol_water <= vol_box, "The volume of water (%f) is superior than the whole box (%f)." % (vol_water, vol_box)

    return vol_box - vol_water


def _add_cosolvent(wat_xyzs, cosolvents, box_origin, box_size, volume, receptor_xyzs=None, concentration=0.25):
    """Add cosolvent to the water box.
    """
    distance_from_edges = 3.
    distance_from_receptor = 4.5
    distance_from_cosolvent = 1.5
    concentration_water = 55.
    cosolv_xyzs = {name: [] for name in cosolvents}
    cosolv_names = cosolvents.keys()

    # Put aside water molecules that are next to the edges because we don't
    # them to be replaced by cosolvent, otherwise they will go out the box
    too_close_edge = _is_water_close_to_edge(wat_xyzs, distance_from_edges, box_origin, box_size)
    to_keep_wat_xyzs = wat_xyzs[too_close_edge]
    # We will work on those ones
    wat_xyzs = wat_xyzs[~too_close_edge]

    # Do the same also for water molecules too close from the receptor
    if receptor_xyzs is not None:
        too_close_protein = _is_water_close_from_receptor(wat_xyzs, receptor_xyzs, distance_from_receptor)
        to_keep_wat_xyzs = np.vstack((to_keep_wat_xyzs, wat_xyzs[too_close_protein]))
        wat_xyzs = wat_xyzs[~too_close_protein]

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
        to_be_removed = kdtree.query_ball_point(cosolv_xyz, distance_from_cosolvent)
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
            final_concentration = concentration_water / (n_water / (i + 1))

            if final_concentration >= concentration:
                break

    # Add back water molecules we put aside at the beginning
    wat_xyzs = np.vstack((to_keep_wat_xyzs, wat_xyzs))

    return wat_xyzs, cosolv_xyzs, final_concentration


def _apply_neutral_patches(receptor_data, peptides_terminus):
    """Apply neutral patches where the protein was truncated
    
    The angles (140) are a bit weird because something is off in my math or because the normals
    are not completely straight so that might explain why the angles (between 120 and 130, which 
    is okay-ish as a starting point for MD simulations) are different when calculated in PyMOL.

    """
    dtype = [("name", "U4"), ("resname", "U3"), ("resid", "i4"), ('chain', 'U1'), ("xyz", "f4", (3))]

    for chain_id in peptides_terminus.keys():
        for peptide in peptides_terminus[chain_id]:
            if peptide[0] is not None:
                selection = (receptor_data["resid"] == peptide[0]) & (receptor_data["chain"] == chain_id)
                residue_atom_ids = np.where(selection == True)
                residue = receptor_data[residue_atom_ids]
                first_atom_id = np.min(residue_atom_ids)

                if all(np.isin(["N", "CA", "C"], residue["name"])):
                    n_xyz = residue[residue["name"] == "N"]["xyz"][0]
                    ca_xyz = residue[residue["name"] == "CA"]["xyz"][0]
                    c_xyz = residue[residue["name"] == "C"]["xyz"][0]

                    # N atom
                    ca_normal = np.cross(c_xyz - ca_xyz, n_xyz - ca_xyz)
                    n_normal = n_xyz + ca_normal
                    c_ace_xyz = utils.rotate_point(ca_xyz, n_xyz, n_normal, np.degrees(-140.))
                    c_ace_xyz = utils.resize_vector(c_ace_xyz, 1.5, n_xyz)
                    # CH3 atom and O atom
                    c_ace_normal = c_ace_xyz + ca_normal
                    # CH3 atom
                    ch3_ace_xyz = utils.rotate_point(n_xyz, c_ace_xyz, c_ace_normal, np.degrees(140))
                    ch3_ace_xyz = utils.resize_vector(ch3_ace_xyz, 1.3, c_ace_xyz)
                    # O atom
                    o_ace_xyz = utils.rotate_point(n_xyz, c_ace_xyz, c_ace_normal, np.degrees(-140))
                    o_ace_xyz = utils.resize_vector(o_ace_xyz, 1.2, c_ace_xyz)

                    data = [("CH3", "ACE", peptide[0] - 1, chain_id, ch3_ace_xyz),
                            ("C", "ACE", peptide[0] - 1, chain_id, c_ace_xyz),
                            ("O", "ACE", peptide[0] - 1, chain_id, o_ace_xyz)]
                    ace_residue = np.array(data, dtype=dtype)
                    
                    receptor_data = np.insert(receptor_data, first_atom_id, ace_residue, axis=0)
                else:
                    print("Warning: Cannot apply neutral patch ACE on residue %d:%s" % (peptide[0], chain_id))

            if peptide[1] is not None:
                selection = (receptor_data["resid"] == peptide[1]) & (receptor_data["chain"] == chain_id)
                residue_atom_ids = np.where(selection == True)
                residue = receptor_data[residue_atom_ids]
                last_atom_id = np.max(residue_atom_ids)

                if all(np.isin(["CA", "C", "O"], residue["name"])):
                    ca_xyz = residue[residue["name"] == "CA"]["xyz"][0]
                    c_xyz = residue[residue["name"] == "C"]["xyz"][0]
                    o_xyz = residue[residue["name"] == "O"]["xyz"][0]

                    # N atom
                    n_nme_xyz = utils.atom_to_move(c_xyz, [ca_xyz, o_xyz])
                    n_nme_xyz = utils.resize_vector(n_nme_xyz, 1.3, c_xyz)
                    # CH3 atom
                    c_normal = np.cross(ca_xyz - c_xyz, o_xyz - c_xyz)
                    n_nme_normal = n_nme_xyz + c_normal
                    ch3_nme_xyz = utils.rotate_point(c_xyz, n_nme_xyz, n_nme_normal, np.degrees(140))
                    ch3_nme_xyz = utils.resize_vector(ch3_nme_xyz, 1.5, n_nme_xyz)

                    data = [("N", "NME", peptide[1] + 1, chain_id, n_nme_xyz),
                            ("CH3", "NME", peptide[1] + 1, chain_id, ch3_nme_xyz)]
                    nme_residue = np.array(data, dtype=dtype)

                    # last_atom_id + 1 because we want to insert the new residue after the last atom
                    receptor_data = np.insert(receptor_data, last_atom_id + 1, nme_residue, axis=0)
                else:
                    print("Warning: Cannot apply neutral patch NME on residue %d:%s" % (peptide[1], chain_id))

    return receptor_data


class CoSolventBox:

    def __init__(self, concentration=0.25, cutoff=12, box="cubic", center=None, box_size=None, min_size_peptide=4):
        """Initialize the cosolvent box
        """
        assert box in ["cubic", "orthorombic"], "Error: the water box can be only cubic or orthorombic."

        self._concentration = concentration
        self._cutoff = cutoff
        self._min_size_peptide = min_size_peptide
        self._box = box
        self._use_existing_waterbox = False
        self._receptor_data = None
        self._water_data = None
        self._receptor_atom_in_box_ids = []
        self._peptides_terminus = {}
        self._cosolvents = {}
        self._wat_xyzs = None
        self._cosolv_xyzs = None
        self._pdb_filename = None

        if center is not None and box_size is not None:
            box_size = np.array(box_size)

            # Check center
            assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
            # Check gridsize
            assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
            assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."

            self._center = center
            # It's easier to work with integers for grid size
            self._box_size = np.ceil(box_size).astype(np.int)
            self._origin = self._center - (self._box_size  / 2.)
        elif (center is not None and box_size is None) or (center is None and box_size is not None):
            print("Error: cannot define the size of the grid without defining its center. Et vice et versa !")
            sys.exit(1)
        else:
            self._center = None
            self._box_size = None
            self._origin = None

        # Read the reference water box
        d = utils.path_module("cosolvkit")
        waterbox_filename = os.path.join(d, "data/waterbox.pdb")
        self._watref_xyzs = _read_pdb(waterbox_filename)['xyz']
        self._watref_dims = [18.856, 18.856, 18.856]

    def add_receptor(self, receptor_filename, use_existing_waterbox=False):
        """Add receptor
        """
        cutoff = self._cutoff
        need_to_truncate = False
        self._use_existing_waterbox = use_existing_waterbox
        self._receptor_filename = receptor_filename

        system_data = _read_pdb(receptor_filename)
        # Separate water molecules from the receptor (protein, ions, membrane, etc...)
        self._receptor_data = system_data[(system_data['resname'] != 'WAT') & (system_data['resname'] != 'HOH')]
        self._water_data = system_data[(system_data['resname'] == 'WAT') | (system_data['resname'] == 'HOH')]

        if self._use_existing_waterbox:
            assert self._center is None and self._box_size is None, 'Error: cannot define center and dimensions when using an existing waterbox.'
            assert self._water_data.shape[0] > 0, 'Error: no water molecules present in the existing waterbox.'

            # If we have an existing waterbox, it is more accurate to use water molecules (only oxygen) to 
            # get the right box dimensions. In the presence of a lipid membrane we can have lipids sticking out the box.
            water_oxygen = self._water_data[(self._water_data['is_hydrogen'] == False)]

            # We assume that all the receptor atoms are in the box when using a pre-existing waterbox
            self._receptor_atom_in_box_ids = list(range(self._receptor_data.shape[0]))

            xmin = np.min(water_oxygen['xyz'][:, 0])
            xmax = np.max(water_oxygen['xyz'][:, 0])
            ymin = np.min(water_oxygen['xyz'][:, 1])
            ymax = np.max(water_oxygen['xyz'][:, 1])
            zmin = np.min(water_oxygen['xyz'][:, 2])
            zmax = np.max(water_oxygen['xyz'][:, 2])
        else:
            # We want to identify all the atoms (per residue) that are in the box.
            # Also knowing where are the true and artificial N and C terminus will be useful
            # after for adding the neutral patches
            results = _atoms_in_box(self._receptor_data, self._origin, self._box_size, self._min_size_peptide)
            self._receptor_atom_in_box_ids, _, self._peptides_terminus = results

            if self._receptor_data.shape[0] != len(self._receptor_atom_in_box_ids):
                need_to_truncate = True

            atoms_in_box = self._receptor_data[self._receptor_atom_in_box_ids]

            xmin = np.min(atoms_in_box['xyz'][:, 0]) - cutoff
            xmax = np.max(atoms_in_box['xyz'][:, 0]) + cutoff
            ymin = np.min(atoms_in_box['xyz'][:, 1]) - cutoff
            ymax = np.max(atoms_in_box['xyz'][:, 1]) + cutoff
            zmin = np.min(atoms_in_box['xyz'][:, 2]) - cutoff
            zmax = np.max(atoms_in_box['xyz'][:, 2]) + cutoff

        # _origin is instanciated only when _center and _box_size are also instanciated
        # That's why we just have to verify that _origin is None
        # If we need to truncate the protein, we have to redefine the box
        # definition to fit the truncated protein
        if self._origin is None or need_to_truncate:
            if self._box == "orthorombic" or use_existing_waterbox:
                self._box_size = np.ceil(np.array([xmax - xmin, ymax - ymin, zmax - zmin])).astype(np.int)
            else:
                lmax = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
                self._box_size = np.ceil(np.array([lmax, lmax, lmax])).astype(np.int)

            self._center = np.mean([[xmin, ymin, zmin], [xmax, ymax, zmax]], axis=0)
            self._origin = self._center - (self._box_size / 2)
                
    def add_cosolvent(self, name, smiles=None, mol2_filename=None, lib_filename=None, frcmod_filename=None, charge=0, resname=None):
        """Add cosolvent and parametrize it
        """
        c = CoSolvent(name, smiles, mol2_filename, lib_filename, frcmod_filename, charge, resname)
        self._cosolvents[name] = c
    
    def build(self):
        """Build the cosolvent box
        """
        if self._origin is not None:
            if self._receptor_data is not None:
                receptor_xyzs = self._receptor_data[self._receptor_atom_in_box_ids]['xyz']
            else:
                receptor_xyzs = None

            if self._use_existing_waterbox:
                self._wat_xyzs = self._water_data['xyz']
            else:
                self._wat_xyzs = _create_waterbox(self._origin, self._box_size, receptor_xyzs,
                                                  self._watref_xyzs, self._watref_dims)

            n_water = np.int(self._wat_xyzs.shape[0] / 3)
            self._volume = _volume_water(n_water)
            volume_protein = _volume_protein(n_water, self._box_size)

            print("------------------------------------")
            print("Volume box: %10.4f A**3" % self._volume)
            if self._receptor_data is not None:
                print("Volume protein (box - water): %10.4f A**3" % volume_protein)
            print("Water (before cosolvent): %d" % n_water)
            if self._use_existing_waterbox:
                print("Box type: pre-existing")
            else:
                print("Box type: %s" % self._box)
            print("Box center: %8.3f %8.3f %8.3f" % (self._center[0], self._center[1], self._center[2]))
            print("Box dimensions: x %d y %d z %d (A)" % (self._box_size[0], self._box_size[1], self._box_size[2]))

            if self._cosolvents:
                wat_xyzs, cosolv_xyzs, final_concentration = _add_cosolvent(self._wat_xyzs, self._cosolvents,
                                                                            self._origin, self._box_size, self._volume,
                                                                            receptor_xyzs, self._concentration)

                self._wat_xyzs = wat_xyzs
                self._cosolv_xyzs = cosolv_xyzs
                n_water = np.int(self._wat_xyzs.shape[0] / 3)

                print("")
                print("Target concentration (M): %5.3f" % self._concentration)
                print("Final concentration (M): %5.3f" % final_concentration)
                print("Water (WAT): %3d" % (n_water))
                for cosolv_name in self._cosolvents:
                    print("%s (%s): %3d" % (cosolv_name.capitalize(), 
                                            self._cosolvents[cosolv_name].resname,
                                            len(self._cosolv_xyzs[cosolv_name])))

        else:
            print("Error: box dimensions was not defined.")
            sys.exit(1)

        print("------------------------------------")

    def export_pdb(self, filename='cosolv_system.pdb'):
        """Export pdb file for tleap
        """
        n_atom = 0
        n_residue = 1
        n_atom_water = 1
        self._pdb_filename = filename
        # We get ride of the segid, otherwise the number of atoms cannot exceed 9.999
        template = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:>4s}{:>2s}{:2s}\n"

        if self._receptor_data is not None:
            receptor_data = self._receptor_data[self._receptor_atom_in_box_ids]
            receptor_data = _apply_neutral_patches(receptor_data, self._peptides_terminus)
            resid_peptide = self._receptor_data[0]['resid']
        else:
            receptor_data = None
            resid_peptide = 0

        # Write pdb file
        with open(self._pdb_filename, 'w') as w:
            # Write protein first
            if receptor_data is not None:
                for i, atom in enumerate(receptor_data):
                    x, y, z = atom['xyz']

                    # Special case when the atom types is 4 caracters long
                    if len(atom['name']) <= 3:
                        name = ' ' + atom['name']
                    else:
                        name = atom['name']

                    w.write(template.format("ATOM", (i + 1) % 100000, name, " ",
                                            atom['resname'], atom['chain'], atom['resid'] % 10000,
                                            " ", x, y, z, 0., 0., " ", atom['name'][0], " "))

                    try:
                        # We are looking for gap in the sequence, maybe due to the truncation
                        if (atom['resid'] != receptor_data[i + 1]['resid']) and (atom['resid'] + 1 != receptor_data[i + 1]['resid']):
                            w.write("TER\n")

                            if (atom['resid'] - resid_peptide + 1) < 14:
                                warning_msg = "Warning: Isolated short peptide of length %d, resid %d to %d."
                                print(warning_msg % (atom['resid'] - resid_peptide + 1, resid_peptide, atom['resid']))

                                resid_peptide = receptor_data[i + 1]['resid']
                    except:
                        continue

                    if atom['is_ter']:
                        w.write('TER\n')

            # Write cosolvent molecules
            if self._cosolv_xyzs is not None:
                for name in self._cosolvents:
                    cosolv_xyzs = self._cosolv_xyzs[name]
                    resname = self._cosolvents[name].resname
                    atom_names = self._cosolvents[name].atom_names

                    for residue_xyzs in cosolv_xyzs:
                        for atom_xyz, atom_name in zip(residue_xyzs, atom_names):
                            x, y, z = atom_xyz
                            w.write(template.format("ATOM", n_atom % 100000, atom_name, " ",
                                                    resname, " ", n_residue  % 10000,
                                                    " ", x, y, z, 0., 0., resname, atom_name[0], " "))
                            n_atom += 1
                        n_residue += 1

                        w.write("TER\n")

            # Write water molecules
            if self._wat_xyzs is not None:
                water_atom_names = ["O", "H1", "H2"] * int(self._wat_xyzs.shape[0] / 3)

                # And water molecules at the end
                for wat_xyz, atom_name in zip(self._wat_xyzs, water_atom_names):
                    x, y, z = wat_xyz
                    w.write(template.format("ATOM", n_atom % 100000, atom_name, " ",
                                            'WAT', " ", n_residue % 10000,
                                            " ", x, y, z, 0., 0., "WAT", atom_name[0], " "))

                    if n_atom_water % 3 == 0:
                        n_residue += 1

                    n_atom_water += 1
                    n_atom += 1

                w.write('TER\n')

            w.write("END\n")

    def write_tleap_input(self, filename='tleap.cmd', prmtop_filename='cosolv_system.prmtop', 
                          inpcrd_filename='cosolv_system.inpcrd', protein_ff='ff19SB', dna_ff='OL15', rna_ff='OL3',
                          glycam_ff='GLYCAM_06j-1', lipid_ff='lipid21', water_ff='tip3p', gaff='gaff2'):

        # Create tleap template
        TLEAP_TEMPLATE = ("source leaprc.protein.%s\n"
                          "source leaprc.DNA.%s\n"
                          "source leaprc.RNA.%s\n"
                          "source leaprc.%s\n"
                          "source leaprc.%s\n"
                          "source leaprc.water.%s\n"
                          "source leaprc.%s\n")
        TLEAP_TEMPLATE = TLEAP_TEMPLATE % (protein_ff, dna_ff, rna_ff, glycam_ff, lipid_ff, water_ff, gaff)

        if self._cosolvents is not None:
            for name in self._cosolvents:
                frcmod_filename = os.path.basename(self._cosolvents[name]._frcmod_filename)
                lib_filename = os.path.basename(self._cosolvents[name]._lib_filename)

                TLEAP_TEMPLATE += "loadamberparams %s\n" % frcmod_filename
                TLEAP_TEMPLATE += "loadoff %s\n" % lib_filename

        TLEAP_TEMPLATE += "set default nocenter on\n"
        TLEAP_TEMPLATE += "m = loadpdb %s\n" % self._pdb_filename
        if self._wat_xyzs is not None:
            TLEAP_TEMPLATE += "charge m\n"
            TLEAP_TEMPLATE += "addIonsRand m Cl- 0\n"
            TLEAP_TEMPLATE += "addIonsRand m Na+ 0\n"
            TLEAP_TEMPLATE += "check m\n"
        TLEAP_TEMPLATE += "set m box {%d %d %d}\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
        TLEAP_TEMPLATE += "saveamberparm m %s %s\n" % (prmtop_filename, inpcrd_filename)
        TLEAP_TEMPLATE += "quit\n"

        with open(filename, 'w') as w:
            w.write(TLEAP_TEMPLATE)

        # cmd = 'tleap -f tleap.cmd'
        # outputs, errors = utils.execute_command(cmd)
