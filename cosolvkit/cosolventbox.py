#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to create the CoSolvent box
#

import itertools
import os
import string
import sys
from operator import itemgetter

import numpy as np
from scipy import spatial

from .cosolvent import CoSolvent
from . import utils


AVOGADRO_CONSTANT_NA = 6.02214179e+23
digits_upper = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
digits_lower = digits_upper.lower()
digits_upper_values = dict([pair for pair in zip(digits_upper, range(36))])
digits_lower_values = dict([pair for pair in zip(digits_lower, range(36))])


def encode_pure(digits, value):
    "encodes value using the given digits"
    assert value >= 0
    
    if (value == 0): return digits[0]
    n = len(digits)
    
    result = []
    
    while (value != 0):
        rest = value // n
        result.append(digits[value - rest * n])
        value = rest
    
    result.reverse()
    
    return "".join(result)


def hy36encode(width, value):
    "encodes value as base-10/upper-case base-36/lower-case base-36 hybrid"
    
    i = value
    
    if (i >= 1-10**(width-1)):
        if (i < 10**width):
            return ("%%%dd" % width) % i
        i -= 10**width
        if (i < 26*36**(width-1)):
            i += 10*36**(width-1)
            return encode_pure(digits_upper, i)
        i -= 26*36**(width-1)
        if (i < 26*36**(width-1)):
            i += 10*36**(width-1)
            return encode_pure(digits_lower, i)
    
    raise ValueError("value out of range.")


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
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
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
    ids = np.unique(np.hstack(ids)).astype(int)

    close_to = np.zeros(len(wat_xyzs), bool)
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


def _atoms_in_box(atom_data, box_origin=None, box_size=None, min_size_peptide=10):
    atoms_in_box = []
    protein_terminus = {}
    peptides_terminus = {}

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

            if not len(peptide_resids) < min_size_peptide:
                resids_to_keep.extend(peptide_resids)

                # We need that information to know where to put the neutral patchs
                if peptide_resids[0] != protein_terminus[chain_id][0]:
                    peptide_terminus[0] = peptide_resids[0]
                if peptide_resids[-1] != protein_terminus[chain_id][1]:
                    peptide_terminus[1] = peptide_resids[-1]

                peptides_terminus[chain_id].append(peptide_terminus)
            else:
                warning_msg = "Warning: peptide %s will be ignored (minimum size allowed: %d)."
                print(warning_msg % (peptide_resids, min_size_peptide))

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


def _add_cosolvent_as_concentrations(wat_xyzs, cosolvents, box_origin, box_size, target_concentrations, receptor_xyzs=None):
    """Add cosolvent to the water box based on the target concentrations.

    Parameters
    ----------
    wat_xyzs : np.ndarray
        Coordinates of the water molecules.
    cosolvents : dict
        Dictionary of cosolvents. The keys are the names of the cosolvents and
        the values are the coordinates of the cosolvent molecules.
    box_origin : np.ndarray
        Coordinates of the origin of the box.
    box_size : np.ndarray
        Size of the box.
    target_concentrations : dict
        Dictionary of target concentrations. The keys are the names of the cosolvents
        and the values are the target concentrations.
    receptor_xyzs : np.ndarray
        Coordinates of the receptor.

    Returns
    -------
    wat_xyzs : np.ndarray
        Coordinates of the water molecules.
    cosolv_xyzs : dict
        Dictionary of cosolvent molecules. The keys are the names of the cosolvents
        and the values are the coordinates of the cosolvent molecules.
    final_concentrations : dict
        Dictionary of final concentrations. The keys are the names of the cosolvents
        and the values are the final concentrations.

    """
    distance_from_edges = 3.
    distance_from_receptor = 4.5
    distance_from_cosolvent = 2.5
    concentration_water = 55.
    cosolv_xyzs = {name: [] for name in cosolvents}
    cosolv_names = cosolvents.keys()
    current_number_copies = {}
    final_number_copies = {}
    final_concentrations = {}

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

    # Calculate the number of copies of each cosolvent according to the target concentration
    n_water = wat_xyzs.shape[0] / 3
    for cosolv_name in cosolv_names:
        current_number_copies[cosolv_name] = 0
        final_concentrations[cosolv_name] = 0
        # 1 cosolvent molecule per 55 waters correspond to a concentration of 1 M
        final_number_copies[cosolv_name] = int((target_concentrations[cosolv_name] / concentration_water) * n_water)

    # Generate a random placement order of cosolvents
    placement_order = []
    for cosolv_name, n in final_number_copies.items():
        placement_order += [cosolv_name] * n
    np.random.shuffle(placement_order)

    for cosolv_name in itertools.cycle(placement_order):
        # Update kdtree
        kdtree = spatial.cKDTree(wat_xyzs)

        if final_concentrations[cosolv_name] <= target_concentrations[cosolv_name]:
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
            mask = np.ones(len(wat_xyzs), bool)
            mask[to_be_removed] = 0
            wat_xyzs = wat_xyzs[mask]

            # Compute the new concentration for that cosolvent
            n_water = (to_keep_wat_xyzs.shape[0] + wat_xyzs.shape[0]) / 3
            current_number_copies[cosolv_name] += 1
            final_concentrations[cosolv_name] = concentration_water / (n_water / (current_number_copies[cosolv_name]))

        # Stop when the target concentration is reached for all cosolvents
        if all([final_concentrations[cosolv_name] >= target_concentrations[cosolv_name] for cosolv_name in cosolv_names]):
            break

    # Add back water molecules we put aside at the beginning
    wat_xyzs = np.vstack((to_keep_wat_xyzs, wat_xyzs))

    return wat_xyzs, cosolv_xyzs, final_concentrations


def _add_cosolvent_as_copies(wat_xyzs, cosolvents, box_origin, box_size, target_number_copies, center_positions=None, receptor_xyzs=None):
    """Add cosolvent to the water box based on the number of copies requested.

    Parameters
    ----------
    wat_xyzs : np.ndarray
        Coordinates of the water molecules.
    cosolvents : dict
        Dictionary of cosolvents. The keys are the names of the cosolvents and
        the values are the coordinates of the cosolvent molecules.
    box_origin : np.ndarray
        Coordinates of the origin of the box.
    box_size : np.ndarray
        Size of the box.
    target_number_copies : dict
        Dictionary of target number of copies. The keys are the names of the cosolvents
        and the values are the target copy numbers.
    center_positions : np.ndarray
        Coordinates of the center of each cosolvent to be added.
    receptor_xyzs : np.ndarray
        Coordinates of the receptor.

    Returns
    -------
    wat_xyzs : np.ndarray
        Coordinates of the water molecules.
    cosolv_xyzs : dict
        Dictionary of cosolvent molecules. The keys are the names of the cosolvents
        and the values are the coordinates of the cosolvent molecules.
    final_number_copies : dict
        Dictionary of final number of copies. The keys are the names of the cosolvents
        and the values are the final copy numbers that was added.

    """
    distance_from_edges = 3.
    distance_from_receptor = 4.5
    distance_from_cosolvent = 2.5
    cosolv_xyzs = {name: [] for name in cosolvents}
    final_number_copies = {}

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

    # Generate a random placement order of cosolvents
    placement_order = []
    for cosolv_name, n in target_number_copies.items():
        final_number_copies[cosolv_name] = 0
        placement_order += [cosolv_name] * n
    np.random.shuffle(placement_order)

    for cosolv_name in placement_order:
        # Update kdtree
        kdtree = spatial.cKDTree(wat_xyzs)

        if center_positions[cosolv_name] is None:
            # Choose a random water molecule
            wat_o = wat_xyzs[::3]
            wat_id = np.random.choice(range(0, wat_o.shape[0]))
            wat_xyz = wat_o[wat_id]

            # Translate fragment on the top of the selected water molecule
            cosolv_xyz = cosolvents[cosolv_name].positions + wat_xyz
        else:
            center = center_positions[cosolv_name][final_number_copies[cosolv_name]]
            cosolv_xyz = cosolvents[cosolv_name].positions + center

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
        mask = np.ones(len(wat_xyzs), bool)
        mask[to_be_removed] = 0
        wat_xyzs = wat_xyzs[mask]

        # Increment current number of copies of that cosolvent
        final_number_copies[cosolv_name] += 1

    # Add back water molecules we put aside at the beginning
    wat_xyzs = np.vstack((to_keep_wat_xyzs, wat_xyzs))

    return wat_xyzs, cosolv_xyzs, final_number_copies


def _apply_neutral_patches(receptor_data, peptides_terminus):
    """Apply neutral patches where the protein was truncated
    
    The angles (140) are a bit weird because something is off in my math or because the normals
    are not completely straight so that might explain why the angles (between 120 and 130, which 
    is okay-ish as a starting point for MD simulations) are different when calculated in PyMOL.

    """
    dtype = [("name", "U4"), ("resname", "U3"), ("resid", "i4"), ('chain', 'U1'), ("xyz", "f4", (3)), ('is_hydrogen', '?'), ('is_ter', "?")]

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

                    data = [("CH3", "ACE", peptide[0] - 1, chain_id, ch3_ace_xyz, False, False),
                            ("C", "ACE", peptide[0] - 1, chain_id, c_ace_xyz, False, False),
                            ("O", "ACE", peptide[0] - 1, chain_id, o_ace_xyz, False, False)]
                    ace_residue = np.array(data, dtype=dtype)

                    # The atom before the N-ter patch becomes automatically a TER atom
                    receptor_data[first_atom_id - 1]['is_ter'] = True
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
                    # c_normal = np.cross(ca_xyz - c_xyz, o_xyz - c_xyz)
                    # n_nme_normal = n_nme_xyz + c_normal
                    # ch3_nme_xyz = utils.rotate_point(c_xyz, n_nme_xyz, n_nme_normal, np.degrees(140))
                    # ch3_nme_xyz = utils.resize_vector(ch3_nme_xyz, 1.5, n_nme_xyz)

                    """
                    data = [("N", "NHE", peptide[1] + 1, chain_id, n_nme_xyz, False, False),
                            ("CH3", "NME", peptide[1] + 1, chain_id, ch3_nme_xyz, False, True)]
                    """

                    # Issue in tleap with CH3 atom in NME residue, so we just write the first
                    # atom, and let tleap reconstructs the whole thing...
                    data = [("N", "NME", peptide[1] + 1, chain_id, n_nme_xyz, False, True)]
                    nme_residue = np.array(data, dtype=dtype)

                    # last_atom_id + 1 because we want to insert the new residue after the last atom
                    receptor_data = np.insert(receptor_data, last_atom_id + 1, nme_residue, axis=0)
                else:
                    print("Warning: Cannot apply neutral patch NHE on residue %d:%s" % (peptide[1], chain_id))

    return receptor_data


def _generate_pdb(receptor_data=None, cosolvents=None, cosolv_xyzs=None, wat_xyzs=None):
    """Generate the pdb string in hybrid-36 PDB format.

    The output PDB follows the hybrid36 format in order to handle >100k atoms, which means 
    that the atom/residue numbers are encoded in base 36. See https://cci.lbl.gov/hybrid_36/ 
    and https://cci.lbl.gov/hybrid_36/pdb_format_evolution.pdf for more information about
    this particular PDB format. This format is supported by MDAnalysis, OpenMM, Pymol, VMD, etc.

    Parameters
    ----------
    receptor_data : np.ndarray
        Coordinates of the receptor.
    cosolvents : dict
        Dictionary of cosolvents. The keys are the names of the cosolvents and
        the values are the coordinates of the cosolvent molecules.
    cosolv_xyzs : dict
        Dictionary of cosolvent molecules. The keys are the names of the cosolvents
        and the values are the coordinates of the cosolvent molecules.
    wat_xyzs : np.ndarray
        Coordinates of the water molecules.

    Returns
    -------
    pdb_string : str
        The output pdb string (hybrid-36 PDB format).

    """
    n_atom = 1
    receptor_n_atom = 0
    chain_alphabet = list(string.ascii_uppercase)
    pdb_string = ""
    pdb_conects = []
    # We get ride of the segid, otherwise the number of atoms cannot exceed 9.999
    template = "{:6s}{:>5s} {:^4s}{:1s}{:3s} {:1s}{:>4s}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}      {:>4s}{:>2s}{:2s}\n"

    # Write protein first
    if receptor_data is not None:
        receptor_n_atom = receptor_data.shape[0]

        for i, atom in enumerate(receptor_data):
            x, y, z = atom['xyz']

            # Special case when the atom types is 4 caracters long
            if len(atom['name']) <= 3:
                name = ' ' + atom['name']
            else:
                name = atom['name']

            pdb_string += template.format("ATOM", hy36encode(5, n_atom), name, " ",
                                          atom['resname'], atom['chain'], hy36encode(4, atom['resid']),
                                          " ", x, y, z, 0., 0., " ", atom['name'][0], " ")

            if atom['is_ter']:
                pdb_string += 'TER\n'
            
            n_atom += 1
    
    # We look for the highest chain index in the receptor, in 
    # case chains are not in alphabetical order. And we add 1.
    current_chain_index = chain_alphabet.index(max(receptor_data['chain'])) + 1

    # Write cosolvent molecules
    if cosolv_xyzs is not None:
        cosolvent_n_atom = 0

        for name in cosolvents:
            n_residue = 1
            selected_cosolv_xyzs = cosolv_xyzs[name]
            resname = cosolvents[name].resname
            atom_names = cosolvents[name].atom_names
            n_atoms = len(atom_names)
            pdb_conect = cosolvents[name].pdb_conect

            for i, residue_xyzs in enumerate(selected_cosolv_xyzs):
                for atom_xyz, atom_name in zip(residue_xyzs, atom_names):
                    x, y, z = atom_xyz
                    pdb_string += template.format("HETATM", hy36encode(5, n_atom), atom_name, " ",
                                                  resname, chain_alphabet[current_chain_index], hy36encode(4, n_residue),
                                                  " ", x, y, z, 0., 0., resname, atom_name[0], " ")
                    n_atom += 1

                # CONECT records
                pdb_conects.append([(c + (i * n_atoms)) + receptor_n_atom + cosolvent_n_atom for c in pdb_conect])

                n_residue += 1
                pdb_string += "TER\n"

            cosolvent_n_atom += len(selected_cosolv_xyzs) * n_atoms
            current_chain_index += 1

    # Write water molecules
    if wat_xyzs is not None:
        n_residue = 1
        n_atom_water = 1
        water_atom_names = ["O", "H1", "H2"] * int(wat_xyzs.shape[0] / 3)

        # And water molecules at the end
        for wat_xyz, atom_name in zip(wat_xyzs, water_atom_names):
            x, y, z = wat_xyz
            pdb_string += template.format("HETATM", hy36encode(5, n_atom), atom_name, " ",
                                          'WAT', chain_alphabet[current_chain_index], hy36encode(4, n_residue),
                                          " ", x, y, z, 0., 0., "WAT", atom_name[0], " ")

            if n_atom_water % 3 == 0:
                n_residue += 1
                pdb_string += 'TER\n'

            n_atom_water += 1
            n_atom += 1
    
    if pdb_conects:
        for pdb_conect in pdb_conects:
            for conect in pdb_conect:
                pdb_string += "CONECT" + "".join(["{:>5d}".format(c) for c in conect]) + "\n"

    pdb_string += "END\n"

    return pdb_string


class CoSolventBox:

    def __init__(self, box="cubic", cutoff=12, center=None, box_size=None,
                 keep_existing_water=False, use_existing_waterbox=False, min_size_peptide=10):
        """Initialize the cosolvent box

        Parameters
        ----------
        box : str
            The type of box to use. Can be either "cubic" or "orthorombic"
        cutoff : float
            The cutoff distance in Angstroms to define the cosolvent box
        center : np.ndarray
            The center of the box in Angstroms
        box_size : np.ndarray
            The size of the box in Angstroms
        keep_existing_water : bool
            If True, the water molecules in the receptor will be kept
        use_existing_waterbox : bool
            If True, the water molecules in the receptor will be kept and the box will be defined
            by the receptor water molecules
        min_size_peptide : int
            The minimum number of residues in the peptide to keep the water molecules

        Raises
        ------
        AssertionError
            If the box is not "cubic" or "orthorombic"
        AssertionError
            If both keep_existing_water and use_existing_waterbox arguments are set to True

        """
        if not use_existing_waterbox:
            assert box in ["cubic", "orthorombic"], "Error: the water box can be only cubic or orthorombic."

        error_msg = "Error: both keep_existing_water and use_existing_waterbox arguments cannot be set to True"
        assert not(keep_existing_water is True and use_existing_waterbox is True), error_msg

        self._cutoff = cutoff
        self._min_size_peptide = min_size_peptide
        self._box = box
        self._use_existing_waterbox = use_existing_waterbox
        self._keep_existing_water = keep_existing_water
        self._receptor_data = None
        self._water_data = None
        self._cosolvents = {}
        self._concentrations = {}
        self._copies = {}
        self._center_positions = {}
        self._added_as_concentrations = []
        self._added_as_copies = []
        self._wat_xyzs = None
        self._cosolv_xyzs = {}
        self._pdb_filename = None

        if center is not None and box_size is not None:
            center = np.asarray(center)
            box_size = np.asarray(box_size)

            # Check center
            assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
            # Check gridsize
            assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
            assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."

            self._center = center
            # It's easier to work with integers for grid size
            self._box_size = np.ceil(box_size).astype(int)
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

    def add_receptor(self, receptor_filename):
        """Add receptor

        Parameters
        ----------
        receptor_filename : str
            PDB filename of the receptor

        """
        receptor_truncated = False

        system_data = _read_pdb(receptor_filename)

        # Separate water molecules from the receptor (protein, ions, membrane, etc...)
        # except if we decided to keep the water molecules already present
        if self._keep_existing_water:
            self._receptor_data = system_data.copy()
        else:
            self._receptor_data = system_data[(system_data['resname'] != 'WAT') & (system_data['resname'] != 'HOH')]

        self._water_data = system_data[(system_data['resname'] == 'WAT') | (system_data['resname'] == 'HOH')]

        if self._use_existing_waterbox:
            assert self._center is None and self._box_size is None, 'Error: cannot define center and dimensions when using an existing waterbox.'
            assert self._water_data.shape[0] > 0, 'Error: no water molecules present in the existing waterbox.'

            # If we have an existing waterbox, it is more accurate to use water molecules (only oxygen) to 
            # get the right box dimensions. In the presence of a lipid membrane we can have lipids sticking out the box.
            water_oxygen = self._water_data[(self._water_data['is_hydrogen'] == False)]

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
            receptor_atom_in_box_ids, _, peptides_terminus = results

            if self._receptor_data.shape[0] != len(receptor_atom_in_box_ids):
                receptor_truncated = True
                # Select only the atoms (residues are complete) that are in the box, and apply neutral patches
                self._receptor_data = self._receptor_data[receptor_atom_in_box_ids]
                self._receptor_data = _apply_neutral_patches(self._receptor_data, peptides_terminus)

            xmin = np.min(self._receptor_data['xyz'][:, 0]) - self._cutoff
            xmax = np.max(self._receptor_data['xyz'][:, 0]) + self._cutoff
            ymin = np.min(self._receptor_data['xyz'][:, 1]) - self._cutoff
            ymax = np.max(self._receptor_data['xyz'][:, 1]) + self._cutoff
            zmin = np.min(self._receptor_data['xyz'][:, 2]) - self._cutoff
            zmax = np.max(self._receptor_data['xyz'][:, 2]) + self._cutoff

        # _origin is instanciated only when _center and _box_size are also instanciated
        # That's why we just have to verify that _origin is None
        # If we need to truncate the protein, we have to redefine the box
        # definition to fit the truncated protein
        if self._origin is None or receptor_truncated:
            if self._box == "orthorombic" or self._use_existing_waterbox:
                self._box_size = np.ceil(np.array([xmax - xmin, ymax - ymin, zmax - zmin])).astype(int)
            else:
                lmax = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
                self._box_size = np.ceil(np.array([lmax, lmax, lmax])).astype(int)

            self._center = np.mean([[xmin, ymin, zmin], [xmax, ymax, zmax]], axis=0)
            self._origin = self._center - (self._box_size / 2)

    def add_cosolvent(self, name, concentration=None, copies=None, smiles=None, mol_filename=None, center_positions=None, resname=None):
        """Add cosolvent and parametrize it

        Parameters
        ----------
        name : str
            Name of the cosolvent
        concentration : float, optional
            Concentration of the cosolvent in M
        copies : int, optional
            Number of copies of the cosolvent to add
        smiles : str, optional
            SMILES string of the cosolvent
        mol_filename : str, optional
            Name of the mol2 file containing the cosolvent
        center_positions : np.ndarray, optional
            Center positions of each cosolvent molecule added
        resname : str, optional
            Residue name of the cosolvent

        """
        assert concentration is not None or copies is not None, 'Either concentration or copies must be defined.'
        assert not all([concentration is not None, copies is not None]), 'Either concentration or copies must be defined, not both.'

        if concentration is not None:
            assert concentration > 0, 'Concentration must be positive.'

        if copies is not None:
            assert copies > 0, 'Number of copies must be positive.'

        if copies is not None and center_positions is not None:
            center_positions = np.asarray(center_positions)
            assert copies == center_positions.shape[0], 'Copies and positions must have the same length.'
            assert center_positions.shape[1] == 3, 'Center positions must be 3D.'

        c = CoSolvent(name, smiles, mol_filename, resname)
        self._cosolvents[name] = c

        if concentration is not None:
            self._added_as_concentrations.append(name)
            copies = 0
        else:
            self._added_as_copies.append(name)
            concentration = 0

        self._concentrations[name] = concentration
        self._copies[name] = copies
        self._center_positions[name] = center_positions
    
    def build(self):
        """Build the cosolvent box. It involves the following steps:

        1. Create the water box (if not using an existing one)
        2. Add cosolvents as copies or concentrations

        """
        if self._origin is not None:
            if self._receptor_data is not None:
                receptor_xyzs = self._receptor_data['xyz']
            else:
                receptor_xyzs = None

            if self._use_existing_waterbox:
                self._wat_xyzs = self._water_data['xyz']
            else:
                self._wat_xyzs = _create_waterbox(self._origin, self._box_size, receptor_xyzs,
                                                  self._watref_xyzs, self._watref_dims)

            n_water = int(self._wat_xyzs.shape[0] / 3)
            self._water_volume = _volume_water(n_water)
            volume_protein = _volume_protein(n_water, self._box_size)

            print("----------------------------------------------")
            print("Volume box                   : %10.3f A**3" % (self._box_size[0] * self._box_size[1] * self._box_size[2]))
            print("Volume water                 : %10.3f A**3" % self._water_volume)
            if self._receptor_data is not None:
                print("Volume protein (box - water) : %10.3f A**3" % volume_protein)
            print("Water (before cosolvent)     : %d" % n_water)
            if self._use_existing_waterbox:
                print("Box type                     : pre-existing")
            else:
                print("Box type                     : %s" % self._box)
            print("Box center                   : x %8.3f y %8.3f z %8.3f (A)" % (self._center[0], self._center[1], self._center[2]))
            print("Box dimensions               : x %8d y %8d z %8d (A)" % (self._box_size[0], self._box_size[1], self._box_size[2]))

            if self._cosolvents and self._added_as_copies:
                # Select only cosolvents that are added as copies
                cosolvents = {k: v for k, v in self._cosolvents.items() if k in self._added_as_copies}
                # Add copies of cosolvents
                wat_xyzs, cosolv_xyzs, final_copies = _add_cosolvent_as_copies(self._wat_xyzs, cosolvents,
                                                              self._origin, self._box_size, self._copies, self._center_positions,
                                                              receptor_xyzs)

                self._wat_xyzs = wat_xyzs
                self._cosolv_xyzs = cosolv_xyzs

                print("")
                print("Target number of copies      :")
                for cosolv_name in self._cosolvents:
                    if self._copies[cosolv_name] > 0:
                        print("%3s - %10d" % (self._cosolvents[cosolv_name].resname, self._copies[cosolv_name]))
                print("Final number of copies       :")
                for cosolv_name in self._cosolvents:
                    if self._copies[cosolv_name] > 0:
                        print("%3s - %10d" % (self._cosolvents[cosolv_name].resname, final_copies[cosolv_name]))

            if self._cosolvents and self._added_as_concentrations:
                # Select only cosolvents that are added as concentrations
                cosolvents = {k: v for k, v in self._cosolvents.items() if k in self._added_as_concentrations}
                # Add cosolvents to reach target concentrations
                wat_xyzs, cosolv_xyzs, final_concentrations = _add_cosolvent_as_concentrations(self._wat_xyzs, cosolvents,
                                                                            self._origin, self._box_size, self._concentrations,
                                                                            receptor_xyzs)

                self._wat_xyzs = wat_xyzs
                # We update the cosolvent xyzs with the new ones
                self._cosolv_xyzs.update(cosolv_xyzs)

                print("")
                print("Target concentration (M)     :")
                for cosolv_name in self._cosolvents:
                    if self._concentrations[cosolv_name] > 0:
                        print("%3s - %10.3f" % (self._cosolvents[cosolv_name].resname, self._concentrations[cosolv_name]))
                print("Final concentration (M)      :")
                for cosolv_name in self._cosolvents:
                    if self._concentrations[cosolv_name] > 0:
                        print("%3s - %10.3f" % (self._cosolvents[cosolv_name].resname, final_concentrations[cosolv_name]))

            n_water = int(self._wat_xyzs.shape[0] / 3)

            print("")
            print("Final composition            :")
            print("WAT - %10d" % (n_water))
            for cosolv_name in self._cosolvents:
                print("%3s - %10d" % (self._cosolvents[cosolv_name].resname, len(self._cosolv_xyzs[cosolv_name])))

        else:
            print("Error                            : box dimensions were not defined.")
            sys.exit(1)

        print("----------------------------------------------")

    def export_pdb(self, filename='cosolv_system.pdb'):
        """Export pdb file (hybrid-36 format) of the whole system.

        The output PDB follows the hybrid36 format in order to handle >100k atoms, which means 
        that the atom/residue numbers are encoded in base 36. See https://cci.lbl.gov/hybrid_36/ 
        and https://cci.lbl.gov/hybrid_36/pdb_format_evolution.pdf for more information about
        this particular PDB format. This format is supported by MDAnalysis, OpenMM, Pymol, VMD, etc.

        Parameters
        ----------
        filename : str, default='cosolv_system.pdb'
            Name of the output pdb file (hybrid-36 PDB format)

        """
        self._pdb_filename = filename
        pdb_string = _generate_pdb(self._receptor_data, self._cosolvents, self._cosolv_xyzs, self._wat_xyzs)

        # Write pdb file
        with open(filename, 'w') as w:
            w.write(pdb_string)
            
    def prepare_system_for_amber(self, filename='tleap.cmd', prmtop_filename='cosolv_system.prmtop', 
                            inpcrd_filename='cosolv_system.inpcrd', pdb_filename='cosolv_system.pdb', 
                            protein_ff='ff19SB', dna_ff='OL15', rna_ff='OL3', glycam_ff='GLYCAM_06j-1', 
                            lipid_ff='lipid21', water_ff='tip3p', gaff='gaff2', 
                            lib_files=None, frcmod_files=None, run_tleap=False):
            """Prepare the system for Amber forcefield. This includes parametrization of the cosolvent 
            molecules using GAFF and the generation of the tleap input file.

            This step does not automatically generate the prmtop and inpcrd files. The user has to run
            tleap manually using the following command: tleap -s -f tleap.cmd. If you know what you are doing,
            you can set the run_tleap argument to True and the tleap command will be executed automatically.

            Parameters
            ----------
            filename : str, default='tleap.cmd'
                Name of the tleap input file
            prmtop_filename : str, default='cosolv_system.prmtop'
                Name of the output prmtop file
            inpcrd_filename : str, default='cosolv_system.inpcrd'
                Name of the output inpcrd file
            pdb_filename : str, default='cosolv_system.pdb'
                Name of the output pdb file (hybrid-36 PDB format)
            protein_ff : str, default='ff19SB'
                Name of the protein force field
            dna_ff : str, default='OL15'
                Name of the DNA force field
            rna_ff : str, default='OL3'
                Name of the RNA force field
            glycam_ff : str, default='GLYCAM_06j-1'
                Name of the glycam force field
            lipid_ff : str, default='lipid21'
                Name of the lipid force field
            water_ff : str, default='tip3p'
                Name of the water force field
            gaff : str, default='gaff2'
                Name of the gaff force field
            lib_files : str or list of str, default=None,
                Extra amber lib files to be included
            frcmod_files : str or list of str, default=None,
                Extra amber frcmod file to be included
            run_tleap : bool, default=False
                If True, run tleap after generating the tleap input file. This step will generate 
                the prmtop and inpcrd files.

            """
            # Create tleap template
            TLEAP_TEMPLATE = ("source leaprc.protein.%s\n"
                            "source leaprc.DNA.%s\n"
                            "source leaprc.RNA.%s\n"
                            "source leaprc.%s\n"
                            "source leaprc.%s\n"
                            "source leaprc.water.%s\n"
                            "source leaprc.%s\n")
            TLEAP_TEMPLATE = TLEAP_TEMPLATE % (protein_ff, dna_ff, rna_ff, glycam_ff, lipid_ff, water_ff, gaff)

            # Write system pdb file
            self.export_pdb(filename=pdb_filename)

            # Parametrize cosolvent molecules
            if self._cosolvents is not None:
                for _, cosolvent in self._cosolvents.items():
                    _, frcmod_filename, lib_filename = cosolvent.parametrize(charge_method="bcc", gaff_version=gaff)

                    TLEAP_TEMPLATE += "loadamberparams %s\n" % os.path.basename(frcmod_filename)
                    TLEAP_TEMPLATE += "loadoff %s\n" % os.path.basename(lib_filename)

            # Add extra lib and frcmod files
            if lib_files is not None and frcmod_files is not None:
                if isinstance(lib_files, str):
                    lib_files = [lib_files]

                if isinstance(frcmod_files, str):
                    frcmod_files = [frcmod_files]

                for lib_file, frcmod_file in zip(lib_files, frcmod_files):
                    TLEAP_TEMPLATE += "loadamberparams %s\n" % os.path.basename(frcmod_file)
                    TLEAP_TEMPLATE += "loadoff %s\n" % os.path.basename(lib_file)

            TLEAP_TEMPLATE += "set default nocenter on\n"
            TLEAP_TEMPLATE += "m = loadpdb %s\n" % pdb_filename

            # Add all disulfide bridges based on the CYX resname
            cyx_cyx_pairs = utils.find_disulfide_bridges(pdb_filename)
            if cyx_cyx_pairs:
                for cyx_cyx_pair in cyx_cyx_pairs:
                    TLEAP_TEMPLATE += 'bond m.%d.SG m.%d.SG\n' % (cyx_cyx_pair[0], cyx_cyx_pair[1])

            # Add counter ions
            if self._wat_xyzs is not None:
                TLEAP_TEMPLATE += "charge m\n"
                TLEAP_TEMPLATE += "addIonsRand m Cl- 0\n"
                TLEAP_TEMPLATE += "addIonsRand m K+ 0\n"
                TLEAP_TEMPLATE += "check m\n"

            TLEAP_TEMPLATE += "set m box {%d %d %d}\n" % (self._box_size[0], self._box_size[1], self._box_size[2])
            TLEAP_TEMPLATE += "saveamberparm m %s %s\n" % (prmtop_filename, inpcrd_filename)
            TLEAP_TEMPLATE += "quit\n"

            with open(filename, 'w') as w:
                w.write(TLEAP_TEMPLATE)

            if run_tleap:
                cmd = 'tleap -s -f %s' % filename
                outputs, errors = utils.execute_command(cmd)

                if errors:
                    print(errors)
