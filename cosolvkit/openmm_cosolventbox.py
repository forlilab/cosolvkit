import json
import numpy as np
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmforcefields.generators import EspalomaTemplateGenerator, GAFFTemplateGenerator, SMIRNOFFTemplateGenerator
from openff.toolkit import Molecule
import pdbfixer
from cosolvkit.cosolventbox import _add_cosolvent_as_concentrations
from cosolvkit.cosolvent import CoSolvent
import parmed
from scipy import spatial
import itertools

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

    return wat_xyzs, cosolv_xyzs, final_number_copies, to_be_removed


class OpenmmCosolventBox:
    
    def __init__(self, pdb_filename, padding):
        if pdb_filename is not None:
            self.pdb_filename = self._fix_pdb(pdb_filename)
        
        self.padding = padding
        self._cosolvents = {}
        self._concentrations = {}
        self._copies = {}
        self._center_positions = {}
        self._added_as_concentrations = []
        self._added_as_copies = []

    def _fix_pdb(self, pdb_filename):
        print("Creating PDBFixer...")
        fixer = pdbfixer.PDBFixer(pdb_filename)
        print("Finding missing residues...")
        fixer.findMissingResidues()

        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                print("ok")
                del fixer.missingResidues[key]
        fixer.findMissingAtoms()
        print("Adding missing atoms...")
        fixer.addMissingAtoms()
        print("Adding missing hydrogens...")
        fixer.addMissingHydrogens(7)
        print("Writing PDB file...")

        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            open(f"{pdb_filename.split('.')[0]}_clean.pdb",
                    "w"),
            keepIds=True)
        return f"{pdb_filename.split('.')[0]}_clean.pdb"
    
    def add_cosolvent(self, name, 
                      concentration=None, 
                      copies=None, smiles=None, 
                      mol_filename=None, center_positions=None, 
                      resname=None):
        
        assert concentration is not None or copies is not None, 'Either concentration or copies must be defined.'
        assert not all([concentration is not None, copies is not None]), 'Either concentration or copies must be defined, not both.'

        if concentration is not None:
            assert concentration > 0, 'Concentration must be positive.'

        if copies is not None:
            assert copies > 0, 'Number of copies must be positive.'

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
