#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Utils functions
#

import contextlib
import os
import tempfile
import shutil
import subprocess
import sys

if sys.version_info >= (3, ):
    import importlib
else:
    import imp

import numpy as np
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import kilocalories_per_mole, angstrom


def path_module(module_name):
    try:
        specs = importlib.machinery.PathFinder().find_spec(module_name)

        if specs is not None:
            return specs.submodule_search_locations[0]
    except:
        try:
            _, path, _ = imp.find_module(module_name)
            abspath = os.path.abspath(path)
            return abspath
        except ImportError:
            return None

    return None


@contextlib.contextmanager
def temporary_directory(suffix=None, prefix=None, dir=None):
    """Create and enter a temporary directory; used as context manager."""
    temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
    cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yield temp_dir
    finally:
        os.chdir(cwd)
        shutil.rmtree(temp_dir)


def execute_command(cmd_line):
    """Simple function to execute bash command."""
    args = cmd_line.split()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, errors = p.communicate()
    return output, errors


def add_repulsive_centroid_force(prmtop, inpcrd, system, residue_names, 
                                 epsilon=-0.01*kilocalories_per_mole, sigma=12*angstrom):
    """Add centroid to residues and add repulsive forces between them
    
    Args:
        prmtop (AmberPrmtopFile): Amber topology object
        inpcrd (AmberInpcrdFile): Amber coordinates object
        system (System): OpenMM system object created from the Amber topology object
        residue_names (list): list of residue names to which centroids will be attached (ex: BEN)
        epsilon (Quantity): depth of the potential well (default: -0.01 * kilocalories_per_mole)
        sigma (Quantity): inter-particle distance (default: 12 * angstrom)

    Returns:
        (list, int): list of all the virtual sites index, index within the System of the force that was added

    """
    virtual_site_idxs = []

    if not isinstance(residue_names, (list, tuple)) and isinstance(residue_names, str):
        residue_names = [residue_names]

    element = Element(0, "EP", "EP", 0)
    epsilon = (epsilon * epsilon).sqrt()

    # Select NonBondedForce
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        if force.__class__.__name__ == 'NonbondedForce':
            nonbonded_force = system.getForce(i)
            break

    # Add centroid
    for i, residue in enumerate(prmtop.topology.residues()):
        if not residue.name in residue_names: continue

        virtual_site_index = system.addParticle(mass=0)
        nonbonded_site_index = nonbonded_force.addParticle(charge=0, sigma=0, epsilon=0)
        # We want to make sure we have the same numbers of particles and nonbonded forces
        assert virtual_site_index == nonbonded_site_index

        # Compute centroid based on the heavy atoms
        xyzs = [inpcrd.positions[a.index].in_units_of(angstrom) for a in residue._atoms if not "H" in a.name]
        idxs = [a.index for a in residue._atoms if not "H" in a.name]
        o_weights = np.ones(len(idxs)) / len(idxs)
        x_weights = np.zeros(len(idxs))
        y_weights = np.zeros(len(idxs))
        x, y, z = np.mean(xyzs, axis=0)
        centroid = Vec3(x._value, y._value, z._value) * angstrom

        # Add atom to topology and coordinates
        tmp_residue = prmtop.topology.addResidue("EP", residue.chain)
        prmtop.topology.addAtom("EP", element, tmp_residue)
        inpcrd.positions.append(centroid)

        # Add virtual site to the system
        system.setVirtualSite(virtual_site_index, LocalCoordinatesSite(idxs, o_weights, x_weights, y_weights, centroid))
        virtual_site_idxs.append(virtual_site_index)

    if not virtual_site_idxs:
        print("Warning: no centroids were added to the system (residue names: %s)" % residue_names)
        return ([], None)

    # Add repulsive potential between centroids
    repulsive_force = CustomBondForce('epsilon * ((sigma / r)^12 - 2 * (sigma / r)^6)')
    repulsive_force.addPerBondParameter('sigma')
    repulsive_force.addPerBondParameter('epsilon')
    # It does not work properly if we don't have the PBC activated
    repulsive_force.setUsesPeriodicBoundaryConditions(True)

    for i in range(0, len(virtual_site_idxs)):
        for j in range(i + 1, len(virtual_site_idxs)):
            repulsive_force.addBond(virtual_site_idxs[i], virtual_site_idxs[j], [sigma, epsilon])

    repulsive_force_id = system.addForce(repulsive_force)

    return virtual_site_idxs, repulsive_force_id


def write_pdb(pdb_filename, prmtop, inpcrd):
    """ Write PDB file

    Args:
        pdb_filename (str): output pdb filename
        prmtop (AmberPrmtopFile): Amber topology object
        inpcrd (AmberInpcrdFile): Amber coordinates object

    """
    PDBFile.writeFile(prmtop.topology, inpcrd.getPositions(), open(pdb_filename, 'w'))
