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
import importlib

import numpy as np
import mdtraj
from MDAnalysis import Universe
from openmm.app import *
from openmm import *
import pdbfixer


def path_module(module_name):
    specs = importlib.machinery.PathFinder().find_spec(module_name)

    if specs is not None:
        return specs.submodule_search_locations[0]

    return None


@contextlib.contextmanager
def temporary_directory(suffix=None, prefix=None, dir=None, clean=True):
    """Create and enter a temporary directory; used as context manager."""
    temp_dir = tempfile.mkdtemp(suffix, prefix, dir)
    cwd = os.getcwd()
    os.chdir(temp_dir)
    try:
        yield temp_dir
    finally:
        os.chdir(cwd)
        if clean:
            shutil.rmtree(temp_dir)


def execute_command(cmd_line):
    """Simple function to execute bash command."""
    args = cmd_line.split()
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    output, errors = p.communicate()
    return output, errors


def vector(a, b):
    """
    Return the vector between a and b
    """
    return b - a


def normalize(a):
    """
    Return a normalized vector
    """
    return a / np.sqrt(np.sum(np.power(a, 2)))


def atom_to_move(o, p):
    """
    Return the coordinates xyz of an atom just above acceptor/donor atom o
    """
    # It will not work if there is just one dimension
    p = np.atleast_2d(p)
    return o + normalize(-1. * vector(o, np.mean(p, axis=0)))


def rotate_point(p, p1, p2, angle):
    """ Rotate the point p around the axis p1-p2
    Source: http://paulbourke.net/geometry/rotate/PointRotate.py"""
    # Translate the point we want to rotate to the origin
    pn = p - p1

    # Get the unit vector from the axis p1-p2
    n = p2 - p1
    n = normalize(n)

    # Setup the rotation matrix
    c = np.cos(angle)
    t = 1. - np.cos(angle)
    s = np.sin(angle)
    x, y, z = n[0], n[1], n[2]

    R = np.array([[t*x**2 + c, t*x*y - s*z, t*x*z + s*y],
                 [t*x*y + s*z, t*y**2 + c, t*y*z - s*x],
                 [t*x*z - s*y, t*y*z + s*x, t*z**2 + c]])

    # ... and apply it
    ptr = np.dot(pn, R)

    # And to finish, we put it back
    p = ptr + p1

    return p


def resize_vector(v, length, origin=None):
    """ Resize a vector v to a new length in regard to a origin """
    if origin is not None:
        return (normalize(v - origin) * length) + origin
    else:
        return normalize(v) * length


def add_harmonic_restraints(prmtop, inpcrd, system, atom_selection, k=1.0):
    """Add harmonic restraints to the system.

    Args:
        prmtop (AmberPrmtopFile): Amber topology object
        inpcrd (AmberInpcrdFile): Amber coordinates object
        system (System): OpenMM system object created from the Amber topology object
        atom_selection (str): Atom selection (see MDTraj documentation)
        k (float): harmonic force restraints in kcal/mol/A**2 (default: 1 kcal/mol/A**2)

    Returns:
        (list, int): list of all the atom ids on which am harmonic force is applied, index with the System of the force that was added

    """
    mdtop = mdtraj.Topology.from_openmm(prmtop.topology)
    atom_idxs = mdtop.select(atom_selection)
    positions = inpcrd.positions

    # Tranform constant to the right unit
    k = k * unit.kilocalories_per_mole / unit.angstroms**2
    k = k.value_in_unit_system(unit.md_unit_system)

    if atom_idxs.size == 0:
        print("Warning: no atoms selected using: %s" % atom_selection)
        return ([], None)

    # Take into accoun the periodic condition
    # http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomExternalForce.html
    force = CustomExternalForce("k * periodicdistance(x, y, z, x0, y0, z0)^2")
    
    harmonic_force_idx = system.addForce(force)
    
    force.addGlobalParameter("k", k)
    print(force.getGlobalParameterDefaultValue(0))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    
    for atom_idx in atom_idxs:
        #print(atom_idx, positions[atom_idx].value_in_unit_system(units.md_unit_system))
        force.addParticle(int(atom_idx), positions[atom_idx].value_in_unit_system(unit.md_unit_system))

    return atom_idxs


def update_harmonic_restraints(simulation, k=1.0):
    """Update harmonic restraints force
    
    Args:
        simulation (Simulation): OpenMM simulation object
        k (float): new harmonic force constraint in kcal/mol/A**2 (default: 1 kcal/mol/A**2)

    """
    # Tranform constant to the right unit
    k = k * unit.kilocalories_per_mole / unit.angstroms**2
    k = k.value_in_unit_system(unit.md_unit_system)
    
    simulation.context.setParameter('k', k)


def add_repulsive_centroid_force(prmtop, inpcrd, system, residue_names, epsilon=-0.01, sigma=12):
    """Add centroid to residues and add repulsive forces between them
    
    Args:
        prmtop (AmberPrmtopFile): Amber topology object
        inpcrd (AmberInpcrdFile): Amber coordinates object
        system (System): OpenMM system object created from the Amber topology object
        residue_names (list): list of residue names to which centroids will be attached (ex: BEN)
        epsilon (float): depth of the potential well in kcal/mol (default: -0.01 kcal/mol)
        sigma (float): inter-particle distance in Angstrom (default: 12 A)

    Returns:
        (int, list, int): original number of particles, list of all the virtual sites index, index within the System of the force that was added

    """
    virtual_site_idxs = []
    n_particles = system.getNumParticles()

    # Tranform constants to the right unit
    epsilon = np.sqrt(epsilon * epsilon) * unit.kilocalories_per_mole
    epsilon = epsilon.value_in_unit_system(unit.md_unit_system)
    sigma = sigma * unit.angstroms
    sigma = sigma.value_in_unit_system(unit.md_unit_system)

    if not isinstance(residue_names, (list, tuple)) and isinstance(residue_names, str):
        residue_names = [residue_names]

    element = Element(0, "EP", "EP", 0)

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
        xyzs = [inpcrd.positions[a.index] * unit.angstroms for a in residue._atoms if not "H" in a.name]
        idxs = [a.index for a in residue._atoms if not "H" in a.name]
        o_weights = np.ones(len(idxs)) / len(idxs)
        x_weights = np.zeros(len(idxs))
        y_weights = np.zeros(len(idxs))
        x, y, z = np.mean(xyzs, axis=0)
        centroid = Vec3(x._value, y._value, z._value) * unit.angstroms

        # Add atom to topology and coordinates
        tmp_residue = prmtop.topology.addResidue("EP", residue.chain)
        prmtop.topology.addAtom("EP", element, tmp_residue)
        inpcrd.positions.append(centroid)

        # Add virtual site to the system
        system.setVirtualSite(virtual_site_index, LocalCoordinatesSite(idxs, o_weights, x_weights, y_weights, centroid))
        virtual_site_idxs.append(virtual_site_index)

    if not virtual_site_idxs:
        print("Warning: no centroids were added to the system (residue names: %s)" % residue_names)
        return (None, [], None)

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

    return n_particles, virtual_site_idxs, repulsive_force_id


def write_pdb(pdb_filename, prmtop, inpcrd):
    """ Write PDB file

    Args:
        pdb_filename (str): output pdb filename
        prmtop (AmberPrmtopFile): Amber topology object
        inpcrd (AmberInpcrdFile): Amber coordinates object

    """
    PDBFile.writeFile(prmtop.topology, inpcrd.getPositions(), open(pdb_filename, 'w'))


def find_disulfide_bridges(pdb_filename):
    """ Find disulfide bridges based on CYX resnames.
    
    Args:
        pdb_filename (str): input pdb filename

    """
    cyx_cyx_pairs = set()

    u = Universe(pdb_filename)

    cyx_sg_atoms = u.select_atoms('resname CYX and name SG')

    resid_first_residue = u.residues[0].resid

    for cyx_sg_atom in cyx_sg_atoms:
        sg_x, sg_y, sg_z = cyx_sg_atom.position
        sel_str = '(resname CYX and name SG and not id %d) and point %f %f %f 2.5' % (cyx_sg_atom.id, sg_x, sg_y, sg_z)
        paired_cyx_residue = u.select_atoms(sel_str)[0]

        pair = (resid_first_residue + cyx_sg_atom.residue.ix, resid_first_residue + paired_cyx_residue.residue.ix)
        cyx_cyx_pairs.add(tuple(sorted(pair)))

    cyx_cyx_pairs = sorted(list(cyx_cyx_pairs))

    return cyx_cyx_pairs

def fix_pdb(pdb_filename: str, save=False):
    """ Fixes common problems in PDB such as:
            - missing atoms
            - missing residues
            - missing hydrogens
            - replace nonstandard residues
    
    Args:
        pdb_filename (str): input pdb filename
    """
    path = os.getcwd()
    # print("Creating PDBFixer...")
    fixer = pdbfixer.PDBFixer(pdb_filename)
    # print("Finding missing residues...")
    fixer.findMissingResidues()

    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            print("ok")
            del fixer.missingResidues[key]

    # print("Finding nonstandard residues...")
    # fixer.findNonstandardResidues()
    # print("Replacing nonstandard residues...")
    # fixer.replaceNonstandardResidues()
    # print("Removing heterogens...")
    # fixer.removeHeterogens(keepWater=True)

    # print("Finding missing atoms...")
    fixer.findMissingAtoms()
    # print("Adding missing atoms...")
    fixer.addMissingAtoms()
    # print("Adding missing hydrogens...")
    fixer.addMissingHydrogens(7)
    if save:
        # print("Writing PDB file...")

        PDBFile.writeFile(
            fixer.topology,
            fixer.positions,
            open(os.path.join(path, f"{pdb_filename.split('.')[0]}_clean.pdb"),
                    "w"),
            keepIds=True)
    return fixer.topology, fixer.positions
