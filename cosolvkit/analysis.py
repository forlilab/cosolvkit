#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to analyze cosolvent MD
#

import numpy as np
from scipy import spatial
from gridData import Grid
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase

from . import utils


BOLTZMANN_CONSTANT_KB = 0.0019872041


def _grid_free_energy(hist, volume, gridsize, n_atoms, n_frames, temperature=300.):
    # Avoid 0 in the histogram
    hist = hist + 1E-10
    # The volume here is the volume of water and not the entire box
    n_voxel = volume / (gridsize **3)
    # Bulk probability without protein
    N_o = n_atoms / n_voxel
    # Probability with the protein
    N = hist / n_frames
    # Free energy
    gfe = (-BOLTZMANN_CONSTANT_KB * temperature) * np.log(N / N_o)

    return gfe


def _grid_density(hist):
    return (hist - np.mean(hist)) / np.std(hist)


def _subset_grid(grid, gridsize=0.5, center=None, box_size=None):
    center = np.array(center)
    box_size = np.array(box_size)

    # Check center and box size
    assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
    assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
    assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."

    x, y, z = center
    sd = box_size / 2.
    edges = (np.arange(x - sd[0], x + sd[0] + gridsize, gridsize), 
             np.arange(y - sd[1], y + sd[1] + gridsize, gridsize), 
             np.arange(z - sd[2], z + sd[2] + gridsize, gridsize))

    sub_grid = grid.resample(edges)

    return sub_grid


class Analysis(AnalysisBase):

    def __init__(self, atomgroup, gridsize=0.5, center=None, box_size=None, **kwargs):
        super(Analysis, self).__init__(atomgroup.universe.trajectory, **kwargs)
        self._u = atomgroup.universe
        self._ag = atomgroup
        self._gridsize = gridsize
        self._nframes = 0
        self._n_atoms = atomgroup.n_atoms

        if center is None:
            receptor = self._u.select_atoms("protein or nucleic")
            self._center = np.mean(receptor.positions, axis=0)
        else:
            center = np.array(center)
            # Check center
            assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
            self._center = center

        if box_size is None:
            self._box_size = self._u.trajectory.ts.dimensions[:3]
        else:
            box_size = np.array(box_size)
            # Check gridsize
            assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
            assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."
            self._box_size = box_size

    def _prepare(self):
        self._positions = []

    def _single_frame(self):
        self._positions.append(self._ag.atoms.positions.astype(np.float32))
        self._nframes += 1

    def _conclude(self):
        self._positions = np.array(self._positions, dtype=np.float32)

        x, y, z = self._center
        sd = self._box_size / 2.
        self._hrange = ((x - sd[0], x + sd[0]), (y - sd[1], y + sd[1]), (z - sd[2], z + sd[2]))
        self._hbins = np.round(self._box_size / self._gridsize).astype(np.int)

        positions = self._get_positions()
        hist, edges = np.histogramdd(positions, bins=self._hbins, range=self._hrange)

        self._histogram = Grid(hist, edges=edges)
        self._density = Grid(_grid_density(hist), edges=edges)

    def _get_positions(self, start=0, stop=None):
        positions = self._positions[start:stop,:,:]
        new_shape = (positions.shape[0] * positions.shape[1], 3)
        positions = positions.reshape(new_shape)
        return positions

    def atomic_grid_free_energy(self, volume, temperature=300.):
        """Compute grid free energy.
        """
        agfe = _grid_free_energy(self._histogram.grid, volume, self._gridsize, self._n_atoms, self._nframes, temperature)
        self._agfe = Grid(agfe, edges=self._histogram.edges)

    def export_histogram(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export density maps
        """
        if center is None:
            center = self._center

        if box_size is None:
            box_size = self._box_size

        sub_grid = _subset_grid(self._histogram, gridsize, center, box_size)
        sub_grid.export(fname)

    def export_density(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export density maps
        """
        if center is None:
            center = self._center

        if box_size is None:
            box_size = self._box_size

        sub_grid = _subset_grid(self._density, gridsize, center, box_size)
        sub_grid.export(fname)

    def export_atomic_grid_free_energy(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export atomic grid free energy
        """
        if center is None:
            center = self._center

        if box_size is None:
            box_size = self._box_size

        sub_grid = _subset_grid(self._agfe, gridsize, center, box_size)
        sub_grid.export(fname)

    def convergence_density(self, sigma=20, n_steps=10):
        """Compute convergence of the density
        """
        overlap_coefficents = []
        n_frames = np.linspace(int(self._nframes / n_steps), self._nframes, n_steps, dtype=np.int)

        # Initialize the first density as reference
        positions = self._get_positions(stop=n_frames[0])
        hist, edges = np.histogramdd(positions, bins=self._hbins, range=self._hrange)
        # We want only the positions with a density >= sigma
        density = _grid_density(hist)
        x, y, z = np.where(density >= sigma)
        density_xyz_ref = np.stack([edges[0][x], edges[1][y], edges[2][z]], axis=1)

        for n_frame in n_frames[1:]:
            positions = self._get_positions(stop=n_frame)
            hist, edges = np.histogramdd(positions, bins=self._hbins, range=self._hrange)
            # We want only the positions with a density >= sigma
            density = _grid_density(hist)
            x, y, z = np.where(density >= sigma)
            density_xyz = np.stack([edges[0][x], edges[1][y], edges[2][z]], axis=1)

            # Compute the overlap coefficient
            k1 = spatial.cKDTree(density_xyz_ref)
            k2 = spatial.cKDTree(density_xyz)

            pairs = k1.count_neighbors(k2, 0.0, p=2)
            overlap_coefficent = pairs * 2 / (density_xyz_ref.shape[0] + density_xyz.shape[0])
            overlap_coefficents.append(overlap_coefficent)

            # The current density becomes the reference now
            density_xyz_ref = density_xyz

        overlap_coefficents = np.array(overlap_coefficents)

        return n_frames[1:], overlap_coefficents

    def convergence_energy(self, volume, temperature=300., n_steps=10, favorable_only=True):
        """Compute convergence of the grid free energy
        """
        grid_free_energies = []
        n_frames = np.linspace(int(self._nframes / n_steps), self._nframes, n_steps, dtype=np.int)

        for n_frame in n_frames:
            # Get the position of the first n_frame
            positions = self._get_positions(stop=n_frame)
            hist, edges = np.histogramdd(positions, bins=self._hbins, range=self._hrange)

            agfe = _atomic_grid_free_energy(hist, volume, self._gridsize, self._n_atoms, n_frame, temperature)

            if favorable_only:
                grid_free_energies.append(agfe[agfe < 0].sum())
            else:
                grid_free_energies.append(agfe.sum())

        grid_free_energies = np.array(grid_free_energies)

        return n_frames, grid_free_energies
