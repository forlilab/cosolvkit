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


class Analysis(AnalysisBase):

    def __init__(self, atomgroup, gridsize=1., center=None, dimensions=None, **kwargs):
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
            self._center = center

        if dimensions is None:
            self._dimensions = self._u.trajectory.ts.dimensions[:3]
        else:
            self._dimensions = dimensions

    def _prepare(self):
        self._positions = []

    def _single_frame(self):
        self._positions.append(self._ag.atoms.positions.astype(np.float32))
        self._nframes += 1

    def _conclude(self):
        self._positions = np.array(self._positions, dtype=np.float32)

        x, y, z = self._center
        sd = self._dimensions / 2.
        self._hrange = ((x - sd[0], x + sd[0]), (y - sd[1], y + sd[1]), (z - sd[2], z + sd[2]))
        self._hbins = np.round(self._dimensions / self._gridsize).astype(np.int)

        positions = self._get_positions()
        hist, edges = np.histogramdd(positions, bins=self._hbins, range=self._hrange)

        origin = (edges[0][0], edges[1][0], edges[2][0])
        self.histogram = Grid(hist, origin=origin, delta=self._gridsize)
        self.density = Grid(_grid_density(hist), origin=origin, delta=self._gridsize)

    def _get_positions(self, start=0, stop=None):
        positions = self._positions[start:stop,:,:]
        new_shape = (positions.shape[0] * positions.shape[1], 3)
        positions = positions.reshape(new_shape)
        return positions

    def grid_free_energy(self, volume, temperature=300.):
        """Compute grid free energy.
        """
        gfe = _grid_free_energy(self.histogram.grid, volume, self._gridsize, self._n_atoms, self._nframes, temperature)
        self.gfe = Grid(gfe, origin=self.histogram.origin, delta=self._gridsize)

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

            gfe = _grid_free_energy(hist, volume, self._gridsize, self._n_atoms, n_frame, temperature)

            if favorable_only:
                grid_free_energies.append(gfe[gfe < 0].sum())
            else:
                grid_free_energies.append(gfe.sum())

        grid_free_energies = np.array(grid_free_energies)

        return n_frames, grid_free_energies
