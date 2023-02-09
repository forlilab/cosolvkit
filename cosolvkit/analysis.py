#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to analyze cosolvent MD
#

import sys

import numpy as np
from scipy import spatial
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from gridData import Grid
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase

from . import utils


BOLTZMANN_CONSTANT_KB = 0.0019872041


def _normalization(data, a=0, b=0):
    min_data = np.min(data)
    max_data = np.max(data)
    return a + ((data - min_data) * (b - a)) / (max_data - min_data)


def _smooth_grid_free_energy(gfe, sigma=1):
    """ Empirical grid free energy smoothing
    """
    # We keep only the favorable spots with energy < 0 kcal/mol
    gfe_fav = np.copy(gfe)
    gfe_fav[gfe > 0.] = 0.
    # We smooth the energy a bit
    gfe_fav_smooth = gaussian_filter(gfe_fav, sigma=sigma)
    # Normalize data between the original energy minima value and 0
    gfe_smooth_norm = _normalization(gfe_fav_smooth, np.min(gfe_fav), np.max(gfe_fav))
    # Put the favorable smoothed data in the original grid
    gfe[gfe_smooth_norm < 0] = gfe_smooth_norm[gfe_smooth_norm < 0]

    return gfe


def _grid_free_energy(hist, volume_water, gridsize, n_atoms, n_frames, temperature=300.):
    # Avoid 0 in the histogram for the log function
    hist = hist + 1E-20
    # The volume here is the volume of water and not the entire box
    volume_voxel = gridsize **3
    n_voxel = volume_water / volume_voxel
    # Probability of the solute in the bulk (without protein)
    N_o = n_atoms / n_voxel
    # Probability of the solute (with the protein)
    N = hist / n_frames
    # Atomic grid free energy
    gfe = -(BOLTZMANN_CONSTANT_KB * temperature) * np.log(N / N_o)

    return gfe


def _grid_density(hist):
    return (hist - np.mean(hist)) / np.std(hist)


def _subset_grid(grid, center, box_size, gridsize=0.5):
    # Create grid interpolator
    # Number of midpoints is equal to the number of grid points
    grid_interpn = RegularGridInterpolator(grid.midpoints, grid.grid)

    # Create sub grid coordinates
    # We get first the edges of the grid box, and after the midpoints
    # So this we are sure (I guess) that the sub grid is well centered on center
    # There might be a better way of doing this... Actually I tried, but didn't worked very well.
    x, y, z = center
    sd = box_size / 2.
    hbins = np.round(box_size / gridsize).astype(int)
    edges = (np.linspace(0, box_size[0], num=hbins[0] + 1, endpoint=True) + (x - sd[0]),
             np.linspace(0, box_size[1], num=hbins[1] + 1, endpoint=True) + (y - sd[1]),
             np.linspace(0, box_size[2], num=hbins[2] + 1, endpoint=True) + (z - sd[2]))
    midpoints = (edges[0][:-1] + np.diff(edges[0]) / 2.,
                 edges[1][:-1] + np.diff(edges[1]) / 2.,
                 edges[2][:-1] + np.diff(edges[2]) / 2.)
    X, Y, Z = np.meshgrid(midpoints[0], midpoints[1], midpoints[2])
    xyzs = np.stack((X.ravel(), Y.ravel(), Z.ravel()), axis=-1)
    # Configuration of the sub grid
    origin_subgrid = (midpoints[0][0], midpoints[1][0], midpoints[2][0])
    shape_subgrid = (midpoints[0].shape[0], midpoints[1].shape[0], midpoints[2].shape[0])

    # Do interpolation
    sub_grid_values = grid_interpn(xyzs)
    sub_grid_values = sub_grid_values.reshape(shape_subgrid)
    sub_grid_values = np.swapaxes(sub_grid_values, 0, 1)
    sub_grid = Grid(sub_grid_values, origin=origin_subgrid, delta=gridsize)

    return sub_grid


def _export(fname, grid, gridsize=0.5, center=None, box_size=None):
    assert (center is None and box_size is None) or (center is not None and box_size is not None), \
           "Both center and box size have to be defined, or none of them."

    if center is None and box_size is None:
        grid.export(fname)
    elif center is not None and box_size is not None:
        center = np.array(center)
        box_size = np.array(box_size)

        assert np.ravel(center).size == 3, "Error: center should contain only (x, y, z)."
        assert np.ravel(box_size).size == 3, "Error: grid size should contain only (a, b, c)."
        assert (box_size > 0).all(), "Error: grid size cannot contain negative numbers."

        sub_grid = _subset_grid(grid, center, box_size, gridsize)
        sub_grid.export(fname)


class Analysis(AnalysisBase):

    def __init__(self, atomgroup, gridsize=0.5, **kwargs):
        super(Analysis, self).__init__(atomgroup.universe.trajectory, **kwargs)

        if atomgroup.n_atoms == 0:
            print("Error: no atoms were selected.")
            sys.exit(1)

        self._u = atomgroup.universe
        self._ag = atomgroup
        self._gridsize = gridsize
        self._nframes = 0
        self._n_atoms = atomgroup.n_atoms
        self._center = None
        self._box_size = None

    def _prepare(self):
        self._positions = []
        self._centers = []
        self._dimensions = []

    def _single_frame(self):
        self._positions.append(self._ag.atoms.positions.astype(float))
        self._dimensions.append(self._u.dimensions[:3])
        self._centers.append(self._u.atoms.center_of_geometry())
        self._nframes += 1

    def _conclude(self):
        self._positions = np.array(self._positions, dtype=float)
        self._box_size = np.mean(self._dimensions, axis=0)
        self._center = np.mean(self._centers, axis=0)

        # Get all the positions
        positions = self._get_positions()

        # Get grid edges and origin
        x, y, z = self._center
        sd = self._box_size / 2.
        hbins = np.round(self._box_size / self._gridsize).astype(int)
        self._edges = (np.linspace(0, self._box_size[0], num=hbins[0] + 1, endpoint=True) + (x - sd[0]),
                       np.linspace(0, self._box_size[1], num=hbins[1] + 1, endpoint=True) + (y - sd[1]),
                       np.linspace(0, self._box_size[2], num=hbins[2] + 1, endpoint=True) + (z - sd[2]))
        origin = (self._edges[0][0], self._edges[1][0], self._edges[2][0])

        hist, edges = np.histogramdd(positions, bins=self._edges)
        self._histogram = Grid(hist, origin=origin, delta=self._gridsize)
        self._density = Grid(_grid_density(hist), origin=origin, delta=self._gridsize)

    def _get_positions(self, start=0, stop=None):
        positions = self._positions[start:stop,:,:]
        new_shape = (positions.shape[0] * positions.shape[1], 3)
        positions = positions.reshape(new_shape)
        return positions

    def atomic_grid_free_energy(self, volume, temperature=300., atom_radius=1.4, smoothing=True):
        """Compute grid free energy.
        """
        agfe = _grid_free_energy(self._histogram.grid, volume, self._gridsize, self._n_atoms, self._nframes, temperature)

        if smoothing:
            # We divide by 3 in order to have radius == 3 sigma
            agfe = _smooth_grid_free_energy(agfe, atom_radius / 3.)

        self._agfe = Grid(agfe, edges=self._histogram.edges)

    def export_histogram(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export histogram maps
        """
        _export(fname, self._histogram, gridsize, center, box_size)

    def export_density(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export density maps
        """
        _export(fname, self._density, gridsize, center, box_size)

    def export_atomic_grid_free_energy(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export atomic grid free energy
        """
        _export(fname, self._agfe, gridsize, center, box_size)

    def convergence_density(self, sigma=20, n_steps=10):
        """Compute convergence of the density
        """
        overlap_coefficents = []
        n_frames = np.linspace(int(self._nframes / n_steps), self._nframes, n_steps, dtype=int)

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
        n_frames = np.linspace(int(self._nframes / n_steps), self._nframes, n_steps, dtype=int)

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
