#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to analyze cosolvent MD
#

import numpy as np
from gridData import Grid
from MDAnalysis import Universe
from MDAnalysis.analysis.base import AnalysisBase

from . import utils


BOLTZMANN_CONSTANT_KB = 0.0019872041


class Analysis(AnalysisBase):

    def __init__(self, atomgroup, gridsize=1., center=None, dimensions=None, **kwargs):
        super(Analysis, self).__init__(atomgroup.universe.trajectory, **kwargs)
        self._u = atomgroup.universe
        self._ag = atomgroup
        self._gridsize = gridsize
        self._nframes = 0

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
        self._positions.extend(self._ag.atoms.positions.astype(np.float32))
        self._nframes += 1

    def _conclude(self):
        self._positions = np.array(self._positions, dtype=np.float32)

        x, y, z = self._center
        sd = self._dimensions / 2.
        hrange = ((x - sd[0], x + sd[0]), (y - sd[1], y + sd[1]), (z - sd[2], z + sd[2]))
        hbins = np.round(self._dimensions / self._gridsize).astype(np.int)

        hist, edges = np.histogramdd(self._positions, bins=hbins, range=hrange)

        self.histogram = Grid(hist, origin=(edges[0][0], edges[1][0], edges[2][0]), delta=self._gridsize)
        self.density = Grid((hist - np.mean(hist)) / np.std(hist), origin=(edges[0][0], edges[1][0], edges[2][0]), delta=self._gridsize)

    def grid_free_energy(self, volume, temperature=300., n_atoms=None):
        """Compute grid free energy.
        """
        if n_atoms is None:
            n_atoms = 0
            # Hydrogen atoms does not count
            ag = self._ag.select_atoms("not name H*")
            for resn in np.unique(ag.resnames):
                ag_tmp = ag.select_atoms("resname %s" % resn)
                n_atoms += ag_tmp.n_atoms / ag_tmp.n_residues

        # Avoid 0 in the histogram
        hist = self.histogram + 1E-10
        # The volume here is the volume of water and not the entire box
        n_voxel = volume / self._gridsize
        # Bulk probability without protein
        N_o = (self._ag.n_residues / n_voxel)
        # Probability with the protein
        N = hist.grid / self._nframes
        # Free energy
        gfe = (-BOLTZMANN_CONSTANT_KB * temperature) * np.log(N / (n_atoms * N_o))

        self.gfe = Grid(gfe, origin=self.histogram.origin, delta=self._gridsize)
