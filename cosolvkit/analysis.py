#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# CoSolvKit
#
# Class to analyze cosolvent MD
#

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Union

from scipy.ndimage import gaussian_filter,binary_dilation
from scipy.signal import correlate
from scipy.interpolate import RegularGridInterpolator
from gridData import Grid

from MDAnalysis import Universe
from MDAnalysis.analysis import rdf, align, rms
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.rms import RMSF

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 3})

from pymol import cmd


BOLTZMANN_CONSTANT_KB = 0.0019872041  # kcal/(mol*K)

def _read_dx(filepath:str=None) -> Grid:
    """Reads a .dx map using gridData.Grid."""
    return Grid(str(filepath))

def combine_dx_maps(filepaths: List[str] = None, method:str= 'mean', out_fname:str='combined.dx') -> Grid:
    """Combines multiple .dx map files into one using a specified method."""

    grids = [_read_dx(path) for path in filepaths]

    # Validate all grids match in shape. This is kinda clunky, but it works.
    shape = grids[0].grid.shape
    for g in grids:
        if g.grid.shape != shape:
            raise ValueError("All input maps must have the same shape.")

    stacked = np.stack([g.grid for g in grids])

    agg_fn = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min,
        'sum': np.sum,
        'median': np.median
    }.get(method)

    if agg_fn is None:
        raise ValueError(f"Unsupported combination method: {method}")

    combined_data = agg_fn(stacked, axis=0)
    combined_grid = Grid(combined_data, grids[0].edges)

    combined_grid.export(out_fname)

    return combined_grid

def _normalization(data, a:float=0, b:float=1):
    """_summary_

    :param data: list of data points
    :type data: list
    :param a: int a, defaults to 0
    :type a: int, optional
    :param b: int b, defaults to 1
    :type b: int, optional
    :return: normalized data
    :rtype: list
    """
    min_data = np.min(data)
    max_data = np.max(data)
    epsilon = 1e-20  # small value to avoid division by zero
    return a + ((data - min_data) * (b - a)) / (max_data - min_data + epsilon)

def _grid_free_energy(hist, n_atoms, n_frames, n_accessible_voxels, temperature=300):
    """
    Compute the atomic grid free energy (GFE) from a given histogram.
    
    :param hist: Histogram of cosolvent occupancy in each voxel
    :param n_atoms: Total number of cosolvent atoms (not total system atoms). Also this is per atom-type
    :param n_frames: Number of frames in the trajectory
    :param n_accessible_voxels: Number of solvent accessible voxels in the grid
    :param temperature: Temperature in Kelvin (default 300K)
    :return: 3D numpy array of free energy values (same shape as `hist`)
    """
    # Apply occupancy filtering: remove low-occupancy grid points
    # occupancy = hist / n_frames
    # occupancy_threshold = 0.001
    # hist[occupancy < occupancy_threshold] = 0
    # hist[hist < 2] = 0

    N_o = n_atoms / n_accessible_voxels  # Bulk probability of cosolvent
    N = hist / n_frames  # Local probability in the grid

    #if hist contains very low values (or zeros), N = hist / n_frames can be much smaller than N_o
    # making log(N / N_o) too negative and gfe extremely large.
    N = np.maximum(N, 1E-10)
   
    gfe = -(BOLTZMANN_CONSTANT_KB * temperature) * np.log(N / N_o)
    
    return gfe

def _smooth_grid_free_energy(gfe, 
                             energy_cutoff: float = 0, 
                             sigma: float = 1, 
                            ):
    """
    Smooths and filters the grid free energy (GFE) map.

    :param gfe: 3D numpy array of grid free energy values.
    :param energy_cutoff: Cutoff energy (default: .0 kcal/mol). Only values below this are retained.
    :param sigma: Standard deviation for Gaussian smoothing (default: 1).
    :return: Smoothed and filtered grid free energy map (new array).
    """

    gfe_filtered = np.copy(gfe)
 
    # Apply Gaussian smoothing BEFORE filtering.
    gfe_smoothed = gaussian_filter(gfe_filtered, sigma=sigma)

    # Keep only favorable energy values after smoothing
    gfe_smoothed[gfe_smoothed >= energy_cutoff] = 0.0

    # Normalization has not no effect
    gfe_smoothed = _normalization(gfe_smoothed, np.min(gfe_filtered), 0.0)

    return gfe_smoothed

def _grid_density(hist):
    return (hist - np.mean(hist)) / np.std(hist)

def _subset_grid(grid, center, box_size, gridsize=0.5):

    #FIXME I think this part of the code is never triggered, not sure if we need this

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
    return
    
class Analysis(AnalysisBase):
    """Analysis class to generate density grids

    :param AnalysisBase: Base MDAnalysis class
    :type AnalysisBase: AnalysisBase
    """
    def __init__(self, atomgroup,
                        gridsize:float=0.5, 
                        use_atomtypes:bool=True, 
                        atomtypes_definitions:dict=None, 
                        **kwargs):
        super(Analysis, self).__init__(atomgroup.universe.trajectory, **kwargs)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

        self._u = atomgroup.universe
        self._ag = atomgroup
        self._gridsize = gridsize
        self._nframes = 0
        self._n_atoms = atomgroup.n_atoms
        self._center = None
        self._box_size = None
        self.use_atomtypes = use_atomtypes
        self.atomtypes_definitions = atomtypes_definitions

        if use_atomtypes and atomtypes_definitions is None:
            self.logger.error("Error: Atom types definitions are required for atom type density analysis.")
            sys.exit(1)

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

        # Get grid edges and origin
        x, y, z = self._center
        sd = self._box_size / 2.
        hbins = np.round(self._box_size / self._gridsize).astype(int)
        self._edges = (np.linspace(0, self._box_size[0], num=hbins[0] + 1, endpoint=True) + (x - sd[0]),
                    np.linspace(0, self._box_size[1], num=hbins[1] + 1, endpoint=True) + (y - sd[1]),
                    np.linspace(0, self._box_size[2], num=hbins[2] + 1, endpoint=True) + (z - sd[2]))
        origin = (self._edges[0][0], self._edges[1][0], self._edges[2][0])

        # get the mask of accesible voxels that will be used for the free energy calculation
        self._build_accessible_mask()

        # Get positions and atom types
        positions = self._get_positions()

        if self.use_atomtypes: # turn on for atomtype density
            self._type_histograms = {}  # Create per-type histograms

            # Map atom types to atoms in the system
            mapped_atomtypes = self._map_atomtypes(self.atomtypes_definitions)

            # Get atom types for all frames as a single array
            atom_types_array = np.tile(mapped_atomtypes, self._nframes)

            for atom_type in self.atomtypes_dict.keys():

                self.logger.info(f"Processing atom type: {atom_type}")

                # Select positions for this atom type
                mask = np.char.startswith(atom_types_array.astype(str), atom_type)

                type_positions = positions[mask]

                # Skip empty positions for a type
                if len(type_positions) == 0:
                    self.logger.warning(f"Skipping atom type {atom_type} as it has no positions.")
                    continue

                # Generate histogram for this type
                hist, _ = np.histogramdd(type_positions, bins=self._edges)
                self._type_histograms[atom_type] = Grid(hist, origin=origin, delta=self._gridsize)

            # Create a combined density grid by summing all atom types
            total_hist = sum(grid.grid for grid in self._type_histograms.values())
            self._histogram = Grid(total_hist, origin=origin, delta=self._gridsize)
            self._density = Grid(_grid_density(total_hist), origin=origin, delta=self._gridsize)
        else:
            hist, _ = np.histogramdd(positions, bins=self._edges)
            self._histogram = Grid(hist, origin=origin, delta=self._gridsize)
            self._density = Grid(_grid_density(hist), origin=origin, delta=self._gridsize)

        # Calculate the number of accessible voxels, once per trajectory
        self._build_accessible_mask()

    def _get_positions(self, start=0, stop=None):
        positions = self._positions[start:stop, :, :]
        new_shape = (positions.shape[0] * positions.shape[1], 3)
        positions = positions.reshape(new_shape)

        return positions           
    
    def _build_accessible_mask(self, traj_step=5, probe_radius=1.4, export=True):
        """
        Build a boolean grid where True = voxel is solvent-accessible.
        Use both water-oxygen and one cosolvent heavy atom per mol to capture small cavities and
        also hydrophobic pockets not accessed by water.
        The grid is dilated by `probe_radius` to account for the size of the probe.

        Parameters
        ----------
        traj_step   : int   use every `traj_step`-th frame to save time
        probe_radius: float Å, radius you want to allow beyond sampled O positions
        export      : bool  if True, export the grid to a .dx file
        """
        if hasattr(self, "_n_accessible_voxels"):
            return  # already built

        # collect water-oxygen + one cosolvent heavy-atom per mol
        O_sel  = self._u.select_atoms("resname HOH WAT and name O")
        # probeH = self._u.select_atoms("not resname HOH WAT and not name H* and prop q<0.1").unique  # e.g. any neutral heavy atom

        coords = []
        for ts in self._u.trajectory[::traj_step]: # this stride saves time
            coords.append(O_sel.positions.copy())
            # coords.append(probeH.positions.copy())
        coords = np.vstack(coords)

        # histogram into current grid
        hist, _ = np.histogramdd(coords, bins=self._edges)
        mask = hist > 0

        # dilate by ≈ probe_radius
        n_iter = int(round(probe_radius / self._gridsize))
        mask = binary_dilation(mask, iterations=max(1, n_iter))

        # count and save the mask
        self._n_accessible_voxels = int(mask.sum())
        grid_vol = self._gridsize ** 3
        self.logger.info(f"Number of accessible voxels: {self._n_accessible_voxels:.2f}")
        self.logger.info(f"Volume of accessible voxels: {self._n_accessible_voxels/1000 * grid_vol:.2f} nm³")
        
        if export:
            mask_grid = mask.astype(float)
            grid = Grid(mask_grid, edges=self._edges)
            grid.export(f"solvent_accessible_map.dx")

        return

    def _map_atomtypes(self, atomtypes_definitions:list=None) -> np.ndarray:
        """Maps atom types to their respective categories based on SMARTS patterns.
        Some useful definitions here:  https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
        :param atomtypes_definitions: A list of atom types definitions based on SMARTS patterns.
        :type atomtypes_definitions: list
        :return: Array of mapped atom types.
        :rtype: np.ndarray
        """
        
        # select atoms based on SMARTS patterns
        self.atomtypes_dict = {atomtype['atype']: self._ag.select_atoms(f"smarts {atomtype['smarts']}") for atomtype in atomtypes_definitions}
        # Count the number of atoms by type, this is required for the free energy calculation
        self._n_atoms_by_type = {key: ag.n_atoms for key, ag in self.atomtypes_dict.items()}
        self.logger.debug(f"Atom types count: {self._n_atoms_by_type}")

        self.atomtypes_dict = {key: np.unique(ag.atoms.types) for key, ag in self.atomtypes_dict.items()}

        mapped_atomtypes = np.zeros_like(self._ag.atoms.types, dtype=object)

        # Map atom types to their respective categories
        for atom in self._ag.atoms.types:
            for key, atomtypes in self.atomtypes_dict.items():
                if atom in atomtypes:
                    mapped_atomtypes[np.where(self._ag.atoms.types == atom)] = key
                    break
            # else:
            #     mapped_atomtypes[np.where(self._ag.atoms.types == atom)] = 'OTHER'
        
        return mapped_atomtypes
    
    def atomic_grid_free_energy(self, temperature=300., atom_radius=1.4, smoothing=True):
        """Compute grid free energy by boltzmann inversion of the occupancy histogram at a given temperature.
        Optionally, the free energy map can be smoothed using a Gaussian filter and some tricks.
        
        :param temperature: Temperature in Kelvin (default 300K)
        :param atom_radius: Atomic radius for smoothing (default 1.4A)
        :param smoothing: Apply smoothing to the free energy map (default True)

        """
        
        if self.use_atomtypes:
            for atom_type, grid in self._type_histograms.items():
                n_atoms_type = self._n_atoms_by_type[atom_type]
                agfe = _grid_free_energy(grid.grid, n_atoms_type, self._nframes, self._n_accessible_voxels, temperature)
                # self.logger.debug(f"Free energy for {atom_type}: MIN: {np.min(agfe):.2f} kcal/mol, MAX: {np.max(agfe):.2f} kcal/mol")
                if smoothing:
                    agfe = _smooth_grid_free_energy(agfe, sigma=atom_radius /35., energy_cutoff=0)
                
                self.logger.info(f"Free energy for {atom_type}: MIN: {np.min(agfe):.2f} kcal/mol, MAX: {np.max(agfe):.2f} kcal/mol")
                self._type_histograms[atom_type] = Grid(agfe, edges=grid.edges)
        else:
            agfe = _grid_free_energy(self._histogram.grid, self._n_atoms, self._nframes, self._n_accessible_voxels, temperature)

            if smoothing:
                # We divide by 3 in order to have radius == 3 sigma
                agfe = _smooth_grid_free_energy(agfe, sigma=atom_radius / 3., energy_cutoff=0)
                self.logger.info(f"Free energy: MIN: {np.min(agfe):.2f} kcal/mol, MAX: {np.max(agfe):.2f} kcal/mol")

            self._agfe = Grid(agfe, edges=self._histogram.edges)

        return

    def export_histogram(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export histogram maps
        """
        _export(fname, self._histogram, gridsize, center, box_size)

    def export_density(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export density maps, either for the total density or for each atom type
        """
        if self.use_atomtypes:
            for atom_type, grid in self._type_histograms.items():
                density_fname=fname.replace('map_rawdensity', f'map_density_{atom_type}')
                _export(density_fname, grid, gridsize, center, box_size)
        else:
            _export(fname, self._density, gridsize, center, box_size)

    def export_atomic_grid_free_energy(self, fname, gridsize=0.5, center=None, box_size=None):
        """ Export atomic grid free energy, either for the total free energy or for each atom type
        """
        if self.use_atomtypes:
            for atom_type, grid in self._type_histograms.items():
                gfe_fname=fname.replace('map_agfe', f'map_agfe_{atom_type}')
                _export(gfe_fname, grid, gridsize, center, box_size)
        else:
            _export(fname, self._agfe, gridsize, center, box_size)
        
class Report:
    """Report class. This is the main class that takes care of post MD simulation processing and analysis.
    """
    def __init__(self, statistics_file, traj_file, top_file, cosolvent_names, out_path):
        """_summary_

        :param statistics_file: log file generated by MD Simulation. In CosolvKit this is called statistics.csv
        :type statistics_file: str
        :param traj_file: Trajectory file generated by MD Simulation.
        :type traj_file: str
        :param top_file: Topology file generated by CosolvKit.
        :type top_file: str
        :param cosolvents_names: list of cosolvent names to analyze.
        :type cosolvents_names: list[str]
        :param out_path: path to where to save the results.
        :type out_path: str
        """
        
        # Setup logging
        self.logger = logging.getLogger(__name__)

        self.cosolvent_names = cosolvent_names
        if cosolvent_names is None or len(cosolvent_names) == 0:
            self.logger.info("No cosolvents specified for the density analysis. At least one cosolvent is required.")
            sys.exit(1)

        self.trajectory = traj_file
        self.topology = top_file
        self.universe = Universe(self.topology, self.trajectory)

        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
        # this will be checked when creating the sessions
        self.avg_pdb_path = os.path.join(self.out_path, "averaged_trajectory.pdb")

        self._volume = None
        self._temperature = None
        self._potential_energy = None
        
        self.statistics = statistics_file
        if statistics_file is not None:
            self._potential_energy, self._temperature, self._volume = self._get_temp_vol_pot(self.statistics)
        
        return
    
    def _plot_rmsf(self, rmsf_df):
        """Plots the RMSF of the protein residues.

        :param rmsf_df: dataframe with the RMSF data per atom.
        :type rmsf_df: pd.DataFrame
        """
        # Group by residue and calculate the mean RMSF
        rmsf_df = rmsf_df.groupby('residue').mean().reset_index()

        fig, ax = plt.subplots()
        ax.plot(rmsf_df['residue'], rmsf_df['RMSF'])
        ax.set_xlabel('Residue');        ax.set_ylabel('RMSF (A)')
        ax.set_title('RMSF by residue')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_path, "rmsf_by_residue.png"))
        plt.close()
        return
    
    def _plot_rmsd(self, rmsd_df, avg_selection):
        """Plots the RMSD of the protein residues.

        :param rmsd_df: dataframe with the RMSD data per frame.
        :type rmsd_df: pd.DataFrame
        :param avg_selection: selection string to average the trajectory.
        :type avg_selection: str
        """
        fig, ax = plt.subplots()
        ax.plot(rmsd_df['Frame'], rmsd_df[avg_selection])
        ax.set_xlabel('Frame');        ax.set_ylabel('RMSD (A)')
        ax.set_title(f'RMSD to Avg Structure - {avg_selection}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_path, "rmsd_by_frame.png"))
        plt.close()
        return


    def _rmsf_analysis(self, avg_selection, align_selection):
        """Computes the RMSF of the protein residues. 
        The funciton also generates the average structure of the trajectory and colors the residues by RMSF.
        This conformtion will be used as a reference for the pymol session.
        As for the density analysis, this function also asumes that the trajectory is already aligned.
        :param avg_selection: selection string to average the trajectory.
        :type avg_selection: str
        :param align_selection: selection string to align the trajectory to the average.
        :type align_selection: str
        """
        self.logger.info("Computing RMSF...")
        average = align.AverageStructure(self.universe, None,
                                        select=avg_selection,
                                        ).run()

        u_avg = average.results.universe

        R = rms.RMSD(self.universe, u_avg,  
                    select=align_selection,  
                    groupselections=[avg_selection],
                    ).run()

        avg_selection = avg_selection + ' and not name H*' # remove hydrogens from the selection
        rmsd_df = pd.DataFrame(R.rmsd,
                  columns=['Frame', 'Time (ps)', align_selection, avg_selection])

        rmsd_df.to_csv(os.path.join(self.out_path, "rmsd_by_frame.csv"))
        self._plot_rmsd(rmsd_df, avg_selection)

        selection = self.universe.select_atoms(avg_selection)
        residues = selection.resids
        rmsf = RMSF(selection).run()

        rmsf_df = pd.DataFrame({'residue': residues, 'RMSF': rmsf.results.rmsf})
        rmsf_df.index.name = 'atom'

        self.universe.add_TopologyAttr('tempfactors') # add empty attribute for all atoms
        for residue, r_value in zip(selection.residues, rmsf.results.rmsf):
            residue.atoms.tempfactors = r_value
        
        selection.write(self.avg_pdb_path)
        rmsf_df.to_csv(os.path.join(self.out_path, "rmsf_by_atom.csv"))    
        self._plot_rmsf(rmsf_df)

        return

    def survivalProbability_analysis(self, 
                                    cosolvent_names: list[str] = None,
                                    candidate_residues: list[tuple] = None, 
                                    radius: float = 5, 
                                    max_tau: int = 100,
                                    intermittency: int = 1
                                    ):
        """Computes the survival probability of the cosolvent around a spherical zone centered 
        at the COM of the candidate residues. Uses the waterdynamics package to compute the survival 
        probability. The results are saved in a csv file and a plot is generated.
        More info: https://www.mdanalysis.org/waterdynamics/api.html#waterdynamics.SurvivalProbability
        
        ProTip: pass the residue name of water to analyze the water survival probability.

        :param cosolvent_names: list of cosolvent names to analyze.
        :type cosolvent_names: list[str]
        :param candidate_residues: list of tuples with the candidate residues to analyze.
        :type candidate_residues: list[tuple]
        :param radius: radius of the sphere to analyze.
        :type radius: float
        :param max_tau: maximum tau to analyze.
        :type max_tau: int
        :param intermittency: intermittency of the interaction, defaults to 1.
        :type intermittency: int, optional
        """
        try:
            from waterdynamics import SurvivalProbability as SP
        except ImportError:
            raise ImportError("waterdynamics package is required for Survival Probability analysis. Please install it.")
        
        assert candidate_residues is not None, "Error! You need to pass the residues to analyze for the survival probability."
        if cosolvent_names is None:
            self.logger.warning("No cosolvent specified for the survival probability analysis. Using all cosolvents...")
            cosolvent_names = self.cosolvent_names

        for cosolvent_name in cosolvent_names:
            data = []
            for res_idx, residue_group in enumerate(candidate_residues):
                # Ensure residue_group is not an int when only one residue is passed
                if isinstance(residue_group, int):
                    residue_group = (residue_group,)
                
                resids = ' or '.join([f'resid {res}' for res in residue_group])
                self.logger.info(f"Analyzing residues: {' '.join([str(i) for i in residue_group])} for cosolvent {cosolvent_name}")
                select = f"resname {cosolvent_name} and sphzone {radius} ({resids})"
                # self.logger.info(f"Selection string: {select}")

                sp = SP(self.universe, select, verbose=True)
                # The default intermittency is continuous (0).
                sp.run(tau_max=max_tau, residues=False, intermittency=intermittency)

                for tau, sp_value in zip(sp.tau_timeseries,  sp.sp_timeseries):
                        data.append({'Group': res_idx, 'Residues':residue_group, 
                                     'Time': tau, 'SP': sp_value, 'Cosolvent': cosolvent_name})

            df_sp = pd.DataFrame(data)
            df_sp.to_csv(os.path.join(self.out_path, f"survival_probability_{cosolvent_name}.csv"))

            # Plot the survival probability
            sns.lineplot(data=df_sp, x='Time', y='SP', hue='Group', palette='flare')
            plt.xlabel('lagtimes'); plt.ylabel('Survival Probability')
            plt.title(f'Cosolvent {cosolvent_name} Survival Probability by Residue Group')

            # Update legend to show residue indices instead of group numbers
            handles, labels = plt.gca().get_legend_handles_labels()
            new_labels = [f"Residues: {', '.join(map(str, candidate_residues[int(label)] if isinstance(candidate_residues[int(label)], (list, tuple)) else (candidate_residues[int(label)],)))}" for label in labels]
            plt.legend(handles, new_labels, title='')
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_path, f"survival_probability_{cosolvent_name}.png"))
            plt.close()

        return

    def generate_report(self, 
                        equilibration:bool=True, rmsf:bool=True, rdf:bool=True,
                        avg_selection:str="protein",
                        align_selection:str="protein and name CA"
                        ):
        """Creates the main plots for RDFs, autocorrelations and equilibration.
        :param equilibration: if True, the equilibration analysis will be performed, defaults to True
        :type equilibration: bool, optional
        :param rmsf: if True, the RMSF analysis will be performed, defaults to True
        :type rmsf: bool, optional
        :param rdf: if True, the RDF analysis will be performed, defaults to True
        :type rdf: bool, optional
        :param avg_selection: selection string to average the trajectory, defaults to "protein". Change this if you have other molecules in the system or things like DNA/RNA.
        :type avg_selection: str, optional
        :param align_selection: selection string to align the trajectory to the average, defaults to "protein and name CA". Change this if you have other molecules in the system or things like DNA/RNA.
        :type align_selection: str, optional
        """
        self.logger.info("Generating report...")

        if equilibration:
            if self.statistics is None:
                self.logger.warning("No statistics file found. Skipping equilibration analysis.")
            else:
                self._equilibration_analysis()
        if rmsf:
            self._rmsf_analysis(avg_selection, align_selection)
        if rdf:
            self._rfd_analysis(self.universe, self.cosolvent_names)
   
        return
    
    def _load_atomtype_definitions(self, atomtypes_fname:str=None) -> list:
        """Loads atom type definitions from a json file.
        :param atomtypes_fname: Path to the json file with atom type definitions.
        :type atomtypes_fname: str
        :return: A list of atom types definitions based on SMARTS patterns.
        :rtype: list
        """
        DARC_default_location = os.path.join(os.path.dirname(__file__), 'data/dacar_atomtypes.json')
        if (atomtypes_fname is None) or (not os.path.exists(atomtypes_fname)):
            self.logger.warning("Warning: Atom types definitions file not found or not provided.\n Using default DACar atom types definitions.")
            try:
                with open(DARC_default_location) as fi:
                    data = json.load(fi)
                    typer_name = next(iter(data))
                    atomtypes_definitions = data[typer_name]
                    self.logger.info(f"Loaded {typer_name} atom types definitions.")
            except FileNotFoundError:
                self.logger.error(f"Error: Default DACar atom types definitions not found @ {DARC_default_location}")
                sys.exit(1)
        else:
            with open(atomtypes_fname) as fi:
                data = json.load(fi)
                typer_name = next(iter(data))
                atomtypes_definitions = data[typer_name]
                self.logger.info(f"Loaded {typer_name} atom types definitions.")

        return atomtypes_definitions
        
    def generate_density_maps(self, 
                              cosolvent_names:list[str]=None,
                              use_atomtypes:bool=True,
                              atomtypes_definitions:str=None, 
                              gridsize:float=0.5,
                              temperature:float=None, 
                              ):
        """Generates the density maps for all the cosolvents especified. It is possible to use atomtypes for the analysis.
        If no atomtypes are specified, the default atomtypes are used: HBD, HBA, Car. 
        :param cosolvent_names: list of cosolvent names to analyze, defaults to None.
        :type cosolvent_names: list[str], optional
        :param use_atomtypes: if True, the density analysis will be performed using atomtypes, defaults to True
        :type use_atomtypes: bool, optional
        :param atomtypes_definitions: path to the json file with the atom types definitions, defaults to None
        :type atomtypes_definitions: str, optional
        :param gridsize: gridsize to use for the analysis, defaults to 0.5
        :type gridsize: float, optional
        :param temperature: temperature to use for the analysis, defaults to None
        :type temperature: float, optional

        """
        self.logger.info("Generating density maps...")

        if cosolvent_names is None or len(cosolvent_names) == 0:
            self.logger.warning("No cosolvents specified for the density analysis. Using all cosolvents specified in the Report class...")
            cosolvent_names = self.cosolvent_names

        if temperature is None: # If temperature is not passed, so we take the last one from statistics
            if self._temperature is None:
                self.logger.error("No temperature found. Please provide a temperature or a statistics file for the density analysis.")
                sys.exit(1)
            else:
                self.logger.warning(f'No temperature provided. Using the last temperature from the statistics file: {self._temperature[-1]}')
                temperature = self._temperature[-1]

        # load the atomtypes definitions
        if use_atomtypes:
            atomtypes_definitions = self._load_atomtype_definitions(atomtypes_definitions)

        for cosolvent in cosolvent_names:
            atomgroup = self.universe.select_atoms(f"resname {cosolvent}")
            if atomgroup.n_atoms == 0:
                self.logger.error("Error: the provided selection didn't match any atoms.")
                sys.exit(1)

            analysis = Analysis(atomgroup, 
                                gridsize=gridsize,
                                use_atomtypes=use_atomtypes, 
                                atomtypes_definitions=atomtypes_definitions, 
                                verbose=True)
            analysis.run()
            analysis.export_density(os.path.join(self.out_path, f"map_rawdensity_{cosolvent}.dx"))
            analysis.atomic_grid_free_energy(temperature, smoothing=True)
            analysis.export_atomic_grid_free_energy(os.path.join(self.out_path, f"map_agfe_{cosolvent}.dx"))

        return
    
    def _get_temp_vol_pot(self, statistics_file):
        """Returns temperature, volume and potential energy of the system during the MD simulation.

        :param statistics_file: log file generated by the MD simulation. In CosolvKit is statistics.csv.
        :type statistics_file: str
        :return: potential energy, temperature and volume of the system for each frame.
        :rtype: tuple(list, list, list)
        """
        df = pd.read_csv(statistics_file)
        pot_e = list(df["Potential Energy (kJ/mole)"])
        temp = list(df["Temperature (K)"])
        vol = list(df["Box Volume (nm^3)"])
        return pot_e, temp, vol

    def _equilibration_analysis(self):
        """Plot equilibration data: potential energy, temperature and volume.

        """
        self.logger.info('Plotting equilibration data..')

        fig, axs = plt.subplots(3, 1, figsize=(12, 6))

        axs[0].plot(self._potential_energy, color='green', linewidth=2)
        axs[0].set_title('Potential Energy',)
        axs[0].set_xlabel('Frames')
        axs[0].set_ylabel('Energy (kJ/mole)')
    
        axs[1].plot(self._volume, color='blue', linewidth=2)
        axs[1].set_title('Volume')
        axs[1].set_xlabel('Frames')
        axs[1].set_ylabel('Volume (nm³)')

        axs[2].plot(self._temperature, color='red', linewidth=2)
        axs[2].set_title('Temperature')
        axs[2].set_xlabel('Frames')
        axs[2].set_ylabel('Temperature (K)')

        plt.tight_layout()
        plt.savefig(os.path.join(self.out_path, "equilibration_statistics.png"))
        plt.close()

        return 
    
    def _rfd_analysis(self, universe: Universe, cosolvent_names: list):
        """Generates the plots for RDFs and Autocorrelations.

        :param universe: MD Analysis Universe that is created from the topology and trajectories.
        :type universe: Universe
        :param cosolvent_names: list of cosolvent resnames in the system
        :type cosolvent_names: list
        """
        np.seterr(divide='ignore', invalid='ignore')

        outpath = os.path.join(self.out_path, "RDFs")
        os.makedirs(outpath, exist_ok=True)

        self.logger.info("Running RDF analysis...")

        wat_resname = "HOH"
        # if top.endswith("cosolv_system.prmtop"):
        #     wat_resname = "WAT"
        oxygen_atoms = universe.select_atoms(f"resname {wat_resname} and name O")
        sim_frames = len(universe.trajectory)
        step_size = int(sim_frames/250)
        if step_size < 1:
            step_size = 1
        n_bins = 150
        for cosolvent_name in cosolvent_names:
            r_max = 15
                
            cosolvent_residues = universe.select_atoms(f'resname {cosolvent_name}')
            atoms_names = cosolvent_residues.residues[0].atoms.names
            for cosolvent_atom in set(atoms_names):
                max_y = 0
                if "H" in cosolvent_atom: continue
                self.logger.info(f"Analysing {cosolvent_name}-{cosolvent_atom}")
                fig, ax = plt.subplots(2, 2, sharex=False, sharey=False)
                plt.tight_layout(pad=3.0)
                # Here compute RDF between same atoms and different molecules
                atoms = cosolvent_residues.select_atoms(f'name {cosolvent_atom}')
                irdf = rdf.InterRDF(atoms, atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
                irdf.run(start=0, step=step_size)
                max_y = max(irdf.results.rdf)
                ax[0][0].plot(irdf.results.bins, irdf.results.rdf, label="RDF")
                ax[0][0].set_xlabel(r'$r$ $\AA$')
                ax[0][0].set_ylabel("$g(r)$")
                ax[0][0].set_title(f"RDF-{cosolvent_name} {cosolvent_atom}")
                # ax[0][0].set_title(f"RDF-{cosolvent_name} {cosolvent_atom} every {step_size} frames")
                leg = ax[0][0].legend(handlelength=0, handletextpad=0, fancybox=True)
                for item in leg.legendHandles:
                    item.set_visible(False)
                
                ax[1][0] = self._plot_autocorrelation(data=irdf.results.rdf,
                                                        ax=ax[1][0], 
                                            cosolvent_name1=cosolvent_name, 
                                            cosolvent_atom1=cosolvent_atom, 
                                            cosolvent_name2=cosolvent_name, 
                                            cosolvent_atom2=cosolvent_atom)
                # Here compute RDF between atom and water's oxygen
                irdf = rdf.InterRDF(atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max))
                irdf.run(start=0, step=step_size)
                max_y = max(irdf.results.rdf)
                irdf.run()
                ax[0][1].plot(irdf.results.bins, irdf.results.rdf, label="RDF")
                ax[0][1].set_xlabel(r'$r$ $\AA$')
                ax[0][1].set_ylabel("$g(r)$")
                ax[0][1].set_title(f"RDF {cosolvent_name} {cosolvent_atom}-HOH O")
                # ax[0][1].set_title(f"RDF {cosolvent_name} {cosolvent_atom}-HOH O every {step_size} frames")
                leg = ax[0][1].legend(handlelength=0, handletextpad=0, fancybox=True)
                for item in leg.legendHandles:
                    item.set_visible(False)

                self._plot_autocorrelation(data=irdf.results.rdf, 
                                             ax=ax[1][1], 
                                             cosolvent_name1=cosolvent_name, 
                                             cosolvent_atom1=cosolvent_atom, 
                                             cosolvent_name2="HOH", 
                                             cosolvent_atom2="O")
                if outpath is not None:
                    plt.savefig(f"{outpath}/rdf_{cosolvent_name}_{cosolvent_atom}.png")
                plt.close()
        
        # Finally do waters
        self.logger.info("Analysing water")
        r_max = 8.5
        fig, ax = plt.subplots()
        plt.setp(ax, xlim=(0, r_max+1))
        irdf = rdf.InterRDF(oxygen_atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
        irdf.run(start=0, step=50)
        # irdf.run()
        ax.plot(irdf.results.bins, irdf.results.rdf, label="RDF")
        ax.set_xlabel(r'$r$ $\AA$')
        ax.set_ylabel("$g(r)$")
        ax.set_title(f"RDF-HOH O every 50 frames")
        leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
        for item in leg.legendHandles:
            item.set_visible(False)
        if outpath is not None:
            plt.savefig(f"{outpath}/rdf_HOH_O.png")
        plt.close()
        return

    def _autocorrelation(self, data):        
        """Gets the autocorrelation values.

        :param data: list of data for which the autocorrelation has to be computed.
        :type data: list
        :return: list of autocorrelations.
        :rtype: list
        """
        n = len(data)
        mean = np.mean(data)
        autocorr = correlate(data - mean, data - mean, mode='full', method='auto')
        return autocorr[n - 1:]
    
    def _plot_autocorrelation(self, data, ax, cosolvent_name1=None, cosolvent_atom1=None, cosolvent_name2=None, cosolvent_atom2=None):
        """Plots autocorrelations.

        :param data: list of data points for which we want to plot autocorrelations.
        :type data: list
        :param ax: matplotlib axis to add the autocorrelation to the RDF plot.
        :type ax: matplotlib.pyplot.axisß
        :param cosolvent_name1: name of the first cosolvent molecule, defaults to None
        :type cosolvent_name1: str, optional
        :param cosolvent_atom1: name of the first atom, defaults to None
        :type cosolvent_atom1: str, optional
        :param cosolvent_name2: name of the second cosolvent molecule, defaults to None
        :type cosolvent_name2: str, optional
        :param cosolvent_atom2: name of the second atom, defaults to None
        :type cosolvent_atom2: str, optional
        :return: the axis with the autocorrelation plot
        :rtype: matplotlib.pyplot.axis
        """
        title = f"{cosolvent_name1} {cosolvent_atom1}-{cosolvent_name2} {cosolvent_atom2}"
        data = data[0::2]
        autocorr_values = self._autocorrelation(data)
        # Normalize autocorrelation values for better plotting
        normalized_autocorr = autocorr_values / np.max(np.abs(autocorr_values))
        lags = np.arange(0, len(autocorr_values))
        pd.plotting.autocorrelation_plot(pd.Series(normalized_autocorr), ax=ax, label="Autocorrelation")
        ax.grid(False)
        ax.set_xlim([0, len(autocorr_values)])
        ax.set_title(title)
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        leg = ax.legend(handlelength=0, handletextpad=0, fancybox=True)
        for item in leg.legendHandles:
            item.set_visible(False)
        return ax
    
    def generate_pymol_session(self, 
                               density_files:Union[str, list]=None,
                               selection_string:str=None, 
                               reference_pdb:str=None):
        """Generate a PyMol session from the density maps. The average structure is always used as a reference. 
        You can also include a reference pdb file and specify the residues of interest. 

        :param density_files: list of density files to include in the PyMol session, or a directory with the density files, or a single density file.
        :type density_files: Union[str, list]
        :param reference_pdb: reference pdb file to load in PyMol.
        :type reference_pdb: str
        :param selection_string: PyMol selection string if willing to specify target residues.
        :type selection_string: str
        """

        if os.path.isfile(density_files):
            density_files = [density_files]
        elif os.path.isdir(density_files):
            density_files = [os.path.join(density_files, f) for f in os.listdir(density_files) if f.endswith('.dx')]
        elif isinstance(density_files, list):
            pass
        else:
            self.logger.error("Please provide a list of density files to include in the PyMol session.")
            return
        
        colors = ['marine', 
                  'orange', 
                  'magenta',
                  'salmon',
                  'purple']

        assert len(density_files) <= len(colors), "Error! Too many density files, not enough colors available!"
        
        if not os.path.exists(self.avg_pdb_path):
            # if the average pdb was not generated in the report, we generate it here
            self._rmsf_analysis(avg_selection='protein')

        structures = {'average_structure': self.avg_pdb_path}
        if reference_pdb is not None and reference_pdb.endswith('.pdb'):
            reference_pdb_name = os.path.basename(reference_pdb).split('.')[0]
            structures[reference_pdb_name] = reference_pdb

        cmd_string = ""

        for structure_name, pdb_path in structures.items():

            # Load topology and first frame of the trajectory
            cmd.load(pdb_path, structure_name)
            cmd_string += f"cmd.load('{pdb_path}', {structure_name})\n"

            # Set structure's color
            cmd.color("grey50", f"{structure_name} and name C*")
            cmd_string += f"cmd.color('grey50', '{structure_name} and name C*')\n"

        for color, density in zip(colors, density_files):
            dens_name = os.path.basename(density).split('.')[0]
            # self.logger.info(f"Loading density map: {dens_name}")

            dx_data = _read_dx(density)
            # calculate 0.001 quantile. This works for agfe maps
            dx_01 = np.quantile(dx_data.grid, 0.001)
            # self.logger.info(f"0.1% of the density map is: {dx_01}")

            cmd.load(density, f'{dens_name}_map')
            cmd_string += f"cmd.load('{density}', '{dens_name}_map')\n"

            # Create isomesh for hydrogen bond probes
            cmd.isomesh(f'{dens_name}_mesh', f'{dens_name}_map', dx_01)
            cmd_string += f"cmd.isomesh('{dens_name}_mesh', '{dens_name}_map', {dx_01})\n"

            # Color the hydrogen bond isomesh
            cmd.color(color, f'{dens_name}_mesh')
            cmd_string += f"cmd.color('{colors}', '{dens_name}_mesh')\n"
            
        # Show sticks for the residues of interest
        if selection_string != '':
            cmd.show("sticks", selection_string)
            cmd_string += f"cmd.show('sticks', '{selection_string}')\n"

        cmd.hide("spheres")
        # Set valence to 0 - no double bonds
        # cmd.set("valence", 0)
        cmd.set('specular', 1)
        # Set cartoon_side_chain_helper to 1 - less messy
        cmd.set("cartoon_side_chain_helper", 1)
        # color protein by b-factor
        cmd.spectrum("b", "blue_white_red", selection_string)
        # Set background color
        cmd.bg_color("white") #grey80

        cmd_string += "cmd.hide('spheres')\n"
        # cmd_string += "cmd.set('valence', 0)\n"
        cmd_string += "cmd.set('specular', 1)\n"
        cmd_string += "cmd.set('cartoon_side_chain_helper', 1)\n"
        cmd_string += f"cmd.spectrum('b', 'blue_white_red', '{selection_string}')\n"
        cmd_string += "cmd.bg_color('white')"
        
        with open(os.path.join(self.out_path, "pymol_session_cmd.pml"), "w") as fo:
            fo.write(cmd_string)
            
        cmd.save(os.path.join(self.out_path, "pymol_results_session.pse"))
        return
    
    def generate_vmd_session(self, 
                             density_files:Union[str, list]=None, 
                            ):
        """
        Generate a VMD session script to visualize the trajectory and density.
        This i very basic and can be improved in the future, but I like more VMD or ChimeraX 
        for visualizing the densities becuase they have sliders and are more interactive.

        :param output_vmd_file: Path to save the VMD session script (.vmd)
        :type output_vmd_file: str
        """
        # FIXME at some point like for pymol 
        isovalue = 1.0 
        output_vmd_file = os.path.join(self.out_path, "vmd_session.vmd")

        # Get absolute paths for the files
        topology_abs_path = os.path.abspath(self.topology)
        trajectory_abs_path = os.path.abspath(self.trajectory)

        vmd_script = f"""
    # VMD visualization script

    # Load topology and trajectory
    mol new {topology_abs_path} type parm7
    mol addfile {trajectory_abs_path} type netcdf waitfor all

    # Set up protein visualization
    mol delrep 0 top
    mol representation NewCartoon
    mol color Structure
    mol selection "protein"
    mol material Opaque
    mol addrep top"""
        
        # Load density maps
        for i,density in enumerate(density_files):
            density_dx_abs_path = os.path.abspath(density)
            vmd_script += f"""

    # Load density map
    mol new {density_dx_abs_path} type dx waitfor all
    mol representation Isosurface {isovalue} 0 0 0 1
    mol color ColorID {i}
    mol material Transparent
    mol addrep top"""
        
        vmd_script += f"""

    color Display Background white

    save_state {output_vmd_file}
    """

        # Write script to file
        with open(output_vmd_file, "w") as f:
            f.write(vmd_script)

        self.logger.info(f"VMD session script saved as {output_vmd_file}")
        
        return