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
import numpy as np
from scipy import spatial
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate
from scipy.interpolate import RegularGridInterpolator
from gridData import Grid
import MDAnalysis as mda
from MDAnalysis import Universe
from MDAnalysis.analysis import rdf
from MDAnalysis.analysis.base import AnalysisBase
import matplotlib.pyplot as plt
import pandas as pd
import pymol
from pymol import cmd, stored
from cosolvkit.cosolvent_system import CosolventMolecule


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

class Report:
    def __init__(self, log_file, traj_file, top_file, cosolvents_path):
        self.statistics = log_file
        self.trajectory = traj_file
        self.topology = top_file
        self.universe = Universe(self.topology, self.trajectory)
        self.cosolvents = list()

        with open(cosolvents_path) as fi:
            cosolvents_d = json.load(fi)
        for cosolvent in cosolvents_d:
            self.cosolvents.append(CosolventMolecule(**cosolvent))
        
        self._volume = None
        self._temperature = None
        self._potential_energy = None
        self._potential_energy, self._temperature, self._volume = self._get_temp_vol_pot(self.statistics)
        return
    
    def generate_report(self, out_path):
        print("Generating report...")
        # setup results folders
        report_path = os.path.join(out_path, "report")
        rdf_path = os.path.join(report_path, "rdf")
        autocorrelation_path = os.path.join(report_path, "autocorrelation")
        os.makedirs(report_path, exist_ok=True)
        os.makedirs(rdf_path, exist_ok=True)
        os.makedirs(autocorrelation_path, exist_ok=True)

        # Generate equilibration plot
        self._plot_temp_vol_pot(report_path)
        print("Plotting RDFs")
        irdf = self._rdf_mda(self.universe, self.cosolvents, rdf_path)
        print("Plotting RDF autocorrelations")
        for pair in irdf:
            cosolvent_name1, cosolvent_atom1, cosolvent_name2, cosolvent_atom2 = pair
            self._plot_autocorrelation(data=irdf[pair],
                                       cosolvent_name1=cosolvent_name1,
                                       cosolvent_atom1=cosolvent_atom1,
                                       cosolvent_name2=cosolvent_name2,
                                       cosolvent_atom2=cosolvent_atom2,
                                       outpath=autocorrelation_path)
        return
    
    def generate_density_maps(self, out_path, analysis_selection_string=""):
        print("Generating density maps...")
        volume = self._volume[-1]
        temperature = self._temperature[-1]
        if analysis_selection_string == "":
            print("No cosolvent specified for the densities analysis. Generating a density map for each cosolvent.")
            for cosolvent in self.cosolvents:
                selection_string = f"resname {cosolvent.resname}"
                self._run_analysis(selection_string=selection_string,
                                   volume=volume,
                                   temperature=temperature,
                                   out_path=out_path,
                                   cosolvent_name=cosolvent.resname)
        else:
            print(f"Generating density maps for the following selection string: {analysis_selection_string}")
            self._run_analysis(selection_string=analysis_selection_string, 
                               volume=volume,
                               temperature=temperature,
                               out_path=out_path,
                               cosolvent_name=None)
        return
    
    def generate_pymol_reports(self, topology, trajectory, density_files, selection_string, out_path):
        colors = ['marine', 
                  'orange', 
                  'magenta',
                  'salmon',
                  'purple']
        
        assert len(density_files) < len(colors), "Error! Too many density files, not enough colors available!"
        base_path = os.getcwd()
        topology = os.path.join(base_path, topology)
        trajectory = os.path.join(base_path, trajectory)
        # density_file = os.path.join(base_path, density_file)
        cmd_string = ""
        # Load topology and first frame of the trajectory
        cmd.load(topology, "structure")
        cmd.load_traj(trajectory, start=0, stop=1)
        cmd_string += f"cmd.load('{topology}', 'structure')\n"
        cmd_string += f"cmd.load_traj('{trajectory}', start=0, stop=1)\n"
        # Load density
        for idx, density in enumerate(density_files):
            cmd.load(density, f"density_map_{idx}")
            cmd_string += f"cmd.load('{density}', 'density_map_{idx}')\n"

        # Set structure's color
        cmd.color("grey50", "structure and name C*")
        # Remove solvent and organic molecules
        cmd.remove("solvent")
        cmd.remove("org")
        cmd_string += f"cmd.color('grey50', 'structure and name C*')\n"
        cmd_string += f"cmd.remove('solvent')\n"
        cmd_string += f"cmd.remove('org')\n"

        for idx in range(len(density_files)):
            # Create isomesh for hydrogen bond probes
            cmd.isomesh(f"hbonds_{idx}", f"density_map_{idx}", 10)
            # Color the hydrgen bond isomesh
            cmd.color(colors[idx], f"hbonds_{idx}")
            cmd_string += f"cmd.isomesh('hbonds_{idx}', 'density_map_{idx}', 10)\n"
            cmd_string += f"cmd.color('{colors[idx]}', 'hbonds_{idx}')\n"

        # Show sticks for the residues of interest
        if selection_string != '':
            cmd.show("sticks", selection_string)
            cmd_string += f"cmd.show('sticks', '{selection_string}')\n"

        cmd.hide("spheres")
        # Set valence to 0 - no double bonds
        cmd.set("valence", 0)
        # Set cartoon_side_chain_helper to 1 - less messy
        cmd.set("cartoon_side_chain_helper", 1)
        # Set background color
        cmd.bg_color("grey80")
        cmd_string += "cmd.hide('spheres')\n"
        cmd_string += "cmd.set('valence', 0)\n"
        cmd_string += "cmd.set('cartoon_side_chain_helper', 1)\n"
        cmd_string += "cmd.bg_color('grey80')"
        
        with open(os.path.join(out_path, "pymol_session_cmd.pml"), "w") as fo:
            fo.write(cmd_string)
            
        cmd.save(os.path.join(out_path, "pymol_results_session.pse"))
        return
    
    def _run_analysis(self, selection_string, volume, temperature, out_path, cosolvent_name=None):
        fig_density_name = os.path.join(out_path, f"map_density.dx")
        fig_energy_name =  os.path.join(out_path, f"map_agfe.dx")
        if cosolvent_name is not None:
            fig_density_name = os.path.join(out_path, f"map_density_{cosolvent_name}.dx")
            fig_energy_name =  os.path.join(out_path, f"map_agfe_{cosolvent_name}.dx")
        analysis = Analysis(self.universe.select_atoms(selection_string), verbose=True)
        analysis.run()
        analysis.atomic_grid_free_energy(volume, temperature)
        analysis.export_density(fig_density_name)
        analysis.export_atomic_grid_free_energy(fig_energy_name)
        self.density_file = fig_density_name
        return
        

    def _get_temp_vol_pot(self, log_file):
        df = pd.read_csv(log_file)
        pot_e = list(df["Potential Energy (kJ/mole)"])
        temp = list(df["Temperature (K)"])
        vol = list(df["Box Volume (nm^3)"])
        return pot_e, temp, vol

    def _plot_temp_vol_pot(self, outpath=None):
        fig_name = f"{outpath}/equilibration.png"
        lim = len(self._potential_energy)
        if len(self._potential_energy) > 75:
            lim = 75

        t = range(0, lim)
        plt.close("all")
        fig, ax = plt.subplots()
        ax.plot(t, [self._potential_energy[x] / 1000 for x in range(0, lim)], label="potential")
        ax.plot(t, [self._volume[x] for x in range(0, lim)], label="volume")
        ax.plot(t, [self._temperature[x] for x in range(0, lim)], label="temperature")

        ax.set(**{
            "title": "Energy",
            "xlabel": "time / ps",
            "xlim": (0, 75),
            })

        ax.legend(
            framealpha=1,
            edgecolor="k",
            fancybox=False
        )
        if outpath is not None:
            plt.savefig(fig_name)
        plt.close()
        return
    
    def _rdf_mda(self, universe: Universe, cosolvents: list, outpath=None, n_frames=250):
        np.seterr(divide='ignore', invalid='ignore')
        wat_resname = "HOH"
        # if top.endswith("cosolv_system.prmtop"):
        #     wat_resname = "WAT"
        oxygen_atoms = universe.select_atoms(f"resname {wat_resname} and name O")
        sim_frames = len(universe.trajectory)
        n_bins = 150
        irdf_results = {}
        for cosolvent in cosolvents:
            cosolvent_name = cosolvent.resname
            r_max = 15
                
            cosolvent_residues = universe.select_atoms(f'resname {cosolvent_name}')
            atoms_names = cosolvent_residues.residues[0].atoms.names
            for cosolvent_atom in set(atoms_names):
                max_y = 0
                if "H" in cosolvent_atom: continue
                print(f"Analysing {cosolvent_name}-{cosolvent_atom}")
                fig, ax = plt.subplots(3, 2, sharex=True, sharey=False)
                plt.tight_layout(pad=3.0)
                # Here compute RDF between same atoms and different molecules
                atoms = cosolvent_residues.select_atoms(f'name {cosolvent_atom}')
                irdf = rdf.InterRDF(atoms, atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
                irdf.run(start=0, step=1000)
                irdf_results[(cosolvent_name, cosolvent_atom, cosolvent_name, cosolvent_atom)] = irdf.results.rdf
                # irdf.run()
                max_y = max(irdf.results.rdf)
                ax[0][0].plot(irdf.results.bins, irdf.results.rdf, label="RDF", alpha=0.5)
                ax[0][0].set_xlabel(r'$r$ $\AA$')
                ax[0][0].set_ylabel("$g(r)$")
                ax[0][0].set_title(f"RDF-{cosolvent_name} {cosolvent_atom} every 1k frames")
                ax[0][0].legend()         

                irdf = rdf.InterRDF(atoms, atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
                irdf.run(start=0, stop=250)
                ax[1][0].plot(irdf.results.bins, irdf.results.rdf, label="RDF", alpha=0.5)
                ax[1][0].set_xlabel(r'$r$ $\AA$')
                ax[1][0].set_ylabel("$g(r)$")
                ax[1][0].set_title(f"RDF-{cosolvent_name} {cosolvent_atom} first 250 frames")
                ax[1][0].legend()
                
                # last frames
                irdf = rdf.InterRDF(atoms, atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
                irdf.run(start=sim_frames-n_frames, stop=sim_frames)
                ax[2][0].plot(irdf.results.bins, irdf.results.rdf, label="RDF", alpha=0.5)
                ax[2][0].set_xlabel(r'$r$ $\AA$')
                ax[2][0].set_ylabel("$g(r)$")
                ax[2][0].set_title(f"RDF-{cosolvent_name} {cosolvent_atom} last 250 frames")
                ax[2][0].legend()

                for i in range(len(ax)):
                    plt.setp(ax[i][0], ylim=(0, max_y), xlim=(0, r_max+1))
                
                # Here compute RDF between atom and water's oxygen
                irdf = rdf.InterRDF(atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max))
                irdf.run(start=0, step=1000)
                max_y = max(irdf.results.rdf)
                irdf_results[(cosolvent_name, cosolvent_atom, "HOH", "O")] = irdf.results.rdf
                # irdf.run()
                ax[0][1].plot(irdf.results.bins, irdf.results.rdf, label="RDF", alpha=0.5)
                ax[0][1].set_xlabel(r'$r$ $\AA$')
                ax[0][1].set_ylabel("$g(r)$")
                ax[0][1].set_title(f"RDF {cosolvent_name} {cosolvent_atom}-HOH O every 1k frames")
                ax[0][1].legend()

                irdf = rdf.InterRDF(atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max))
                irdf.run(start=0, stop=250)
                ax[1][1].plot(irdf.results.bins, irdf.results.rdf, label="RDF", alpha=0.5)
                ax[1][1].set_xlabel(r'$r$ $\AA$')
                ax[1][1].set_ylabel("$g(r)$")
                ax[1][1].set_title(f"RDF {cosolvent_name} {cosolvent_atom}-HOH O first 250 frames")
                ax[1][1].legend()
                
                # last frames
                irdf = rdf.InterRDF(atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max))
                irdf.run(start=sim_frames-n_frames, stop=sim_frames)
                ax[2][1].plot(irdf.results.bins, irdf.results.rdf, label="RDF", alpha=0.5)
                ax[2][1].set_xlabel(r'$r$ $\AA$')
                ax[2][1].set_ylabel("$g(r)$")
                ax[2][1].set_title(f"RDF {cosolvent_name} {cosolvent_atom}-HOH O last 250 frames")
                ax[2][1].legend()
                
                for i in range(len(ax)):
                    plt.setp(ax[i][1], ylim=(0, max_y), xlim=(0, r_max+1))

                # for ax in fig.get_axes():
                #     ax.label_outer()
                    
                if outpath is not None:
                    plt.savefig(f"{outpath}/rdf_{cosolvent_name}_{cosolvent_atom}.png")
                plt.close()
        
        # Finally do waters
        print("Analysing water")
        r_max = 8.5
        fig, ax = plt.subplots()
        plt.setp(ax, ylim=(0, 4.5), xlim=(0, r_max+1))
        irdf = rdf.InterRDF(oxygen_atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
        irdf.run(start=0, step=50)
        # irdf.run()
        ax.plot(irdf.results.bins, irdf.results.rdf, label="RDF", alpha=0.5)
        ax.set_xlabel(r'$r$ $\AA$')
        ax.set_ylabel("$g(r)$")
        ax.set_title(f"RDF-HOH O every 50 frames")
        ax.legend()
        if outpath is not None:
            plt.savefig(f"{outpath}/rdf_HOH_O.png")
        plt.close()
        irdf_results[("HOH", "O", "HOH", "O")] = irdf.results.rdf
        return irdf_results

    def _autocorrelation(self, data):        
        """Autocorrelation function"""
        n = len(data)
        mean = np.mean(data)
        autocorr = correlate(data - mean, data - mean, mode='full', method='auto')
        return autocorr[n - 1:]

    def _plot_autocorrelation(self, data, cosolvent_name1=None, cosolvent_atom1=None, cosolvent_name2=None, cosolvent_atom2=None, outpath=""):
        """

        """
        figname = f"{outpath}/ac_{cosolvent_name1}_{cosolvent_atom1}_{cosolvent_name2}_{cosolvent_atom2}.png"
        title = f"{cosolvent_name1} {cosolvent_atom1}-{cosolvent_name2} {cosolvent_atom2}"
        data = data[0::2]
        autocorr_values = self._autocorrelation(data)
        # Normalize autocorrelation values for better plotting
        normalized_autocorr = autocorr_values / np.max(np.abs(autocorr_values))
        lags = np.arange(0, len(autocorr_values))
        ax = pd.plotting.autocorrelation_plot(pd.Series(normalized_autocorr))
        ax.set_xlim([0, len(autocorr_values)])
        plt.title(title)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.savefig(figname)
        plt.close()
        return