import os
import argparse
import json
from collections import defaultdict
import mdtraj
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
# import freud
import matplotlib.pyplot as plt

from cosolvkit.cosolvent_system import CoSolvent

def bo(data):
    from pymbar import timeseries
    from pymbar.timeseries import statistical_inefficiency_multiple, subsample_correlated_data
    t0, g, Neff_max = timeseries.detect_equilibration(data) # compute indices of uncorrelated timeseries
    # A_t_equil = data[t0:]
    A_t_equil = data
    indices = timeseries.subsample_correlated_data(A_t_equil, g=g)
    A_n = A_t_equil[indices]
    print(f't0 is {t0} and indices are {len(indices)} frames')
    print(f't0 is {t0} and A_t_equil is {len(A_t_equil)} frames')
    return A_n

def get_temp_vol_pot(log_file):
    df = pd.read_csv(log_file)
    pot_e = list(df["Potential Energy (kJ/mole)"])
    temp = list(df["Temperature (K)"])
    vol = list(df["Box Volume (nm^3)"])
    return pot_e, temp, vol

def plot_temp_vol_pot(pot_e, temp, vol, outpath=None):
    lim = len(pot_e)
    if len(pot_e) > 75:
        lim = 75

    t = range(0, lim)
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot(t, [pot_e[x] / 1000 for x in range(0, lim)], label="potential")
    ax.plot(t, [vol[x] for x in range(0, lim)], label="volume")
    ax.plot(t, [temp[x] for x in range(0, lim)], label="temperature")
    # ax.plot(t, [pot_e[x] / 1000 for x in range(0, len(pot_e))], label="potential")
    # ax.plot(t, [vol[x] for x in range(0, len(pot_e))], label="volume")
    # ax.plot(t, [temp[x] for x in range(0, len(pot_e))], label="temperature")

    ax.set(**{
        "title": "Energy",
        "xlabel": "time / ps",
        "xlim": (0, 75),
        # "ylabel": "energy / 10$^{3}$ kJ mol$^{-1}$"
        })

    ax.legend(
        framealpha=1,
        edgecolor="k",
        fancybox=False
    )
    if outpath is not None:
        plt.savefig(outpath)
    plt.close()
    return

def rdf_mda(traj: str, top: str, cosolvents: list, outpath=None, n_frames=250):
    np.seterr(divide='ignore', invalid='ignore')
    u = mda.Universe(top, traj)
    wat_resname = "HOH"
    if top.endswith("cosolv_system.prmtop"):
        wat_resname = "WAT"
    oxygen_atoms = u.select_atoms(f"resname {wat_resname} and name O")
    sim_frames = len(u.trajectory)
    n_bins = sim_frames
    irdf_results = {}
    for cosolvent in cosolvents:
        cosolvent_name = cosolvent.resname
        r_max = 15
            
        cosolvent_residues = u.select_atoms(f'resname {cosolvent_name}')
        atoms_names = cosolvent_residues.residues[0].atoms.names
        for cosolvent_atom in set(atoms_names):
            if "H" in cosolvent_atom: continue
            print(f"Analysing {cosolvent_name}-{cosolvent_atom}")
            fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
            plt.setp(ax, ylim=(0, 4.5), xlim=(0, r_max+1))
            plt.tight_layout(pad=2.0)
            # Here compute RDF between same atoms and different molecules
            atoms = cosolvent_residues.select_atoms(f'name {cosolvent_atom}')
            irdf = rdf.InterRDF(atoms, atoms, nbins=n_bins, range=(0.0, r_max), exclusion_block=(1, 1))
            irdf.run(start=0, step=1000)
            irdf_results[(cosolvent_name, cosolvent_atom, cosolvent_name, cosolvent_atom)] = irdf.results.rdf
            # irdf.run()
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
            
            # Here compute RDF between atom and water's oxygen
            irdf = rdf.InterRDF(atoms, oxygen_atoms, nbins=n_bins, range=(0.0, r_max))
            irdf.run(start=0, step=1000)
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
            
            for ax in fig.get_axes():
                ax.label_outer()
                
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

def autocorrelation(data):
    from scipy.signal import correlate
    
    """Autocorrelation function"""
    n = len(data)
    mean = np.mean(data)
    autocorr = correlate(data - mean, data - mean, mode='full', method='auto')
    return autocorr[n - 1:]

def plot_autocorrelation(data, cosolvent_name1=None, cosolvent_atom1=None, cosolvent_name2=None, cosolvent_atom2=None, outpath=""):
    equilibration = cosolvent_name1 is None and cosolvent_atom1 is None and cosolvent_name2 is None and cosolvent_atom2 is None
    if equilibration:
        figname = f"{outpath}/ac_equilibration.png"
        title = f"Equilibration Auto correlation"
    else:
        figname = f"{outpath}/ac_{cosolvent_name1}_{cosolvent_atom1}_{cosolvent_name2}_{cosolvent_atom2}.png"
        title = f"{cosolvent_name1} {cosolvent_atom1}-{cosolvent_name2} {cosolvent_atom2}"
    if equilibration:
        labels = ["Potential Energy", "Volume", "Temperature"]
        colors = ["g", "b", "r"]
        for d in range(len(data)):
            autocorr_values = autocorrelation(data[d])
            # Normalize autocorrelation values for better plotting
            normalized_autocorr = autocorr_values / np.max(np.abs(autocorr_values))
            lags = np.arange(0, len(autocorr_values))
            plt.stem(lags, normalized_autocorr, colors[d], markerfmt='o', label=labels[d])
    else:
        autocorr_values = autocorrelation(data)
        # Normalize autocorrelation values for better plotting
        normalized_autocorr = autocorr_values / np.max(np.abs(autocorr_values))
        lags = np.arange(0, len(autocorr_values))
        plt.stem(lags, normalized_autocorr, basefmt="k-")
    plt.title(title)
    plt.legend()
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.savefig(figname)
    plt.close()
    return

def cmd_lineparser():
    parser = argparse.ArgumentParser(description="Runs cosolvkit and MD simulation afterwards.")
    parser.add_argument('-l', '--log_file', dest='log_file', required=True,
                        action='store', help='path to the log file from MD simulation')
    parser.add_argument('-traj', dest='traj_file', required=True,
                        action='store', help='path to the traj <.dcd> file from MD simulation')
    parser.add_argument('-topo', dest='top_file', required=True,
                        action='store', help='path to the topology file from MD simulation')
    parser.add_argument('-o', '--out_path', dest='out_path', required=True,
                        action='store', help='path where to store output plots')
    parser.add_argument('-c', '--cosolvents', dest='cosolvents', required=True,
                        action='store', help='path to the json file containing cosolvents used for the simulation')
    return parser.parse_args()


if __name__ == "__main__":
    args = cmd_lineparser()
    log_file = args.log_file
    traj_file = args.traj_file
    top_file = args.top_file
    out_path = args.out_path
    cosolvents_path = args.cosolvents

    with open(cosolvents_path) as fi:
        cosolvents_d = json.load(fi)
    
    cosolvents = list()
    for cosolvent in cosolvents_d:
        cosolvents.append(CoSolvent(**cosolvent))
    rdf_path = os.path.join(out_path, "rdf")
    ac_path = os.path.join(out_path, "autocorrelation")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(rdf_path):
        os.makedirs(rdf_path)
    if not os.path.exists(ac_path):
        os.makedirs(ac_path)

    # Equilibration
    pot_e, temp, vol = get_temp_vol_pot(log_file)
    plot_temp_vol_pot(pot_e, temp, vol, outpath=out_path+"/equilibration_plot.png")
    
    irdf_results = rdf_mda(traj_file, top_file, cosolvents, rdf_path)
    for pair in irdf_results:
        cosolvent_name1, cosolvent_atom1, cosolvent_name2, cosolvent_atom2 = pair
        plot_autocorrelation(data=irdf_results[pair],
                             cosolvent_name1=cosolvent_name1,
                             cosolvent_atom1=cosolvent_atom1,
                             cosolvent_name2=cosolvent_name2,
                             cosolvent_atom2=cosolvent_atom2,
                             outpath=ac_path)
    plot_autocorrelation([pot_e, vol, temp], outpath=ac_path)
    # plot_autocorrelation(data=pot_e, parameter="Potential Energy", outpath=ac_path)
    # plot_autocorrelation(data=temp, parameter="Temperature (K)", outpath=ac_path)
    # plot_autocorrelation(data=vol, parameter="Volume", outpath=ac_path)