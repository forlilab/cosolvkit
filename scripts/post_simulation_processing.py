import argparse
import json
import mdtraj
import numpy as np
import pandas as pd
import freud
import matplotlib.pyplot as plt

from cosolvkit.cosolvent_system import CoSolvent

def get_temp_vol_pot(log_file):
    df = pd.read_csv(log_file)
    pot_e = list(df["Potential Energy (kJ/mole)"])
    temp = list(df["Temperature (K)"])
    vol = list(df["Box Volume (nm^3)"])
    return pot_e, temp, vol

def plot_temp_vol_pot(pot_e, temp, vol, outpath=None):
    t = range(1, len(pot_e)+1)
    plt.close("all")
    fig, ax = plt.subplots()
    ax.plot(t, [x / 1000 for x in pot_e], label="potential")
    ax.plot(t, [x for x in vol], label="volume")
    ax.plot(t, [x for x in temp], label="temperature")

    ax.set(**{
        "title": "Energy",
        "xlabel": "time / ps",
        "xlim": (0, 100),
        # "ylabel": "energy / 10$^{3}$ kJ mol$^{-1}$"
        })

    ax.legend(
        framealpha=1,
        edgecolor="k",
        fancybox=False
    )
    if outpath is not None:
        plt.savefig(outpath)
    plt.show()
    return

def radial_distribution_function(cosolvents, traj, outpath=None):
    # pairs = traj.top.select_pairs("name O", "name O")
    # r, gr = mdtraj.compute_rdf(
    #     traj,
    #     pairs,
    #     r_range=(0.1, 0.5),
    #     bin_width=0.0005
    # )

    for cosolvent in cosolvents:
        cosolvent_name = cosolvent.resname
        cosolvent_indices = [atom.index for atom in traj.top.atoms if atom.name == "" and atom.residue.name == cosolvent_name]

    oxygen_indices = [atom.index for atom in traj.top.atoms if atom.name == "O" and atom.residue.name == "HOH"]
    freud_rdf = freud.density.RDF(bins=300, r_min=0.01, r_max=1)
    for system in zip(np.asarray(traj.unitcell_vectors), traj.xyz[:, oxygen_indices, :]):
        freud_rdf.compute(system, reset=False)

    fig, ax = plt.subplots()
    ax.plot(freud_rdf.bin_centers, freud_rdf.rdf, "o", label="freud", alpha=0.5)
    ax.set_xlabel("$r$")
    ax.set_ylabel("$g(r)$")
    ax.set_title("RDF")
    ax.legend()

    # plt.close("all")
    # fig, ax = plt.subplots(num=3)
    # ax.plot(r, gr)
    # ax.set_xlabel("Radius [nm]")
    # ax.set_ylabel("Radial distribution function [gr]")
    if outpath is not None:
        plt.savefig(outpath)
    plt.show()
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


    pot_e, temp, vol = get_temp_vol_pot(log_file)
    plot_temp_vol_pot(pot_e, temp, vol, outpath=out_path+"/equilibration_plot.png")
    traj = mdtraj.load(traj_file, top=top_file)
    print("Traj loaded in memory")
    radial_distribution_function(cosolvents, traj, outpath=out_path+"/rdf.png")