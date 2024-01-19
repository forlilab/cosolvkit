import sys
import mdtraj
import numpy as np
import pandas as pd
import freud
import matplotlib.pyplot as plt

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

def radial_distribution_function(traj, outpath=None):
    # pairs = traj.top.select_pairs("name O", "name O")
    # r, gr = mdtraj.compute_rdf(
    #     traj,
    #     pairs,
    #     r_range=(0.1, 0.5),
    #     bin_width=0.0005
    # )
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


if __name__ == "__main__":
    # pot_e, temp, vol = get_temp_vol_pot(sys.argv[1])
    # plot_temp_vol_pot(pot_e, temp, vol, outpath=sys.argv[2])
    traj = "/home/nick/phd/cosolvkit_validation/diogo_target/find_and_replace/simulation.dcd"
    topo = "/home/nick/phd/cosolvkit_validation/diogo_target/find_and_replace/cosolv_system.prmtop"
    traj = mdtraj.load(traj, top=topo)
    print("Traj loaded in memory")
    radial_distribution_function(traj, outpath="/home/nick/phd/cosolvkit_validation/diogo_target/find_and_replace/rdf.png")