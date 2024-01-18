import sys
import pandas as pd
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


if __name__ == "__main__":
    pot_e, temp, vol = get_temp_vol_pot(sys.argv[1])
    plot_temp_vol_pot(pot_e, temp, vol, outpath=sys.argv[2])