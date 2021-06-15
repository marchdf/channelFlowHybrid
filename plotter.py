#!/usr/bin/env python3

# ========================================================================
#
# Imports
#
# ========================================================================
import argparse
import os
import glob
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import utilities
from scipy.interpolate import griddata


# ========================================================================
#
# Some defaults variables
#
# ========================================================================
plt.rc("text", usetex=True)
cmap_med = [
    "#F15A60",
    "#7AC36A",
    "#5A9BD4",
    "#FAA75B",
    "#9E67AB",
    "#CE7058",
    "#D77FB4",
    "#737373",
]
cmap = [
    "#EE2E2F",
    "#008C48",
    "#185AA9",
    "#F47D23",
    "#662C91",
    "#A21D21",
    "#B43894",
    "#010202",
]
dashseq = [
    (None, None),
    [10, 5],
    [10, 4, 3, 4],
    [3, 3],
    [10, 4, 3, 4, 3, 4],
    [3, 3],
    [3, 3],
]
markertype = ["s", "d", "o", "p", "h"]


# ========================================================================
#
# Functions
#
# ========================================================================
def read_dns_data(fdir):
    lst = []
    for fname in glob.glob(fdir + "/*.dat"):
        with open(fname, "r") as f:
            for line in f:
                if line.startswith("# x/h"):
                    x = float(line.split("=")[-1])
                    break

        df = pd.read_csv(
            fname,
            header=None,
            names=["y", "u", "v", "upup", "vpvp", "upvp"],
            comment="#",
        )
        df["x"] = x
        lst.append(df)

    return pd.concat(lst, ignore_index=True)


# ========================================================================


# ========================================================================
#
# Main
#
# ========================================================================
if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="A simple plot tool")
    parser.add_argument("-f", "--fdir", nargs="+", help="Folder to plot", required=True)
    args = parser.parse_args()

    Retau = 5200

    # Reference data
    refdir = os.path.abspath("refdata")
    ddir = os.path.join(refdir, "dns")
    ddf = pd.read_csv(
        os.path.join(ddir, f"Re{Retau}.txt"),
        header=None,
        names=["y", "yp", "u", "dudy", "v", "p"],
        skiprows=1,
        delim_whitespace=True,
    )
    figsize = (15, 6)

    # plot stuff
    fname = "plots.pdf"
    legend_elements = []

    # DNS
    legend_elements += (Line2D([0], [0], lw=2, color=cmap[-1], label="DNS"),)
    plt.figure("u")
    p = plt.plot(ddf.u, ddf.y, lw=2, color=cmap[-1],)

    plt.figure("up")
    p = plt.semilogy(ddf.u, ddf.yp, lw=2, color=cmap[-1],)

    # Nalu data
    for i, fdir in enumerate(args.fdir):

        yname = os.path.join(os.path.dirname(fdir), "channel.yaml")
        u0, rho0, mu, turb_model = utilities.parse_ic(yname)
        utau = Retau * mu / rho0
        model = turb_model.upper().replace("_", "-")
        legend_elements += [Line2D([0], [0], lw=2, color=cmap[i], label=f"{model}")]
        pfx = "channel-fine"

        h = 1.0
        tau = h / u0
        dynPres = rho0 * 0.5 * u0 * u0
        ndf = pd.read_csv(os.path.join(fdir, f"{pfx}-profiles.dat"))
        ndf["yp"] = ndf.y * utau / (mu / rho0)

        plt.figure("u")
        p = plt.plot(ndf.u, ndf.y, lw=2, color=cmap[i])
        p[0].set_dashes(dashseq[i])

        plt.figure("up")
        p = plt.semilogy(ndf.u, ndf.yp, lw=2, color=cmap[i])
        p[0].set_dashes(dashseq[i])

        inlet = pd.read_csv(os.path.join(fdir, "inlet.dat"))
        plt.figure("u_inlet")
        p = plt.plot(inlet.t / tau, inlet.u, lw=2, color=cmap[i], label=f"{model}")
        p[0].set_dashes(dashseq[i])

        plt.figure("tke_inlet")
        p = plt.plot(inlet.t / tau, inlet.tke, lw=2, color=cmap[i], label=f"{model}")
        p[0].set_dashes(dashseq[i])

        plt.figure("sdr_inlet")
        p = plt.plot(inlet.t / tau, inlet.sdr, lw=2, color=cmap[i], label=f"{model}")
        p[0].set_dashes(dashseq[i])

        front = pd.read_csv(os.path.join(fdir, f"{pfx}-f_front.dat")).sort_values(by=["x","y"])
        nx = len(front.x.unique())
        ny = len(front.y.unique())
        xmin, xmax = front.x.min(), front.x.max()
        ymin, ymax = front.y.min(), front.y.max()

        fields = {
            "u": {"vmin": -0.2, "vmax": 30},
            "beta": {"vmin": 0.0, "vmax": 1},
            "rk": {"vmin": 0.5, "vmax": 7},
        }
        for name, opt in fields.items():
            if name in front.columns:
                plt.figure(f"{name}-front-{model}", figsize=figsize)
                dat = np.reshape(front[name].values, (nx,ny))
                plt.imshow(
                    dat.T,
                    origin="lower",
                    extent=[xmin, xmax, ymin, ymax],
                    vmin=opt["vmin"],
                    vmax=opt["vmax"],
                )

    # Save the plots
    with PdfPages(fname) as pdf:

        plt.figure("u")
        ax = plt.gca()
        plt.xlabel(r"$\langle u \rangle / u_0$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$y / h$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.xlim([0, 30])
        plt.ylim([0, 1.0])
        legend = ax.legend(handles=legend_elements, loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("up")
        ax = plt.gca()
        plt.xlabel(r"$\langle u \rangle / u_0$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$y^+$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        plt.xlim([0, 30])
        plt.ylim([1, Retau])
        legend = ax.legend(handles=legend_elements, loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("u_inlet")
        ax = plt.gca()
        plt.xlabel(r"$t / \tau$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\bar{u} (x=0)$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("tke_inlet")
        ax = plt.gca()
        plt.xlabel(r"$t / \tau$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\bar{k} (x=0)$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        plt.figure("sdr_inlet")
        ax = plt.gca()
        plt.xlabel(r"$t / \tau$", fontsize=22, fontweight="bold")
        plt.ylabel(r"$\bar{k} (x=0)$", fontsize=22, fontweight="bold")
        plt.setp(ax.get_xmajorticklabels(), fontsize=18, fontweight="bold")
        plt.setp(ax.get_ymajorticklabels(), fontsize=18, fontweight="bold")
        legend = ax.legend(loc="best")
        plt.tight_layout()
        pdf.savefig(dpi=300)

        for i in plt.get_figlabels():
            if "-front-" in i:
                plt.figure(i)
                ax = plt.gca()
                plt.colorbar()
                plt.xlabel(r"$x / h$", fontsize = 22, fontweight = "bold")
                plt.ylabel(r"$y / h$", fontsize = 22, fontweight = "bold")
                plt.setp(ax.get_xmajorticklabels(), fontsize = 18, fontweight = "bold")
                plt.setp(ax.get_ymajorticklabels(), fontsize = 18, fontweight = "bold")
                plt.tight_layout()
                pdf.savefig(dpi = 300)
