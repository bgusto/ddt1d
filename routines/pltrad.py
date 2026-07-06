#!/usr/bin/env python3
"""Detonation radius of the initial-condition plotfile for one case.

Usage:  python3 pltrad.py [plotfile]
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from routines.readflash import readflash
from routines.dcj import dcj
from routines.detrad import detrad


def pltrad(pltfile):
    d = readflash(pltfile)
    rdet, xc, xL, xR = detrad(d["x"], d["igtm"], d["uspx"], dcj(d["gamc"]))
    return rdet


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else \
        "ddt1d_ta17249_tm20811_rd00563_hdf5_plt_cnt_0000"
    d = readflash(fname)
    x, usp, cj = d["x"], d["uspx"], dcj(d["gamc"])
    rdet, xc, xL, xR = detrad(x, d["igtm"], usp, cj)

    print(f"file : {fname.split('/')[-1]}")
    print(f"dcj  : {cj[np.argmin(d['igtm'])]:.4g} cm/s")
    print(f"region : [{xL:.6g}, {xR:.6g}] cm  (center {xc:.6g})")
    print(f"rdet : {rdet:.6g} cm")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(x, np.abs(usp), color="C0", label="|usp|")
    ax.semilogy(x, cj, color="C3", ls="--", label="dcj")
    ax.axvspan(xL, xR, color="C2", alpha=0.2, label=f"rdet = {rdet:.0f} cm")
    ax.axvline(xc, color="k", ls=":", lw=0.8)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("speed (cm/s)")
    ax.set_xlim(xc - 4 * rdet, xc + 4 * rdet)
    ax.legend(loc="upper right")
    ax.set_title(fname.split("/")[-1])
    fig.tight_layout()
    plt.show()
