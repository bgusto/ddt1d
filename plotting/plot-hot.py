#!/usr/bin/env python3
"""Hot = ready to burn: igtm within 3x the case's own minimum.

Animates T(x) and igtm(x) over time, right half of the domain only (the
profile is symmetric), red where hot, black elsewhere. The hot region and
the axis limits are both fixed from all frames so nothing jumps around.

Usage:  python3 plot-hot.py [casedir]
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("GTK3Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from routines.readflash import readflash

MULT = 3.0     # hot = igtm <= MULT * igtm.min(), always non-empty by construction
PAUSE = 0.05   # seconds between frames

casedir = sys.argv[1] if len(sys.argv) > 1 else \
    "../ddt1d-n123e04-mr-tol100e08-gauss-dens100e07-temp178e09-ampl0130-stdv1250"

files = sorted(glob.glob(os.path.join(casedir, "ddt1d_hdf5_plt_cnt_[0-9]*")))
print(len(files), "frames")

print("reading frames...")
frames = [readflash(f) for f in files]

# color once from frame 0, hold it -- the grid is AMR so cell count/position
# shifts frame to frame, so "hold the mask" means hold the x-range, not indices
d0 = frames[0]
hot0 = d0["igtm"] <= MULT * d0["igtm"].min()
xlo, xhi = d0["x"][hot0].min(), d0["x"][hot0].max()

# right half of the domain only; fixed axis limits from every frame so the
# view doesn't rescale as the burn evolves
xmax = max(d["x"].max() for d in frames)
Tlo = min(d["temp"][d["x"] >= 0].min() for d in frames)
Thi = max(d["temp"][d["x"] >= 0].max() for d in frames)
ilo = min(d["igtm"][d["x"] >= 0].min() for d in frames)
ihi = max(d["igtm"][d["x"] >= 0].max() for d in frames)

plt.ion()
fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

for fname, d in zip(files, frames):
    x, T, igtm = d["x"], d["temp"], d["igtm"]
    half = x >= 0
    hot = (x >= xlo) & (x <= xhi)

    ax0.clear()
    ax0.plot(x[half & ~hot], T[half & ~hot], ".", color="k", ms=2)
    ax0.plot(x[half & hot], T[half & hot], ".", color="r", ms=2)
    ax0.set_xlim(0, xmax)
    ax0.set_ylim(Tlo, Thi)
    ax0.set_ylabel("T (K)")
    ax0.set_title(fname.split("/")[-1], fontsize=9)

    ax1.clear()
    ax1.semilogy(x[half & ~hot], igtm[half & ~hot], ".", color="k", ms=2)
    ax1.semilogy(x[half & hot], igtm[half & hot], ".", color="r", ms=2)
    ax1.axhline(d0["igtm"].min(), color="0.5", ls=":", lw=0.8)
    ax1.axhline(MULT * d0["igtm"].min(), color="0.5", ls=":", lw=0.8)
    ax1.set_xlim(0, xmax)
    ax1.set_ylim(ilo, ihi)
    ax1.set_ylabel("igtm (s)")
    ax1.set_xlabel("x (cm)")

    plt.pause(PAUSE)

plt.ioff()
plt.show()
