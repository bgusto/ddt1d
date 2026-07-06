#!/usr/bin/env python3
"""Closeup of a hotspot: temperature and the spontaneous-wave-speed criterion.

Plots T(x) and |u_sp| vs D_CJ near the induction-time minimum, shading where
|u_sp| < D_CJ (the sub-critical region the gradient mechanism cannot couple).

Usage:  python3 plot-hotspot.py [plotfile]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from routines.readflash import readflash
from routines.dcj import dcj

fname = sys.argv[1] if len(sys.argv) > 1 else \
    "ddt1d-n123e04-mr-tol100e08-extr-tamb1611-tmax2106-rad0375/ddt1d_hdf5_plt_cnt_0000"
d = readflash(fname)
x, usp, dcj, igtm, T = d["x"], np.abs(d["uspx"]), dcj(d["gamc"]), d["igtm"], d["temp"]
sub = usp < dcj                                   # sub-critical: cannot couple

# window on the hotspot: where T rises above the ambient floor
Tamb = np.median(T)
hot = T > Tamb + 0.05 * (T.max() - Tamb)
xc = x[np.argmin(igtm)]
half = 3.0 * max(x[hot].max() - xc, xc - x[hot].min())
win = (x > xc - half) & (x < xc + half)

fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

ax0.plot(x[win], T[win], color="C1")
ax0.set_ylabel("T (K)")
ax0.set_title(fname.split("/")[-1], fontsize=9)

ax1.semilogy(x[win], usp[win], color="C0", lw=1.0, label=r"$|u_{sp}|$")
ax1.semilogy(x[win], dcj[win], color="C3", ls="--", label=r"$D_{CJ}$")
lo, hi = ax1.get_ylim()
ax1.fill_between(x[win], lo, hi, where=sub[win], color="0.7", alpha=0.4,
                 step="mid", label=r"$|u_{sp}| < D_{CJ}$")
ax1.set_ylim(usp[win].min() * 0.5, dcj[win].max() * 10)   # clip the divide-by-zero spikes
ax1.set_ylabel("speed (cm/s)")
ax1.set_xlabel("x (cm)")
ax1.legend(loc="upper right", fontsize=8)

fig.tight_layout()
out = "hotspot.png"
fig.savefig(out, dpi=110)
print(f"coupled cells in window: {(~sub & win).sum()} / {win.sum()};  wrote {out}")
