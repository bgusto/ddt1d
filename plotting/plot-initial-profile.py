#!/usr/bin/env python3
"""Read a 1D FLASH plotfile (HDF5) and plot the temperature profile."""
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt

fname = "../raw-data/ddt1d-n123e04-mr-tol100e08-gauss-dens100e07-temp178e09-ampl0190-stdv450/ddt1d_hdf5_plt_cnt_0000"

with h5py.File(fname, "r") as f:
    temp = f["temp"][:]            # (nblocks, nzb, nyb, nxb)
    bbox = f["bounding box"][:]    # (nblocks, ndim, 2)
    ntype = f["node type"][:]      # 1 == leaf

    nblk, nzb, nyb, nxb = temp.shape
    leaves = np.where(ntype == 1)[0]

    xs, ts = [], []
    for b in leaves:
        xlo, xhi = bbox[b, 0, 0], bbox[b, 0, 1]
        # cell-centered x within this block
        edges = np.linspace(xlo, xhi, nxb + 1)
        xc = 0.5 * (edges[:-1] + edges[1:])
        xs.append(xc)
        ts.append(temp[b, 0, 0, :])   # 1D: y,z indices are 0

    x = np.concatenate(xs)
    t = np.concatenate(ts)
    order = np.argsort(x)
    x, t = x[order], t[order]

print(f"{len(leaves)} leaf blocks, {x.size} cells")
print(f"x range: [{x.min():.4g}, {x.max():.4g}] cm")
print(f"T range: [{t.min():.4g}, {t.max():.4g}] K")

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, t, lw=1.0)
ax.set_xlabel("x (cm)")
ax.set_ylabel("Temperature (K)")
ax.set_title(fname.split("/")[-1])
fig.tight_layout()
plt.show()
