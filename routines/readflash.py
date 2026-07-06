#!/usr/bin/env python3
"""Read a 1D FLASH plotfile (HDF5) into arrays sorted by x."""
import sys
import numpy as np
import h5py


def readflash(fname):
    with h5py.File(fname, "r") as f:
        keymap = {k.strip(): k for k in f.keys()}          # names are space-padded
        ntype = f["node type"][:]
        bbox = f["bounding box"][:]
        leaves = np.where(ntype == 1)[0]
        nxb = f[keymap["temp"]].shape[-1]

        xs = []
        for b in leaves:
            xlo, xhi = bbox[b, 0, 0], bbox[b, 0, 1]
            edges = np.linspace(xlo, xhi, nxb + 1)
            xs.append(0.5 * (edges[:-1] + edges[1:]))
        x = np.concatenate(xs)
        order = np.argsort(x)

        d = {"x": x[order]}
        for v in ("temp", "dens", "pres", "gamc", "igtm", "uspx", "velx", "c12", "o16"):
            if v in keymap:
                d[v] = np.concatenate([f[keymap[v]][b, 0, 0, :] for b in leaves])[order]
    d["cs"] = np.sqrt(d["gamc"] * d["pres"] / d["dens"])
    return d


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "ddt1d_hdf5_plt_cnt_0000"
    d = readflash(fname)
    print(fname)
    for k, v in d.items():
        print(f"  {k:5s} [{v.min():.4g}, {v.max():.4g}]")
