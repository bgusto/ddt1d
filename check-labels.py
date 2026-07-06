#!/usr/bin/env python3
"""Manual DDT relabeling audit against the automatic (max local Mach) labeler.

Picks a random sample of cases, shows the last plotfile's Mach / temperature /
pressure profile for each, and lets you flip through and type your own
detonation call. At the end, tallies your manual labels against the automatic
label (max local Mach > MACHTHRESH) and the old .dat-vote label, and reports
where they disagree.

For each case:  key d = detonation, n = no detonation, s = skip, q = quit early
"""
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
import h5py

import routines.detonate as det

datadir = "/mnt/elements/ddt1d/extr"
ncase = 50
seed = 1
machthresh = 0.8
outfile = "checklabels.out"

#------- index the plotfiles once (fast: one listdir, not one glob per case) -------#

print("indexing plotfiles in", datadir, "...")
allfiles = os.listdir(datadir)
pltre = re.compile(r"^(.+)_hdf5_plt_cnt_(\d+)$")
bystem = {}
for fn in allfiles:
    m = pltre.match(fn)
    if m:
        st, idx = m.group(1), int(m.group(2))
        if st not in bystem or idx > bystem[st][1]:
            bystem[st] = (fn, idx)

stems = [f[:-4] for f in allfiles if f.endswith(".dat")]
random.seed(seed)
sample = random.sample(stems, ncase)

#------- helpers -------#

def readvars(fname, vars):
    with h5py.File(fname, "r") as f:
        keymap = {k.strip(): k for k in f.keys()}
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
        for v in vars:
            d[v] = np.concatenate([f[keymap[v]][b, 0, 0, :] for b in leaves])[order]
    return d

#------- flip through cases -------#

rows = []
for i, st in enumerate(sample):
    if st not in bystem:
        print(st, ": no plotfile found, skipping")
        continue
    lastpf, lastidx = bystem[st]
    d = readvars(os.path.join(datadir, lastpf), ["velx", "gamc", "pres", "dens", "temp"])
    cs = np.sqrt(d["gamc"] * d["pres"] / d["dens"])
    mach = np.abs(d["velx"]) / cs
    maxmach = float(mach.max())
    tjump = float(d["temp"].max() / d["temp"].min())
    pjump = float(d["pres"].max() / d["pres"].min())
    automach = int(maxmach > machthresh)

    df = os.path.join(datadir, st + ".dat")
    olddeton = int(det.detonate(df)["deton"])

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 7.5))
    ax[0].plot(d["x"], mach)
    ax[0].axhline(machthresh, color="r", ls="--", lw=1)
    ax[0].set_ylabel("local Mach")
    ax[1].plot(d["x"], d["temp"])
    ax[1].set_ylabel("temp (K)")
    ax[2].plot(d["x"], d["pres"])
    ax[2].set_ylabel("pres (erg/cc)")
    ax[2].set_xlabel("x (cm)")
    fig.suptitle(f"[{i+1}/{len(sample)}] {st}  lastidx={lastidx}\n"
                 f"maxmach={maxmach:.2f}  Tjump={tjump:.2f}x  Pjump={pjump:.2f}x  "
                 f"auto={'DETON' if automach else 'no'}  olddat={'DETON' if olddeton else 'no'}")
    fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    ans = input("your label [d=deton / n=no / s=skip / q=quit]: ").strip().lower()
    plt.close(fig)
    if ans == "q":
        break
    if ans == "s":
        continue
    user = 1 if ans == "d" else 0
    rows.append((st, lastidx, maxmach, tjump, pjump, automach, olddeton, user))

#------- tally -------#

with open(outfile, "w") as f:
    f.write("stem lastidx maxmach tjump pjump auto olddat user\n")
    for r in rows:
        f.write(f"{r[0]} {r[1]} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} {r[5]} {r[6]} {r[7]}\n")
print("wrote", outfile)

if rows:
    n = len(rows)
    autoagree = sum(r[5] == r[7] for r in rows)
    oldagree = sum(r[6] == r[7] for r in rows)
    print(f"\n{n} cases labeled")
    print(f"auto (mach>{machthresh}) vs manual : {autoagree}/{n} agree")
    print(f"old dat-vote      vs manual : {oldagree}/{n} agree")

    print("\ndisagreements with auto:")
    for r in rows:
        if r[5] != r[7]:
            print(f"  {r[0]}  maxmach={r[2]:.2f}  auto={r[5]}  manual={r[7]}")

    print("\ndisagreements with old dat-vote:")
    for r in rows:
        if r[6] != r[7]:
            print(f"  {r[0]}  olddat={r[6]}  manual={r[7]}")
