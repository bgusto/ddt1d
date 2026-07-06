#!/usr/bin/env python3
"""Build the curated Khokhlov-scalar dataset for training/testing.

Sweeps DDT1D_DATA/extr for cases (flat layout: stem_hdf5_plt_cnt_0000 +
stem.dat + stem_hdf5_plt_cnt_NNNN...), computes the Khokhlov scalars + rdet
from the IC plotfile and the deton label from the last plotfile's peak local
Mach number (see routines/machdet.py), writes one compressed npz plus an
info file. Cases whose dat file hasn't landed yet, or that are flagged in
badcases.txt (known-corrupted runs), are skipped.
"""
import os
import re
import glob
import numpy as np
from routines.khscal import khscal
from routines.pltrad import pltrad
from routines.machdet import machdetpf, MACHTHRESH

datadir = os.path.join(os.environ["DDT1D_DATA"], "extr")
outnpz = "khokhlov.npz"
outinfo = "khokhlov.info"
feat = ["tmax", "taumin", "dens", "tamb", "csnd", "pres", "xc12", "xo16", "rdet"]

badcases = set(open("badcases.txt").read().split())

# index every case's last plotfile once (a glob per case is far too slow at 25k+ cases)
pltre = re.compile(r"^(.+)_hdf5_plt_cnt_(\d+)$")
lastpf = {}
for fn in os.listdir(datadir):
    m = pltre.match(fn)
    if m:
        st, idx = m.group(1), int(m.group(2))
        if st not in lastpf or idx > lastpf[st][1]:
            lastpf[st] = (os.path.join(datadir, fn), idx)

pltfiles = sorted(glob.glob(os.path.join(datadir, "*_hdf5_plt_cnt_0000")))
print(len(pltfiles), "plotfiles in", datadir)

stem = []
X = []
y = []
nskip = 0
nbad = 0
for pf in pltfiles:
    st = pf[:-len("_hdf5_plt_cnt_0000")]
    df = st + ".dat"
    if os.path.basename(st) in badcases:
        nbad += 1
        continue
    if not os.path.exists(df):
        nskip += 1
        continue

    s = khscal(pf)
    s["rdet"] = pltrad(pf)
    deton, maxmach = machdetpf(lastpf[os.path.basename(st)][0])

    stem.append(os.path.basename(st))
    X.append([s[k] for k in feat])
    y.append(int(deton))

    if len(stem) % 500 == 0:
        print(len(stem), "done")

X = np.array(X)
y = np.array(y)
stem = np.array(stem)

np.savez_compressed(outnpz, X=X, y=y, stem=stem, feat=np.array(feat))
print("wrote", outnpz, "-", X.shape[0], "cases,", nskip, "skipped (no .dat yet),",
      nbad, "excluded (badcases.txt)")

with open(outinfo, "w") as f:
    f.write(f"source    {datadir}\n")
    f.write(f"ncase     {X.shape[0]}\n")
    f.write(f"nskip     {nskip}  (plotfile present, .dat not yet landed)\n")
    f.write(f"nbad      {nbad}  (excluded via badcases.txt)\n")
    f.write(f"ndeton    {int(y.sum())}\n")
    f.write(f"features  {' '.join(feat)}\n")
    f.write(f"label     last plotfile, max local Mach (|velx|/cs) > {MACHTHRESH}; "
            f"validated 49/49 vs manual review (excl. known-bad cases), see routines/machdet.py\n")
print("wrote", outinfo)
