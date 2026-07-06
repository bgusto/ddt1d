#!/usr/bin/env python3
"""Load the saved DDT ensemble and score raw Khokhlov-scalar samples.

Output per sample is P(DDT) (ensemble mean probability) and its uncertainty
(ensemble std). The hard class is just P >= 0.5.

Input rows must be the raw 9 features in khokhlov.npz order:
  tmax taumin dens tamb csnd pres xc12 xo16 rdet
Here we reuse the dataset itself as the example input.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

modelfile = "ensemble.pt"
npzfile = "khokhlov.npz"

#------- rebuild the ensemble from the saved state -------#

ck = torch.load(modelfile, weights_only=False)
nnode, nlayr, drpout, nin = ck["nnode"], ck["nlayr"], ck["drpout"], ck["nin"]

models = []
for sd in ck["members"]:
    # activation MUST match fit-nn-khokhlov.py (GELU) or the loaded weights are wrong
    layrs = [nn.Linear(nin, nnode[0]), nn.GELU(), nn.Dropout(drpout[0])]
    for l in range(nlayr - 1):
        layrs += [nn.Linear(nnode[l], nnode[l+1]), nn.GELU(), nn.Dropout(drpout[l+1])]
    layrs += [nn.Linear(nnode[nlayr-1], 1)]
    m = nn.Sequential(*layrs)
    m.load_state_dict(sd)
    m.eval()
    models.append(m)

#------- example raw input (reuse the dataset) -------#

d = np.load(npzfile, allow_pickle=True)
Xraw = d["X"].astype(float)
stem = d["stem"]
y = d["y"]

# reproduce the exact held-out test split from fit-nn-khokhlov.py (same params ->
# same partition), already shuffled by train_test_split, and score only those rows
idx = np.arange(len(y))
_, ite = train_test_split(idx, test_size=0.25, stratify=y, random_state=1)
Xraw = Xraw[ite]
stem = stem[ite]
ytrue = y[ite]

#------- preprocess exactly as training did -------#

X = Xraw[:, ck["keepmask"]]
for j, xf in enumerate(ck["xform"]):
    if xf == "log10":
        X[:, j] = np.log10(X[:, j])
    elif xf == "log1p":
        X[:, j] = np.log1p(X[:, j])
X = (X - ck["mean"]) / ck["scale"]
Xt = torch.tensor(X, dtype=torch.float32)

#------- score: ensemble mean = P(DDT), std = uncertainty -------#

with torch.no_grad():
    ps = np.stack([torch.sigmoid(m(Xt).squeeze(1)).numpy() for m in models])
prob = ps.mean(axis=0)
unc = ps.std(axis=0)

ncorrect = 0
for i in range(len(prob)):
    cls = int(prob[i] >= 0.5)
    ok = "OK " if cls == ytrue[i] else "XX "
    ncorrect += cls == ytrue[i]
    print(f"{ok} {stem[i]}  P(DDT)={prob[i]:.3f} +/- {unc[i]:.3f}  class={cls}  true={ytrue[i]}")
print(f"\n{ncorrect}/{len(prob)} correct = {ncorrect/len(prob):.3f}")
