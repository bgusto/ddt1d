#!/usr/bin/env python3
"""Deep-ensemble dense NN classifier for DDT (deton) from the Khokhlov scalars.

Reads khokhlov.npz (X, y, stem, feat), log-transforms wide-range features,
standardizes, splits train/test, then trains NENS independent MLPs (different
seeds -> different init + batch order) on the same train set. The ensemble
mean predicted probability is the P(DDT) estimate; the spread across members
is the predictive (epistemic) uncertainty. Reports ensemble AUC / accuracy /
Brier, a reliability table + ECE, and the mean ensemble std as an uncertainty
summary.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             brier_score_loss, classification_report)

#------- params -------#

npzfile = "khokhlov.npz"
nens = 8
nlayr = 2
nnode = [32, 16]
drpout = [0.2, 0.2]
lernrt = 1e-3
wdecay = 1e-3
nbatsz = 64
nepoch = 200
probfile = "ensembleprobs.dat"
modelfile = "ensemble.pt"

#------- load + transform -------#

d = np.load(npzfile, allow_pickle=True)
X, y, feat = d["X"], d["y"].astype(np.float32), list(d["feat"])

keepmask = X.std(axis=0) > 0
for i, f in enumerate(feat):
    if f in ("xc12", "xo16"):     # near-constant (~0.5) and mutually redundant; no signal here
        keepmask[i] = False
dropped = [f for f, k in zip(feat, keepmask) if not k]
if dropped:
    print("dropping constant features:", dropped)
feat = [f for f, k in zip(feat, keepmask) if k]
X = X[:, keepmask]

xform = []                            # record per-feature transform so predict.py can replay it
for j, f in enumerate(feat):
    col = X[:, j]
    if f == "rdet":
        X[:, j] = np.log1p(col); xform.append("log1p")
    elif (col > 0).all():
        X[:, j] = np.log10(col); xform.append("log10")
    else:
        xform.append("none")
scaler = StandardScaler()
Xs = scaler.fit_transform(X).astype(np.float32)

npos, nneg = int(y.sum()), int((y == 0).sum())
print(f"{len(y)} cases | deton: {npos} pos / {nneg} neg ({npos/len(y):.1%} positive)")

#------- split -------#

Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.25,
                                      stratify=y, random_state=1)
Xtr = torch.from_numpy(Xtr)
Xte = torch.from_numpy(Xte)
ytr = torch.from_numpy(ytr)
yteT = torch.from_numpy(yte)
ntrain = Xtr.shape[0]
nin = Xtr.shape[1]

#------- train ensemble -------#

memberprobs = np.zeros((nens, Xte.shape[0]))
models = []
for m in range(nens):
    torch.manual_seed(m)

    layrs = [nn.Linear(nin, nnode[0]), nn.GELU(), nn.Dropout(drpout[0])]
    for l in range(nlayr - 1):
        layrs += [nn.Linear(nnode[l], nnode[l+1]), nn.GELU(), nn.Dropout(drpout[l+1])]
    layrs += [nn.Linear(nnode[nlayr-1], 1)]
    model = nn.Sequential(*layrs)

    lossfn = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lernrt, weight_decay=wdecay)

    for epoch in range(nepoch):
        model.train()
        perm = torch.randperm(ntrain)
        for i in range(0, ntrain, nbatsz):
            idx = perm[i:i+nbatsz]
            opt.zero_grad()
            loss = lossfn(model(Xtr[idx]).squeeze(1), ytr[idx])
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pm = torch.sigmoid(model(Xte).squeeze(1)).numpy()
    memberprobs[m] = pm
    models.append(model)
    print(f"  member {m+1}/{nens}  test AUC {roc_auc_score(yte, pm):.3f}")

#------- save ensemble + preprocessing (needed to score new samples) -------#

torch.save({
    "members": [mm.state_dict() for mm in models],
    "nnode": nnode, "nlayr": nlayr, "drpout": drpout, "nin": nin,
    "keepmask": keepmask, "xform": xform,
    "mean": scaler.mean_, "scale": scaler.scale_,
}, modelfile)
print("wrote", modelfile)

#------- ensemble prediction + uncertainty -------#

p = memberprobs.mean(axis=0)
pstd = memberprobs.std(axis=0)
yhat = (p >= 0.5).astype(int)

np.savetxt(probfile, np.column_stack([yte, p, pstd]),
           header="ytrue ensmean ensstd")
print("wrote", probfile)

print(f"\nensemble ({nens} members)")
print(f"test AUC   = {roc_auc_score(yte, p):.3f}")
print(f"test acc   = {accuracy_score(yte, yhat):.3f}")
print(f"test brier = {brier_score_loss(yte, p):.3f}  (lower = better calibrated)")
print(f"mean pred std (epistemic uncertainty) = {pstd.mean():.3f}")
print(classification_report(yte, yhat, digits=3))

#------- calibration: reliability table + ECE -------#

nbin = 10
edges = np.linspace(0, 1, nbin + 1)
print("reliability  (bin  n   mean_p   frac_pos)")
ece = 0.0
for b in range(nbin):
    inbin = (p >= edges[b]) & (p < edges[b+1] if b < nbin-1 else p <= edges[b+1])
    if inbin.sum() == 0:
        continue
    conf = p[inbin].mean()
    acc = yte[inbin].mean()
    ece += inbin.mean() * abs(acc - conf)
    print(f"  [{edges[b]:.1f},{edges[b+1]:.1f})  {inbin.sum():5d}  {conf:.3f}  {acc:.3f}")
print(f"ECE = {ece:.3f}  (lower = better calibrated)")
