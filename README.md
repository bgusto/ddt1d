# DDT1D — Learning the probability of deflagration-to-detonation transition

A machine-learned subgrid model for one of the longest-standing unresolved
problems in thermonuclear (Type Ia) supernova theory: **when does a subsonic
flame in degenerate carbon–oxygen matter spontaneously run away into a
supersonic detonation?** The transition is governed by microphysics far below
the reach of any multidimensional simulation, so it has to be *modeled*, not
resolved.

We build a probabilistic classifier that maps a handful of resolved-scale
hotspot descriptors — peak temperature, induction-time minimum, background
thermodynamic state, and the reactive-gradient (detonation) radius — to
**P(DDT)**, the probability that the hotspot detonates. It is designed to act as
a subgrid closure inside large-scale supernova models, where the ignition scale
lives entirely below the grid.

- **Dataset** — 25,000+ one-dimensional FLASH/PROTEUS direct numerical
  simulations of reactive hotspots, with initial states drawn from
  turbulence-extracted temperature profiles.
- **Model** — a deep ensemble of compact neural networks. The ensemble mean is a
  well-calibrated P(DDT) (expected calibration error ≈ 0.02) and the disagreement
  across members is a principled epistemic uncertainty. Held-out AUC ≈ 0.95.

📄 Paper: *(TK: citation / arXiv link)*

## Quick start

The curated dataset (`khokhlov.npz`, ~1 MB) ships with the repository, so you can
reproduce the model and every figure without any raw simulation data:

```
python3 fit-nn-khokhlov.py          # train the ensemble -> ensemble.pt, ensembleprobs.dat
python3 predict.py                  # score the held-out test set
cd plotting/manuscript             # paper figures -> figures/
for f in roc calib prc confus; do python3 $f.py; done
```

## The dataset

Raw FLASH/PROTEUS data for each 1D DNS hotspot simulation live in `$DDT1D_DATA`.
You only need this if you want to **rebuild the training dataset from scratch**;
otherwise the shipped `khokhlov.npz` is enough.

The DNS were run with two main initial temperature profiles: (1) Gaussian and
(2) extracted from a 3D stirred compressible turbulence simulation. `$DDT1D_DATA`
holds the FLASH/PROTEUS plot files (`plt_*`) for each case plus the integrated
quantities file (`*.dat`). The **initial** plotfile of each run supplies the
tabular input features; the **final** plotfile is post-processed to robustly
determine whether a detonation occurred — i.e. it provides the label.

To rebuild from scratch, point `$DDT1D_DATA` at the raw data (TK: insert download
location) and run:

```
python3 build-dataset-khokhlov.py   # -> khokhlov.npz  (~4 hours)
```

This parses the raw data, computes the Khokhlov scalars (peak temperature,
minimum induction time, and the far-field background state), assigns the binary
detonation label, and writes `khokhlov.npz`.

## Training

`fit-nn-khokhlov.py` reads `khokhlov.npz`, log-scales the wide-range features,
standardizes them, and carves off a random train/test split. It then trains a
deep ensemble of eight small GELU networks (AdamW, plain unweighted
cross-entropy). The ensemble mean is the detonation probability P(DDT); the
spread across the eight members is the uncertainty. The script reports test AUC,
accuracy, Brier score, and a reliability table with its ECE, then saves:

- `ensemble.pt` — the trained ensemble (weights **and** the exact preprocessing)
- `ensembleprobs.dat` — the per-case test predictions

## Prediction

`predict.py` loads `ensemble.pt`, replays the saved preprocessing bit-for-bit,
and scores samples — returning P(DDT) and its uncertainty. This is the piece
intended to be embedded as a subgrid closure.

## Paper figures

Four scripts under `plotting/manuscript/` each read `ensembleprobs.dat` and draw
one metric:

| script | figure |
| --- | --- |
| `roc.py` | ROC curve + AUC |
| `calib.py` | reliability diagram + ECE |
| `prc.py` | precision–recall curve + average precision |
| `confus.py` | confusion matrix |

Each writes a vector `.pdf` (for the paper) and a `.png` (for a quick look) into
`figures/`.
