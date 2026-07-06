# DDT1D — Learning the probability of deflagration-to-detonation transition

A machine-learned subgrid model for one of the longest-standing unresolved
problems in thermonuclear (Type Ia) supernova theory: when does a subsonic
flame in degenerate carbon–oxygen matter spontaneously run away into a
supersonic detonation?  The transition is determined by microphysics far below
the reach of any multidimensional simulation so it has to be modeled (not
resolved).

We create a probabilistic classifier that maps a handful of resolved-scale
hotspot descriptors — peak temperature, induction-time minimum, background
thermodynamic state, and the reactive-gradient (detonation) radius — to P(DDT),
the probability that the hotspot detonates. It is built to act as a subgrid
closure inside large-scale supernova models.

The training dataset consists of 25,000+ one-dimensional FLASH/PROTEUS direct
numerical simulations of reactive hotspots whose initial states are drawn from
turbulence-extracted initial temperature profiles.

The model is a deep ensemble of compact neural networks. The ensemble mean is a
well-calibrated P(DDT) (expected calibration error ≈ 0.02); the disagreement
across members is a principled epistemic uncertainty.  On held-out data it
reaches an AUC of about 0.95.

## Rebuilding the dataset

Raw FLASH/PROTEUS data for each 1D DNS hotspot simulation are stored in
`$DDT1D_DATA`.  If the user wants to rebuild the training dataset from scratch,
the user must set this environment variable to point to the data, which they
can download from (TK: insert location to find data online).  The DNS were run
with two main initial temperature profiles: 1.) Gauss shaped 2.) Extracted from
an actual 3D stirred compressible turbulence simulation In `$DDT1D_DATA` we
store the FLASH/PROTEUS "plot" files (`plt_*`) for each DNS case as well as the
integrated quantities file `*.dat`.  The initial plotfile for each run is used
to build the tabular input dataset.  The final plotfile is used to robustly
determine whether a detonation occured - in other words it is used for
labeling.  To build the training dataset from scratch run:

```
python3 build-dataset-khokhlov.py
```

This parses the raw data and computes scalars: peak temperature, minimum
induction time, and the far-field background state.  It also produces the binary
label: detonation or no detonation. It takes four hours or so.  It should output
khokhlov.npz.

By default we store khokhlov.npz in the repo - it is only 1MB of storage.

## Training

The training code `fit-nn-khokhlov.py` reads khokhlov.npz, log-scales
the wide-range features, standardizes them, and carves off a random train/test
split.  It then trains a deep ensemble of eight small GELU networks (AdamW,
plain unweighted cross-entropy).  The ensemble mean is your detonation
probability P(DDT); the spread across the eight members is the uncertainty.
The script prints the test AUC, accuracy, Brier score, a reliability table and
its ECE, then tucks the trained ensemble away in ensemble.pt (weights AND the
exact preprocessing) and the per-case test predictions in ensembleprobs.dat.

## Prediction

To predict new cases:  `predict.py` loads ensemble.pt, replays the
saved preprocessing bit-for-bit, and scores samples, handing back P(DDT) and
its uncertainty.

## Paper figures

To make the paper figures there are four scripts that each read
ensembleprobs.dat.  They live in `plotting/manuscript/`.  They are:

```
roc.py     ROC curve + AUC
calib.py   reliability diagram + ECE
prc.py     precision-recall curve + average precision
confus.py  confusion matrix
```

Each drops a vector .pdf (for the paper) and a .png (for a quick look) into
figures/.
