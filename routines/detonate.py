#!/usr/bin/env python3
"""Detonation label for a 1D FLASH hotspot case from its ddt1d.dat log.

Definition
----------
A case is labeled a detonation by a majority vote (>= MINVOTE of 4) over four
indicators taken from the FLASH integrated-quantities log (ddt1d.dat).  The
indicators are dimensionless or relative so they transfer across initial
conditions:

  front : max|maxT_x| exceeds a floor well above grid jitter (~20 cm)
  cburn : fractional drop in total C12 mass
  mach  : peak mean Mach
  trise : max(maxT_temp) / initial

PKMACH/TRISE are calibrated by grid search against front (the one unambiguous
indicator) over the 400 cases in raw-data/: 0.0138 / 1.549 split propagating
vs. non-propagating with 88.5% / 96.0% accuracy respectively.

The metric tests sustained / propagating behavior, not peaks: the nuclear
burning rate spikes on the first step even in non-detonating runs (an initial
burner transient that fizzles), and FLASH's shockedCells counter reads 0 even
in a clear detonation -- neither is used.

dettime is the moment the reaction front launches past the propagation floor.
"""
import sys
import numpy as np

# ddt1d.dat column order (FLASH integrated-quantities log)
COLS = ('time mass xmom ymom zmom Etot Ekin Eint Enuc meanMach rmsMach stdMach '
        'meanMachMwt rmsMachMwt meanTmwt rmsvelx rmsvely rmsvelz maxT_dens maxT_temp '
        'maxT_C12 maxT_O16 maxT_igtmC maxT_x maxT_y maxT_z min_igtm shockedCells '
        'totalCells fillfac ar36 c12 ca40 cr48 fe52 he4 mg24 ne20 ni56 o16 s32 si28 ti44').split()
IDX = {c: i for i, c in enumerate(COLS)}

XTRAV = 1.0e3     # cm; grid jitter floor is ~20 cm
CBURN = 0.02      # fractional carbon consumed
PKMACH = 1.4e-2   # peak mean Mach
TRISE = 1.55      # max(maxT_temp) / initial
MINVOTE = 3       # of 4 indicators


def detonate(datfile):
    a = np.loadtxt(datfile, comments="#")
    col = lambda c: a[:, IDX[c]]
    t = col("time")
    maxT = col("maxT_temp")
    maxTx = np.abs(col("maxT_x"))
    c12 = col("c12")
    mach = col("meanMach")

    front = float(maxTx.max())
    burn = float((c12[0] - c12.min()) / c12[0]) if c12[0] > 0 else 0.0
    pkmach = float(mach.max())
    trise = float(maxT.max() / maxT[0]) if maxT[0] > 0 else float("inf")

    votes = {
        "front": front > XTRAV,
        "cburn": burn > CBURN,
        "mach": pkmach > PKMACH,
        "trise": trise > TRISE,
    }
    nvote = sum(votes.values())
    deton = nvote >= MINVOTE

    if deton:
        launched = np.nonzero(maxTx > XTRAV)[0]
        dettime = float(t[launched[0]]) if launched.size else float(t[-1])
    else:
        dettime = float("nan")

    return {
        "deton": deton,
        "dettime": dettime,
        "nvote": nvote,
        "votes": votes,
        "metrics": {"front": front, "burn": burn, "pkmach": pkmach, "trise": trise},
        "tfinal": float(t[-1]),
    }


if __name__ == "__main__":
    for datfile in (sys.argv[1:] or ["ddt1d.dat"]):
        r = detonate(datfile)
        m = r["metrics"]
        flag = "DETONATION" if r["deton"] else "no detonation"
        print(f"\n{datfile}")
        print(f"  -> {flag}  ({r['nvote']}/4 votes)   "
              f"dettime={r['dettime']:.3e}s  tfinal={r['tfinal']:.3e}s")
        print(f"     front={m['front']:.3e} cm   cburn={m['burn']*100:.2f}%   "
              f"pkmach={m['pkmach']:.3e}   trise={m['trise']:.3f}x")
        print("     votes: " + ", ".join(f"{k}={'Y' if v else 'n'}" for k, v in r["votes"].items()))
