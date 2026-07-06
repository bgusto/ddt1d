#!/usr/bin/env python3
"""Detonation radius of a hotspot from its induction-time/wave-speed profile.

Definition
----------
rdet = half-width of the contiguous region, containing the induction-time
minimum, over which the spontaneous reaction wave outruns the Chapman-Jouguet
speed:  |usp| >= dcj,  where  usp = 1/|grad tau_ind|  is the phase velocity
of sequential ignition.  Equivalently the region where the induction-time
gradient is sub-critical, |grad tau| <= 1/dcj.

This is the Zel'dovich reactivity-gradient / SWACER condition: a detonation can
be initiated where the spontaneous wave couples to the gas dynamics near dcj.
There is no tunable threshold -- the gate is a physical speed.  The radius is
defined by the region edges (where the gradient crosses critical), so it does
not depend on pinpointing the center, which makes it robust on flat-topped
hotspots.
"""
import numpy as np


def detrad(x, igtm, usp, dcj):
    """Returns (rdet, xc, xL, xR): half-width, region center, and the
    interpolated left/right edges where |usp| = dcj."""
    g = np.abs(usp) - dcj                  # >= 0 where the phase wave outruns dcj
    coupled = g >= 0.0
    i0 = int(np.argmin(igtm))               # ignition point

    if not coupled[i0]:                     # gradient already supercritical at center
        return 0.0, x[i0], x[i0], x[i0]

    iL = i0
    while iL - 1 >= 0 and coupled[iL - 1]:
        iL -= 1
    iR = i0
    while iR + 1 < len(x) and coupled[iR + 1]:
        iR += 1

    # sub-cell crossings by linear interpolation of g; clamp at domain edge
    xL = x[iL] if iL == 0 else x[iL] - (x[iL] - x[iL - 1]) * g[iL] / (g[iL] - g[iL - 1])
    xR = x[iR] if iR == len(x) - 1 else x[iR] + (x[iR + 1] - x[iR]) * g[iR] / (g[iR] - g[iR + 1])
    xc = 0.5 * (xL + xR)
    return 0.5 * (xR - xL), xc, xL, xR
