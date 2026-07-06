#!/usr/bin/env python3
"""Detonation label from the last plotfile's peak local Mach number.

Validated 2026-07-04 against 50 manually-reviewed cases: mach > MACHTHRESH
agreed with a human call on 49/50; the one miss (ta16579_tm20560_rd00344)
turned out to be a corrupted run (checkerboard pattern), not a labeler
error -- excluding known-bad cases, agreement is 49/49. This supersedes the
old .dat multi-vote label in detonate.py, which only agreed with manual
review on 33/50 (it over-calls detonation: meanMach in the .dat log tops
out ~0.4 and never actually discriminates detonating from non-detonating
runs).

Usage:  python3 machdet.py [stem]
"""
import sys
import numpy as np

from routines.lastplt import lastplt
from routines.readflash import readflash

MACHTHRESH = 0.8


def machdetpf(pf):
    """Same as machdet(), but takes the last-plotfile path directly -- for
    bulk callers that already have a stem -> last-plotfile index (a glob per
    case is too slow at dataset scale)."""
    d = readflash(pf)
    maxmach = float((np.abs(d["velx"]) / d["cs"]).max())
    return maxmach > MACHTHRESH, maxmach


def machdet(stem):
    pf = lastplt(stem)
    if pf is None:
        return None
    return machdetpf(pf)


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else \
        "ddt1d_ta17249_tm20811_rd00563"
    deton, maxmach = machdet(stem)
    print(f"{stem}  maxmach={maxmach:.3f}  {'DETON' if deton else 'no'}")
