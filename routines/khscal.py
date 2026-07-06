#!/usr/bin/env python3
"""Khokhlov scalar inputs for one hotspot case, from its initial-condition plotfile.

Extracts: tmax (peak temperature), taumin (minimum induction time), and the
background state (dens, tamb, csnd, pres, xc12, xo16) taken as the median of
the far field outside the hotspot. Hotspot = ready to burn: igtm within MULT
times the case's own minimum induction time (always non-empty by construction,
and unlike a temperature threshold doesn't assume an injected T perturbation
-- works the same for gauss and extr cases).
"""
import sys
from routines.readflash import readflash
from routines.bg import bg

MULT = 3.0   # hot = igtm <= MULT * igtm.min()


def khscal(pltfile):
    d = readflash(pltfile)
    igtm = d["igtm"]

    hot = igtm <= MULT * igtm.min()
    out = ~hot

    return {
        "tmax": float(d["temp"].max()),
        "taumin": float(igtm.min()),
        "dens": bg(d["dens"], out),
        "tamb": bg(d["temp"], out),
        "csnd": bg(d["cs"], out),
        "pres": bg(d["pres"], out),
        "xc12": bg(d["c12"], out),
        "xo16": bg(d["o16"], out),
    }


if __name__ == "__main__":
    for pltfile in (sys.argv[1:] or ["."]):
        s = khscal(pltfile)
        print(pltfile)
        for k, v in s.items():
            print(f"  {k:7s} {v:.6e}")
