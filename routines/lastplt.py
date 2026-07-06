#!/usr/bin/env python3
"""Find the last FLASH plotfile for a case, by plot index.

Dumps are named <stem>_hdf5_plt_cnt_NNNN. Runs can terminate early (the
solver's internal DDT detector firing) so the last dump isn't always index
0040 -- but every case in this dataset uses that one plain numbering scheme
(checked: no out-of-sequence/forced dump files actually exist), so the last
dump is reliably the one with the highest NNNN. Picking by filename index
instead of opening every candidate file to compare simulation time is much
faster over ~25k cases.

Usage:  python3 lastplt.py [stem]
"""
import sys
import glob
import re

pltre = re.compile(r"_hdf5_plt_cnt_(\d+)$")


def lastplt(stem):
    cands = glob.glob(stem + "_hdf5_plt_cnt_*")
    if not cands:
        return None
    return max(cands, key=lambda c: int(pltre.search(c).group(1)))


if __name__ == "__main__":
    stem = sys.argv[1] if len(sys.argv) > 1 else \
        "ddt1d_ta17249_tm20811_rd00563"
    print(lastplt(stem))
