#!/usr/bin/env python3
"""Median of an array outside the hot region -- the ambient/background value."""
import numpy as np


def bg(v, out):
    return float(np.median(v[out]))
