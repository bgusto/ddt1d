#!/usr/bin/env python3
"""Chapman-Jouguet speed from a strong-detonation estimate.

QNUC (erg/g) is a Tier-1 fixed value for C/O -> IME/NSE at rho~1e7; upgrade to
a Helmholtz-EOS CJ solve for a state/composition-consistent DCJ if needed.
"""
import numpy as np

QNUC = 6.0e17


def dcj(gamc, q=QNUC):
    return np.sqrt(2.0 * (gamc ** 2 - 1.0) * q)
