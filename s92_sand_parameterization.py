"""Evaluate the final S92_sand soil surface resistance, Eqs. (28)-(32).

This module is independent of the data-processing scripts: supply soil water
content, porosity, and sand fraction directly to soilress_S92_sand.
fit_s92_sand_parameters.py estimates the sand-parameter relationship; this
implementation uses the fixed manuscript coefficients 20.45 and -4.12.
It neither reads fitted CSVs nor reruns parameter estimation.
The returned r_ss is in s/m. With r_d in s/m, beta = 1/(1 + r_ss/r_d)."""
import numpy as np

def soilress_S92_sand(vol_liq, porsl, bsw, psi0, dz, t, hksati, sand, om):
    """Return r_ss (s/m); vol_liq/porsl and sand are fractions. Other inputs are unused."""
    # Relative wetness is dimensionless; clipping defines the implemented boundaries.
    fac = np.clip(vol_liq / porsl, 0.001, 1)
    # Smooth transitions at the dry and wet ends (Eqs. 30-31).
    f_transition = 1 / (1 + np.exp(-50 * (fac - 0.35)))
    f_sigmoid = 1 / (1 + np.exp(25 * (fac - 0.7)))
    # Calculate the original S92 and sand-dependent resistances.
    rss_basic = np.exp(8.206 - 6.0 * fac)
    rss_adjusted = np.exp(8.206 - np.log(1 + np.exp(20.45 * sand - 4.12)) * fac)
    # Combine the two resistances using the transition weights (Eq. 32).
    rss = (1 - f_transition) * rss_basic + f_transition * rss_adjusted * f_sigmoid
    return rss
