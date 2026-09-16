"""Calculate S92_sand resistance (s/m) using the paper coefficients."""
import numpy as np

def soilress_S92_sand(vol_liq, porsl, bsw, psi0, dz, t, hksati, sand, om):
    """Return r_ss (s/m) from relative wetness vol_liq/porsl and sand fraction."""
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
