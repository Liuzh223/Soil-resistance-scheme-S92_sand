"""Physical helpers. Units: K, Pa, kg/kg, s/m, and mm (water potential)."""
import numpy as np
vonkar = 0.4

def soilress_dg(porsl, bsw, psi0, t):
    """Effective soil vapor diffusivity (m2/s)."""
    aird = porsl * (psi0 / -10000000.0) ** (1.0 / bsw)
    d0 = 2.12e-05 * (t / 273.15) ** 1.75
    eps = porsl - aird
    tao = eps * eps * (eps / porsl) ** (3 / np.maximum(3, bsw))
    dg = d0 * tao
    return dg

def soilress_aird(porsl, bsw, psi0):
    """Residual volumetric water content at the prescribed water-potential threshold."""
    aird = porsl * (psi0 / -10000000.0) ** (1.0 / bsw)
    return aird

def soilress_wp(porsl, bsw, psi0):
    """Wilting-point volumetric water content."""
    wp = porsl * (psi0 / -150000) ** (1.0 / bsw)
    return wp

def soilress_fc(porsl, bsw, psi0):
    """Field-capacity volumetric water content."""
    fc = porsl * (psi0 / -3400) ** (1.0 / bsw)
    return fc

def qsadv(T, p):
    # Saturation vapor-pressure polynomial coefficients over liquid water.
    a = [6.11213476, 0.444007856, 0.0143064234, 0.000264461437, 3.05903558e-06, 1.96237241e-08, 8.92344772e-11, -3.7320841e-13, 2.09339997e-16]
    # Corresponding coefficients over ice; select the branch by temperature.
    c = [6.11123516, 0.503109514, 0.0188369801, 0.000420547422, 6.14396778e-06, 6.02780717e-08, 3.87940929e-10, 1.49436277e-12, 2.62655803e-15]
    td = T - 273.16
    td = np.clip(td, -75.0, 75.0)
    es = np.where(td >= 0, np.polyval(a[::-1], td), np.polyval(c[::-1], td)) * 100
    vp = 1.0 / (p - 0.378 * es)
    vp1 = 0.622 * vp
    # Convert vapor pressure to specific humidity (kg/kg).
    qs = es * vp1
    return qs

def calculate_rhoair(PA_F, qair, TA_F_MDS, Rd=287.04):
    """Moist-air density (kg/m3) from pressure (Pa), humidity (kg/kg), and temperature (K)."""
    rhoair = (PA_F - 0.378 * qair * PA_F / (0.622 + 0.378 * qair)) / (Rd * TA_F_MDS)
    return rhoair

def calculate_Qg(Qair, Tsurf, Psurf, psi0, Wsurf, porsl, bsw):
    """Soil-pore specific humidity (kg/kg), including the soil-water-potential correction."""
    Qsrf_sat = qsadv(Tsurf, Psurf)
    psit = np.maximum(-100000000.0, psi0 * (Wsurf / porsl) ** (-bsw))
    hr = np.exp(psit / 47104.7 / Tsurf)
    Qg = np.where((Qsrf_sat > Qair) & (Qair > hr * Qsrf_sat), Qair, hr * Qsrf_sat)
    return Qg

def psi(k, zeta):
    """Unstable similarity correction: k=1 for momentum, k=2 for scalars."""
    chik = (1.0 - 16.0 * zeta) ** 0.25
    if k == 1:
        return 2.0 * np.log((1.0 + chik) / 2.0) + np.log((1.0 + chik ** 2) / 2.0) - 2.0 * np.arctan(chik) + np.pi / 2.0
    else:
        return 2.0 * np.log((1.0 + chik ** 2) / 2.0)

def moninobuk(hq, displa, z0m, z0h, z0q, obu):
    """Return humidity and heat profile factors for the specified stability regime."""
    zldis = hq - displa
    zeta = zldis / obu
    zetat = 0.465
    if zeta < -zetat:
        fq = np.log(-zetat * obu / z0q) - psi(2, -zetat) + psi(2, z0q / obu) + 0.8 * (zetat ** (-0.333) - (-zeta) ** (-0.333))
    elif zeta < 0:
        fq = np.log(zldis / z0q) - psi(2, zeta) + psi(2, z0q / obu)
    elif zeta <= 1:
        fq = np.log(zldis / z0q) + 5 * zeta - 5 * z0q / obu
    else:
        fq = np.log(obu / z0q) + 5 - 5 * z0q / obu + (5 * np.log(zeta) + zeta - 1)
    if zeta < -zetat:
        fh = np.log(-zetat * obu / z0h) - psi(2, -zetat) + psi(2, z0h / obu) + 0.8 * (zetat ** (-1 / 3) - (-zeta) ** (-1 / 3))
    elif zeta < 0:
        fh = np.log(zldis / z0h) - psi(2, zeta) + psi(2, z0h / obu)
    elif zeta <= 1:
        fh = np.log(zldis / z0h) + 5 * zeta - 5 * z0h / obu
    else:
        fh = np.log(obu / z0h) + 5 - 5 * z0h / obu + (5 * np.log(zeta) + zeta - 1)
    return (fq, fh)

def calculate_raw(forc_t, forc_psrf, LAI, SAI, qm, rhoair, ustar, LE, H, hq, Hcan):
    """Aerodynamic vapor resistance (s/m); LE and H are in W/m2, heights in m."""
    # Physical constants: dry-air gas constant, heat capacity, and latent heat use SI units.
    vonkar = 0.4
    grav = 9.81
    p0 = 100000.0
    rgas = 287.04
    cpair = 1004.64
    Lambda = 2510400.0
    zlnd = 0.01
    z0mg = zlnd
    # Vegetation roughness length and displacement height (m).
    z0m = 0.1 * Hcan
    displa = 0.667 * Hcan
    z0mv = z0m
    lt = min(LAI + SAI, 2.0)
    egvf = (1.0 - np.exp(-lt)) / (1.0 - np.exp(-2.0))
    displa = egvf * displa
    z0mv = np.exp(egvf * np.log(z0mv) + (1.0 - egvf) * np.log(z0mg))
    z0hv = z0mv
    z0qv = z0mv
    # Potential and virtual potential temperature (K).
    th = forc_t * (p0 / forc_psrf) ** (rgas / cpair)
    thv = th * (1.0 + 0.61 * qm)
    # Convert latent heat flux to water-vapor mass flux (kg/m2/s).
    E = LE / Lambda
    zldis = hq - displa
    if H == 0 or E == 0:
        H = 1e-06
        E = 1e-06
    tstar = H / (-cpair * rhoair * ustar)
    qstar = E / (-rhoair * ustar)
    thvstar = tstar * (1.0 + 0.61 * qm) + 0.61 * th * qstar
    obu = 1 / (vonkar * grav * thvstar / (ustar ** 2 * thv))
    zeta = zldis / obu
    # Limit stability and update the Monin-Obukhov length.
    if zeta >= 0.0:
        zeta = min(2.0, max(zeta, 1e-06))
    else:
        zeta = max(-100.0, min(zeta, -1e-06))
    obu = zldis / zeta
    fq, _ = moninobuk(hq, displa, z0mv, z0hv, z0qv, obu)
    raw = 1.0 / (vonkar / fq * ustar)
    return raw

def calc_rd(ustar, lai, sai):
    """Surface-to-canopy-air vapor transfer resistance (s/m)."""
    z0mg = 0.01
    W = np.exp(-(lai + sai))
    Cs_bare = vonkar / (0.13 * (z0mg * ustar / 1.5e-05) ** 0.45)
    Cs_dense = 0.004
    Cs = Cs_bare * W + Cs_dense * (1 - W)
    rd = 1 / (Cs * ustar)
    return rd

def calculate_emis_veg(IGBP, lai, sai):
    if IGBP == 1 or IGBP == 3 or 6 <= IGBP <= 9:
        chil = 0.01
    elif IGBP == 2:
        chil = 0.1
    elif IGBP == 4:
        chil = 0.25
    elif IGBP == 5:
        chil = 0.125
    elif IGBP == 10 or IGBP == 12:
        chil = -0.3
    else:
        chil = 0
    # Leaf-angle coefficients for the vegetation radiation calculation.
    phi1 = 0.5 - 0.633 * chil - 0.33 * chil * chil
    phi2 = 0.877 * (1.0 - 2.0 * phi1)
    # Compute the angular integral, including near-zero coefficient cases.
    if abs(phi1) > 1e-06 and abs(phi2) > 1e-06:
        zmu = 1.0 / phi2 * (1.0 - phi1 / phi2 * np.log((phi1 + phi2) / phi1))
    elif abs(phi1) <= 1e-06:
        zmu = 1.0 / 0.877
    elif abs(phi2) <= 1e-06:
        zmu = 1.0 / (2.0 * phi1)
    # Canopy transmissivity and emissivity.
    power3 = (lai + sai) / zmu
    power3 = np.minimum(50.0, power3)
    power3 = np.maximum(1e-05, power3)
    thermk = np.exp(-power3)
    emis_veg = 1.0 - thermk
    return emis_veg

def calc_surface_temperature_from_canopy_obs_ratio(Rld, mRlu, Rlu, Tleaf, emis_surf, IGBP, lai, sai):
    """Infer surface temperature (K) using modeled longwave component fractions."""
    sb = 5.670373e-08
    emis_veg = calculate_emis_veg(IGBP, lai, sai)
    tao_v = 1 - emis_veg
    mRlu_veg = emis_veg * sb * Tleaf ** 4
    mRlu_ref = (1 - emis_surf) * (tao_v ** 2 * Rld + tao_v * emis_veg * sb * Tleaf ** 4)
    mRlu_srf = mRlu - mRlu_veg - mRlu_ref
    srf_ratio = mRlu_srf / mRlu
    Rlu_srf = Rlu * srf_ratio
    T_surf = (Rlu_srf / (tao_v * emis_surf * sb)) ** 0.25
    return T_surf
