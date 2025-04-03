"""
Constants for geophysical calculations in the lenapy library.

This module defines various constants related to geophysical and gravitational properties of the Earth.
These constants are used throughout the lenapy library for consistent and accurate calculations.

The default Earth ellipsoidal model used in lenapy is GRS80

Constants
---------
DAY_YEAR : float
    Floating number of days per year (including leap years).
SECONDS_DAY : float
    Number of seconds per day.
LNPY_A_EARTH_GRS80 : float
    Earth semi-major axis [m] of [GRS80]_.
LNPY_F_EARTH_GRS80 : float
    Earth flattening of [GRS80]_.
LNPY_EARTH_SURFACE : float
    Surface area of the Earth [m²].
LNPY_G : float
    Gravitational constant [m³.kg⁻¹.s⁻²] defined in [CODATA2018]_.
LNPY_GM_EARTH : float
    Standard gravitational parameter for Earth [m³.s⁻²] defined in [IERS2010]_.
LNPY_GM_ATMO : float
    Standard gravitational parameter for the atmosphere of the Earth [m³.s⁻²] deduced from [Trenberth2005]_.
LNPY_G_WMO : float
    Standard gravitational acceleration [m.s⁻²] defined in [WMO1988]_.

LNPY_RAVERAGE_EARTH : float
    Radius of the Earth for which the spherical Earth volume equals the ellipsoidal Earth volume.
LNPY_RHO_EARTH : float
    Average Earth density [kg.m⁻³].

References
----------
    .. [CODATA2018] E. Tiesinga, P. Mohr, D. Newell, B. Taylor,
        "CODATA Recommended Values of the Fundamental Physical Constants: 2018",
        *Reviews of Modern Physics*, 93, (2018).
        `doi: 10.1103/RevModPhys.93.025010 <https://doi.org/10.1103/RevModPhys.93.025010>`_
    .. [GRS80] H. Moritz,
        "Geodetic reference system 1980",
        *Bulletin géodésique*, 54, 395-405, (1980).
        `doi: 10.1007/BF02521480 <https://doi.org/10.1007/BF02521480>`_
    .. [IERS2010] G. Petit and B. Luzum,
        "IERS Conventions (2010)",
        *IERS Technical Note*, 36, 88--89, (2010).
        `doi: 10.1007/s00190-002-0216-2 <https://doi.org/10.1007/s00190-002-0216-2>`_
    .. [Trenberth2005] K. Trenberth, L. Smith,
        "The Mass of the Atmosphere: A Constraint on Global Analyses",
        *Journal of Climate*, 18, 6, 864-875, (2005).
        `doi: 10.1175/JCLI-3299.1 <https://doi.org/10.1175/JCLI-3299.1>`_
    .. [WMO1988] World Meteorological Organization
        *WMO Technical Regulations*, 49, (1988).
"""

from numpy import pi

LNPY_DAYS_YEAR = 365.24219  # d/y
LNPY_SECONDS_DAY = 86400.0  # s/d

LNPY_A_EARTH_GRS80 = 6378137.0  # m, Earth semi-major axis [GRS80]_
LNPY_F_EARTH_GRS80 = 1 / 298.257222101  # Earth flattening [GRS80]_

LNPY_EARTH_SURFACE = 510065621718491.4  # m²
# equals to 2*np.pi*LNPY_A_EARTH_GRS80**2 + np.pi*(LNPY_A_EARTH_GRS80 - LNPY_A_EARTH_GRS80*LNPY_F_EARTH_GRS80)**2/
# np.sqrt(2*LNPY_F_EARTH_GRS80 - LNPY_F_EARTH_GRS80**2)*
# np.log((1 + np.sqrt(2*LNPY_F_EARTH_GRS80 - LNPY_F_EARTH_GRS80**2))/
# (1 - np.sqrt(2*LNPY_F_EARTH_GRS80 - LNPY_F_EARTH_GRS80**2)))  ## note that np.sqrt(2*F - F**2) = Excentricity

LNPY_G = 6.67430e-11  # m³.kg⁻¹.s⁻² [CODATA2018]_
# GM value is consistent with Terrestrial Time (TT) as the time argument, TT(TAI) = TAI + 32.184 seconds
LNPY_GM_EARTH = 3.986004415e14  # m³.s⁻², IERS-2010 Standards [IERS2010]_
LNPY_GM_ATMO = (
    3.436e8  # m³.s⁻², mass of the atmosphere from [Trenberth2005]_ multiplied by LNPY_G
)
LNPY_G_WMO = 9.80665  # m.s², standard gravitational acceleration from [WMO1988]_

# Radius of the Earth for which spherical Earth volume equals the ellipsoidal Earth volume
LNPY_RAVERAGE_EARTH = LNPY_A_EARTH_GRS80 * (1 - LNPY_F_EARTH_GRS80) ** (
    1 / 3
)  # m, Average radius of the Earth

# Deduced from LNPY other constants with rho = 3*M_Earth / 4*pi*R**3 approx equals to 5513.4
LNPY_RHO_EARTH = (
    0.75 * (LNPY_GM_EARTH - LNPY_GM_ATMO) / (LNPY_G * pi * LNPY_RAVERAGE_EARTH**3)
)  # kg.m-3 , Earth density

LNPY_SSO = 35.16404  # psu - ref TEOS10 seawater equations (gsw)
