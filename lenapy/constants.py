"""
GRS80 choice

Constants
---------
DAY_YEAR : float
    Floating number of day per year (including leap year)
SECONDS_DAY : float
    Number of seconds per day
A_EARTH_GRS80 : float
    Earth semi-major axis [m] of [GRS80]_
F_EARTH_GRS80 : float
    Earth flattening of [GRS80]_
LNPY_SURFTER : float
    Earth surface area [m²]
LNPY_G : float
    Gravitational constant [m³.kg⁻¹.s⁻²] define in [CODATA2018]_
LNPY_GM_EARTH : float
    Standard gravitational parameter for Earth [m³.s⁻²] define in [IERS2010]_
LNPY_GM_ATMO : float
    Standard gravitational parameter for the atmosphere of the Earth [m³.s⁻²] deduced from [Trenberth2005]_
LNPY_G_WMO : float
    Standard gravitational acceleration [m.s⁻²] define in [WMO1988]_

LNPY_RAVERAGE_EARTH : float
    Radius of the Earth for which spherical Earth volume = ellipsoidal Earth volume
LNPY_RHO_EARTH : float
    Earth density [kg.m⁻³]

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
        doi: 10.1175/JCLI-3299.1 <https://doi.org/10.1175/JCLI-3299.1>`_
    .. [WMO1988] World Meteorological Organization
        *WMO Technical Regulations*, 49, (1988).
"""
import numpy as np

LNPY_DAYS_YEAR = 365.24219 # d/y
LNPY_SECONDS_DAY = 86400.  # s/d
LNPY_MEAN_EARTH_RADIUS = 6378137. # m
LNPY_EARTH_FLATTENING = 1/298.257222 # -
LNPY_EARTH_SURFACE = 5.10074897e+14 # m²
A_EARTH_GRS80 = 6378137.  # m, earth semi-major axis [GRS80]_
F_EARTH_GRS80 = 1/298.257222101  # earth flattening [GRS80]_
LNPY_SURFTER = 510065621718491.4  # m²
# 2*np.pi*A_EARTH_GRS80**2 + np.pi*(A_EARTH_GRS80 - A_EARTH_GRS80*F_EARTH_GRS80)**2/
# np.sqrt(2*F_EARTH_GRS80 - F_EARTH_GRS80**2)*np.log((1+np.sqrt(2*F_EARTH_GRS80 - F_EARTH_GRS80**2))/
# (1-np.sqrt(2*F_EARTH_GRS80 - F_EARTH_GRS80**2)))
LNPY_G = 6.67430e-11  # m3.kg-1.s-2 [CODATA2018]
# GM value is consistent with Terrestrial Time (TT) as the time argument, TT(TAI) = TAI + 32.184 seconds
LNPY_GM_EARTH = 3.986004415e14  # m3.s-2 IERS-2010 Standards [IERS2010]_
LNPY_GM_ATMO = 3.436e8  # m3.s-2 mass of the atmosphere from [Trenberth2005]_ multiplied by LNPY_G
LNPY_G_WMO = 9.80665  # standard gravitational acceleration [WMO1988]

# Radius of the Earth for which spherical Earth volume = ellipsoidal Earth volume
LNPY_RAVERAGE_EARTH = A_EARTH_GRS80*(1 - F_EARTH_GRS80)**(1/3)  # Earth average radius

# Deduce from LNPY other constants with rho = 3*M / 4*pi*R**3
LNPY_RHO_EARTH = 0.75*(LNPY_GM_EARTH - LNPY_GM_ATMO)/(LNPY_G * np.pi * LNPY_RAVERAGE_EARTH**3)  # kg.m-3 , Earth density
