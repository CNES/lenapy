import numpy as np

"""
References
    ----------
    .. [WMO1988] World Meteorological Organization
        *WMO Technical Regulations*, 49, (1988).
    .. [CODATA2018] E. Tiesinga, P. Mohr, D. Newell, B. Taylor,
        "CODATA Recommended Values of the Fundamental Physical Constants: 2018",
        *Reviews of Modern Physics*, 93, (2018).
        `doi: 10.1103/RevModPhys.93.025010 <https://doi.org/10.1103/RevModPhys.93.025010>`_
    .. [GRS80] H. Moritz,
        "Geodetic reference system 1980",
        *Bulletin géodésique*, 54, 395-405, (1980).
        `doi: 10.1007/BF02521480 <https://doi.org/10.1007/BF02521480>`_
"""

DAY_YEAR = 365.24219
SECONDS_DAY = 86400.
LNPY_RTER = 6378137.  # m
LNPY_F_EARTH = 1/298.257222101  # earth flattening [GRS80]
LNPY_SURFTER = 5.10074897e+14  # m²
LNPY_A_EARTH = 6378137.  # m, earth semi-major axis [GRS80]
LNPY_G = 6.67430e-11  # m3.kg-1.s-2 [CODATA2018]
LNPY_GM_EARTH = 3.986004415e14  # m3.s-2
LNPY_GM_ATMO = 3.44e8  # m3.s-2
LNPY_G_WMO = 9.80665  # standard gravitational acceleration [WMO1988]

LNPY_E_EARTH = np.sqrt(2*LNPY_F_EARTH - LNPY_F_EARTH**2)  # earth eccentricity deduce from [GRS80]
# Radius of the Earth for which spherical Earth volume = ellipsoidal Earth volume
LNPY_RAVERAGE_EARTH = LNPY_A_EARTH*(1 - LNPY_F_EARTH)**(1/3)  # Earth average radius
LNPY_RHO_EARTH = 0.75*(LNPY_GM_EARTH - LNPY_GM_ATMO)/(LNPY_G * np.pi * LNPY_RAVERAGE_EARTH**3)  # kg.m-3 , Earth density
