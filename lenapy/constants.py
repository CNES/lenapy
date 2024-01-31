import numpy as np
from numpy import pi
import pathlib
import inspect

DAY_YEAR = 365.24219
SECONDS_DAY = 86400.
LNPY_RTER = 6378137. #m
LNPY_f = 1/298.257222
LNPY_SURFTER = 5.10074897e+14 # m²
a_earth = 6378137.  # m (IAG GRS80)
f_earth = 1/298.257222101  # (IAG GRS80)
e_earth = np.sqrt(2*f_earth - f_earth**2)  # eccentricity (IAG GRS80)
R_earth = a_earth*(1 - f_earth)**(1/3)  # Earth average radius
G_cst = 6.67430e-11  # m3.kg-1.s-2 (CODATA 2018)
GM_earth = 3.986004415e14  # m3.s-2
GM_atmo = 3.44e8  # m3.s-2
rho_earth = 0.75*(GM_earth - GM_atmo)/(G_cst*pi*R_earth**3)  # kg.m-3 , Earth density
g_wmo = 9.80665  # standard gravitational acceleration (World Meteorological Organization)


def read_love_numbers(love_file=None):
    # verif love file, catch error, deal with kl or hl, kl, ll
    # change reference
    # test kwargs given love or not or love_file
    # hl, ll

    if love_file is None:
        current_file = inspect.getframeinfo(inspect.currentframe()).filename
        folderpath = pathlib.Path(current_file).absolute().parent.parent
        love_file = folderpath.joinpath('data', 'love', 'LoveNumbers_Gegout97.txt')

    return np.genfromtxt(love_file)


def l_factor_gravi(l, unit='mewh', include_elastic=True, ellispoidal_earth=False, geocentric_colat=None,
                   love_file=None):
    """
    Compute scale factor for a transformation between spherical harmonics and grid data.
    Spatial data over the grid are associated with a specific unit.
    The scale factor is degree dependant and is computed for the given list of degree l.
    The scale factor can be estimated using elastic or none elastic Earth as well as a spherical or ellipsoidal Earth.

    Parameters
    ----------
    l : np.ndarray
        degree for which the scale factor is estimated.
    unit : str, optional, optional
        'mewh', 'geoid', 'microGal', 'pascal', 'mvcu', or 'norm'
        Unit of the spatial data used in the transformation. Default is 'mewh' for meters of Equivalent Water Height.
        'geoid' represents millimeters geoid height, 'microGal' represents microGal gravity perturbations,
        'pascal' represents equivalent surface pressure in pascal and
        'mvcu' represents meters viscoelastic crustal uplift
    include_elastic : bool, optional
        If True, the Earth behavior is elastic.
    ellispoidal_earth : bool, optional
        If True, consider the Earth as an ellispoid following [Ditmar2018] and if False as a sphere.
    geocentric_colat : list, optional
        List of geocentric colatitude for ellispoidal earth radius computation.
    love_file : str / path, optional
        File with Love numbers that can be read by read_love_numbers() function.
        Default Love numbers used are from Gegout97.

    Returns
    -------
    l_factor : np.ndarray
        Degree dependant scale factor

    References
    ----------
    .. [Ditmar2018] P. Ditmar,
        "Conversion of time-varying Stokes coefficients into mass anomalies at the
        Earth’s surface considering the Earth’s oblateness",
        *Journal of Geodesy*, 92, 1401--1412, (2018).
        `doi: 10.1007/s00190-018-1128-0 <https://doi.org/10.1007/s00190-018-1128-0>`_
    """
    fraction = np.ones(l.shape)

    # include elastic redistribution with kl Love numbers
    if include_elastic:
        # verif love file, catch error, deal with kl or hl, kl, ll
        # change reference
        # test kwargs given love or not or love_file
        # hl, ll
        kl = read_love_numbers(love_file)
        fraction += kl[l]

    # test for ellispoidal_earth
    if ellispoidal_earth and geocentric_colat is None:
        raise ValueError("For ellipsoidal Earth, you need to set "
                         "the parameter 'geocentric_colat' in l_factor_gravi function")

    # l_factor is degree dependant
    if unit == 'norm':
        # norm, fully normalized spherical harmonics
        l_factor = np.ones(l.shape)

    elif unit == 'mewh':
        # mewh, meters equivalent water height [kg.m-2]
        l_factor = rho_earth * a_earth * (2 * l + 1) / (3 * fraction * 1e3)
        if ellispoidal_earth:
            l_factor *= ((R_earth/a_earth)**3 *
                         (np.sqrt(1 - e_earth**2*np.sin(geocentric_colat)**2) / (1 - f_earth))**(l + 2))

    elif unit == 'geoid':
        # geoid, millimeters geoid height
        l_factor = np.ones(l.shape) * a_earth * 1e3
        if ellispoidal_earth:
            l_factor *= (np.sqrt(1 - e_earth**2*np.sin(geocentric_colat)**2) / (1 - f_earth))**(l + 1)

    elif unit == 'microGal':
        # microGal, microGal gravity perturbations
        l_factor = GM_earth * (l + 1) / (a_earth ** 2) * 1e8
        if ellispoidal_earth:
            l_factor *= (np.sqrt(1 - e_earth**2*np.sin(geocentric_colat)**2) / (1 - f_earth))**(l + 2)

    elif unit == 'pascal':
        # pascal, equivalent surface pressure
        l_factor = g_wmo * rho_earth * a_earth * (2*l + 1) / (3 * fraction)
        if ellispoidal_earth:
            l_factor *= (np.sqrt(1 - e_earth**2*np.sin(geocentric_colat)**2) / (1 - f_earth))**(l + 1)

    elif unit == 'mvcu':
        # mVCU, meters viscoelastic crustal uplift
        l_factor = a_earth * (2*l + 1) / 2
        if ellispoidal_earth:
            l_factor *= (np.sqrt(1 - e_earth**2*np.sin(geocentric_colat)**2) / (1 - f_earth))**(l + 1)

    # mCU, meters elastic crustal deformation (uplift)
    # mCH, meters elastic crustal deformation (horizontal)
    # Gt, Gigatonnes

    else:
        raise ValueError("Invalid 'unit' parameter value in l_factor_gravi function, valid values are: "
                         "(norm, mewh, geoid, microGal, bar, mvcu)")

    return l_factor
