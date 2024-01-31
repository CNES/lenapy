import numpy as np
import xarray as xr
import pathlib
import inspect
from .constants import *


# TODO see if output in xr.Dataset or in array
def compute_plm(lmax, z, mmax=None, normalization='4pi'):
    """
    Compute all the associated Legendre functions up to a maximum degree and
    order using the recursion relation from [Holmes2002]
    (Adapted from SHTOOLS/pyshtools tools [Wieczorek2018])

    Parameters
    ----------
    lmax : int
        maximum degree of legrendre functions
    z : np.ndarray
        argument of the associated Legendre functions
    mmax : int or NoneType, optional
        maximum order of associated legrendre functions
    normalization : str, optional, optional
        '4pi', 'ortho', or 'schmidt' for use with geodesy 4pi normalized, orthonormalized, or Schmidt semi-normalized
        spherical harmonic functions, respectively. Default is '4pi'.

    Returns
    -------
    plm : np.ndarray
        fully-normalized legendre functions

    References
    ----------
    .. [Holmes2002] S. A. Holmes and W. E. Featherstone,
        "A unified approach to the Clenshaw summation and the
        recursive computation of very high degree and order
        normalised associated Legendre functions",
        *Journal of Geodesy*, 76, 279--299, (2002).
        `doi: 10.1007/s00190-002-0216-2 <https://doi.org/10.1007/s00190-002-0216-2>`_
    .. [Wieczorek2018]  M. A. Wieczorek and M. Meschede,
        "SHTools: Tools for working with spherical harmonics",
        *Geochemistry, Geophysics, Geosystems*, 19, 2574–2592, (2018).
        `doi: 10.1029/2018GC007529 <https://doi.org/10.1029/2018GC007529>`_
    """
    # removing singleton dimensions of x
    z = np.atleast_1d(z).flatten()

    # if default mmax, set mmax to be maximal degree
    if mmax is None:
        mmax = lmax

    # scale factor based on Holmes2002
    scalef = 1e-280

    # create multiplicative factors and p
    f1 = np.zeros(((lmax + 1) * (lmax + 2) // 2))
    f2 = np.zeros(((lmax + 1) * (lmax + 2) // 2))
    p = np.zeros(((lmax + 1) * (lmax + 2) // 2, len(z)))

    k = 2
    if normalization in ('4pi', 'ortho'):
        norm_p10 = np.sqrt(3)

        for l in range(2, lmax + 1):
            k += 1
            f1[k] = np.sqrt(2 * l - 1) * np.sqrt(2 * l + 1) / l
            f2[k] = (l - 1) * np.sqrt(2 * l + 1) / (np.sqrt(2 * l - 3) * l)
            for m in range(1, l - 1):
                k += 1
                f1[k] = np.sqrt(2 * l + 1) * np.sqrt(2 * l - 1) / (np.sqrt(l + m) * np.sqrt(l - m))
                f2[k] = (np.sqrt(2 * l + 1) * np.sqrt(l - m - 1) * np.sqrt(l + m - 1) /
                         (np.sqrt(2 * l - 3) * np.sqrt(l + m) * np.sqrt(l - m)))
            k += 2

        if normalization == '4pi':
            norm_4pi = 1
        else:
            norm_4pi = 4 * np.pi

    elif normalization == 'schmidt':
        norm_p10 = 1
        norm_4pi = 1

        for l in range(2, lmax + 1):
            k += 1
            f1[k] = (2 * l - 1) / l
            f2[k] = (l - 1) / l
            for m in range(1, l - 1):
                k += 1
                f1[k] = (2 * l - 1) / (np.sqrt(l + m) * np.sqrt(l - m))
                f2[k] = np.sqrt(l - m - 1) * np.sqrt(l + m - 1) / (np.sqrt(l + m) * np.sqrt(l - m))
            k += 2

    else:
        raise AssertionError("Unknown normalization given: ", normalization, ". It should be either "
                                                                             "'4pi', 'ortho' or 'schmidt'")

    # u is sine of colatitude (cosine of latitude), for z=cos(th): u=sin(th)
    u = np.sqrt(1 - z ** 2)
    # update where u==0 to minimal numerical precision different from 0 to prevent invalid divisions
    u[u == 0] = np.finfo(np.float64).eps

    # Calculate P(l,0) (not scaled)
    p[0, :] = 1 / np.sqrt(norm_4pi)
    p[1, :] = norm_p10 * z / np.sqrt(norm_4pi)
    k = 1
    for l in range(2, lmax + 1):
        k += l
        p[k, :] = f1[k] * z * p[k - l, :] - f2[k] * p[k - 2 * l + 1, :]

    # Calculate P(m,m), P(m+1,m), and P(l,m)
    pmm = np.sqrt(2) * scalef / np.sqrt(norm_4pi)
    rescalem = 1 / scalef
    kstart = 0

    # case lmax == 1, does not go into the 'for' and need m a value for after
    m = 1
    # elif lmax != 1
    for m in range(1, lmax + 1):
        rescalem = rescalem * u
        # Calculate P(m,m)
        kstart += m + 1
        pmm = pmm * np.sqrt(2 * m + 1) / np.sqrt(2 * m)
        if normalization in ('4pi', 'ortho'):
            p[kstart, :] = pmm
        elif normalization == 'schmidt':
            p[kstart, :] = pmm / np.sqrt(2 * m + 1)

        if m != lmax:  # test if P(m+1,m) exist
            # Calculate P(m+1,m)
            k = kstart + m + 1
            if normalization in ('4pi', 'ortho'):
                p[k, :] = z * np.sqrt(2 * m + 3) * pmm
            elif normalization == 'schmidt':
                p[k, :] = z * pmm
        else:
            # set up k for rescale P(lmax,lmax)
            k = kstart

        # Calculate P(l,m)
        for l in range(m + 2, lmax + 1):
            k += l
            p[k, :] = z * f1[k] * p[k - l, :] - f2[k] * p[k - 2 * l + 1, :]
            p[k - 2 * l + 1, :] = p[k - 2 * l + 1, :] * rescalem

        # rescale
        p[k, :] = p[k, :] * rescalem
        p[k - lmax, :] = p[k - lmax, :] * rescalem

    # reshape Legendre polynomials to output dimensions (lower triangle array)
    plm = np.zeros((lmax + 1, lmax + 1, len(z)))
    ind = np.tril_indices(lmax + 1)
    plm[ind] = p

    # return the legendre polynomials and truncating orders to mmax
    return plm[:, :mmax + 1, :]


def sh_to_grid(data, unit='cmwe', love_file=None, **kwargs):
    """
    Transform Spherical Harmonics (SH) dataset into spatial DataArray.
    With choice for unit, love_numbers, degree/order, spatial grid latitude and longitude, Earth hypothesis.

    Parameters
    ----------
    data : xr.Dataset
        xr.Dataset that corresponds to SH data to convert into spatial representation
    unit : str, optional
        'mewh', 'geoid', 'microGal', 'bar', 'mvcu', or 'norm'
        Unit of the spatial data used in the transformation. Default is 'mewh' for meters of Equivalent Water Height
        See constants.l_factor_gravi() doc for details on the units
    love_file : str / path, optional
        File with Love numbers that can be read by read_love_numbers() function.
        Default Love numbers used are from Gegout97.
    **kwargs : dict
        Extra keyword arguments, for degree/order choice, spatial grid choice, Earth choice and plm pre-computation.
        See below

    Keywords Arguments
    ------------------
    lmax : int, optional
        maximal degree of the SH coefficients to be used.
    mmax : int, optional
        maximal order of the SH coefficients to be used.
    lmin : int, optional
        minimal degree of the SH coefficients to be used.
    mmin : int, optional
        minimal order of the SH coefficients to be used.
    used_l : np.ndarray, optional
        list of degree to use for the grid computation (if given, lmax and lmin are not considered).
    used_m : np.ndarray, optional
        list of order to use for the grid computation (if given, mmax and mmin are not considered).

    lonmin : float, optional
        minimal longitude of the future grid.
    lonmax : float, optional
        maximal longitude of the future grid.
    latmin : float, optional
        minimal latitude of the future grid.
    latmax : float, optional
        maximal latitude of the future grid.
    bounds : list, optional
        list of 4 elements with [lonmin, lonmax, latmin, latmax] (if given min/max information are not considered).
    radians : bool, optional
        True if the unit of the given latitude and longitude information is radians. Default is False for degree unit.
    dlon : float, optional
        spacing of the longitude values.
    dlat : float, optional
        spacing of the latitude values.
    longitude : np.ndarray, optional
        list of longitude to use for the grid computation (if given, others longitude information are not considered).
    latitude : np.ndarray, optional
        list of latitude to use for the grid computation (if given, others latitude information are not considered).

    ellispoidal_earth : bool, optional
        If True, consider the Earth as an ellispoid following [Ditmar2018] and if False as a sphere. Default is False
    include_elastic : bool, optional
        If True, the Earth behavior is elastic. Default is True

    plm : xr.DataArray, optional
        Precomputed plm values as a xr.DataArray variable. For example with the code :
        plm = xr.DataArray(compute_plm(lmax, sin_latitude), dims=['l', 'm', 'latitude'],
                           coords={'l': data.l, 'm': data.m, 'latitude': latitude})
    normalization_plm : str, optional
        If plm need to be computed, choice of the norm corresponding to the SH dataset.
        '4pi', 'ortho', or 'schmidt' for use with geodesy
        4pi normalized, orthonormalized, or Schmidt semi-normalized SH functions, respectively.

    Returns
    -------
    xgrid : xr.DataArray
        Spatial representation of the SH Dataset in the chosen unit.
    """
    # addition error propagation, add mask in output variable

    # -- set degree and order default parameters
    # it prioritizes used_l and used_m if given over lmin, lmax, mmin and mmax
    kwargs.setdefault('lmax', int(data.l.max()))
    kwargs.setdefault('mmax', int(data.m.max()))
    kwargs.setdefault('lmin', 0)
    kwargs.setdefault('mmin', 0)
    kwargs.setdefault('used_l', np.arange(kwargs['lmin'], kwargs['lmax'] + 1))
    kwargs.setdefault('used_m', np.arange(kwargs['mmin'], kwargs['mmax'] + 1))

    # -- set grid output latitude and longitude
    # Verify input variable to create bounds of grid information
    kwargs.setdefault('lonmin', -180)
    kwargs.setdefault('lonmax', 180)
    kwargs.setdefault('latmin', -90)
    kwargs.setdefault('latmax', 90)
    kwargs.setdefault('bounds', [kwargs['lonmin'], kwargs['lonmax'], kwargs['latmin'], kwargs['latmax']])

    # verify integrity of the argument "bounds" if given
    try:
        test_iter = iter(kwargs['bounds'])
        if len(kwargs['bounds']) != 4:
            raise TypeError
    except TypeError:
        raise TypeError('Given argument "bounds" has to be a list with 4 elements [lonmin, lonmax, latmin, latmax]')

    # Convert bounds in radians if necessary
    # work only if bounds or all lonmin, lonmax, latmin, latmax are given in the announced unit
    kwargs.setdefault('radians', False)
    if kwargs['radians']:
        kwargs['bounds'] = [np.rad2deg(i) for i in kwargs['bounds']]

    # convert dlon, dlat from radians to degree if necessary
    if 'dlon' in kwargs and kwargs['radians']:
        kwargs['dlon'] = np.rad2deg(kwargs['dlon'])
    if 'dlat' in kwargs and kwargs['radians']:
        kwargs['dlat'] = np.rad2deg(kwargs['dlat'])
    kwargs.setdefault('dlon', 1)
    kwargs.setdefault('dlat', 1)

    # load longitude and latitude if given
    # if not : compute longitude and latitude in degrees between given or defaults bounds
    if 'longitude' in kwargs:
        if kwargs['radians']:
            longitude = np.rad2deg(kwargs['longitude'])
        else:
            longitude = kwargs['longitude']
    else:
        longitude = np.arange(kwargs['bounds'][0] + kwargs['dlon'] / 2.0,
                              kwargs['bounds'][1] + kwargs['dlon'] / 2.0, kwargs['dlon'])
    if 'latitude' in kwargs:
        if kwargs['radians']:
            latitude = np.rad2deg(kwargs['latitude'])
        else:
            latitude = kwargs['latitude']
    else:
        latitude = np.arange(kwargs['bounds'][3] - kwargs['dlat'] / 2.0,
                             kwargs['bounds'][2] - kwargs['dlat'] / 2.0, -kwargs['dlat'])

    cos_latitude = np.cos(np.deg2rad(latitude))
    sin_latitude = np.sin(np.deg2rad(latitude))
    geocentric_colat = np.arctan2(cos_latitude, (1 - f_earth) ** 2 * sin_latitude)

    # -- load Earth representation parameters
    # default Earth is spherical
    kwargs.setdefault('ellispoidal_earth', False)
    # default Earth is elastic (only serve for particular unit)
    kwargs.setdefault('include_elastic', True)

    # -- beginning of computation
    # Computing plm for converting to spatial domain
    if 'plm' in kwargs:
        plm = kwargs['plm']
        if (isinstance(plm, xr.DataArray) or
                'l' not in plm.coords or 'm' not in plm.coords or 'latitude' not in plm.coords):
            raise TypeError('Given argument "plm" has to be a DataArray with 3 coordinates [l, m, latitude]')

        if plm.l.max < data.l.max:
            raise AssertionError('Given argument "plm" maximal degree is too small ', plm.l.max, '<', data.l.max)
    else:
        kwargs.setdefault('normalization_plm', '4pi')
        if kwargs['ellispoidal_earth']:
            plm = compute_plm(kwargs['lmax'], np.cos(geocentric_colat), normalization=kwargs['normalization_plm'])
        else:
            plm = compute_plm(kwargs['lmax'], sin_latitude, normalization=kwargs['normalization_plm'])
        plm = xr.DataArray(plm, dims=['l', 'm', 'latitude'],
                           coords={'l': np.arange(kwargs['lmax'] + 1), 'm': np.arange(kwargs['lmax'] + 1),
                                   'latitude': latitude})

    # scale factor for each degree
    lfactor = l_factor_gravi(kwargs['used_l'], unit, kwargs['include_elastic'],
                             kwargs['ellispoidal_earth'], geocentric_colat, love_file)

    # convolve unit over degree
    plm_lfactor = plm.sel(l=kwargs['used_l'], m=kwargs['used_m']) * lfactor[:, np.newaxis, np.newaxis]

    # summation over all spherical harmonic degrees
    d_cslm = (plm_lfactor * data.sel(l=kwargs['used_l'], m=kwargs['used_m'])).sum(dim='l')

    # Calculating cos(m*phi) and sin(m*phi)
    c_cos = xr.DataArray(np.cos(kwargs['used_m'][:, np.newaxis] @ np.deg2rad(longitude)[np.newaxis, :]),
                         dims=['m', 'longitude'], coords={'m': data.m, 'longitude': longitude})
    s_sin = xr.DataArray(np.sin(kwargs['used_m'][:, np.newaxis] @ np.deg2rad(longitude)[np.newaxis, :]),
                         dims=['m', 'longitude'], coords={'m': data.m, 'longitude': longitude})

    # Final calcul on the grid
    xgrid = c_cos.dot(d_cslm.clm) + s_sin.dot(d_cslm.slm)
    xgrid = xgrid.transpose("latitude", "longitude", ...)

    return xgrid


def grid_to_sh(grid, lmax, unit='mewh', love_file=None, **kwargs):
    """
    Transform gravity field spatial representation DataArray into Spherical Harmonics (SH) dataset.
    With choice for unit of the spatial DataArray, love_numbers, degree/order, Earth hypothesis.

    Parameters
    ----------
    grid : xr.DataArray
        xr.Dataset that corresponds a gravity field spatial representation in a unit to convert into SH
    lmax : int
        maximal degree of the SH coefficients to be computed.
    unit : str, optional
        'mewh', 'geoid', 'microGal', 'bar', 'mvcu', or 'norm'
        Unit of the spatial data used in the transformation. Default is 'mewh' for meters of Equivalent Water Height
        See constants.l_factor_gravi() doc for details on the units
    love_file : str / path, optional
        File with Love numbers that can be read by read_love_numbers() function.
        Default Love numbers used are from Gegout97.
    **kwargs : dict
        Extra keyword arguments, for degree/order choice, Earth choice and plm pre-computation.
        See below

    Keywords Arguments
    ------------------
    mmax : int, optional
        maximal order of the SH coefficients to be computed.
    lmin : int, optional
        minimal degree of the SH coefficients to be computed.
    mmin : int, optional
        minimal order of the SH coefficients to be computed.
    used_l : np.ndarray, optional
        list of degree to compute for the SH Dataset (if given, lmax and lmin are not considered).
    used_m : np.ndarray, optional
        list of order to compute for the SH Dataset (if given, mmax and mmin are not considered).

    ellispoidal_earth : bool, optional
        If True, consider the Earth as an ellispoid following [Ditmar2018] and if False as a sphere. Default is False
    include_elastic : bool, optional
        If True, the Earth behavior is elastic. Default is True

    plm : xr.DataArray, optional
        Precomputed plm values as a xr.DataArray variable. For example with the code :
        plm = xr.DataArray(compute_plm(lmax, sin_latitude), dims=['l', 'm', 'latitude'],
                           coords={'l': data.l, 'm': data.m, 'latitude': latitude})
    normalization_plm : str, optional
        If plm need to be computed, choice of the norm. '4pi', 'ortho', or 'schmidt' for use with geodesy
        4pi normalized, orthonormalized, or Schmidt semi-normalized SH functions, respectively.
        Output SH coefficient will be normalized according to this parameter.

    Returns
    -------
    xharmo : xr.DataArray
        SH Dataset computed from the grid with the chosen unit.
    """
    # -- set degree and order default parameters
    # it prioritizes used_l and used_m if given over lmin, lmax, mmin and mmax
    kwargs.setdefault('mmax', lmax)
    kwargs.setdefault('lmin', 0)
    kwargs.setdefault('mmin', 0)
    kwargs.setdefault('used_l', np.arange(kwargs['lmin'], lmax + 1))
    kwargs.setdefault('used_m', np.arange(kwargs['mmin'], kwargs['mmax'] + 1))

    # -- create integration factor over the grid
    # longitude degree spacing of each cell in radians
    diffphi = np.abs(grid.longitude.diff(dim='longitude').values)
    # deal with case where longitude goes from 180 to -180 in the array
    diffphi[diffphi > np.pi] = np.abs(2 * np.pi - diffphi[diffphi > np.pi])
    # size of the cell is half of the diff with both adjacent grid cell + convert to radians
    dphi = np.deg2rad(np.concatenate((diffphi[[0]], (diffphi[1:] + diffphi[:-1]) / 2, diffphi[[-1]])))

    # latitude degree spacing in radians
    diffth = np.abs(grid.latitude.diff(dim='latitude').values)
    # size of the cell is half of the diff with both adjacent grid cell + convert to radians
    dth = np.deg2rad(np.concatenate((diffth[[0]], (diffth[1:] + diffth[:-1]) / 2, diffth[[-1]])))

    cos_latitude = np.cos(np.deg2rad(grid.latitude.values))
    sin_latitude = np.sin(np.deg2rad(grid.latitude.values))
    geocentric_colat = np.arctan2(cos_latitude, (1 - f_earth) ** 2 * sin_latitude)

    # create DataArray corresponding to the integration factor [sin(theta) * dtheta * dphi] for each cell
    # Possible test if Gt
    int_fact = xr.DataArray(cos_latitude[np.newaxis, :] * dphi[:, np.newaxis] * dth / (4 * np.pi),
                            dims=['longitude', 'latitude'],
                            coords={'longitude': grid.longitude, 'latitude': grid.latitude})

    # -- load Earth representation parameters
    # default Earth is spherical
    kwargs.setdefault('ellispoidal_earth', False)
    # default Earth is elastic (only serve for particular unit)
    kwargs.setdefault('include_elastic', True)

    # scale factor for each degree
    lfactor = l_factor_gravi(kwargs['used_l'], unit, kwargs['include_elastic'],
                             kwargs['ellispoidal_earth'], geocentric_colat, love_file)

    # -- prepare variables for the computation of SH
    # Computing plm for converting to spatial domain
    if 'plm' in kwargs:
        plm = kwargs['plm']
        if (isinstance(plm, xr.DataArray) or
                'l' not in plm.coords or 'm' not in plm.coords or 'latitude' not in plm.coords):
            raise TypeError('Given argument "plm" has to be a DataArray with 3 coordinates [l, m, latitude]')

        if plm.l.max < lmax:
            raise AssertionError('Given argument "plm" maximal degree is too small ', plm.l.max, '<', lmax)
    else:
        kwargs.setdefault('normalization_plm', '4pi')
        if kwargs['ellispoidal_earth']:
            plm = compute_plm(lmax, np.cos(geocentric_colat), normalization=kwargs['normalization_plm'])
        else:
            plm = compute_plm(lmax, sin_latitude, normalization=kwargs['normalization_plm'])
        plm = xr.DataArray(plm, dims=['l', 'm', 'latitude'],
                           coords={'l': np.arange(lmax + 1), 'm': np.arange(lmax + 1), 'latitude': grid.latitude})

    # convolve unit over degree
    plm_lfactor = plm.sel(l=kwargs['used_l'], m=kwargs['used_m']) / lfactor[:, np.newaxis, np.newaxis]

    # Calculating cos/sin of phi arrays, [m,phi]
    c_cos = xr.DataArray(np.cos(kwargs['used_m'][:, np.newaxis] @ np.deg2rad(grid.longitude.values)[np.newaxis, :]),
                         dims=['m', 'longitude'], coords={'m': kwargs['used_m'], 'longitude': grid.longitude})
    s_sin = xr.DataArray(np.sin(kwargs['used_m'][:, np.newaxis] @ np.deg2rad(grid.longitude.values)[np.newaxis, :]),
                         dims=['m', 'longitude'], coords={'m': kwargs['used_m'], 'longitude': grid.longitude})

    # -- Computation of SH
    # Multiplying data and integral factor with sin/cos of m*longitude. This will sum over longitude, [m,theta]
    dcos = c_cos.dot(grid * int_fact)
    dsin = s_sin.dot(grid * int_fact)

    # WARNING, will have to change dims to dim in next xarray version
    # Multiplying plm and degree scale factors with last variable to sum over latitude, [l, m, ...]
    clm = plm_lfactor.dot(dcos, dims=['latitude'])
    slm = plm_lfactor.dot(dsin, dims=['latitude'])

    # add name for the merge into xr.Dataset
    clm.name = 'clm'
    slm.name = 'slm'

    return xr.merge([clm, slm], join='exact')
