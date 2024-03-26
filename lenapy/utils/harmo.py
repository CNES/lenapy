import datetime
import pathlib
import inspect
import numpy as np
import xarray as xr
import cf_xarray as cfxr
from lenapy.constants import *


def sh_to_grid(data, unit='mewh', love_file=None,
               a_earth=None, gm_earth=None, f_earth=F_EARTH_GRS80, rho_earth=LNPY_RHO_EARTH,
               lmax=None, mmax=None, lmin=None, mmin=None, used_l=None, used_m=None,
               lonmin=-180, lonmax=180, latmin=-90, latmax=90, bounds=None,
               dlon=1, dlat=1, longitude=None, latitude=None, radians_in=False,
               ellispoidal_earth=False, include_elastic=True, plm=None, normalization_plm='4pi'):
    """
    Transform Spherical Harmonics (SH) dataset into spatial DataArray.
    With choice for constants, unit, love_numbers, degree/order, spatial grid latitude and longitude, Earth hypothesis.

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

    a_earth : float | None, optional
        Earth semi-major axis [m]. Default is data.attrs['radius'] and if this does not exist, default is A_EARTH_GRS80
    gm_earth : float | None, optional
        Standard gravitational parameter for Earth [m³.s⁻²]. Default is data.attrs['earth_gravity_constant'] and
        if this does not exist, default is LNPY_GM_EARTH
        gm_earth is used if unit='microGal'
    f_earth : float | None, optional
        Earth flattening. Default is F_EARTH_GRS80
        f_earth is used if ellispoidal_earth=True
    rho_earth : float, optional
        Earth density [kg.m⁻³]. Default is LNPY_RHO_EARTH
        rho_earth is used if unit='mewh' or if unit='pascal'

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
    radians_in : bool, optional
        True if the unit of the given latitude and longitude information is radians. Default is False for degree unit.
        If radians_in is True and dlat or dlon are given, they are considered as radians.
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
        '4pi', 'ortho', or 'schmidt' for use with geodesy. Default is '4pi'
        4pi normalized, orthonormalized, or Schmidt semi-normalized SH functions, respectively.

    Returns
    -------
    xgrid : xr.DataArray
        Spatial representation of the SH Dataset in the chosen unit.
    """
    # addition error propagation, add mask in output variable

    # -- define constants
    if a_earth is None:
        if 'radius' in data.attrs:
            a_earth = float(data.attrs['radius'])
        else:
            a_earth = A_EARTH_GRS80
    if gm_earth is None:
        if 'earth_gravity_constant' in data.attrs:
            gm_earth = float(data.attrs['earth_gravity_constant'])
        else:
            gm_earth = LNPY_GM_EARTH

    # -- set degree and order default parameters
    # it prioritizes used_l and used_m if given over lmin, lmax, mmin and mmax
    if lmax is None:
        lmax = int(data.l.max())
    if mmax is None:
        mmax = int(min(lmax, data.m.max()))
    if lmin is None:
        lmin = int(data.l.min())
    if mmin is None:
        mmin = int(data.m.min())
    if used_l is None:
        used_l = np.arange(lmin, lmax + 1)
    if used_m is None:
        used_m = np.arange(mmin, mmax + 1)

    # -- set grid output latitude and longitude
    # Verify input variable to create bounds of grid information
    if bounds is None:
        bounds = [lonmin, lonmax, latmin, latmax]

    # verify integrity of the argument "bounds" if given
    try:
        test_iter = iter(bounds)
        if len(bounds) != 4:
            raise TypeError
    except TypeError:
        raise TypeError('Given argument "bounds" has to be a list with 4 elements [lonmin, lonmax, latmin, latmax]')

    # Convert bounds in radians if necessary
    # work only if bounds or all lonmin, lonmax, latmin, latmax are given in the announced unit
    if radians_in:
        bounds = [np.rad2deg(i) for i in bounds]

    # convert dlon, dlat from radians to degree if necessary
    if radians_in and dlon != 1:
        dlon = np.rad2deg(dlon)
    if radians_in and dlat != 1:
        dlat = np.rad2deg(dlat)

    # load longitude and latitude if given
    # if not : compute longitude and latitude in degrees between given or defaults bounds
    if longitude is None:
        longitude = np.arange(bounds[0] + dlon / 2.0,
                              bounds[1] + dlon / 2.0, dlon)
    else:
        if radians_in:
            longitude = np.rad2deg(longitude)

    if latitude is None:
        latitude = np.arange(bounds[3] - dlat / 2.0,
                             bounds[2] - dlat / 2.0, - dlat)
    else:
        if radians_in:
            latitude = np.rad2deg(latitude)

    cos_latitude = np.cos(np.deg2rad(latitude))
    sin_latitude = np.sin(np.deg2rad(latitude))
    geocentric_colat = np.arctan2(cos_latitude, (1 - f_earth) ** 2 * sin_latitude)

    # -- beginning of computation
    # Computing plm for converting to spatial domain
    if plm is None:
        if ellispoidal_earth:
            plm = compute_plm(lmax, np.cos(geocentric_colat),
                              mmax=mmax, normalization=normalization_plm)
        else:
            plm = compute_plm(lmax, sin_latitude,
                              mmax=mmax, normalization=normalization_plm)
        plm = xr.DataArray(plm, dims=['l', 'm', 'latitude'],
                           coords={'l': np.arange(lmax + 1), 'm': np.arange(mmax + 1), 'latitude': latitude})

    else:
        if (not isinstance(plm, xr.DataArray) or
                'l' not in plm.coords or 'm' not in plm.coords or 'latitude' not in plm.coords):
            raise TypeError('Given argument "plm" has to be a DataArray with 3 coordinates [l, m, latitude]')
        else:
            plm = plm.transpose("l", "m", "latitude")

        if plm.l.max() < data.l.max():
            raise AssertionError('Given argument "plm" maximal degree is too small ', plm.l.max(), '<', data.l.max())
        if (plm.latitude.values != latitude).all():
            raise AssertionError('Given argument "plm" latitude does not correspond to the wanted latitude ', latitude)

    # scale factor for each degree
    lfactor = l_factor_gravi(used_l, unit=unit, include_elastic=include_elastic, ellispoidal_earth=ellispoidal_earth,
                             geocentric_colat=geocentric_colat, love_file=love_file,
                             a_earth=a_earth, f_earth=f_earth, gm_earth=gm_earth, rho_earth=rho_earth)

    # convolve unit over degree
    plm_lfactor = plm.sel(l=used_l, m=used_m) * lfactor[:, np.newaxis, np.newaxis]

    # summation over all spherical harmonic degrees
    d_clm = (plm_lfactor * data.sel(l=used_l, m=used_m).clm).sum(dim='l')
    d_slm = (plm_lfactor * data.sel(l=used_l, m=used_m).slm).sum(dim='l')

    # Calculating cos(m*phi) and sin(m*phi)
    c_cos = xr.DataArray(np.cos(used_m[:, np.newaxis] @ np.deg2rad(longitude)[np.newaxis, :]),
                         dims=['m', 'longitude'], coords={'m': np.arange(mmax + 1), 'longitude': longitude})
    s_sin = xr.DataArray(np.sin(used_m[:, np.newaxis] @ np.deg2rad(longitude)[np.newaxis, :]),
                         dims=['m', 'longitude'], coords={'m': np.arange(mmax + 1), 'longitude': longitude})

    # Final calcul on the grid
    xgrid = c_cos.dot(d_clm) + s_sin.dot(d_slm)
    xgrid = xgrid.transpose("latitude", "longitude", ...)

    xgrid.attrs = {'units': unit, 'max_degree': int(data.l.max())}
    if 'radius' in data.attrs:
        xgrid.attrs['radius'] = data.attrs['radius']
    if 'earth_gravity_constant' in data.attrs:
        xgrid.attrs['earth_gravity_constant'] = data.attrs['earth_gravity_constant']

    return xgrid


def grid_to_sh(grid, lmax, unit='mewh', love_file=None,
               a_earth=None, gm_earth=None, f_earth=F_EARTH_GRS80, rho_earth=LNPY_RHO_EARTH,
               mmax=None, lmin=0, mmin=0, used_l=None, used_m=None,
               ellispoidal_earth=False, include_elastic=True, plm=None, normalization_plm='4pi'):
    """
    Transform gravity field spatial representation DataArray into Spherical Harmonics (SH) dataset.
    With choice for constants, unit of the spatial DataArray, love_numbers, degree/order, Earth hypothesis.

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

    a_earth : float | None, optional
        Earth semi-major axis [m]. Default is grid.attrs['radius'] and if this does not exist, default is A_EARTH_GRS80
    gm_earth : float | None, optional
        Standard gravitational parameter for Earth [m³.s⁻²]. Default is grid.attrs['earth_gravity_constant'] and
        if this does not exist, default is LNPY_GM_EARTH
        gm_earth is used if unit='microGal'
    f_earth : float | None, optional
        Earth flattening. Default is F_EARTH_GRS80
        f_earth is used if ellispoidal_earth=True
    rho_earth : float, optional
        Earth density [kg.m⁻³]. Default is LNPY_RHO_EARTH
        rho_earth is used if unit='mewh' or if unit='pascal'

    mmax : int, optional
        maximal order of the SH coefficients to be computed.
    lmin : int, optional
        minimal degree of the SH coefficients to be computed. Default is 0
    mmin : int, optional
        minimal order of the SH coefficients to be computed. Default is 0
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
        plm = xr.DataArray(compute_plm(lmax, np.sin(np.deg2rad(latitude))), dims=['l', 'm', 'latitude'],
        coords={'l': np.arange(lmax+1), 'm': np.arange(lmax+1), 'latitude': latitude})
    normalization_plm : str, optional
        If plm need to be computed, choice of the norm. '4pi', 'ortho', or 'schmidt' for use with geodesy
        4pi normalized, orthonormalized, or Schmidt semi-normalized SH functions, respectively.
        Output SH coefficient will be normalized according to this parameter.

    Returns
    -------
    ds_out : xr.DataArray
        SH Dataset computed from the grid with the chosen unit.
    """
    # -- define constants
    if a_earth is None:
        if 'radius' in grid.attrs:
            a_earth = float(grid.attrs['radius'])
        else:
            a_earth = A_EARTH_GRS80
    if gm_earth is None:
        if 'earth_gravity_constant' in grid.attrs:
            gm_earth = float(grid.attrs['earth_gravity_constant'])
        else:
            gm_earth = LNPY_GM_EARTH

    # -- set degree and order default parameters
    # it prioritizes used_l and used_m if given over lmin, lmax, mmin and mmax
    if mmax is None:
        mmax = lmax
    if used_l is None:
        used_l = np.arange(lmin, lmax + 1)
    if used_m is None:
        used_m = np.arange(mmin, mmax + 1)

    # -- create integration factor over the grid
    # longitude degree spacing of each cell in radians
    diff_phi = np.abs(grid.cf["longitude"].diff(dim='longitude').values)
    # deal with case where longitude goes from 180 to -180 in the array
    diff_phi[diff_phi > np.pi] = np.abs(2 * np.pi - diff_phi[diff_phi > np.pi])
    # size of the cell is half of the diff with both adjacent grid cell + convert to radians
    dphi = np.deg2rad(np.concatenate((diff_phi[[0]], (diff_phi[1:] + diff_phi[:-1]) / 2, diff_phi[[-1]])))

    # latitude degree spacing in radians
    diff_th = np.abs(grid.cf["latitude"].diff(dim='latitude').values)
    # size of the cell is half of the diff with both adjacent grid cell + convert to radians
    dth = np.deg2rad(np.concatenate((diff_th[[0]], (diff_th[1:] + diff_th[:-1]) / 2, diff_th[[-1]])))

    cos_latitude = np.cos(np.deg2rad(grid.cf["latitude"].values))
    sin_latitude = np.sin(np.deg2rad(grid.cf["latitude"].values))
    geocentric_colat = np.arctan2(cos_latitude, (1 - f_earth) ** 2 * sin_latitude)

    # create DataArray corresponding to the integration factor [sin(theta) * dtheta * dphi] for each cell
    # Possible test if Gt
    int_fact = xr.DataArray(cos_latitude[np.newaxis, :] * dphi[:, np.newaxis] * dth / (4 * np.pi),
                            dims=['longitude', 'latitude'],
                            coords={'longitude': grid.cf["longitude"], 'latitude': grid.cf["latitude"]})
    # Create a readable cf_xarray DataArray
    int_fact['latitude'].attrs = dict(standard_name='latitude')
    int_fact['longitude'].attrs = dict(standard_name='longitude')

    # scale factor for each degree
    lfactor = l_factor_gravi(used_l, unit=unit, include_elastic=include_elastic, ellispoidal_earth=ellispoidal_earth,
                             geocentric_colat=geocentric_colat, love_file=love_file,
                             a_earth=a_earth, f_earth=f_earth, gm_earth=gm_earth, rho_earth=rho_earth)

    # -- prepare variables for the computation of SH
    # Computing plm for converting to spatial domain
    if plm is None:
        if ellispoidal_earth:
            plm = compute_plm(lmax, np.cos(geocentric_colat),
                              mmax=mmax, normalization=normalization_plm)
        else:
            plm = compute_plm(lmax, sin_latitude,
                              mmax=mmax, normalization=normalization_plm)
        plm = xr.DataArray(plm, dims=['l', 'm', 'latitude'],
                           coords={'l': np.arange(lmax + 1), 'm': np.arange(lmax + 1), 'latitude': grid.cf["latitude"]})

    else:
        if (not isinstance(plm, xr.DataArray) or
                'l' not in plm.coords or 'm' not in plm.coords or 'latitude' not in plm.coords):
            raise TypeError('Given argument "plm" has to be a DataArray with 3 coordinates [l, m, latitude]')
        else:
            plm = plm.transpose("l", "m", "latitude")

        if plm.l.max() < lmax:
            raise AssertionError('Given argument "plm" maximal degree is too small ', plm.l.max(), '<', lmax)
        align = xr.align(plm.latitude, grid.cf["latitude"])  # return the intersection of latitudes for each input
        if align[0] != plm.latitude.size or align[1].size != grid.cf["latitude"].size:
            raise AssertionError('Given argument "plm" latitude does not correspond to the grid latitude')

    # convolve unit over degree
    plm_lfactor = plm.sel(l=used_l, m=used_m) / lfactor[:, np.newaxis, np.newaxis]

    # Calculating cos/sin of phi arrays, [m,phi]
    c_cos = xr.DataArray(np.cos(used_m[:, np.newaxis] @ np.deg2rad(grid.cf["longitude"].values)[np.newaxis, :]),
                         dims=['m', 'longitude'], coords={'m': used_m, 'longitude': grid.cf["longitude"]})
    s_sin = xr.DataArray(np.sin(used_m[:, np.newaxis] @ np.deg2rad(grid.longitude.values)[np.newaxis, :]),
                         dims=['m', 'longitude'], coords={'m': used_m, 'longitude': grid.cf["longitude"]})

    # -- Computation of SH
    # WARNING, will have to change dims to dim in next xarray version
    # Multiplying data and integral factor with sin/cos of m*longitude. This will sum over longitude, [m,theta]
    dcos = c_cos.dot(grid.cf.rename_like(int_fact) * int_fact, dims=['longitude'])
    dsin = s_sin.dot(grid.cf.rename_like(int_fact) * int_fact, dims=['longitude'])

    # WARNING, will have to change dims to dim in next xarray version
    # Multiplying plm and degree scale factors with last variable to sum over latitude, [l, m, ...]
    clm = plm_lfactor.dot(dcos, dims=['latitude'])
    slm = plm_lfactor.dot(dsin, dims=['latitude'])

    # add name for the merge into xr.Dataset
    clm.name = 'clm'
    slm.name = 'slm'

    ds_out = xr.merge([clm, slm], join='exact')
    ds_out.attrs = {'earth_gravity_constant': gm_earth, 'radius': a_earth, 'max_degree': lmax,
                    'norm': normalization_plm}
    return ds_out


def compute_plm(lmax, z, mmax=None, normalization='4pi'):
    """
    Compute all the associated Legendre functions up to a maximum degree and
    order using the recursion relation from [Holmes2002]_
    (Adapted from SHTOOLS/pyshtools tools [Wieczorek2018]_)

    Parameters
    ----------
    lmax : int
        maximum degree of legrendre functions
    z : np.ndarray
        argument of the associated Legendre functions
    mmax : int or NoneType, optional
        maximum order of associated legrendre functions
    normalization : str, optional
        '4pi', 'ortho', or 'schmidt' for use with geodesy 4pi normalized, orthonormalized, or Schmidt semi-normalized
        spherical harmonic functions, respectively. Default is '4pi'.

    Returns
    -------
    plm : np.ndarray
        fully-normalized legendre functions as a 3D array with "l", "m" and z dimensions

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
    # update type to provide more memory for computation (np.float32 create some problems)
    z = z.astype(np.float128)

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
        if m != lmax:
            p[k - lmax, :] = p[k - lmax, :] * rescalem

    # reshape Legendre polynomials to output dimensions (lower triangle array)
    plm = np.zeros((lmax + 1, lmax + 1, len(z)))
    ind = np.tril_indices(lmax + 1)
    plm[ind] = p

    # return the legendre polynomials and truncating orders to mmax
    return plm[:, :mmax + 1, :]


def mid_month_grace_estimate(begin_time, end_time):
    """
    Calculate middle of the month date based on begin_time and end_time for GRACE products
    begin_time is round to equal to the first day of the month and
    end_time is round to equal to the first day of the month after.

    Parameters
    ----------
    begin_time : datetime.datetime
        Date of the beginning of the month
    end_time : datetime.datetime
        Date of the end of the month + 1 day

    Returns
    -------
    mid_month : datetime.datetime
        Date of the middle of the month
    """
    # to compute mid_month, need to round begin_time to the 1st of the month
    # deal with GRACE month when the begin date is not between 16 of month before and 15 of the month
    # it includes May 2015, Dec 2011 (JPL), Mar 2017 and Oct 2018 that cover second half of month
    if ((begin_time.day <= 15 and begin_time.strftime('%Y%j') != '2015102') or
            begin_time.strftime('%Y%j') == '2011351' or begin_time.strftime('%Y%j') == '2017076' or
            begin_time.strftime('%Y%j') == '2018295'):
        tmp_begin = begin_time.replace(day=1)
    else:
        tmp_begin = (begin_time.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)

    # to compute mid_month, need to round end_time to the 1st of the month after
    # deal with GRACE month when the end date is not between 16 of month and 15 of the month after
    # it includes Janv 2004, Nov 2011 (CSR, GFZ) and May 2015 and Dec 2011 for TN14
    if ((end_time.day <= 15 and end_time.strftime('%Y%j') not in ('2004014', '2011320', '2015132')) or
            end_time.strftime('%Y%j') == '2012016'):
        tmp_end = end_time.replace(day=1)
    else:
        tmp_end = (end_time.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)

    return tmp_begin + (tmp_end - tmp_begin) / 2


def l_factor_gravi(l, unit='mewh', include_elastic=True, ellispoidal_earth=False, geocentric_colat=None, love_file=None,
                   a_earth=A_EARTH_GRS80, f_earth=F_EARTH_GRS80, gm_earth=LNPY_GM_EARTH, rho_earth=LNPY_RHO_EARTH):
    """
    Compute scale factor for a transformation between spherical harmonics and grid data.
    Spatial data over the grid are associated with a specific unit.
    The scale factor is degree dependant and is computed for the given list of degree l.
    The scale factor can be estimated using elastic or none elastic Earth as well as a spherical or ellipsoidal Earth.

    Parameters
    ----------
    l : np.ndarray
        degree for which the scale factor is estimated.
    unit : str, optional
        'mewh', 'geoid', 'microGal', 'pascal', 'mvcu', or 'norm'
        Unit of the spatial data used in the transformation. Default is 'mewh' for meters of Equivalent Water Height.
        'geoid' represents millimeters geoid height, 'microGal' represents microGal gravity perturbations,
        'pascal' represents equivalent surface pressure in pascal and
        'mvcu' represents meters viscoelastic crustal uplift
    include_elastic : bool, optional
        If True, the Earth behavior is elastic.
    ellispoidal_earth : bool, optional
        If True, consider the Earth as an ellispoid following [Ditmar2018]_ and if False as a sphere.
    geocentric_colat : list, optional
        List of geocentric colatitude for ellispoidal earth radius computation.
    love_file : str / path, optional
        File with Love numbers that can be read by read_love_numbers() function.
        Default Love numbers used are from Gegout97.
    a_earth : float, optional
        Earth semi-major axis [m]. Default is A_EARTH_GRS80
    f_earth : float, optional
        Earth flattening. Default is F_EARTH_GRS80
    gm_earth : float, optional
        Standard gravitational parameter for Earth [m³.s⁻²]. Default is LNPY_GM_EARTH
    rho_earth : float, optional
        Earth density [kg.m⁻³]. Default is LNPY_RHO_EARTH

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
    if ellispoidal_earth:
        # test if geocentric_colat is set
        if geocentric_colat is None:
            raise ValueError("For ellipsoidal Earth, you need to set "
                             "the parameter 'geocentric_colat' in l_factor_gravi function")

        # compute variable for ellispoidal_earth
        # e = sqrt(2f - f**2)
        e_earth = np.sqrt(2 * f_earth - f_earth ** 2)
        # a_div_r_lat = a / r(theta)  with r(theta) = a(1-f)/sqrt(1 - e**2*sin(theta)**2)
        a_div_r_lat = np.sqrt(1 - e_earth ** 2 * np.sin(geocentric_colat) ** 2) / (1 - f_earth)
        # Earth average radius = a*(1-f)**(1/3) for which spherical Earth volume = ellipsoidal Earth volume
        raverage_radius = a_earth * (1 - f_earth) ** (1 / 3)

    # l_factor is degree dependant
    if unit == 'norm':
        # norm, fully normalized spherical harmonics
        l_factor = np.ones(l.shape)

    elif unit == 'mewh':
        # mewh, meters equivalent water height [kg.m-2]
        l_factor = rho_earth * a_earth * (2 * l + 1) / (3 * fraction * 1e3)
        if ellispoidal_earth:
            l_factor *= ((raverage_radius/a_earth)**3 * a_div_r_lat**(l + 2))

    elif unit == 'geoid':
        # geoid, millimeters geoid height
        l_factor = np.ones(l.shape) * a_earth * 1e3
        if ellispoidal_earth:
            l_factor *= a_div_r_lat**(l + 1)

    elif unit == 'microGal':
        # microGal, microGal gravity perturbations
        l_factor = gm_earth * (l + 1) / (a_earth ** 2) * 1e8
        if ellispoidal_earth:
            l_factor *= a_div_r_lat**(l + 2)

    elif unit == 'pascal':
        # pascal, equivalent surface pressure
        l_factor = LNPY_G_WMO * rho_earth * a_earth * (2*l + 1) / (3 * fraction)
        if ellispoidal_earth:
            l_factor *= a_div_r_lat**(l + 1)

    elif unit == 'mvcu':
        # mVCU, meters viscoelastic crustal uplift
        l_factor = a_earth * (2*l + 1) / 2
        if ellispoidal_earth:
            l_factor *= a_div_r_lat**(l + 1)

    # mCU, meters elastic crustal deformation (uplift)
    # mCH, meters elastic crustal deformation (horizontal)
    # Gt, Gigatonnes

    else:
        raise ValueError("Invalid 'unit' parameter value in l_factor_gravi function, valid values are: "
                         "(norm, mewh, geoid, microGal, bar, mvcu)")

    return l_factor


def read_love_numbers(love_file=None):
    # verif love file, catch error, deal with kl or hl, kl, ll
    # change reference
    # test kwargs given love or not or love_file
    # hl, ll

    if love_file is None:
        current_file = inspect.getframeinfo(inspect.currentframe()).filename
        folderpath = pathlib.Path(current_file).absolute().parent.parent
        love_file = folderpath.joinpath('data', 'LoveNumbers_Gegout97.txt')

    return np.genfromtxt(love_file)


def assert_sh(ds):
    """
    Verify if ds have dimensions l and m as well as variables clm and slm
    Raise Assertion error if not

    Returns
    -------
    True : bool
        return True is ds have dimensions l and m as well as variables clm and slm

    Raises
    ------
    AssertionError
        This function raise AssertionError is ds is not a xr.Dataset corresponding to spherical harmonics
    """
    if 'l' not in ds.coords:
        raise AssertionError("The degree coordinates that should be named 'l' does not exist")
    if 'm' not in ds.coords:
        raise AssertionError("The order coordinates that should be named 'm' does not exist")
    if 'clm' not in ds.keys():
        raise AssertionError("The Dataset have to contain 'clm' variable")
    if 'slm' not in ds.keys():
        raise AssertionError("The Dataset have to contain 'slm' variable")
    return True


def assert_grid(ds):
    """
    Verify if ds have dimensions longitude and latitude
    Raise Assertion error if not

    Returns
    -------
    True : bool
        return True is ds have dimensions longitude and latitude

    Raises
    ------
    AssertionError
        This function raise AssertionError is self._obj is not a xr.Dataset corresponding to spherical harmonics
    """
    if 'latitude' not in ds.coords:
        raise AssertionError("The latitude coordinates that should be named 'latitude' does not exist")
    if 'longitude' not in ds.coords:
        raise AssertionError("The longitude coordinates that should be named 'longitude' does not exist")
    return True
