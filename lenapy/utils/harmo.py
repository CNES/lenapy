"""
The harmo module provides functions for transforming spherical harmonics datasets into spatial grid representations
and vice versa.

This module includes functions to:
  * Convert spherical harmonics datasets into spatial grid dataarray.
  * Convert spatial grid dataarray into spherical harmonics datasets.
  * Compute associated Legendre functions.
  * Calculate mid-month estimates for GRACE data products.
  * Compute scaling factors for unit conversions between spherical harmonics and grid data.
  * Validate spherical harmonics and grid datasets.

"""

import datetime
import inspect
import pathlib
from typing import Literal

import cf_xarray as cfxr
import numpy as np
import pandas as pd
import scipy as sc
import xarray as xr

from lenapy.constants import *


def _compute_factors(
    lmax: int, normalization: Literal["4pi", "ortho", "schmidt"]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute recurrence coefficients f1 and f2 for the Legendre recursion.

    Parameters
    ----------
    lmax : int
        Maximum degree.
    normalization : {'4pi', 'ortho', 'schmidt'}
        Normalization scheme.

    Returns
    -------
    f1 : np.ndarray
        Recurrence factor f1.
    f2 : np.ndarray
        Recurrence factor f2.
    norm_p10 : float
        Normalization for P(1,0).
    norm_4pi : float
        Overall normalization factor.
    """
    size = (lmax + 1) * (lmax + 2) // 2
    f1 = np.zeros(size)
    f2 = np.zeros(size)

    k = 2
    if normalization in ("4pi", "ortho"):
        norm_p10 = np.sqrt(3)
        norm_4pi = 1 if normalization == "4pi" else 4 * np.pi
        for l in range(2, lmax + 1):
            k += 1
            f1[k] = np.sqrt(2 * l - 1) * np.sqrt(2 * l + 1) / l
            f2[k] = (l - 1) * np.sqrt(2 * l + 1) / (np.sqrt(2 * l - 3) * l)
            for m in range(1, l - 1):
                k += 1
                f1[k] = (
                    np.sqrt(2 * l + 1)
                    * np.sqrt(2 * l - 1)
                    / (np.sqrt(l + m) * np.sqrt(l - m))
                )
                f2[k] = (
                    np.sqrt(2 * l + 1)
                    * np.sqrt(l - m - 1)
                    * np.sqrt(l + m - 1)
                    / (np.sqrt(2 * l - 3) * np.sqrt(l + m) * np.sqrt(l - m))
                )
            k += 2
    elif normalization == "schmidt":
        norm_p10 = 1
        norm_4pi = 1
        for l in range(2, lmax + 1):
            k += 1
            f1[k] = (2 * l - 1) / l
            f2[k] = (l - 1) / l
            for m in range(1, l - 1):
                k += 1
                f1[k] = (2 * l - 1) / (np.sqrt(l + m) * np.sqrt(l - m))
                f2[k] = (
                    np.sqrt(l - m - 1)
                    * np.sqrt(l + m - 1)
                    / (np.sqrt(l + m) * np.sqrt(l - m))
                )
            k += 2
    else:
        raise ValueError(
            (
                f"Unknown normalization '{normalization}'. "
                "It should be either "
                "'4pi', 'ortho' or 'schmidt'"
            )
        )

    return f1, f2, norm_p10, norm_4pi


def _generate_grid(
    bounds: list[float],
    dlon: float,
    dlat: float,
    longitude: np.ndarray | None,
    latitude: np.ndarray | None,
    radians_in: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate longitude and latitude arrays for the grid.

    load longitude and latitude if given
    if not : compute longitude and latitude in degrees between given or defaults bounds

    Parameters
    ----------
    bounds : list of float
        [lonmin, lonmax, latmin, latmax]
    dlon, dlat : float
        Grid spacing in degrees.
    longitude, latitude : np.ndarray or None
        Optionally provided coordinate arrays.
    radians_in : bool
        Whether the input arrays are in radians.

    Returns
    -------
    longitude : np.ndarray
        Array of longitudes in degrees.
    latitude : np.ndarray
        Array of latitudes in degrees.
    """
    if longitude is None:
        longitude = np.arange(bounds[0] + dlon / 2.0, bounds[1] + dlon / 2.0, dlon)
    elif radians_in:
        longitude = np.rad2deg(longitude)

    if latitude is None:
        latitude = np.arange(bounds[2] + dlat / 2.0, bounds[3] + dlat / 2.0, dlat)
    elif radians_in:
        latitude = np.rad2deg(latitude)

    return longitude, latitude


def _init_bounds_and_grid(
    bounds: list[float] | None,
    lonmin: float,
    lonmax: float,
    latmin: float,
    latmax: float,
    dlon: float,
    dlat: float,
    radians_in: bool,
) -> tuple[list[float], float, float]:
    """
    Initialize bounds and grid resolution.

    Parameters
    ----------
    bounds: list of float or None
        [lonmin, lonmax, latmin, latmax]
    lonmin, lonmax, latmin, latmax: float
        Default bounds if `bounds` is None.
    dlon, dlat: float
        Grid resolution in degrees or radians.
    radians_in: bool
        Whether the inputs are in radians.

    Returns
    -------
    bounds: list of float
        Bounds in degrees.
    dlon: float
        Longitude resolution in degrees.
    dlat: float
        Latitude resolution in degrees.
    """
    if bounds is None:
        bounds = [lonmin, lonmax, latmin, latmax]
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise TypeError('"bounds" must be a list of 4 elements')

    if radians_in:
        bounds = [np.rad2deg(b) for b in bounds]
        if dlon != 1:
            dlon = np.rad2deg(dlon)
        if dlat != 1:
            dlat = np.rad2deg(dlat)
    return bounds, dlon, dlat


def _handle_mass_conservation(
    used_l: np.ndarray,
    force_mass_conservation: bool,
) -> tuple[np.ndarray, bool, bool]:
    """
    Test if mass conservation has to be forced to remove mass induced by the projection of C2n,0 coefficients

    Parameters
    ----------
    used_l: np.ndarray
        Degrees used in the computation.
    force_mass_conservation: bool
        Whether to enforce mass conservation.

    Returns
    -------
    used_l: np.ndarray
        Possibly modified degrees list.
    use_czero_coef: bool
        Whether to re-add degree 0 later.
    force_mass_conservation: bool
        Updated flag (can be False if nothing to conserve).
    """
    if force_mass_conservation and 0 in used_l:
        if len(used_l) > 1:
            return used_l[1:], True, force_mass_conservation
        elif len(used_l) == 1:
            return used_l, False, False
    return used_l, False, force_mass_conservation


def _init_degrees(
    data: xr.Dataset,
    lmin: int = None,
    lmax: int = None,
    mmin: int = None,
    mmax: int = None,
    used_l: np.ndarray = None,
    used_m: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Initialize spherical harmonic degrees and orders to be used.

    It prioritizes used_l and used_m if given over lmin, lmax, mmin and mmax

    Parameters
    ----------
    data : xr.Dataset
        Spherical harmonics dataset with 'l' and 'm' dimensions.
    lmin, lmax : int or None
        Minimum and maximum degrees.
    mmin, mmax : int or None
        Minimum and maximum orders.
    used_l, used_m : np.ndarray or None
        Explicit list of degrees and orders to use.

    Returns
    -------
    used_l: np.ndarray
        Degrees to use.
    used_m: np.ndarray
        Orders to use.
    lmin: int
        Minimum degree
    lmax: int
        Maximum degree.
    mmin:
        Minimum order.
    mmax:
        Maximum order.
    """
    lmax = int(data.l.max()) if lmax is None else lmax
    mmax = int(min(lmax, data.m.max())) if mmax is None else mmax
    lmin = int(data.l.min()) if lmin is None else lmin
    mmin = int(data.m.min()) if mmin is None else mmin
    used_l = np.arange(lmin, lmax + 1) if used_l is None else used_l
    used_m = np.arange(mmin, mmax + 1) if used_m is None else used_m
    return used_l, used_m, lmin, lmax, mmin, mmax


def sh_to_grid(
    data,
    unit="mewh",
    errors=False,
    lmax=None,
    mmax=None,
    lmin=None,
    mmin=None,
    used_l=None,
    used_m=None,
    lonmin=-180,
    lonmax=180,
    latmin=-90,
    latmax=90,
    bounds=None,
    dlon=1,
    dlat=1,
    longitude=None,
    latitude=None,
    radians_in=False,
    force_mass_conservation=False,
    ellipsoidal_earth=False,
    include_elastic=True,
    plm=None,
    normalization_plm="4pi",
    **kwargs,
):
    """
    Transform Spherical Harmonics (SH) dataset into spatial DataArray.
    With choice for constants, unit, love numbers, degree/order, spatial grid latitude and longitude, Earth hypothesis.

    For details on unit transformations, see :func:`l_factor_conv`.

    Parameters
    ----------
    data : xr.Dataset
        xr.Dataset that corresponds to SH data to convert into spatial representation.
    unit : str, optional
        'mewh', 'mmgeoid', 'microGal', 'bar', 'mvcu', or 'norm'
        Unit of the spatial data used in the transformation. Default is 'mewh' for meters of Equivalent Water Height.
        See utils.harmo.l_factor_conv() doc for details on the units.
    errors : bool, optional
        If True, propagate the errors from eclm and eslm variables to create an error grid. Default is False.
    lmax : int, optional
        Maximal degree of the SH coefficients to be used.
    mmax : int, optional
        Maximal order of the SH coefficients to be used.
    lmin : int, optional
        Minimal degree of the SH coefficients to be used.
    mmin : int, optional
        Minimal order of the SH coefficients to be used.
    used_l : np.ndarray, optional
        List of degree to use for the grid computation (if given, lmax and lmin are not considered).
    used_m : np.ndarray, optional
        List of order to use for the grid computation (if given, mmax and mmin are not considered).

    lonmin : float, optional
        Minimal longitude of the future grid.
    lonmax : float, optional
        Maximal longitude of the future grid.
    latmin : float, optional
        Minimal latitude of the future grid.
    latmax : float, optional
        Maximal latitude of the future grid.
    bounds : list, optional
        List of 4 elements with [lonmin, lonmax, latmin, latmax] (if given min/max information are not considered).
    radians_in : bool, optional
        True if the unit of the given latitude and longitude information is radians. Default is False for degree unit.
        If radians_in is True and dlat or dlon are given, they are considered as radians.
    dlon : float, optional
        Spacing of the longitude values.
    dlat : float, optional
        Spacing of the latitude values.
    longitude : np.ndarray, optional
        List of longitude to use for the grid computation (if given, others longitude information are not considered).
    latitude : np.ndarray, optional
        List of latitude to use for the grid computation (if given, others latitude information are not considered).

    force_mass_conservation : bool, optional
        If True, force that the grid resulting from all coefficients except C0 has a null global mass. Default is False
    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid following [Ditmar2018]. Default is False for a spherical Earth.
    include_elastic : bool, optional
        If True, the Earth behavior is elastic. Default is True

    plm : xr.DataArray, optional
        Precomputed plm values as a xr.DataArray variable. For example with the code :
        plm = xr.DataArray(compute_plm(lmax, sin_latitude), dims=['l', 'm', 'latitude'],
        coords={'l': data.l, 'm': data.m, 'latitude': latitude})
    normalization_plm : str, optional
        If plm need to be computed, choice of the norm corresponding to the SH dataset.
        Either '4pi', 'ortho', or 'schmidt' for 4pi normalized, orthonormalized, or Schmidt semi-normalized SH
        functions, respectively. Default is '4pi'.

    **kwargs :
        Supplementary parameters used by the function l_factor_conv to modify defaults constants used in the computation
        for the unit conversion. These parameters include (see :func:`l_factor_conv` documentation for more details) :
        a_earth, gm_earth, f_earth, rho_earth, ds_love

    Returns
    -------
    xgrid : xr.DataArray
        Spatial representation of the SH Dataset in the chosen unit.
    """
    # addition error propagation, add mask in output variable

    used_l, used_m, lmin, lmax, mmin, mmax = _init_degrees(
        data, lmin, lmax, mmin, mmax, used_l, used_m
    )
    used_l, use_czero_coef, force_mass_conservation = _handle_mass_conservation(
        used_l, force_mass_conservation
    )
    bounds, dlon, dlat = _init_bounds_and_grid(
        bounds, lonmin, lonmax, latmin, latmax, dlon, dlat, radians_in
    )
    longitude, latitude = _generate_grid(
        bounds, dlon, dlat, longitude, latitude, radians_in
    )

    cos_latitude = np.cos(np.deg2rad(latitude))
    sin_latitude = np.sin(np.deg2rad(latitude))
    f_earth = kwargs["f_earth"] if "f_earth" in kwargs else LNPY_F_EARTH_GRS80
    geocentric_colat = xr.DataArray(
        np.arctan2(cos_latitude, (1 - f_earth) ** 2 * sin_latitude),
        dims=["latitude"],
        coords={"latitude": latitude},
    )

    # -- beginning of computation
    # Computing plm for converting to spatial domain
    if plm is None:
        if ellipsoidal_earth:
            plm = compute_plm(
                lmax,
                np.cos(geocentric_colat),
                mmax=mmax,
                normalization=normalization_plm,
            )
        else:
            plm = compute_plm(
                lmax, sin_latitude, mmax=mmax, normalization=normalization_plm
            )
        plm = xr.DataArray(
            plm,
            dims=["l", "m", "latitude"],
            coords={
                "l": np.arange(lmax + 1),
                "m": np.arange(mmax + 1),
                "latitude": latitude,
            },
        )

    else:
        # Verify plm integrity
        if (
            not isinstance(plm, xr.DataArray)
            or "l" not in plm.coords
            or "m" not in plm.coords
            or "latitude" not in plm.coords
        ):
            raise TypeError(
                'Given argument "plm" has to be a DataArray with 3 coordinates [l, m, latitude]'
            )
        elif plm.l.max() < lmax:
            raise AssertionError(
                'Given argument "plm" maximal degree is too small ',
                plm.l.max(),
                "<",
                lmax,
            )
        elif (plm.latitude.values != latitude).all():
            raise AssertionError(
                'Given argument "plm" latitude does not correspond to the wanted latitude ',
                latitude,
            )

    # scale factor for each degree
    lfactor, cst = l_factor_conv(
        used_l,
        unit=unit,
        include_elastic=include_elastic,
        ellipsoidal_earth=ellipsoidal_earth,
        geocentric_colat=geocentric_colat,
        attrs=data.attrs,
        **kwargs,
    )

    # convolve unit over degree
    plm_lfactor = plm.sel(l=used_l, m=used_m) * lfactor

    # Calculating cos(m*phi) and sin(m*phi)
    c_cos = xr.DataArray(
        np.cos(used_m[:, np.newaxis] @ np.deg2rad(longitude)[np.newaxis, :]),
        dims=["m", "longitude"],
        coords={"m": used_m, "longitude": longitude},
    )
    s_sin = xr.DataArray(
        np.sin(used_m[:, np.newaxis] @ np.deg2rad(longitude)[np.newaxis, :]),
        dims=["m", "longitude"],
        coords={"m": used_m, "longitude": longitude},
    )

    # summation over all spherical harmonic degrees
    if not errors:
        d_clm = (plm_lfactor * data.sel(l=used_l, m=used_m).clm).sum(dim="l")
        d_slm = (plm_lfactor * data.sel(l=used_l, m=used_m).slm).sum(dim="l")

        # Final calcul on the grid
        xgrid = c_cos.dot(d_clm) + s_sin.dot(d_slm)
    else:
        d_clm = (plm_lfactor**2 * data.sel(l=used_l, m=used_m).clm ** 2).sum(dim="l")
        d_slm = (plm_lfactor**2 * data.sel(l=used_l, m=used_m).slm ** 2).sum(dim="l")

        # Final calcul of sigma on the grid
        xgrid = np.sqrt((c_cos**2).dot(d_clm) + (s_sin**2).dot(d_slm))

        unit = "Errors in " + unit

    # force mass conservation by removing the global mass computed without C0 coefficient (generated by C2n,0 coeffs)
    if force_mass_conservation and not errors:
        if ellipsoidal_earth:
            raise ValueError(
                'Mass conservation set to True with argument "force_mass_conservation" cannot be'
                " forced on ellipsoidal Earth"
            )

        int_fact = xgrid.lngeo.surface_cell(ellipsoidal_earth=False, a_earth=1) / (
            4 * np.pi
        )
        # remove the mass on the whole grid
        xgrid = (
            xgrid
            - (xgrid * int_fact).sum(dim=["latitude", "longitude"]) / int_fact.sum()
        )

        # restore C0 mass
        if use_czero_coef:
            lfactor_zero = l_factor_conv(
                np.array([0]), unit=unit, attrs=data.attrs, **kwargs
            )[0]
            xgrid = xgrid + (lfactor_zero * data.clm.sel(l=0, m=0)).values

    xgrid = xgrid.transpose("latitude", "longitude", ...)

    xgrid.attrs = {"units": unit, "max_degree": int(lmax)}
    if "radius" in data.attrs:
        xgrid.attrs["radius"] = data.attrs["radius"]
    if "earth_gravity_constant" in data.attrs:
        xgrid.attrs["earth_gravity_constant"] = data.attrs["earth_gravity_constant"]

    return xgrid


def grid_to_sh(
    grid,
    lmax,
    unit="mewh",
    mmax=None,
    lmin=0,
    mmin=0,
    used_l=None,
    used_m=None,
    ellipsoidal_earth=False,
    include_elastic=True,
    plm=None,
    normalization_plm="4pi",
    **kwargs,
):
    """
    Transform gravity field spatial representation DataArray into Spherical Harmonics (SH) dataset.
    With choice for constants, unit of the spatial DataArray, love_numbers, degree/order, Earth hypothesis.

    For details on unit transformations, see :func:`l_factor_conv`.

    Parameters
    ----------
    grid : xr.DataArray
        xr.Dataset that corresponds a gravity field spatial representation in a unit to convert into SH.
    lmax : int
        Maximal degree of the SH coefficients to be computed.
    unit : str, optional
        'mewh', 'mmgeoid', 'microGal', 'bar', 'mvcu', or 'norm'
        Unit of the spatial data used in the transformation. Default is 'mewh' for meters of Equivalent Water Height.
        See constants.l_factor_conv() doc for details on the units.

    mmax : int, optional
        Maximal order of the SH coefficients to be computed.
    lmin : int, optional
        Minimal degree of the SH coefficients to be computed. Default is 0.
    mmin : int, optional
        Minimal order of the SH coefficients to be computed. Default is 0.
    used_l : np.ndarray, optional
        List of degree to compute for the SH Dataset (if given, lmax and lmin are not considered).
    used_m : np.ndarray, optional
        List of order to compute for the SH Dataset (if given, mmax and mmin are not considered).

    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid following [Ditmar2018]. Default is False for a spherical Earth.
    include_elastic : bool, optional
        If True, the Earth behavior is elastic. Default is True.

    plm : xr.DataArray, optional
        Precomputed plm values as a xr.DataArray variable. For example with the code :
        plm = xr.DataArray(compute_plm(lmax, np.sin(np.deg2rad(latitude))), dims=['l', 'm', 'latitude'],
        coords={'l': np.arange(lmax+1), 'm': np.arange(lmax+1), 'latitude': latitude})
    normalization_plm : str, optional
        If plm need to be computed, choice of the norm.  Either '4pi', 'ortho', or 'schmidt' for
        4pi normalized, orthonormalized, or Schmidt semi-normalized SH functions, respectively. Default is '4pi'.
        Output SH coefficient will be normalized according to this parameter.

    **kwargs :
        Supplementary parameters used by the function l_factor_conv to modify defaults constants used in the computation
        for the unit conversion. These parameters include (see :func:`l_factor_conv` documentation for more details) :
        a_earth, gm_earth, f_earth, rho_earth, ds_love

    Returns
    -------
    ds_out : xr.DataArray
        SH Dataset computed from the grid with the chosen unit.
    """
    # -- set degree and order default parameters
    # it prioritizes used_l and used_m if given over lmin, lmax, mmin and mmax
    mmax = lmax if mmax is None else mmax
    used_l = np.arange(lmin, lmax + 1) if used_l is None else used_l
    used_m = np.arange(mmin, mmax + 1) if used_m is None else used_m

    cos_latitude = np.cos(np.deg2rad(grid.cf["latitude"].values))
    sin_latitude = np.sin(np.deg2rad(grid.cf["latitude"].values))

    f_earth = kwargs["f_earth"] if "f_earth" in kwargs else LNPY_F_EARTH_GRS80
    geocentric_colat = xr.DataArray(
        np.arctan2(cos_latitude, (1 - f_earth) ** 2 * sin_latitude),
        dims=["latitude"],
        coords={"latitude": grid.cf["latitude"]},
    )

    # create DataArray corresponding to the integration factor for each cell
    # case for ellipsoidal earth where integration over ellipsoidal cell
    if ellipsoidal_earth:
        surface = grid.lngeo.surface_cell(ellipsoidal_earth=True, f_earth=f_earth)
        int_fact = surface / surface.sum()

    # case for spherical earth where integration over spherical cell
    else:
        int_fact = grid.lngeo.surface_cell(ellipsoidal_earth=False, a_earth=1) / (
            4 * np.pi
        )

    # Create a readable cf_xarray DataArray
    int_fact["latitude"].attrs = dict(standard_name="latitude")
    int_fact["longitude"].attrs = dict(standard_name="longitude")

    # scale factor for each degree
    lfactor, cst = l_factor_conv(
        used_l,
        unit=unit,
        include_elastic=include_elastic,
        ellipsoidal_earth=ellipsoidal_earth,
        geocentric_colat=geocentric_colat,
        attrs=grid.attrs,
        **kwargs,
    )

    # -- prepare variables for the computation of SH
    # Computing plm for converting to spatial domain
    if plm is None:
        if ellipsoidal_earth:
            plm = compute_plm(
                lmax,
                np.cos(geocentric_colat),
                mmax=mmax,
                normalization=normalization_plm,
            )
        else:
            plm = compute_plm(
                lmax, sin_latitude, mmax=mmax, normalization=normalization_plm
            )
        plm = xr.DataArray(
            plm,
            dims=["l", "m", "latitude"],
            coords={
                "l": np.arange(lmax + 1),
                "m": np.arange(lmax + 1),
                "latitude": grid.cf["latitude"],
            },
        )

    else:
        # Verify plm integrity
        align = xr.align(
            plm.latitude, grid.cf["latitude"]
        )  # return the intersection of latitudes for each input
        if (
            not isinstance(plm, xr.DataArray)
            or "l" not in plm.coords
            or "m" not in plm.coords
            or "latitude" not in plm.coords
        ):
            raise TypeError(
                'Given argument "plm" has to be a DataArray with 3 coordinates [l, m, latitude]'
            )
        elif plm.l.max() < lmax:
            raise AssertionError(
                'Given argument "plm" maximal degree is too small ',
                plm.l.max(),
                "<",
                lmax,
            )
        elif align[0] != plm.latitude.size or align[1].size != grid.cf["latitude"].size:
            raise AssertionError(
                'Given argument "plm" latitude does not correspond to the grid latitude'
            )

    # convolve unit over degree
    plm_lfactor = plm.sel(l=used_l, m=used_m) / lfactor

    # Calculating cos/sin of phi arrays, [m,phi]
    c_cos = xr.DataArray(
        np.cos(
            used_m[:, np.newaxis]
            @ np.deg2rad(grid.cf["longitude"].values)[np.newaxis, :]
        ),
        dims=["m", "longitude"],
        coords={"m": used_m, "longitude": grid.cf["longitude"]},
    )
    s_sin = xr.DataArray(
        np.sin(
            used_m[:, np.newaxis] @ np.deg2rad(grid.longitude.values)[np.newaxis, :]
        ),
        dims=["m", "longitude"],
        coords={"m": used_m, "longitude": grid.cf["longitude"]},
    )

    # -- Computation of SH
    # Multiplying data and integral factor with sin/cos of m*longitude. This will sum over longitude, [m,theta]
    dcos = c_cos.dot(grid.cf.rename_like(int_fact) * int_fact, dim=["longitude"])
    dsin = s_sin.dot(grid.cf.rename_like(int_fact) * int_fact, dim=["longitude"])

    # Multiplying plm and degree scale factors with last variable to sum over latitude, [l, m, ...]
    clm = plm_lfactor.dot(dcos, dim=["latitude"])
    slm = plm_lfactor.dot(dsin, dim=["latitude"])

    # add name for the merge into xr.Dataset
    clm.name = "clm"
    slm.name = "slm"

    ds_out = xr.merge([clm, slm], join="exact")
    cst.update({"max_degree": lmax, "norm": normalization_plm})
    ds_out.attrs = cst
    return ds_out


def compute_plm(lmax, z, mmax=None, normalization="4pi"):
    """
    Compute all the associated Legendre functions up to a maximum degree and
    order using the recursion relation from [Holmes2002]_
    (Adapted from SHTOOLS/pyshtools tools [Wieczorek2018]_)

    Parameters
    ----------
    lmax : int
        Maximum degree of legrendre functions.
    z : np.ndarray
        Argument of the associated Legendre functions.
    mmax : int or NoneType, optional
        Maximum order of associated legrendre functions.
    normalization : str, optional
        '4pi', 'ortho', or 'schmidt' for use with geodesy 4pi normalized, orthonormalized, or Schmidt semi-normalized
        spherical harmonic functions, respectively. Default is '4pi'.

    Returns
    -------
    plm : np.ndarray
        Fully-normalized Legendre functions as a 3D array with "l", "m" and z dimensions.

    References
    ----------
    .. [Holmes2002] S. A. Holmes and W. E. Featherstone,
        "A unified approach to the Clenshaw summation and the recursive computation of very high degree and order
        normalised associated Legendre functions",
        *Journal of Geodesy*, 76, 279--299, (2002).
        `doi: 10.1007/s00190-002-0216-2 <https://doi.org/10.1007/s00190-002-0216-2>`_
    .. [Wieczorek2018]  M. A. Wieczorek and M. Meschede,
        "SHTools: Tools for working with spherical harmonics",
        *Geochemistry, Geophysics, Geosystems*, 19, 2574–2592, (2018).
        `doi: 10.1029/2018GC007529 <https://doi.org/10.1029/2018GC007529>`_
    """
    # removing singleton dimensions of x
    # update type to provide more memory for computation (np.float32 create some problems)
    z = np.atleast_1d(z).flatten().astype(np.float128)

    # if default mmax, set mmax to be maximal degree
    mmax = lmax if mmax is None else mmax

    f1, f2, norm_p10, norm_4pi = _compute_factors(lmax, normalization)

    # scale factor based on Holmes2002
    scalef = 1e-280

    p = np.zeros(((lmax + 1) * (lmax + 2) // 2, len(z)))
    # u is sine of colatitude (cosine of latitude), for z=cos(th): u=sin(th)
    u = np.sqrt(1 - z**2)
    # update where u==0 to minimal numerical precision different from 0 to prevent invalid divisions
    u[u == 0] = np.finfo(np.float64).eps

    # Calculate P(l,0) (not scaled)
    p[0, :] = 1 / np.sqrt(norm_4pi)
    if lmax:  # test for the case where lmax=0
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
        if normalization in ("4pi", "ortho"):
            p[kstart, :] = pmm
        elif normalization == "schmidt":
            p[kstart, :] = pmm / np.sqrt(2 * m + 1)

        if m != lmax:  # test if P(m+1,m) exist
            # Calculate P(m+1,m)
            k = kstart + m + 1
            if normalization in ("4pi", "ortho"):
                p[k, :] = z * np.sqrt(2 * m + 3) * pmm
            elif normalization == "schmidt":
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
    return plm[:, : mmax + 1, :]


def mid_month_grace_estimate(begin_time, end_time):
    """
    Calculate middle of the month date based on begin_time and end_time for GRACE products.
    begin_time is rounded to equal to the first day of the month and
    end_time is rounded to equal to the first day of the month after.

    Parameters
    ----------
    begin_time : datetime.datetime
        Date of the beginning of the month.
    end_time : datetime.datetime
        Date of the end of the month + 1 day.

    Returns
    -------
    mid_month : datetime.datetime
        Date of the middle of the month.
    """
    # to compute mid_month, need to round begin_time to the 1st of the month
    # deal with GRACE month when the begin date is not between 16 of month before and 15 of the month
    # it includes May 2015, Dec 2011 (JPL), Mar 2017 and Oct 2018 that cover second half of month
    if (
        (begin_time.day <= 15 and begin_time.strftime("%Y%j") != "2015102")
        or begin_time.strftime("%Y%j") == "2011351"
        or begin_time.strftime("%Y%j") == "2017076"
        or begin_time.strftime("%Y%j") == "2018295"
    ):
        tmp_begin = begin_time.replace(day=1)
    else:
        tmp_begin = (begin_time.replace(day=1) + datetime.timedelta(days=32)).replace(
            day=1
        )

    # to compute mid_month, need to round end_time to the 1st of the month after
    # deal with GRACE month when the end date is not between 16 of month and 15 of the month after
    # it includes Janv 2004, Nov 2011 (CSR, GFZ) and May 2015 and Dec 2011 for TN14
    if (
        end_time.day <= 15
        and end_time.strftime("%Y%j") not in ("2004014", "2011320", "2015132")
    ) or end_time.strftime("%Y%j") == "2012016":
        tmp_end = end_time.replace(day=1)
    else:
        tmp_end = (end_time.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)

    return tmp_begin + (tmp_end - tmp_begin) / 2


def change_normalization(
    ds, new_normalization="4pi", old_normalization=None, apply=False
):
    """
    Spherical Harmonics (SH) dataset are associated with a Legendre polynomial normalization.
    This function return the dataset updated with the new normalization asked in input (as a deep copy by default).
    The current normalization can be given in parameters or can be contained in ds.attrs['norm'].

    Parameters
    ----------
    ds : xr.Dataset
        xr.Dataset that corresponds to SH data associated with a current reference frame (constants whise) to update.
    new_normalization : str
        New normalization for the SH dataset, either '4pi', 'ortho', or 'schmidt'.
    old_normalization : str | None, optional
        Current normalization of the SH dataset.
        Either '4pi', 'ortho', 'schmidt' or 'unnorm' for
        4pi normalized, orthonormalized, Schmidt semi-normalized, or unnormalized SH functions, respectively.
        Default is '4pi'. If not provided, uses `ds.attrs['norm']`.
    apply : bool, optional
        If True, apply the update to the current dataset without making a deep copy. Default is False.

    Returns
    -------
    ds_out : xr.Dataset
        Updated dataset with the new normalization.

    Examples
    --------
    >>> ds_new_norm = change_normalization(ds, new_normalization='schmidt')
    """
    try:
        old_normalization = (
            ds.attrs["norm"] if old_normalization is None else old_normalization
        )

    except ValueError:
        raise ValueError(
            "If you provide no information about the current normalization of your ds dataset using "
            "'old_normalization' parameters, those information need to be contained in ds.attrs dict as "
            "ds.attrs['norm']."
        )

    # -- conversion for each tidal system
    if new_normalization == "4pi" and old_normalization == "schmidt":
        update_factor = 1 / np.sqrt(2 * ds.l + 1)
    elif new_normalization == "schmidt" and old_normalization == "4pi":
        update_factor = np.sqrt(2 * ds.l + 1)
    elif new_normalization == "ortho" and old_normalization == "schmidt":
        update_factor = np.sqrt((4 * np.pi) / (2 * ds.l + 1))
    elif new_normalization == "schmidt" and old_normalization == "ortho":
        update_factor = np.sqrt((2 * ds.l + 1) / (4 * np.pi))
    elif new_normalization == "ortho" and old_normalization == "4pi":
        update_factor = np.sqrt(4 * np.pi)
    elif new_normalization == "4pi" and old_normalization == "ortho":
        update_factor = 1 / np.sqrt(4 * np.pi)
    elif old_normalization == "unnorm":
        fact_l_minus_m = xr.apply_ufunc(sc.special.factorial, ds.l - ds.m)
        fact_l_minus_m = fact_l_minus_m.where(fact_l_minus_m != 0, float("NaN"))
        if new_normalization == "4pi":
            update_factor = np.sqrt(
                xr.apply_ufunc(sc.special.factorial, ds.l + ds.m)
                / fact_l_minus_m
                / (2 * ds.l + 1)
                / (2 - (0 == ds.m).astype(int))
            )
        elif new_normalization == "ortho":
            update_factor = np.sqrt(
                xr.apply_ufunc(sc.special.factorial, ds.l + ds.m)
                * 4
                * np.pi
                / fact_l_minus_m
                / (2 * ds.l + 1)
                / (2 - (0 == ds.m).astype(int))
            )
        elif new_normalization == "schmidt":
            update_factor = np.sqrt(
                xr.apply_ufunc(sc.special.factorial, ds.l + ds.m)
                * 4
                * np.pi
                / fact_l_minus_m
                / (2 - (0 == ds.m).astype(int))
            )
        update_factor = update_factor.fillna(0)
    else:
        update_factor = 1

    # if apply = False : Copy the dataset to avoid modifying the input dataset
    ds_out = ds if apply else ds.copy(deep=True)

    # Update the clm and slm values
    ds_out["clm"] *= update_factor
    ds_out["slm"] *= update_factor

    # Update the attributes in the output dataset
    ds_out.attrs["norm"] = new_normalization

    return ds_out


def get_earth_parameters(
    attrs: dict | None, a_earth: float | None, gm_earth: float | None
) -> tuple[float, float]:
    """
    Retrieve Earth parameters from inputs or fallback constants.

    Parameters
    ----------
    attrs : dict or None
        Attributes potentially containing Earth parameters.
    a_earth : float or None
        Semi-major axis.
    gm_earth : float or None
        Gravitational constant.

    Returns
    -------
    a_earth : float
        Resolved semi-major axis.
    gm_earth : float
        Resolved gravitational constant.
    """
    if attrs is None:
        attrs = {}

    resolved_a = a_earth or float(attrs.get("radius", LNPY_A_EARTH_GRS80))
    resolved_gm = gm_earth or float(attrs.get("earth_gravity_constant", LNPY_GM_EARTH))
    return resolved_a, resolved_gm


def load_default_love_numbers() -> xr.Dataset:
    """
    Load default Love numbers dataset.

    Returns
    -------
    ds_love : xr.Dataset
        Love numbers dataset.
    """
    current_file = inspect.getframeinfo(inspect.currentframe()).filename
    folderpath = pathlib.Path(current_file).absolute().parent.parent
    default_love_file = folderpath / "resources" / "LoveNumbers_Gegout97.txt"

    df = pd.read_csv(default_love_file, names=["kl"])
    ds = xr.Dataset.from_dataframe(df).rename({"index": "l"})
    return ds


def compute_a_div_r_lat(geocentric_colat: np.ndarray, f_earth: float) -> np.ndarray:
    """
    Compute a/r(θ) for ellipsoidal Earth correction.

    Parameters
    ----------
    geocentric_colat : np.ndarray
        Geocentric colatitudes in radians.
    f_earth : float
        Earth flattening.

    Returns
    -------
    a_div_r_lat : np.ndarray
    """
    # e = sqrt(2f - f**2)
    e_earth = np.sqrt(2 * f_earth - f_earth**2)
    # a_div_r_lat = a / r(theta)  with r(theta) = a(1-f)/sqrt(1 - e**2*sin(theta)**2)
    return np.sqrt(1 - e_earth**2 * np.sin(geocentric_colat) ** 2) / (1 - f_earth)


def l_factor_conv(
    l,
    unit="mewh",
    include_elastic=True,
    ellipsoidal_earth=False,
    geocentric_colat=None,
    ds_love=None,
    a_earth=None,
    gm_earth=None,
    f_earth=LNPY_F_EARTH_GRS80,
    rho_earth=LNPY_RHO_EARTH,
    attrs=None,
):
    """
    Compute a scale factor for a transformation between spherical harmonics and grid data.
    Spatial data over the grid are associated with a specific unit.
    The scale factor is degree-dependent and is computed for the given list of degree l.
    The scale factor can be estimated using elastic or non-elastic Earth as well as a spherical or ellipsoidal Earth.

    Parameters
    ----------
    l : np.ndarray
        Degree for which the scale factor is estimated.
    unit : str, optional
        'mewh', 'mmgeoid', 'microGal', 'pascal', 'mvcu', or 'norm'
        Unit of the spatial data used in the transformation. Default is 'mewh' for meters of Equivalent Water Height.
        'mmgeoid' represents millimeters mmgeoid height, 'microGal' represents microGal gravity perturbations,
        'pascal' represents equivalent surface pressure in pascal and
        'mvcu' represents meters viscoelastic crustal uplift
    include_elastic : bool, optional
        If True, the Earth behavior is elastic.
    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid following [Ditmar2018]_ and if False as a sphere.
    geocentric_colat : list, optional
        List of geocentric colatitude for ellipsoidal Earth radius computation in radians.
    ds_love : xr.Dataset | None, optional
        Dataset with a l dimension corresponding to degree and with l (and possibly h and k) variables that
        are Love numbers.
        Default Love numbers used are from Gegout97.
    a_earth : float, optional
        Earth semi-major axis [m]. If not provided, uses `data.attrs['radius']` and
        if it does not exist, uses LNPY_A_EARTH_GRS80.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80.
    gm_earth : float, optional
        Standard gravitational parameter for Earth [m³.s⁻²]. Default is LNPY_GM_EARTH.
    rho_earth : float, optional
        Earth density [kg.m⁻³]. Default is LNPY_RHO_EARTH.
    attrs : dict | None, optional
        ds.attrs information that might help to estimate l_factor if no parameters are given.

    Returns
    -------
    l_factor : np.ndarray
        Degree-dependent scale factor.

    References
    ----------
    .. [Ditmar2018] P. Ditmar,
        "Conversion of time-varying Stokes coefficients into mass anomalies at the
        Earth’s surface considering the Earth’s oblateness",
        *Journal of Geodesy*, 92, 1401--1412, (2018).
        `doi: 10.1007/s00190-018-1128-0 <https://doi.org/10.1007/s00190-018-1128-0>`_
    """

    a_earth, gm_earth = get_earth_parameters(attrs, a_earth, gm_earth)

    l = xr.DataArray(l, dims=["l"], coords={"l": l})
    fraction = xr.ones_like(l)

    # include elastic redistribution with kl Love numbers
    if include_elastic:
        if ds_love is None:
            ds_love = load_default_love_numbers()
        fraction = fraction + ds_love.kl

    a_div_r_lat = None
    if ellipsoidal_earth:
        # test if geocentric_colat is set
        if geocentric_colat is None:
            raise ValueError(
                "For ellipsoidal Earth, you need to set "
                "the parameter 'geocentric_colat' in l_factor_conv function"
            )

        a_div_r_lat = compute_a_div_r_lat(geocentric_colat, f_earth)

    l_factor = compute_l_factor(
        a_div_r_lat,
        a_earth,
        ds_love,
        ellipsoidal_earth,
        fraction,
        gm_earth,
        l,
        rho_earth,
        unit,
    )

    cst = {"gm_earth": gm_earth, "a_earth": a_earth}
    return l_factor, cst


def compute_l_factor(
    a_div_r_lat: np.ndarray | None,
    a_earth: float,
    ds_love: xr.Dataset | None,
    ellipsoidal_earth: bool,
    fraction: xr.DataArray,
    gm_earth: float,
    l: np.ndarray | xr.DataArray,
    rho_earth,
    unit,
):
    """
    Compute the degree-dependent scale factor.

    Parameters
    ----------
    a_div_r_lat: np.ndarray or None
        Ellipsoidal correction factor
    a_earth: float
        Earth radius
    ds_love: xr.Dataset, optional
        Love numbers
    ellipsoidal_earth: bool
        Whether to apply ellipsoidal correction
    fraction: xr.DataArray
        Redistribution factor
    gm_earth: float
        Earth gravitational constant
    l: np.ndarray or xr.DataArray
        Degrees
    rho_earth: float
        Earth density
    unit: str
        Unit type for conversion

    Returns
    -------
    l_factor : xr.DataArray
        Degree-dependent conversion factor.
    """
    # l_factor is degree dependant
    if unit == "norm":
        # norm, fully normalized spherical harmonics
        l_factor = xr.ones_like(l)
        if ellipsoidal_earth:
            l_factor = l_factor * a_div_r_lat**l

    elif unit == "mewh":
        # mewh, meters equivalent water height [kg.m-2]
        # the exact formula is l_factor*(1 - f) (see [Ditmar2018]_ after eq. 17)
        # it is an approximation of the order of 0.3% to be coherent with the common formula from Wahr 1998
        l_factor = rho_earth * a_earth * (2 * l + 1) / (3 * fraction * 1e3)
        if ellipsoidal_earth:
            l_factor = l_factor * a_div_r_lat ** (l + 2)

    elif unit == "mmgeoid":
        # mmgeoid, millimeters geoid height
        l_factor = xr.ones_like(l) * a_earth * 1e3
        if ellipsoidal_earth:
            l_factor = l_factor * a_div_r_lat ** (l + 1)

    elif unit == "microGal":
        # microGal, microGal gravity perturbations
        l_factor = gm_earth * (l + 1) / (a_earth**2) * 1e8
        if ellipsoidal_earth:
            l_factor = l_factor * a_div_r_lat ** (l + 2)

    elif unit == "potential":
        # potential, meters².seconds⁻²
        l_factor = gm_earth / a_earth
        if ellipsoidal_earth:
            l_factor = l_factor * a_div_r_lat ** (l + 1)

    elif unit == "pascal":
        # pascal, equivalent surface pressure
        l_factor = LNPY_G_WMO * rho_earth * a_earth * (2 * l + 1) / (3 * fraction)
        if ellipsoidal_earth:
            l_factor = l_factor * a_div_r_lat ** (l + 1)

    elif unit == "mvcu":
        # mvcu, meters viscoelastic crustal uplift
        l_factor = a_earth * (2 * l + 1) / 2
        if ellipsoidal_earth:
            l_factor = l_factor * a_div_r_lat ** (l + 1)

    elif unit == "mecu":
        # mecu, meters elastic crustal deformation (uplift)
        l_factor = a_earth * ds_love.hl / fraction
        if ellipsoidal_earth:
            l_factor = l_factor * a_div_r_lat ** (l - 1)

    elif unit == "int_radial_mag":
        # internal radial magnetic field in nT
        l_factor = l + 1
        if ellipsoidal_earth:
            pass

    elif unit == "ext_radial_mag":
        # external radial magnetic field in nT
        l_factor = -l
        if ellipsoidal_earth:
            pass

    # Gt, Gigatonnes ?

    else:
        raise ValueError(
            "Invalid 'unit' parameter value in l_factor_conv function, valid values are: "
            "(norm, mewh, mmgeoid, microGal, bar, mvcu)"
        )
    return l_factor


def assert_sh(ds):
    """
    Verify if the dataset ds has dimensions 'l' and 'm' as well as variables 'clm' and 'slm'
    Raise Assertion error if not.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to verify.

    Returns
    -------
    True : bool
        Returns True if the dataset ds has dimensions 'l' and 'm' as well as variables 'clm' and 'slm'.

    Raises
    ------
    AssertionError
        This function raise AssertionError is ds is not a xr.Dataset corresponding to spherical harmonics
    """
    if "l" not in ds.coords:
        raise AssertionError(
            "The degree coordinates that should be named 'l' does not exist"
        )
    if "m" not in ds.coords:
        raise AssertionError(
            "The order coordinates that should be named 'm' does not exist"
        )
    if "clm" not in ds.keys():
        raise AssertionError("The Dataset have to contain 'clm' variable")
    if "slm" not in ds.keys():
        raise AssertionError("The Dataset have to contain 'slm' variable")
    return True
