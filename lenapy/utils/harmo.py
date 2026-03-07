"""
The **harmo** module provides functions for transforming spherical harmonics datasets into spatial grid representations
and vice versa.

This module includes functions to:
  * Convert spherical harmonics datasets into spatial grid DataArray.
  * Convert spatial grid DataArray into spherical harmonics datasets.
  * Compute associated Legendre functions.
  * Calculate mid-month estimates for GRACE data products.
  * Change the associated normalization to a spherical harmonics dataset.
  * Compute scaling factors for unit conversions between spherical harmonics and grid data.
  * Validate spherical harmonics and grid datasets.

"""

import datetime
import inspect
import pathlib
import warnings
from typing import Literal, Optional

import cf_xarray as cfxr
import numpy as np
import pandas as pd
import scipy as sc
import xarray as xr

from lenapy.constants import *
from lenapy.utils.geo import (
    assert_grid,
    assert_latitude,
    latitude_to_geocentric_colatitude,
)


def _generate_grid(
    bounds: list[float] | None,
    lonmin: float,
    lonmax: float,
    latmin: float,
    latmax: float,
    dlon: float,
    dlat: float,
    radians_in: bool,
    longitude: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    grid: xr.DataArray | xr.Dataset | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Initialize bounds and grid resolution.
    Generate longitude and latitude arrays for the grid.

    Load longitude and latitude from the given grid, if not from the given latitude, longitude and
    if not : compute longitude and latitude in degrees between given or defaults bounds

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
    longitude, latitude: np.ndarray or None
        Optionally provided coordinate arrays.
    radians_in: bool
        Whether the input arrays are in radians.
    grid: xr.DataArray | xr.Dataset | float
        Optionally provided grid from which to extract longitude and latitude coordinates.

    Returns
    -------
    longitude : np.ndarray
        Array of longitudes in degrees.
    latitude : np.ndarray
        Array of latitudes in degrees.
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

    if grid is None or type(grid) is not xr.Dataset:
        if longitude is None:
            longitude = np.arange(bounds[0] + dlon / 2, bounds[1] + dlon / 2, dlon)
        elif radians_in:
            longitude = np.rad2deg(longitude)

        if latitude is None:
            latitude = np.arange(bounds[2] + dlat / 2, bounds[3] + dlat / 2, dlat)
        elif radians_in:
            latitude = np.rad2deg(latitude)

    else:
        if "longitude" in grid.coords:
            assert_grid(grid)
            longitude = grid.longitude.values
            latitude = grid.latitude.values
        else:
            assert_latitude(grid)
            latitude = grid.latitude.values
            if longitude is None:
                longitude = np.arange(bounds[0] + dlon / 2, bounds[1] + dlon / 2, dlon)
            elif radians_in:
                longitude = np.rad2deg(longitude)

    return longitude, latitude


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
) -> tuple[xr.Dataset, np.ndarray, np.ndarray, int, int, int, int]:
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
    sub_data : xr.Dataset
        Reduced dataset with selected degrees and orders.
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
    sub_data = data.sel(l=used_l, m=used_m)
    return sub_data, used_l, used_m, lmin, lmax, mmin, mmax


def sh_to_grid(
    data: xr.Dataset,
    unit: str = "mewh",
    errors=False,
    lmax: int | None = None,
    mmax: int | None = None,
    lmin: int | None = None,
    mmin: int | None = None,
    used_l: np.ndarray | None = None,
    used_m: np.ndarray | None = None,
    lonmin: float = -180,
    lonmax: float = 180,
    latmin: float = -90,
    latmax: float = 90,
    bounds: list = None,
    dlon: float = 1,
    dlat: float = 1,
    longitude: np.ndarray | None = None,
    latitude: np.ndarray | None = None,
    radians_in: bool = False,
    radius: xr.DataArray | float | None = None,
    force_mass_conservation: bool = False,
    ellipsoidal_earth: bool = False,
    include_elastic: bool = True,
    plm: xr.DataArray = None,
    normalization_plm: Literal["4pi", "ortho", "schmidt"] = "4pi",
    dtype_plm: type[complex] | type[float] = np.float128,
    use_dask: bool = False,
    chunks_lfactor: dict | None = None,
    chunks_plm: dict | None = None,
    **kwargs,
) -> xr.DataArray:
    """
    Transform Spherical Harmonics (SH) dataset into spatial DataArray.
    With choice for constants, unit, love numbers, degree/order, spatial grid latitude and longitude, Earth hypothesis.

    For details on unit transformations, see :func:`l_factor_conv` and the
    associated `PDF document <../../_static/Mathematics_consideration_for_LENAPY.pdf>`_.

    Parameters
    ----------
    data : xr.Dataset
        xr.Dataset that corresponds to SH data to convert into spatial representation.
    unit : str, optional
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

    radius : xr.DataArray | float, optional
        DataArray or float with the radius of the grid to compute. If not given, the radius is the surface of reference.

    force_mass_conservation : bool, optional
        If True, force that the grid resulting from all coefficients except C0 has a null global mass. Default is False.
    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid following [Ditmar2018]. Default is False for a spherical Earth.
    include_elastic : bool, optional
        If True, the Earth behavior is elastic. Default is True.

    plm : xr.DataArray, optional
        Precomputed plm values as a xr.DataArray variable. For example with the code :
        plm = xr.DataArray(compute_plm(lmax, sin_latitude), dims=['l', 'm', 'latitude'],
        coords={'l': data.l, 'm': data.m, 'latitude': latitude})
    normalization_plm : str, optional
        If plm need to be computed, choice of the norm corresponding to the SH dataset.
        Either '4pi', 'ortho', or 'schmidt' for 4pi normalized, orthonormalized, or Schmidt semi-normalized SH
        functions, respectively. Default is '4pi'.
    dtype_plm : dtype, optional
        Specify the dtype to compute the plm DataArray. Default is np.float128.

    use_dask : bool, optional
        If True, use dask to chunk plm for memory optimization. Default is False.
    chunks_lfactor : dict, optional
        Define the chunking of lfactor when use_dask is True. Default is None, which set the chunking to {'l': 200}.
    chunks_plm : dict, optional
        Define the chunking of plm when use_dask is True. Default is None, which set the chunking to {'latitude': 4}.

    **kwargs :
        Supplementary parameters used by the function l_factor_conv to modify defaults constants used in the computation
        for the unit conversion. These parameters include (see :func:`l_factor_conv` documentation for more details) :
        a_earth, earth_gravity_constant, f_earth, omega_earth, rho_earth, ds_love

    Returns
    -------
    xgrid : xr.DataArray
        Spatial representation of the SH Dataset in the chosen unit.
    """
    # Get parameters for computation of the grid and for the unit conversion
    sub_data, used_l, used_m, lmin, lmax, mmin, mmax = _init_degrees(
        data, lmin, lmax, mmin, mmax, used_l, used_m
    )
    used_l, use_czero_coef, force_mass_conservation = _handle_mass_conservation(
        used_l, force_mass_conservation
    )
    longitude, latitude = _generate_grid(
        bounds,
        lonmin,
        lonmax,
        latmin,
        latmax,
        dlon,
        dlat,
        radians_in,
        longitude,
        latitude,
        radius,
    )

    geocentric_colat = latitude_to_geocentric_colatitude(
        latitude, ellipsoidal_earth=ellipsoidal_earth, **kwargs
    )

    # -- beginning of computation
    # Computing plm for converting to spatial domain
    if plm is None:
        plm = compute_plm(
            lmax,
            np.cos(geocentric_colat),
            latitude=latitude,
            mmax=mmax,
            normalization=normalization_plm,
            dtype=dtype_plm,
            use_dask=use_dask,
            chunks=chunks_plm,
        )

    else:
        _assert_plm(plm, lmax, latitude)

    # scale factor for each degree
    lfactor, cst = l_factor_conv(
        used_l,
        unit=unit,
        include_elastic=include_elastic,
        ellipsoidal_earth=ellipsoidal_earth,
        geocentric_colat=geocentric_colat,
        radius=radius,
        attrs=sub_data.attrs,
        use_dask=use_dask,
        chunks=chunks_lfactor,
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
        d_clm = (plm_lfactor * sub_data.clm).sum(dim="l")
        d_slm = (plm_lfactor * sub_data.slm).sum(dim="l")

        # Final calcul on the grid
        xgrid = c_cos.dot(d_clm, dim=["m"]) + s_sin.dot(d_slm, dim=["m"])

    else:
        d_clm = (plm_lfactor**2 * sub_data.clm**2).sum(dim="l")
        d_slm = (plm_lfactor**2 * sub_data.slm**2).sum(dim="l")

        # Final calcul of sigma on the grid
        xgrid = np.sqrt(
            (c_cos**2).dot(d_clm, dim=["m"]) + (s_sin**2).dot(d_slm, dim=["m"])
        )

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
                np.array([0]), unit=unit, attrs=sub_data.attrs, **kwargs
            )[0]
            xgrid = xgrid + (lfactor_zero * sub_data.clm.sel(l=0, m=0)).values

    xgrid = xgrid.transpose("latitude", "longitude", ...)

    xgrid.attrs = {"units": unit, "max_degree": int(lmax)}
    if "radius" in sub_data.attrs:
        xgrid.attrs["radius"] = sub_data.attrs["radius"]
    if "earth_gravity_constant" in sub_data.attrs:
        xgrid.attrs["earth_gravity_constant"] = sub_data.attrs["earth_gravity_constant"]

    return xgrid


def grid_to_sh(
    grid: xr.DataArray,
    lmax: int,
    unit: str = "mewh",
    mmax: int | None = None,
    lmin: int = 0,
    mmin: int = 0,
    used_l: np.ndarray | None = None,
    used_m: np.ndarray | None = None,
    ellipsoidal_earth: bool = False,
    include_elastic: bool = True,
    plm: xr.DataArray | None = None,
    normalization_plm: Literal["4pi", "ortho", "schmidt"] = "4pi",
    dtype_plm: type[complex] | type[float] = np.float128,
    use_dask: bool = False,
    chunks_lfactor: dict | None = None,
    chunks_plm: dict | None = None,
    **kwargs,
) -> xr.Dataset:
    """
    Transform gravity field spatial representation DataArray into Spherical Harmonics (SH) dataset.
    With choice for constants, unit of the spatial DataArray, love_numbers, degree/order, Earth hypothesis.

    For details on unit transformations, see :func:`l_factor_conv` and the
    associated `PDF document <../../_static/Mathematics_consideration_for_LENAPY.pdf>`_.

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
    dtype_plm : dtype, optional
        Specify the dtype to compute the plm DataArray. Default is np.float128.

    use_dask : bool, optional
        If True, use dask to chunk plm for memory optimization. Default is False.
    chunks_lfactor : dict, optional
        Define the chunking of lfactor when use_dask is True. Default is None, which set the chunking to {'l': 400}.
    chunks_plm : dict, optional
        Define the chunking of plm when use_dask is True. Default is None, which set the chunking to {'latitude': 4}.

    **kwargs :
        Supplementary parameters used by the function l_factor_conv to modify defaults constants used in the computation
        for the unit conversion. These parameters include (see :func:`l_factor_conv` documentation for more details) :
        a_earth, earth_gravity_constant, f_earth, omega_earth, rho_earth, ds_love

    Returns
    -------
    ds_out : xr.Dataset
        SH Dataset computed from the grid with the chosen unit.
    """
    # -- set degree and order default parameters
    # it prioritizes used_l and used_m if given over lmin, lmax, mmin and mmax
    mmax = lmax if mmax is None else mmax
    used_l = np.arange(lmin, lmax + 1) if used_l is None else used_l
    used_m = np.arange(mmin, mmax + 1) if used_m is None else used_m

    geocentric_colat = latitude_to_geocentric_colatitude(
        grid.cf["latitude"].values, ellipsoidal_earth=ellipsoidal_earth, **kwargs
    )

    # create DataArray corresponding to the integration factor for each cell
    # case for ellipsoidal earth where integration over ellipsoidal cell
    if ellipsoidal_earth:
        f_earth = kwargs["f_earth"] if "f_earth" in kwargs else LNPY_F_EARTH_GRS80
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
        use_dask=use_dask,
        chunks=chunks_lfactor,
        **kwargs,
    )

    # -- prepare variables for the computation of SH
    # Computing plm for converting to spatial domain
    if plm is None:
        plm = compute_plm(
            lmax,
            np.cos(geocentric_colat),
            latitude=grid.cf["latitude"],
            mmax=mmax,
            normalization=normalization_plm,
            dtype=dtype_plm,
            use_dask=use_dask,
            chunks=chunks_plm,
        )

    else:
        _assert_plm(plm, lmax, grid.cf["latitude"].values)

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


def _scale_plm_factors(
    lmax: int,
    normalization: Literal["4pi", "ortho", "schmidt"],
    derivative: bool,
    dtype: complex | float | type[complex] | type[float],
) -> tuple[np.ndarray, np.ndarray, float, float, Optional[np.ndarray]]:
    """
    Compute recurrence coefficients f1 and f2 for the Legendre recursion.
    Compute df recurrence coefficients for the derivative if requested.

    Operation used here are the developed and vectorize form of the SHTOOLS implementation
    to minimize the use of sqrt and division operations.

    Parameters
    ----------
    lmax : int
        Maximum degree.
    normalization : {'4pi', 'ortho', 'schmidt'}
        Normalization scheme.
    derivative : bool
        Whether to compute factors for the derivative of plm. Default is False.
    dtype : dtype
        Data type of the output arrays.

    Returns
    -------
    f1 : np.ndarray (lmax+1, lmax+1)
        Recurrence factor f1.
    f2 : np.ndarray (lmax+1, lmax+1)
        Recurrence factor f2.
    norm_p10 : float
        Normalization for P(1,0).
    norm_4pi : float
        Overall normalization factor.
    df : np.ndarray (lmax+1, lmax+1) | None
        Recurrence factor for the derivative of plm, only returned if derivative is True.
    """
    l = np.tile(np.arange(lmax + 1)[:, None], (1, lmax + 1))
    m = np.tile(np.arange(lmax + 1)[None, :], (lmax + 1, 1))
    mask = m < l - 1
    l_mask = l[mask]
    m_mask = m[mask]
    f1 = np.zeros((lmax + 1, lmax + 1), dtype=dtype)
    f2 = np.zeros((lmax + 1, lmax + 1), dtype=dtype)
    df = np.zeros((lmax + 1, lmax + 1), dtype=dtype) if derivative else None

    if normalization in ("4pi", "ortho"):
        # Normalization factors for P(1,0) and overall normalization factor
        norm_p10 = np.sqrt(3)
        norm_4pi = 1 if normalization == "4pi" else 4 * np.pi

        f1[mask] = np.sqrt(4 * l_mask**2 - 1) / np.sqrt(l_mask**2 - m_mask**2)
        f2[mask] = np.sqrt(
            2 * l_mask**3 - 3 * l_mask**2 - 2 * l_mask * m_mask**2 - m_mask**2 + 1
        ) / np.sqrt(
            2 * l_mask**3 + 3 * m_mask**2 - 2 * l_mask * m_mask**2 - 3 * l_mask**2
        )

        if derivative:
            pmask = mask & (m > 0)

            df[2:, 0] = np.sqrt(2 * np.arange(2, lmax + 1) + 1) / np.sqrt(
                2 * np.arange(2, lmax + 1) - 1
            )
            df[pmask] = np.sqrt(
                2 * l[pmask] ** 3
                + l[pmask] ** 2
                - 2 * l[pmask] * m[pmask] ** 2
                - m[pmask] ** 2
            ) / np.sqrt(2 * l[pmask] - 1)

    elif normalization == "schmidt":
        # Normalization factors for P(1,0) and overall normalization factor
        norm_p10 = 1
        norm_4pi = 1

        f1[mask] = (2 * l_mask - 1) / np.sqrt(l_mask**2 - m_mask**2)
        f2[mask] = np.sqrt(l_mask**2 - 2 * l_mask - m_mask**2 + 1) / np.sqrt(
            l_mask**2 - m_mask**2
        )

        if derivative:
            pmask = mask & (m > 0)

            df[2:, 0] = 1
            df[pmask] = np.sqrt(l[pmask] + m[pmask]) * np.sqrt(l[pmask] - m[pmask])

    else:
        raise ValueError(
            (
                f"Unknown normalization '{normalization}'. "
                "It should be either "
                "'4pi', 'ortho' or 'schmidt'"
            )
        )

    return f1, f2, norm_p10, norm_4pi, df


def _compute_plm_vector(
    z: np.ndarray,
    lmax: int,
    normalization: Literal["4pi", "ortho", "schmidt"],
    derivative: bool,
    dtype: complex | float | type[complex] | type[float],
    f1,
    f2,
    norm_p10,
    norm_4pi,
    df,
) -> np.ndarray:
    """
    Compute associated Legendre functions P(l,m) and optionally their derivatives as a array.

    Parameters
    ----------
    lmax : int
        Maximum degree of legrendre functions.
    z : np.ndarray
        Argument of the associated Legendre functions.
    normalization : {'4pi', 'ortho', 'schmidt'}
        '4pi', 'ortho', or 'schmidt' for use with geodesy 4pi normalized, orthonormalized, or Schmidt semi-normalized
        spherical harmonic functions, respectively.
    derivative : bool
        If True, compute the first derivative of the associated Legendre functions.
    dtype : dtype
        Data type of the output array.
    f1 : np.ndarray
        Recurrence factor f1.
    f2 : np.ndarray
        Recurrence factor f2.
    norm_p10 : float
        Normalization for P(1,0).
    norm_4pi : float
        Overall normalization factor.
    df : np.ndarray | None
        Recurrence factor for the derivative of plm, only returned if derivative is True.
    """
    plm = np.zeros((lmax + 1, lmax + 1, len(z)), dtype=dtype)
    dplm = np.zeros((lmax + 1, lmax + 1, len(z)), dtype=dtype) if derivative else None

    # scale factor based on Holmes2002, marginal effect
    scalef = 1e-280

    # u is sine of colatitude (cosine of latitude), for z=cos(th): u=sin(th)
    u = np.sqrt(1 - z**2)
    # update where u==0 to minimal numerical precision different from 0 to prevent invalid divisions
    u[u == 0] = np.finfo(dtype).eps

    # Calculate P(l,0) (not scaled)
    plm[0, 0] = 1 / np.sqrt(norm_4pi)
    if lmax:  # test for the case where lmax=0
        plm[1, 0] = norm_p10 * z / np.sqrt(norm_4pi)

    if derivative:
        dplm[0, 0] = 0
        if lmax:
            dplm[1, 0] = norm_p10 / np.sqrt(norm_4pi)

    # Iterative determination of the rest of P(l,0)
    for l in range(2, lmax + 1):
        plm[l, 0] = f1[l, 0] * z * plm[l - 1, 0] - f2[l, 0] * plm[l - 2, 0]

        if derivative:
            dplm[l, 0] = l * (df[l, 0] * plm[l - 1, 0] - z * plm[l, 0]) / u**2

    # Calculate P(m,m), P(m+1,m), and P(l,m)
    pmm = np.sqrt(2) * scalef / np.sqrt(norm_4pi)
    rescalem = (1 / scalef) * u[None, :] ** np.arange(lmax + 1)[:, None]

    for m in range(1, lmax + 1):
        # Compute P(m,m)
        pmm = pmm * np.sqrt(2 * m + 1) / np.sqrt(2 * m)
        if normalization in ("4pi", "ortho"):
            plm[m, m] = pmm

        elif normalization == "schmidt":
            plm[m, m] = pmm / np.sqrt(2 * m + 1)

        if derivative:
            dplm[m, m] = (-m * z * plm[m, m] / u**2) * rescalem[m]

        # Test if m = lmax to avoid computing P(m+1,m)) that do not exist
        if m < lmax:
            # Compute P(m+1,m)
            if normalization in ("4pi", "ortho"):
                plm[m + 1, m] = z * np.sqrt(2 * m + 3) * pmm

                if derivative:
                    dplm[m + 1, m] = (
                        (np.sqrt(2 * m + 3) * plm[m, m] - (m + 1) * z * plm[m + 1, m])
                        / u**2
                        * rescalem[m]
                    )

            elif normalization == "schmidt":
                plm[m + 1, m] = z * pmm

                if derivative:
                    dplm[m + 1, m] = (
                        (np.sqrt(2 * m + 1) * plm[m, m] - (m + 1) * z * plm[m + 1, m])
                        / u**2
                        * rescalem[m]
                    )

        # Test if m = lmax-1 or lmax to avoid computing P(m+2,m) that do not exist
        if m < lmax - 1:
            # Compute P(m+2, m) for all possible m
            plm[m + 2, 1 : m + 1] = (
                z[None, None, :] * f1[m + 2, 1 : m + 1, None] * plm[m + 1, 1 : m + 1]
                - f2[m + 2, 1 : m + 1, None] * plm[m, 1 : m + 1]
            )

            if derivative:
                dplm[m + 2, 1 : m + 1] = (
                    (
                        df[m + 2, 1 : m + 1, None] * plm[m + 1, 1 : m + 1]
                        - z[None, None, :] * (m + 2) * plm[m + 2, 1 : m + 1]
                    )
                    / u**2
                ) * rescalem[1 : m + 1]

            plm[m, 1 : m + 1] *= rescalem[1 : m + 1]

    # rescale for m=lmax-1 and m=lmax
    plm[-2, 1:] *= rescalem[1:]
    plm[-1, 1:] *= rescalem[1:]

    if derivative:
        return dplm
    else:
        return plm


def _compute_plm_ufunc(
    z, lmax, mmax, normalization, derivative, dtype, f1, f2, norm_p10, norm_4pi, df
):
    """
    Wrapper to use xr.apply_ufunc() to estimate plm with chunked latitudes.
    Compute associated Legendre functions P(l,m) and optionally their derivatives as a array.

    Parameters
    ----------
    see _compute_plm_vector Parameters list.
    """
    z = np.atleast_1d(np.asarray(z, dtype=dtype))

    plm = _compute_plm_vector(
        z, lmax, normalization, derivative, dtype, f1, f2, norm_p10, norm_4pi, df
    )

    # move axis to be consistent with xr.apply_ufunc dimension behavior
    return np.moveaxis(plm[:, : mmax + 1], -1, 0)


def compute_plm(
    lmax: int,
    z: np.ndarray,
    latitude: np.ndarray = None,
    mmax: int = None,
    normalization: Literal["4pi", "ortho", "schmidt"] = "4pi",
    derivative: bool = False,
    dtype: complex | float | type[complex] | type[float] = np.float128,
    use_dask: bool = False,
    chunks: dict | None = None,
) -> xr.DataArray:
    """
    Compute all the associated Legendre functions up to a maximum degree and
    order using the recursion relation from [Holmes2002]_, schematic recursion is in Fig. 2 of the article.
    (Adapted from SHTOOLS/pyshtools tools [Wieczorek2018]_)

    Parameters
    ----------
    lmax : int
        Maximum degree of legrendre functions.
    z : np.ndarray
        Argument of the associated Legendre functions, either cos of the colatitude or sin of the latitude.
    latitude : np.ndarray, optional
        Latitude values in degrees. Default is None and latitude is made from z.
    mmax : int or NoneType, optional
        Maximum order of associated legrendre functions.
    normalization : {'4pi', 'ortho', 'schmidt'}, optional
        '4pi', 'ortho', or 'schmidt' for use with geodesy 4pi normalized, orthonormalized, or Schmidt semi-normalized
        spherical harmonic functions, respectively. Default is '4pi'.
    derivative : bool, optional
        If True, compute the first derivative of the associated Legendre functions. Default is False.
    dtype : dtype, optional
        Data type of the output array. Default is np.float128.

    use_dask : bool, optional
        If True, use dask to chunk plm for memory optimization. Default is False.
    chunks : dict, optional
        Define the chunking of plm when use_dask is True. Default is None, which set the chunking to {'latitude': 4}.

    Returns
    -------
    plm : xr.DataArray
        Fully-normalized Legendre functions (or first derivative if derivative=True)
        as a DataArray with "l", "m" and "latitude" dimensions.

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
    if lmax > 2200:
        warnings.warn(
            "No validation made with lmax > 2200, the "
            "Legendre functions may be unstable near the poles because of IEEE754-2008 norm."
        )

    # update type to provide more memory for computation (np.float32 create some problems)
    z = np.atleast_1d(z).flatten().astype(dtype)

    # if default mmax, set mmax to be maximal degree
    mmax = lmax if mmax is None else mmax

    # if default latitude, set it from z
    latitude = z if latitude is None else latitude

    # Pregenerate scale factors for computation
    f1, f2, norm_p10, norm_4pi, df = _scale_plm_factors(
        lmax, normalization, derivative, dtype
    )

    # Call with vectorized function or call with xr.apply_ufunc for dask compatibility and memory optimization
    if not use_dask:
        plm = _compute_plm_vector(
            z, lmax, normalization, derivative, dtype, f1, f2, norm_p10, norm_4pi, df
        )

        plm_da = xr.DataArray(
            plm[:, : mmax + 1],
            dims=["l", "m", "latitude"],
            coords={
                "l": np.arange(lmax + 1),
                "m": np.arange(mmax + 1),
                "latitude": latitude,
            },
            name="plm",
        )

    else:
        z = xr.DataArray(
            z,
            dims=["latitude"],
            coords={"latitude": latitude if latitude is not None else z},
        )

        # Chunking plm for dask usage and memory optimization
        chunks = {"latitude": 4} if chunks is None else chunks
        z = z.chunk({"latitude": chunks["latitude"]})

        plm_da = xr.apply_ufunc(
            _compute_plm_ufunc,
            z,
            input_core_dims=[[]],
            output_core_dims=[["l", "m"]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[np.dtype(dtype)],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "l": lmax + 1,
                    "m": mmax + 1,
                }
            },
            kwargs={
                "lmax": lmax,
                "mmax": mmax,
                "normalization": normalization,
                "derivative": derivative,
                "dtype": dtype,
                "f1": f1,
                "f2": f2,
                "norm_p10": norm_p10,
                "norm_4pi": norm_4pi,
                "df": df,
            },
        )

        plm_da = (
            plm_da.assign_coords(l=np.arange(lmax + 1), m=np.arange(mmax + 1))
            .rename("plm")
            .transpose("l", "m", "latitude")
        )

        if "l" in chunks:
            plm_da = plm_da.chunk(chunks)

    # return the legendre polynomials and truncating orders to mmax
    return plm_da


def mid_month_grace_estimate(
    begin_time: datetime.datetime, end_time: datetime.datetime
) -> datetime.datetime:
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
    ds: xr.Dataset,
    new_normalization: Literal["4pi", "ortho", "schmidt"] = "4pi",
    old_normalization: Literal["4pi", "ortho", "schmidt"] | None = None,
    apply: bool = True,
) -> xr.Dataset:
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
        If True, apply the update to the current dataset without making a deep copy. Default is True.

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

    except KeyError:
        raise KeyError(
            "If you provide no information about the current normalization of your ds dataset using "
            "'old_normalization' parameters, those information need to be contained in ds.attrs dict as "
            "ds.attrs['norm']."
        )

    # -- conversion for each system
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

    if apply:
        ds_out = ds
    else:
        ds_out = ds.copy(deep=True)
        # need also clm and slm copy because .loc behave weirdly
        ds_out["clm"].data = ds_out["clm"].data.copy()
        ds_out["slm"].data = ds_out["slm"].data.copy()

    # Update the clm and slm values
    ds_out["clm"] *= update_factor
    ds_out["slm"] *= update_factor

    # Update the attributes in the output dataset
    ds_out.attrs["norm"] = new_normalization

    return ds_out


def _get_earth_parameters(
    attrs: dict | None, a_earth: float | None, earth_gravity_constant: float | None
) -> tuple[float, float]:
    """
    Retrieve Earth parameters from inputs or fallback constants.

    Parameters
    ----------
    attrs : dict or None
        Attributes potentially containing Earth parameters.
    a_earth : float or None
        Semi-major axis.
    earth_gravity_constant : float or None
        Gravitational constant.

    Returns
    -------
    a_earth : float
        Resolved semi-major axis.
    earth_gravity_constant : float
        Resolved gravitational constant.
    """
    if attrs is None:
        attrs = {}

    # return first element if not None, otherwise return value for the attrs.key if it exists otherwise return last term
    resolved_a = a_earth or float(attrs.get("radius", LNPY_A_EARTH_GRS80))
    resolved_gm = earth_gravity_constant or float(
        attrs.get("earth_gravity_constant", LNPY_GM_EARTH)
    )
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


def _prefix_parser(unit: str) -> float:
    """
    Parse a SI-style prefix from a unit string and return its numeric multiplier.

    Parameters
    ----------
    unit : str
        Unit name that may include a prefix. The function accepts common ASCII and Unicode forms for the micro prefix
        (both 'u' and 'µ') and is case\-sensitive where relevant (e.g. 'M' for mega vs 'm' for milli).

        Unit cannot be defined using 'm' for milli because it is used for meter unit,
        so 'milli' or 'mm' should be used instead. Same for 'p' for pico, 'pico' should be used instead.

        Goes from 'giga' / 'G' to 'pico', including 'mega', 'kilo', 'hecto', 'deca', 'deci', 'centi', 'milli',
        'micro', and 'nano'.

    Returns
    -------
    scale : float
        Multiplicative factor corresponding to the detected prefix.
        If no recognized prefix is found, returns 1.0 (indicating no scaling).
    """
    prefixes = {
        ("G", "giga"): 1e-9,
        ("M", "mega"): 1e-6,
        ("k", "kilo"): 1e-3,
        ("h", "hecto"): 1e-2,
        ("deca",): 1e-1,
        ("d", "deci"): 1e1,
        ("c", "centi"): 1e2,
        ("milli", "mm"): 1e3,  # cannot accept 'm' because it is used for meter unit
        ("µ", "u", "micro"): 1e6,
        ("nano",): 1e9,  # cannot accept 'n' because it is used for norm unit
        ("pico",): 1e12,  # cannot accept 'p' because it is used for pressure unit
    }

    for prefix, multiplier in prefixes.items():
        if unit.startswith(prefix):
            return multiplier

    return 1.0


def _compute_l_factor(
    l: np.ndarray | xr.DataArray,
    unit: str,
    geocentric_colat: xr.DataArray | None,
    ds_love: xr.Dataset | None,
    a_earth: float,
    earth_gravity_constant: float,
    f_earth: float,
    omega_earth: float,
    rho_earth: float,
    rho_water: float,
    fraction: xr.DataArray,
    a_div_r: np.ndarray | xr.DataArray | None,
) -> xr.DataArray:
    """
    Compute the degree-dependent scale factor.

    Parameters
    ----------
    l: np.ndarray or xr.DataArray
        Degrees
    unit: str
        Unit type for conversion ending with the unit name
    geocentric_colat : xr.DataArray
        Geocentric colatitude with a latitude dimension in degree
    ds_love: xr.Dataset, optional
        Love numbers
    a_earth: float
        Earth radius
    earth_gravity_constant: float
        Earth gravitational constant
    f_earth : float
        Earth flattening
    omega_earth : float, optional
        Earth's rotation rate
    rho_earth: float
        Earth density
    rho_water: float
        Water density
    fraction: xr.DataArray
        Redistribution factor
    a_div_r: np.ndarray | xr.DataArray | None
        Correction factor (used for ellipsoidal correction) corresponding to a_earth divided by radius of the grid.

    Returns
    -------
    l_factor : xr.DataArray
        Degree-dependent conversion factor.
    """
    # l_factor is degree dependant
    if unit.endswith(("norm",)):
        # norm, fully normalized spherical harmonics
        l_factor = xr.ones_like(l)
        if a_div_r is not None:
            l_factor = l_factor * a_div_r**l

    elif unit.endswith(
        ("ewh", "equivalent_water_height", "lwe", "lwe_thickness", "water_column")
    ):
        # equivalent water height [kg.m⁻²]
        # the exact formula is l_factor*(1 - f) (see [Ditmar2018]_ after eq. 17)
        # it is an approximation of the order of 0.3% to be coherent with the common formula from Wahr 1998
        l_factor = rho_earth * a_earth * (2 * l + 1) / (3 * fraction * rho_water)
        if a_div_r is not None:
            l_factor = l_factor * a_div_r ** (l + 2)

    elif unit.endswith(("geoid", "geoid_height")):
        # meter of geoid height [m]
        if a_div_r is not None:
            from lenapy.utils.gravity import estimate_normal_gravity

            gamma_0 = estimate_normal_gravity(
                latitude=geocentric_colat.latitude,
                a_earth=a_earth,
                earth_gravity_constant=earth_gravity_constant,
                f_earth=f_earth,
                omega_earth=omega_earth,
            )
            l_factor = earth_gravity_constant / a_earth / gamma_0 * a_div_r ** (l + 1)

        else:
            # Simplification of the formula for the spherical case
            l_factor = xr.ones_like(l) * a_earth

    elif unit.endswith(("gravity", "potential_gradient")):
        # gravity perturbations [m.s⁻²]
        l_factor = earth_gravity_constant * (l + 1) / (a_earth**2)
        if a_div_r is not None:
            l_factor = l_factor * a_div_r ** (l + 2)

    elif unit.endswith(("Gal", "gal", "galileo")):
        # Gal, Gal gravity perturbations [m.s⁻²]
        l_factor = earth_gravity_constant * (l + 1) / (a_earth**2) * 1e2
        if a_div_r is not None:
            l_factor = l_factor * a_div_r ** (l + 2)

    elif unit.endswith(("potential",)):
        # potential, [m².s⁻²]
        l_factor = earth_gravity_constant / a_earth
        if a_div_r is not None:
            l_factor = l_factor * a_div_r ** (l + 1)

    elif unit.endswith(("pascal",)):
        # pascal, equivalent surface pressure
        l_factor = LNPY_G_WMO * rho_earth * a_earth * (2 * l + 1) / (3 * fraction)
        if a_div_r is not None:
            l_factor = l_factor * a_div_r ** (l + 1)

    elif unit.endswith(("bar",)):
        # bar, equivalent surface pressure
        l_factor = (
            LNPY_G_WMO * rho_earth * a_earth * (2 * l + 1) * 1e-5 / (3 * fraction)
        )
        if a_div_r is not None:
            l_factor = l_factor * a_div_r ** (l + 1)

    elif unit.endswith(("vcu",)):
        # vcu, meters viscoelastic crustal uplift
        l_factor = a_earth * (2 * l + 1) / 2
        if a_div_r is not None:
            l_factor = l_factor * a_div_r ** (l + 1)

    elif unit.endswith(("ecu",)):
        # ecu, meters elastic crustal deformation (uplift)
        l_factor = a_earth * ds_love.hl / fraction
        if a_div_r is not None:
            l_factor = l_factor * a_div_r ** (l - 1)

    # Possible to add magnetic field unit in the future
    else:
        raise ValueError(
            "Invalid unit for l_factor conversion. See documentation or code for accepted units."
        )
    return l_factor


def l_factor_conv(
    l: np.ndarray,
    unit: str = "mewh",
    include_elastic: bool = True,
    ellipsoidal_earth: bool = False,
    geocentric_colat: xr.DataArray | None = None,
    radius: xr.DataArray | None = None,
    ds_love: xr.Dataset | None = None,
    a_earth: float | None = None,
    earth_gravity_constant: float | None = None,
    f_earth: float = LNPY_F_EARTH_GRS80,
    omega_earth: float = LNPY_OMEGA_EARTH_GRS80,
    rho_earth: float = LNPY_RHO_EARTH,
    rho_water: float = 1000,
    attrs: dict | None = None,
    use_dask: bool = False,
    chunks: dict | None = None,
) -> tuple[xr.DataArray, dict]:
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
        Unit of the spatial data used in the transformation. Default is 'mewh' for meters of Equivalent Water Height.
        str composed of a prefix (optional) and a unit type.
        The unit prefix parser function accepts common ASCII and Unicode forms for the micro prefix
        (both 'u' and 'µ') and is case\-sensitive where relevant (e.g. 'M' for mega vs 'm' for milli).
        Unit cannot be defined using 'm' for milli because it is used for meter unit,
        so 'milli' or 'mm' should be used instead. Same for 'p' for pico, 'pico' should be used instead.
        Goes from 'giga' / 'G' to 'pico', including 'mega', 'kilo', 'hecto', 'deca', 'deci', 'centi', 'milli',
        'micro', and 'nano'.
        The unit type can be usual gravity related units such as 'geoid' for geoid height in meter,
        'gravity' for gravity perturbations in m.s⁻², 'Gal' for gravity perturbations in Gal,
        'potential' for potential in m².s⁻², 'pascal' for equivalent surface pressure,
        'vcu' for viscoelastic crustal uplift in meter and 'ecu' for elastic crustal deformation (uplift) in meter.
        It can also be 'ewh', 'equivalent_water_height', 'lwe', 'lwe_thickness' or 'water_column' for equivalent water
        height in kg.m⁻².
    include_elastic : bool, optional
        If True, the Earth behavior is elastic.
    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid following [Ditmar2018]_ and if False as a sphere.
    geocentric_colat : xr.DataArray | None, optional
        Geocentric colatitude for ellipsoidal Earth radius computation in radians,
        the dimension is geographic latitude in degree.
    radius : xr.DataArray | None, optional
        DataArray with the radius of the grid to compute. If not given, the radius is the surface of reference.

    ds_love : xr.Dataset | None, optional
        Dataset with the l dimension corresponding to degree and with l (and possibly h and k) variables that
        are Love numbers.
        Default Love numbers used are from Gegout97.
    a_earth : float, optional
        Earth semi-major axis [m]. If not provided, uses `data.attrs['radius']` and
        if it does not exist, uses LNPY_A_EARTH_GRS80.
    earth_gravity_constant : float, optional
        Standard gravitational parameter for Earth [m³.s⁻²]. If not provided, uses
        `data.attrs['earth_gravity_constant']` and if it does not exist, uses LNPY_GM_EARTH.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80.
    omega_earth : float, optional
        Earth's rotation rate [rad.s⁻¹]. Default is LNPY_OMEGA_EARTH.
    rho_earth : float, optional
        Earth density [kg.m⁻³] for Equivalent Water Height formula. Default is LNPY_RHO_EARTH.
    rho_water : float, optional
        Water density [kg.m⁻³] for Equivalent Water Height formula. Default is 1000 Kg.m⁻³.
    attrs : dict | None, optional
        ds.attrs information that might help to estimate l_factor if no parameters are given.

    use_dask : bool, optional
        If True, use dask to chunk lfactor for memory optimization. Default is False.
    chunks : dict, optional
        Define the chunking of lfactor when use_dask is True. Default is None, which set the chunking to {'l': 200}.

    Returns
    -------
    l_factor : xr.DataArray
        Degree-dependent scale factor.
    cst: dict
        Attributes for the final dataset

    References
    ----------
    .. [Ditmar2018] P. Ditmar,
        "Conversion of time-varying Stokes coefficients into mass anomalies at the
        Earth’s surface considering the Earth’s oblateness",
        *Journal of Geodesy*, 92, 1401--1412, (2018).
        `doi: 10.1007/s00190-018-1128-0 <https://doi.org/10.1007/s00190-018-1128-0>`_
    """
    a_earth, earth_gravity_constant = _get_earth_parameters(
        attrs, a_earth, earth_gravity_constant
    )

    l = xr.DataArray(l, dims=["l"], coords={"l": l})
    fraction = xr.ones_like(l)

    # include elastic redistribution with kl Love numbers
    if include_elastic:
        if ds_love is None:
            ds_love = load_default_love_numbers()
        fraction = fraction + ds_love.kl

    a_div_r = None
    if ellipsoidal_earth and radius is None:
        # test if geocentric_colat is set
        if geocentric_colat is None:
            raise ValueError(
                "For ellipsoidal Earth, you need to set "
                "the parameter 'geocentric_colat' in l_factor_conv function"
            )

        # e = sqrt(2f - f**2)
        e_earth_square = 2 * f_earth - f_earth**2
        # a_div_r_lat = a / r(theta)  with r(theta) = a(1-f)/sqrt(1 - e**2*sin(theta)**2)
        a_div_r = np.sqrt(1 - e_earth_square * np.sin(geocentric_colat) ** 2) / (
            1 - f_earth
        )

    elif radius is not None:
        a_div_r = a_earth / radius

    l_factor = _compute_l_factor(
        l,
        unit,
        geocentric_colat,
        ds_love,
        a_earth,
        earth_gravity_constant,
        f_earth,
        omega_earth,
        rho_earth,
        rho_water,
        fraction,
        a_div_r,
    )

    scale = _prefix_parser(unit)

    l_factor_scale = scale * l_factor

    # Chunking plm for dask usage and memory optimization
    if use_dask and type(l_factor_scale) is not float:
        if chunks is None:
            chunks = {"l": 200}
        l_factor_scale = l_factor_scale.chunk(chunks)

    cst = {"earth_gravity_constant": earth_gravity_constant, "a_earth": a_earth}
    return l_factor_scale, cst


def _assert_plm(plm: xr.DataArray, lmax: int, latitude: np.ndarray) -> bool:
    """
    Verify if the dataset plm is valid for spherical harmonics computations.
    Raise Type error or Assertion error if not.

    Parameters
    ----------
    plm : xr.DataArray
        DataArray to verify.
    lmax : int
        Maximum degree of the Legendre polynomials.
    latitude : np.ndarray
        Latitude values in degrees.

    Returns
    -------
    True : bool
        Returns True if the dataset ds has dimensions 'l' and 'm' as well as variables 'clm' and 'slm'.

    Raises
    ------
    TypeError
        This function raise TypeError if plm is not a xr.DataArray with 3 coordinates [l, m, latitude]
    AssertionError
        This function raise AssertionError if ds is not a xr.Dataset corresponding to spherical harmonics
    """
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
    return True


def assert_sh(ds: xr.Dataset) -> bool:
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
        This function raise AssertionError if ds is not a xr.Dataset corresponding to spherical harmonics
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
