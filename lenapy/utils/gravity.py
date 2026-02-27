"""
The **gravity** module provides functions for manipulating spherical harmonics datasets with respect to different
reference frames, tide systems, normal gravity reference and smoothing techniques.

This module includes functions to:
  * Change the reference frame of spherical harmonics datasets.
  * Change the tide system of spherical harmonics datasets.
  * Modify degree 1 Love numbers for different reference frames.
  * Estimate normal gravity values.
  * Generate Gaussian weights for smoothing spherical harmonics datasets.
  * Estimate time variable gravity field from a .gfc temporal file.

Examples
--------
>>> import xarray as xr
>>> from lenapy.utils.gravity import *
# Load a dataset
>>> ds = xr.open_dataset('example_file.nc')
# Change the reference frame of the dataset
>>> ds_new_ref = change_reference(ds, new_radius=6378137.0, new_earth_gravity_constant=3.986004418e14)
# Change the tide system of the dataset
>>> ds_new_tide_system = change_tide_system(ds, new_tide='tide_free')
# Generate Gaussian weights for smoothing spherical harmonics datasets.
>>> ds_gauss_weights = gauss_weights(radius=100000, lmax=60)
# Filter the dataset with Gaussian weights
>>> ds['clm'] *= ds_gauss_weights
>>> ds['slm'] *= ds_gauss_weights
"""

import warnings
from typing import Literal

import numpy as np
import xarray as xr

from lenapy.constants import *
from lenapy.utils.geo import assert_latitude, latitude_to_geocentric_colatitude
from lenapy.utils.harmo import (
    _assert_plm,
    _generate_grid,
    _get_earth_parameters,
    compute_plm,
    l_factor_conv,
)


def change_reference(
    ds: xr.Dataset,
    new_radius: float = LNPY_A_EARTH_GRS80,
    new_earth_gravity_constant: float = LNPY_GM_EARTH,
    old_radius: float | None = None,
    old_earth_gravity_constant: float | None = None,
    apply: bool = True,
) -> xr.Dataset:
    """
    Spherical Harmonics dataset are associated with an earth radius *a* and *µ* or *GM* the earth gravity constant.
    This function return the dataset updated with the new constants associated to it in input
    (as a deep copy by default). The current reference frame constants can be given in parameters or can be contained in
    ds.attrs['radius'] and ds.attrs['earth_gravity_constant'].

    Warning ! This function must be applied before removing the mean values of the dataset through time.

    Parameters
    ----------
    ds : xr.Dataset
        xr.Dataset that corresponds to SH data associated with a current reference frame (constants whise) to update.
    new_radius : float
        New Earth radius constant in meters. Default is LNPY_A_EARTH_GRS80 define in constants.py
    new_earth_gravity_constant : float
        New gravitational constant of the Earth in m³/s². Default is LNPY_GM_EARTH define in constants.py
    old_radius : float | None, optional
        Current Earth radius constant of the dataset ds in meters. If not provided, uses `ds.attrs['radius']`.
    old_earth_gravity_constant : float | None, optional
        Current gravitational constant of the Earth of the dataset ds in m³/s².
        If not provided, uses `ds.attrs['earth_gravity_constant']`.
    apply : bool, optional
        If True, apply the update to the current dataset without making a deep copy. Default is True.

    Returns
    -------
    ds_out : xr.Dataset
        Updated dataset with the new constants.

    Raises
    ------
    KeyError
        If the current reference frame constants are not provided and not found in the dataset attributes.

    Examples
    --------
    >>> ds_new_ref = change_reference(ds, new_radius=6378137.0, new_earth_gravity_constant=3.986004418e14)
    """
    try:
        old_radius = ds.attrs["radius"] if old_radius is None else old_radius
        old_earth_gravity_constant = (
            ds.attrs["earth_gravity_constant"]
            if old_earth_gravity_constant is None
            else old_earth_gravity_constant
        )

    except KeyError:
        raise KeyError(
            "If you provide no information about the current reference constants of your ds dataset using "
            "'old_radius' and 'old_earth_gravity_constant' parameters, those information need to be "
            "contained in ds.attrs dict as ds.attrs['radius'] and ds.attrs['earth_gravity_constant']."
        )

    gravity_constant_ratio = old_earth_gravity_constant / new_earth_gravity_constant
    update_factor = gravity_constant_ratio * (old_radius / new_radius) ** ds.l

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
    ds_out.attrs["radius"] = new_radius
    ds_out.attrs["earth_gravity_constant"] = new_earth_gravity_constant

    return ds_out


def change_tide_system(
    ds: xr.Dataset,
    new_tide: Literal["tide_free", "zero_tide", "mean_tide"],
    old_tide: Literal["tide_free", "zero_tide", "mean_tide"] | None = None,
    k20: float | None = None,
    apply: bool = True,
) -> xr.Dataset:
    """
    Apply a C20 offset to the dataset to change the tide system.
    Follows [IERS2010]_ convention to convert between tide system ('tide_free', 'zero_tide', 'mean_tide').
    Warning : It should be noted that each center use his one tide offset, it may differ in terms of offset formula
    that can come from [IERS2010]_ and [Smith1998]_. The value of k20 can also change between centers.

    Parameters
    ---------
    ds : xr.Dataset
        xr.Dataset that corresponds to SH data associated with a current tide system to update.
    new_tide : str
        Output tidal system, either 'tide_free', 'zero_tide' or 'mean_tide'.
    old_tide : str | None, optional
        Input tidal system. If not provided, uses `ds.attrs['tide_system']`.
    k20 : float | None, optional
        k20 Earth tide external potential Love numbers. If not provided, the default value from [IERS2010]_ is used.
    apply : bool, optional
        If True, apply the update to the current dataset without making a deep copy. Default is True.

    Returns
    -------
    ds_out : xr.Dataset
        Updated dataset with the new tidal system. Warning, the dataset is also updated in place if copy=False (default)

    Raises
    ------
    ValueError | KeyError
        If the input tidal system is not provided and not found in the dataset attributes.

    Examples
    --------
    >>> ds_new_tide_system = change_tide_system(ds, new_tide='tide_free')

    References
    ----------
    .. [IERS2010] G. Petit and B. Luzum,
        "IERS Conventions (2010)",
        *IERS Technical Note*, 36, 88--89, (2010).
        `doi: 10.1007/s00190-002-0216-2 <https://doi.org/10.1007/s00190-002-0216-2>`_
    .. [Smith1998]  D. A. Smith,
        "There is no such thing as 'The' EGM96 geoid: Subtle points on the use of a global geopotential model",
        *IGeS Bulletin, International Geoid Service*, 8, 17-28, (1998).
        `link to archive <https://www.ngs.noaa.gov/PUBS_LIB/EGM96_GEOID_PAPER/egm96_geoid_paper.html>`_
    """
    if old_tide is None:
        if "tide_system" in ds.attrs:
            old_tide = ds.attrs["tide_system"]
            if ds.attrs["tide_system"] == "missing":
                raise ValueError(
                    "No information about tide in ds.attrs['tide_system'], the info is 'missing'. "
                    "You need to provide 'old_tide' param"
                )
        else:
            raise KeyError(
                "No information ds.attrs['tide_system'] in dataset, you need to provide 'old_tide' param"
            )

    # -- Define IERS 2010 constants
    A0 = 4.4228e-8
    H0 = -0.31460
    if k20 is None:
        k20 = 0.30190

    # -- conversion for each tidal system
    if new_tide == "zero_tide" and "mean" in old_tide:
        conv = -1
    elif new_tide == "mean_tide" and ("zero" in old_tide or "inclusive" in old_tide):
        conv = 1
    elif new_tide == "tide_free" and "mean" in old_tide:
        conv = -(1 + k20)
    elif new_tide == "mean_tide" and "free" in old_tide:
        conv = 1 + k20
    elif new_tide == "tide_free" and ("zero" in old_tide or "inclusive" in old_tide):
        conv = -k20
    elif new_tide == "zero_tide" and ("free" in old_tide or "exclusive" in old_tide):
        conv = k20
    else:
        conv = 0

    if apply:
        ds_out = ds
    else:
        ds_out = ds.copy(deep=True)
        # need also clm copy because .loc behave weirdly
        ds_out["clm"].data = ds_out["clm"].data.copy()

    ds_out.clm.loc[dict(l=2, m=0)] += conv * A0 * H0
    ds_out.attrs["tide_system"] = new_tide

    # -- return ds
    return ds_out


def change_love_reference_frame(
    ds: xr.Dataset,
    new_frame: Literal["CM", "CE", "CF", "CL", "CH"],
    old_frame: Literal["CM", "CE", "CF", "CL", "CH"],
    apply: bool = True,
) -> xr.Dataset:
    """
    Modify degree 1 love numbers of the dataset to change the reference frame.
    The input dataset need to contain 'hl', 'll', 'kl' variables that are potential love numbers with a degree dimension
    named 'l' with at least coordinate l=1.

    This function follows [Blewitt2003]_ to convert between reference frames.
    Reference frames can be :
    * 'CM' for Center of Mass of the Earth System
    * 'CE' for Center of Mass of the Solid Earth
    * 'CF' for Center of Surface Figure
    * 'CL' for Center of Surface Lateral Figure
    * 'CH' for Center of Surface Height Figure

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'hl', 'll', 'kl' variables and 'l' dimension.
    new_frame : str
        Reference frame of the output dataset. Either 'CM', 'CE', 'CF', 'CL' or 'CH'.
    old_frame : str
        Reference frame of the input dataset. Either 'CM', 'CE', 'CF', 'CL' or 'CH'.
    apply : bool
        If True, apply the update to the current dataset without making a deep copy. Default is True.

    Returns
    -------
    ds_out : xr.Dataset
        Dataset with the updated degree 1 love numbers.
        Warning, the dataset is also updated in place if copy=False (default)

    References
    ----------
    .. [Blewitt2003] G. Blewitt, "Self-consistency in reference frames, geocenter definition,
        and surface loading of the solid Earth",
        *Journal of Geophysical Research: Solid Earth*, 108, 2103, (2003).
        doi: 10.1029/2002JB002082 <https://doi.org/10.1029/2002JB002082>`_
    """
    if apply:
        ds_love = ds
    else:
        ds_love = ds.copy(deep=True)
        # need also var copy because .loc behave weirdly
        for var in ["hl", "ll", "kl"]:
            ds_love[var].data = ds_love[var].data.copy()

    # compute k1, h1 and l1 into CE
    if old_frame == "CE":
        k1 = ds_love.kl.sel(l=1).values
        h1 = ds_love.hl.sel(l=1).values
        l1 = ds_love.ll.sel(l=1).values
    elif old_frame == "CM":
        k1 = ds_love.kl.sel(l=1).values + 1
        h1 = ds_love.hl.sel(l=1).values + 1
        l1 = ds_love.ll.sel(l=1).values + 1
    elif old_frame == "CF":
        k1 = 0
        h1 = ds_love.hl.sel(l=1).values + ds_love.kl.sel(l=1).values
        l1 = -ds_love.hl.sel(l=1).values / 2 - ds_love.kl.sel(l=1).values
    elif old_frame == "CL":
        k1 = 0
        h1 = ds_love.hl.sel(l=1).values - ds_love.kl.sel(l=1).values
        l1 = -ds_love.kl.sel(l=1).values
    elif old_frame == "CH":
        k1 = 0
        h1 = -ds_love.kl.sel(l=1).values
        l1 = ds_love.ll.sel(l=1).values - ds_love.kl.sel(l=1).values
    else:
        raise ValueError

    # compute new_frame degree 1 love numbers from CE
    if new_frame == "CE":
        ds_love.kl.loc[{"l": 1}] = k1
        ds_love.hl.loc[{"l": 1}] = h1
        ds_love.ll.loc[{"l": 1}] = l1
    elif new_frame == "CM":
        ds_love.kl.loc[{"l": 1}] = k1 - 1
        ds_love.hl.loc[{"l": 1}] = h1 - 1
        ds_love.ll.loc[{"l": 1}] = l1 - 1
    elif new_frame == "CF":
        ds_love.kl.loc[{"l": 1}] = -h1 / 3 - 2 * l1 / 3
        ds_love.hl.loc[{"l": 1}] = 2 * (h1 - l1) / 3
        ds_love.ll.loc[{"l": 1}] = (l1 - h1) / 3
    elif new_frame == "CL":
        ds_love.kl.loc[{"l": 1}] = -l1
        ds_love.hl.loc[{"l": 1}] = h1 - l1
        ds_love.ll.loc[{"l": 1}] = 0
    elif new_frame == "CH":
        ds_love.kl.loc[{"l": 1}] = -h1
        ds_love.hl.loc[{"l": 1}] = 0
        ds_love.ll.loc[{"l": 1}] = l1 - h1
    else:
        raise ValueError

    return ds_love


def estimate_normal_gravity(
    geographic_latitude: xr.DataArray | np.ndarray | None = None,
    a_earth: float | None = LNPY_A_EARTH_GRS80,
    earth_gravity_constant: float | None = LNPY_GM_EARTH,
    f_earth: float = LNPY_F_EARTH_GRS80,
    omega_earth: float = LNPY_OMEGA_EARTH_GRS80,
) -> xr.DataArray:
    """
    Estimate normal acceleration with the Somigliana equation at given geographic latitude
    for a group of Earth's parameters that represent the ellipsoid.

    Parameters
    ----------
    geographic_latitude: xr.DataArray | np.ndarray, optional
        Geographic / geodetic latitude for ellipsoidal Earth radius computation in radians.
    a_earth : float, optional
        Earth semi-major axis [m]. Default is LNPY_A_EARTH_GRS80.
    earth_gravity_constant : float, optional
        Standard gravitational parameter for Earth [m³.s⁻²]. Default is LNPY_GM_EARTH.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80.
    omega_earth : float, optional
        Earth's rotation rate [rad.s⁻¹]. Default is LNPY_OMEGA_EARTH.

    Returns
    -------
    gamma_0: xr.DataArray | np.ndarray
        Normal gravity at given geographic colatitude.
    """
    e_prime = np.sqrt(2 * f_earth - f_earth**2) / (1 - f_earth)
    q0 = (0.5 + 1.5 / e_prime**2) * np.arctan(e_prime) - 1.5 / e_prime
    q0_prime = 3 * (1 + 1 / e_prime**2) * (1 - 1 / e_prime * np.arctan(e_prime)) - 1
    m = omega_earth**2 * a_earth**3 * (1 - f_earth) / earth_gravity_constant

    gamma_e = (
        earth_gravity_constant
        / a_earth**2
        / (1 - f_earth)
        * (1 - m - m * e_prime * q0_prime / 6 / q0)
    )
    gamma_p = (
        earth_gravity_constant / a_earth**2 * (1 + m * e_prime * q0_prime / 3 / q0)
    )

    return (
        gamma_e * np.cos(geographic_latitude) ** 2
        + (1 - f_earth) * gamma_p * np.sin(geographic_latitude) ** 2
    ) / np.sqrt(
        np.cos(geographic_latitude) ** 2
        + (1 - f_earth) ** 2 * np.sin(geographic_latitude) ** 2
    )


def normal_zonal_correction(
    ds: xr.Dataset,
    a_earth: float | None = None,
    earth_gravity_constant: float | None = None,
    f_earth: float = LNPY_F_EARTH_GRS80,
    omega_earth: float = LNPY_OMEGA_EARTH_GRS80,
    reverse: bool = False,
    apply: bool = True,
) -> xr.Dataset:
    """
    Normal gravity field correction on zonal coefficients on a SH dataset for a specified ellipsoid.

    Parameters
    ----------
    ds : xr.Dataset
        xr.Dataset that corresponds to SH data to be correct for the normal gravity field.
    a_earth : float | None, optional
        Earth radius constant of the dataset ds in meters. If not provided, uses `ds.attrs['radius']`.
    earth_gravity_constant : float | None, optional
        Current gravitational constant of the Earth of the dataset ds in m³/s².
        If not provided, uses `ds.attrs['earth_gravity_constant']`.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80.
    omega_earth : float, optional
        Earth rotation rate. Default is LNPY_OMEGA_EARTH_GRS80.
    reverse : bool, optional
        False to apply the correction, True to remove the correction.
    apply : bool, optional
        If True, apply the update to the current dataset without making a deep copy. Default is True.

    Returns
    -------
    ds_out : xr.Dataset
        Updated dataset with the correction.

    Raises
    ------
    KeyError
        If the current reference frame constants are not provided and not found in the dataset attributes.

    """
    try:
        a_earth = ds.attrs["radius"] if a_earth is None else a_earth
        earth_gravity_constant = (
            ds.attrs["earth_gravity_constant"]
            if earth_gravity_constant is None
            else earth_gravity_constant
        )

    except KeyError:
        raise KeyError(
            "If you provide no information about the current reference constants of your ds dataset using "
            "'radius' and 'earth_gravity_constant' parameters, those information need to be "
            "contained in ds.attrs dict as ds.attrs['radius'] and ds.attrs['earth_gravity_constant']."
        )

    if apply:
        ds_out = ds
    else:
        ds_out = ds.copy(deep=True)
        # need also clm copy because .loc behave weirdly
        ds_out["clm"].data = ds_out["clm"].data.copy()

    l = ds.sel(l=slice(0, None, 2)).l
    if f_earth == 0:
        correction = -(l == 0).astype(int)

    else:
        e_prime = np.sqrt(2 * f_earth - f_earth**2) / (1 - f_earth)
        q0 = (0.5 + 1.5 / e_prime**2) * np.arctan(e_prime) - 1.5 / e_prime

        k = 1 / 3 - (
            2 * omega_earth**2 * a_earth**3 * np.sqrt(2 * f_earth - f_earth**2)
        ) / (45 * earth_gravity_constant * q0)

        correction = (
            (-1) ** (l // 2 + 1)
            * 3
            * np.sqrt(2 * f_earth - f_earth**2) ** l
            * (1 + l / 2 * (5 * k - 1))
            / ((l**2 + 4 * l + 3) * np.sqrt(2 * l + 1))
        )

    sign = -1 if reverse else 1

    ds_out.clm.loc[dict(l=slice(0, None, 2), m=0)] = (
        ds_out.clm.sel(m=0) + sign * correction
    )

    return ds_out


def sh_to_gravity_disturbance(
    data: xr.Dataset,
    surface: Literal[None, "reference", "geoid"] | float | xr.DataArray = None,
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
    ellipsoidal_earth: bool = False,
    normalization_plm: Literal["4pi", "ortho", "schmidt"] = "4pi",
    dtype_plm: type[complex] | type[float] = np.float128,
    use_dask: bool = False,
    chunks_lfactor: dict | None = None,
    chunks_plm: dict | None = None,
    test=True,
    **kwargs,
) -> xr.DataArray:
    """
    Transform Spherical Harmonics (SH) dataset into a grid corresponding to the gravity disturbance.
    This grid can be the reference surface (sphere or ellipsoid) or another surface.
    With choice for constants, unit, love numbers, degree/order, spatial grid latitude and longitude, Earth hypothesis.

    Parameters
    ----------
    data : xr.Dataset
        xr.Dataset that corresponds to SH data to convert into spatial representation.
    surface : float | xr.DataArray | Literal[None, "reference", "geoid"], optional
        Surface to consider for the computation of the grid. If None or "reference", consider the reference sphere or
        ellipsoid. If "geoid", consider the geoid as surface computed from the given field.
        If a float or xr.DataArray is given, consider it as the surface height in meter to add to the reference surface.

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

    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid following [Ditmar2018]. Default is False for a spherical Earth.

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
        a_earth, gm_earth, f_earth, omega_earth

    Returns
    -------
    xgrid : xr.DataArray
        Spatial representation of the SH Dataset in gravity disturbance.
    """
    # Get parameters for computation of the grid and for the unit conversion
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
    )

    data_null = xr.zeros_like(data)
    normal_field = normal_zonal_correction(data_null, reverse=True, **kwargs)

    geoc_colat = latitude_to_geocentric_colatitude(
        latitude, ellipsoidal_earth=ellipsoidal_earth, **kwargs
    )

    # Precomputing plm and dplm for converting to spatial domain
    plm = compute_plm(
        data.l.max().values,
        np.cos(geoc_colat),
        latitude=latitude,
        mmax=data.m.max().values,
        normalization=normalization_plm,
        dtype=dtype_plm,
        use_dask=use_dask,
        chunks=chunks_plm,
    )
    dplm = compute_plm(
        data.l.max().values,
        np.cos(geoc_colat),
        latitude=latitude,
        mmax=data.m.max().values,
        normalization=normalization_plm,
        dtype=dtype_plm,
        use_dask=use_dask,
        chunks=chunks_plm,
        derivative=True,
    )

    # -- beginning of computation
    a_earth = kwargs["a_earth"] if "a_earth" in kwargs else LNPY_A_EARTH_GRS80
    f_earth = kwargs["f_earth"] if "f_earth" in kwargs else LNPY_F_EARTH_GRS80
    omega_earth = (
        kwargs["omega_earth"] if "omega_earth" in kwargs else LNPY_OMEGA_EARTH_GRS80
    )
    if ellipsoidal_earth:
        r_theta = (
            a_earth
            * (1 - f_earth)
            / np.sqrt(1 - (2 * f_earth - f_earth**2) * np.sin(geoc_colat) ** 2)
        )
    else:
        r_theta = a_earth

    if surface == "geoid":
        if test == 1:
            h_geoid = (
                data.lnharmo.normal_zonal_correction(
                    f_earth=f_earth, omega_earth=omega_earth, apply=False
                )
            ).lnharmo.to_grid(
                unit="mmgeoid",
                longitude=longitude,
                latitude=latitude,
                ellipsoidal_earth=ellipsoidal_earth,
                plm=plm,
                normalization_plm=normalization_plm,
                use_dask=use_dask,
                chunks_lfactor=chunks_lfactor,
                **kwargs,
            ) * 1e-3
            radius = r_theta + h_geoid

        else:
            radius = (
                data.lnharmo.to_grid(
                    unit="mmgeoid",
                    longitude=longitude,
                    latitude=latitude,
                    ellipsoidal_earth=ellipsoidal_earth,
                    plm=plm,
                    normalization_plm=normalization_plm,
                    use_dask=use_dask,
                    chunks_lfactor=chunks_lfactor,
                    **kwargs,
                )
                * 1e-3
            )

    elif surface is not None and surface != "reference":
        radius = r_theta + surface

    else:
        radius = r_theta

    ds_centrifugal = centrifugal_potential_partial_derivate_radius(
        radius=radius * xr.ones_like(geoc_colat),
        ellipsoidal_earth=ellipsoidal_earth,
        **kwargs,
    )

    potential_dradius = (
        data.lnharmo.to_grid(
            unit="microGal",
            radius=radius,
            ellipsoidal_earth=ellipsoidal_earth,
            plm=plm,
            normalization_plm=normalization_plm,
            use_dask=use_dask,
            chunks_lfactor=chunks_lfactor,
            **kwargs,
        )
        * 1e-8
    )

    normal_dradius = (
        normal_field.lnharmo.to_grid(
            unit="microGal",
            radius=radius,
            ellipsoidal_earth=ellipsoidal_earth,
            plm=plm,
            normalization_plm=normalization_plm,
            use_dask=use_dask,
            chunks_lfactor=chunks_lfactor,
            **kwargs,
        )
        * 1e-8
    )

    potential_dlongitude = sh_to_potential_partial_derivative_longitude(
        data,
        radius=radius,
        ellipsoidal_earth=ellipsoidal_earth,
        plm=plm,
        normalization_plm=normalization_plm,
        use_dask=use_dask,
        chunks_lfactor=chunks_lfactor,
        **kwargs,
    )

    potential_dlatitude = data.lnharmo.to_grid(
        unit="potential",
        radius=radius,
        ellipsoidal_earth=ellipsoidal_earth,
        plm=dplm,
        normalization_plm=normalization_plm,
        use_dask=use_dask,
        chunks_lfactor=chunks_lfactor,
        **kwargs,
    )

    normal_dlatitude = normal_field.lnharmo.to_grid(
        unit="potential",
        radius=radius,
        ellipsoidal_earth=ellipsoidal_earth,
        plm=dplm,
        normalization_plm=normalization_plm,
        use_dask=use_dask,
        chunks_lfactor=chunks_lfactor,
        **kwargs,
    )

    nabla_w = np.sqrt(
        (potential_dradius + ds_centrifugal.dV_dradius) ** 2
        + potential_dlongitude**2 / (radius**2 * np.sin(geoc_colat) ** 2)
        + (potential_dlatitude + ds_centrifugal.dV_dlatitude) ** 2 / radius**2
    )

    # dlongitude is equal to 0 for the normal field
    nabla_u = np.sqrt(
        (normal_dradius + ds_centrifugal.dV_dradius) ** 2
        + (normal_dlatitude + ds_centrifugal.dV_dlatitude) ** 2 / radius**2
    )

    gravity_disturbance = nabla_w - nabla_u

    gravity_disturbance.attrs = {
        "units": "gravity_disturbance",
        "max_degree": int(data.l.max().values),
    }
    if "radius" in data.attrs:
        gravity_disturbance.attrs["radius"] = data.attrs["radius"]
    if "earth_gravity_constant" in data.attrs:
        gravity_disturbance.attrs["earth_gravity_constant"] = data.attrs[
            "earth_gravity_constant"
        ]
    return gravity_disturbance


def _compute_centrifugal_potential_partial_derivative(
    radius: xr.DataArray,
    geocentric_colatitude: xr.DataArray,
    omega_earth: float,
    f_earth: float,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute the partial derivative of the centrifugal potential on the surface of reference with respect to
    radius, longitude and geographic latitude at given geocentric colatitude.

    Parameters
    ----------
    radius: xr.DataArray
        Earth radius depending on latitude and possibly longitude.
    geocentric_colatitude : xr.DataArray | None, optional
        Geocentric colatitude in radians, the dimension is geographic latitude.
    omega_earth : float, optional
        Earth's rotation rate [rad.s⁻¹]. Default is LNPY_OMEGA_EARTH.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80 if ellipsoidal_earth is True else it is set to 0.
    """
    e_earth_square = 2 * f_earth - f_earth**2

    # Partial derivatives of the centrifugal potential
    centrifugal_r = omega_earth**2 * radius * np.sin(geocentric_colatitude) ** 2
    centrifugal_longitude = radius * 0
    centrifugal_latitude = -(omega_earth**2) * radius**2 * np.cos(
        geocentric_colatitude
    ) * np.sin(geocentric_colatitude) - omega_earth**2 * radius**2 * np.sin(
        geocentric_colatitude
    ) ** 2 * (
        e_earth_square * np.cos(geocentric_colatitude) * np.sin(geocentric_colatitude)
    ) / (
        1 - e_earth_square * np.sin(geocentric_colatitude) ** 2
    )

    return centrifugal_r, centrifugal_longitude, centrifugal_latitude


def centrifugal_potential_partial_derivative(
    latitude: xr.DataArray | np.ndarray | None = None,
    dlat: float = 1,
    ellipsoidal_earth: bool = False,
    a_earth: float | None = None,
    omega_earth: float = LNPY_OMEGA_EARTH_GRS80,
    f_earth: float = LNPY_F_EARTH_GRS80,
    attrs: dict | None = None,
) -> xr.Dataset:
    """
    Estimate the partial derivative of the centrifugal potential on the surface of reference with respect to
    radius, longitude and geographic latitude at given geographic latitude.

    Parameters
    ----------
    latitude: xr.DataArray | np.ndarray | None, optional
        Geographic latitude in degree, either as array or as DataArray.
    dlat: float
        Latitude resolution in degrees.
    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid for the computation of the centrifugal potential.
        Default is False.
    a_earth : float, optional
        Earth radius for spherical case or Earth semi-major axis [m]. If not provided, uses `attrs['radius']` and
        if it does not exist, uses LNPY_A_EARTH_GRS80.
    omega_earth : float, optional
        Earth's rotation rate [rad.s⁻¹]. Default is LNPY_OMEGA_EARTH.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80 if ellipsoidal_earth is True else it is set to 0.
    attrs : dict | None, optional
        ds.attrs information that might help to estimate a_earth if no parameters are given.

    Returns
    -------
    ds_centrifugal : xr.Dataset
        Dataset with the partial derivatives of the centrifugal potential of the surface of reference,
        named 'dV_radius', 'dV_longitude' and 'dV_latitude', with respect to radius, longitude and latitude.
        The dataset has a latitude dimension and coordinate set with the latitude input.
    """
    if latitude is None:
        latitude = np.arange(-90 + dlat / 2, 90 + dlat / 2, dlat)

    # Change latitude to DataArray if it is not already
    if type(latitude) != xr.DataArray:
        latitude = xr.DataArray(
            latitude, dims=["latitude"], coords={"latitude": latitude}
        )

    # get parameters for the computation
    a_earth, _ = _get_earth_parameters(attrs, a_earth, None)

    geoc_colat = latitude_to_geocentric_colatitude(
        latitude, ellipsoidal_earth=ellipsoidal_earth, f_earth=f_earth
    )

    if ellipsoidal_earth:
        r_theta = (
            a_earth
            * (1 - f_earth)
            / np.sqrt(1 - (2 * f_earth - f_earth**2) * np.sin(geoc_colat) ** 2)
        )
    else:
        r_theta = a_earth
        f_earth = 0

    centrifugal_r, centrifugal_lon, centrifugal_lat = (
        _compute_centrifugal_potential_partial_derivative(
            r_theta, geoc_colat, omega_earth, f_earth
        )
    )

    # Return the partial derivatives as a dataset
    return xr.Dataset(
        {
            "dV_dradius": centrifugal_r,
            "dV_dlongitude": centrifugal_lon,
            "dV_dlatitude": centrifugal_lat,
        },
        attrs={
            "name": "centrifugal_potential_partial_derivative",
            "radius": a_earth,
            "omega_earth": omega_earth,
            "f_earth": f_earth,
        },
    )


def centrifugal_potential_partial_derivate_radius(
    radius: xr.DataArray | None = None,
    height: xr.DataArray | None = None,
    ellipsoidal_earth: bool = False,
    a_earth: float | None = None,
    f_earth: float = LNPY_F_EARTH_GRS80,
    omega_earth: float = LNPY_OMEGA_EARTH_GRS80,
    attrs: dict | None = None,
) -> xr.Dataset:
    """
    Estimate the partial derivative of the centrifugal potential on a defined surface (that can be the geoid).
    The surface is defined either by the radius with respect to the Earth's center or by the height above the
    surface of reference (sphere or ellipsoid depending on the `ellipsoidal_earth` parameter).
    The partial derivative are estimated with respect to radius, longitude and geographic latitude.

    Parameters
    ----------
    radius: xr.DataArray | float | None, optional
        Radius from the Earth's center depending on latitude and other dimensions. If not provided, `height` parameter
        is used to compute the radius.
    height: xr.DataArray | float | None, optional
        Height above of the surface of reference (sphere or ellipsoid) depending on latitude and possibly other
        dimensions. If not provided, radius parameter is used for the computation.
    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid for the computation of the centrifugal potential.
        Default is False.
    a_earth : float, optional
        Earth radius for spherical case or Earth semi-major axis [m]. If not provided, uses `attrs['radius']` and
        if it does not exist, uses LNPY_A_EARTH_GRS80.
    omega_earth : float, optional
        Earth's rotation rate [rad.s⁻¹]. Default is LNPY_OMEGA_EARTH.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80 if ellipsoidal_earth is True else it is set to 0.
    attrs : dict | None, optional
        ds.attrs information that might help to estimate a_earth if no parameters are given.

    Returns
    -------
    ds_centrifugal : xr.Dataset
        Dataset with the partial derivatives of the centrifugal potential at the given radius,
        named 'dV_radius', 'dV_longitude' and 'dV_latitude', with respect to radius, longitude and latitude.
        The radius of the reference surface is given in the dataset as 'determination_radius' variable.
        The dataset has a latitude dimension and possiblt a longitude dimension depending on the input.

    Raises
    ------
    ValueError
        If neither radius nor height parameter is provided.
    """
    # get parameters for the computation
    a_earth, _ = _get_earth_parameters(attrs, a_earth, None)

    if radius is None and height is not None:
        assert_latitude(height)
        latitude = height.latitude
    elif radius is not None:
        assert_latitude(radius)
        latitude = radius.latitude
    else:
        raise ValueError("Either radius or height parameter must be provided.")

    geoc_colat = latitude_to_geocentric_colatitude(
        latitude, ellipsoidal_earth=ellipsoidal_earth, f_earth=f_earth
    )

    if radius is None and height is not None:
        if ellipsoidal_earth:
            r_theta = (
                a_earth
                * (1 - f_earth)
                / np.sqrt(1 - (2 * f_earth - f_earth**2) * np.sin(geoc_colat) ** 2)
            )
        else:
            r_theta = a_earth
            f_earth = 0

        radius = height + r_theta

    centrifugal_r, centrifugal_lon, centrifugal_lat = (
        _compute_centrifugal_potential_partial_derivative(
            radius, geoc_colat, omega_earth, f_earth
        )
    )

    # Return the partial derivatives as a dataset
    return xr.Dataset(
        {
            "dV_dradius": centrifugal_r,
            "dV_dlongitude": centrifugal_lon,
            "dV_dlatitude": centrifugal_lat,
            "determination_radius": radius,
        },
        attrs={
            "name": "centrifugal_potential_partial_derivative",
            "radius": a_earth,
            "omega_earth": omega_earth,
            "f_earth": f_earth,
        },
    )


def sh_to_potential_partial_derivative_longitude(
    data: xr.Dataset,
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
    radius: xr.DataArray = None,
    ellipsoidal_earth: bool = False,
    plm: xr.DataArray = None,
    normalization_plm: Literal["4pi", "ortho", "schmidt"] = "4pi",
    dtype_plm: type[complex] | type[float] = np.float128,
    use_dask: bool = False,
    chunks_lfactor: dict | None = None,
    chunks_plm: dict | None = None,
    **kwargs,
) -> xr.DataArray:
    """
    Transform Spherical Harmonics (SH) dataset into a grid corresponding to the potential partial derivative
    in regard to longitude. This function is used for gravity disturbance estimation.
    With choice for constants, unit, love numbers, degree/order, spatial grid latitude and longitude, Earth hypothesis.

    Parameters
    ----------
    data : xr.Dataset
        xr.Dataset that corresponds to SH data to convert into potential partial derivative of longitude
        spatial representation.

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

    radius : xr.DataArray, optional
        DataArray with the radius of the grid to compute. If not given, the radius is the surface of reference.
    ellipsoidal_earth : bool, optional
        If True, consider the Earth as an ellipsoid following [Ditmar2018]. Default is False for a spherical Earth.

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
        a_earth, gm_earth, f_earth, omega_earth

    Returns
    -------
    xgrid : xr.DataArray
        Spatial representation of the SH Dataset in potential partial derivative in regards to longitude.
    """
    # Get parameters for computation of the grid and for the unit conversion
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
    )

    geocentric_colat = latitude_to_geocentric_colatitude(
        latitude, ellipsoidal_earth=ellipsoidal_earth, **kwargs
    )

    # -- beginning of computation
    # Computing plm for converting to spatial domain
    if plm is None:
        plm = compute_plm(
            data.l.max().values,
            np.cos(geocentric_colat),
            latitude=latitude,
            mmax=data.m.max().values,
            normalization=normalization_plm,
            dtype=dtype_plm,
            use_dask=use_dask,
            chunks=chunks_plm,
        )

    else:
        _assert_plm(plm, data.l.max().values, latitude)

    # scale factor for each degree
    lfactor, cst = l_factor_conv(
        data.l.values,
        unit="potential",
        ellipsoidal_earth=ellipsoidal_earth,
        geocentric_colat=geocentric_colat,
        radius=radius,
        attrs=data.attrs,
        use_dask=use_dask,
        chunks=chunks_lfactor,
        **kwargs,
    )

    # convolve unit over degree
    plm_lfactor = plm.sel(l=data.l, m=data.m) * lfactor

    # Calculating cos(m*phi) and sin(m*phi)
    c_cos = xr.DataArray(
        np.cos(data.m.values[:, np.newaxis] @ np.deg2rad(longitude)[np.newaxis, :]),
        dims=["m", "longitude"],
        coords={"m": data.m, "longitude": longitude},
    )
    s_sin = xr.DataArray(
        np.sin(data.m.values[:, np.newaxis] @ np.deg2rad(longitude)[np.newaxis, :]),
        dims=["m", "longitude"],
        coords={"m": data.m, "longitude": longitude},
    )

    # summation over all spherical harmonic degrees
    d_clm = (plm_lfactor * data.clm * data.m).sum(dim="l")
    d_slm = (plm_lfactor * data.slm * data.m).sum(dim="l")

    # Final calcul on the grid
    xgrid = c_cos.dot(d_slm, dim=["m"]) - s_sin.dot(d_clm, dim=["m"])

    xgrid = xgrid.transpose("latitude", "longitude", ...)

    xgrid.attrs = {
        "units": "potential_dlongitude",
        "max_degree": int(data.l.max().values),
    }
    if "radius" in data.attrs:
        xgrid.attrs["radius"] = data.attrs["radius"]
    if "earth_gravity_constant" in data.attrs:
        xgrid.attrs["earth_gravity_constant"] = data.attrs["earth_gravity_constant"]

    return xgrid


def gauss_weights(
    radius: float, lmax: int, a_earth: float = LNPY_A_EARTH_GRS80, cutoff: float = 1e-10
) -> xr.DataArray:
    """
    Generate a xr.DataArray with Gaussian weights as a function of degree.

    This function uses the method described in [Jekeli1981]_ to generate Gaussian weights for spherical harmonics
    coefficients.

    Parameters
    ----------
    radius : float
        Gaussian smoothing radius in meters.
    lmax : int
        Maximum degree of spherical harmonic coefficients.
    a_earth : float, optional
        Radius of the Earth in meters. Default is LNPY_A_EARTH_GRS80 define in constants.py
    cutoff : float, optional
        Minimum value for the tail of the Gaussian averaging function (see [Jekeli1981]_ p.18), default is 1e-10.

    Returns
    -------
    gaussian_weights : xr.DataArray
        Degree-dependent Gaussian weights xr.DataArray.

    Examples
    --------
    >>> ds_gauss_weights = gauss_weights(radius=100000, lmax=60)

    References
    ----------
    .. [Jekeli1981] C. Jekeli, "Alternative Methods to Smooth
        the Earth's Gravity Field", NASA Grant No. NGR 36-008-161,
        OSURF Proj. No. 783210, 48 pp., (1981).
    """
    # Create weights array
    gaussian_weights = np.zeros((lmax + 1))

    # Computation using recursion from [Jekeli1981]_ p.17
    a = np.log(2) / (1 - np.cos(radius / a_earth))
    # Initialize weight for degree 0 and 1
    gaussian_weights[0] = 1
    gaussian_weights[1] = gaussian_weights[0] * (
        (1 + np.exp(-2 * a)) / (1 - np.exp(-2 * a)) - 1 / a
    )

    for l in range(2, lmax + 1):
        # recursion with the two previous terms
        gaussian_weights[l] = (
            gaussian_weights[l - 1] * (1 - 2 * l) / a + gaussian_weights[l - 2]
        )

        # test if weight is less than cutoff
        if gaussian_weights[l] < cutoff:
            # set all weights after the current l to cutoff
            gaussian_weights[l : lmax + 1] = cutoff
            break

    return xr.DataArray(gaussian_weights, dims=["l"], coords={"l": np.arange(lmax + 1)})


def gfct_field_estimation(
    ds: xr.Dataset, time: np.datetime64 | np.ndarray | list | tuple | xr.DataArray
) -> xr.Dataset:
    """
    Compute time-variable gravity field from variation coefficients contain in '.gfc' file
    with icgem1.0 or icgem2.0 format.

    Parameters
    ----------
    ds : xr.Dataset
        Output of the reader from the '.gfc' file with gfct information.
    time : np.datetime64 | list, tuple, np.ndarray of np.datetime64 | xr.DataArray of np.datetime64
        Time information to compute the coefficients variations on.

    Returns
    -------
    ds_out : xr.Dataset
        Estimated coefficients time variations.
    """
    if isinstance(time, xr.DataArray):
        if not np.issubdtype(time.dtype, np.datetime64):
            raise TypeError("The time xr.DataArray does not contain np.datetime64.")

    elif isinstance(time, (list, tuple, np.ndarray)):
        time = xr.DataArray(time, dims=["time"])
        if not np.issubdtype(time.dtype, np.datetime64):
            raise TypeError("The time xr.DataArray does not contain np.datetime64.")

    elif isinstance(time, np.datetime64):
        time = xr.DataArray(np.array([time]), dims=["time"])

    else:
        raise TypeError(
            "The 'time' parameter has to be a xr.DataArray with np.datetime64, "
            "or a list/array of datetime64, or a datetime64 object."
        )

    if "name" in ds.dims:
        if ds.sizes["name"] > 1:
            warnings.warn(
                "Multiple object on the dimension 'name', only the first one is converted to a time dataset."
            )
        ds = ds.isel(name=0)

    # Case of icgem1.0 format: direct computation
    if "icgem1." in ds.attrs["format"]:
        delta_year = (time - ds.ref_time).dt.days / 365.25

        clm = (
            ds.clm
            + ds.trnd_clm * delta_year
            + (ds.acos_clm * np.cos(2 * np.pi * delta_year / ds.periods_acos)).sum(
                "periods_acos"
            )
            + (ds.asin_slm * np.sin(2 * np.pi * delta_year / ds.periods_asin)).sum(
                "periods_asin"
            )
        ).values
        slm = (
            ds.slm
            + ds.trnd_slm * delta_year
            + (ds.acos_slm * np.cos(2 * np.pi * delta_year / ds.periods_acos)).sum(
                "periods_acos"
            )
            + (ds.asin_slm * np.sin(2 * np.pi * delta_year / ds.periods_asin)).sum(
                "periods_asin"
            )
        ).values

        if ds.attrs["errors"] != "no":
            eclm = np.sqrt(
                ds.eclm**2
                + (ds.trnd_eclm * delta_year) ** 2
                + (
                    (ds.acos_eclm * np.cos(2 * np.pi * delta_year / ds.periods_acos))
                    ** 2
                ).sum("periods_acos")
                + (
                    (ds.asin_eslm * np.sin(2 * np.pi * delta_year / ds.periods_asin))
                    ** 2
                ).sum("periods_asin")
            ).values
            eslm = np.sqrt(
                ds.eslm**2
                + (ds.trnd_eslm * delta_year) ** 2
                + (
                    (ds.acos_eslm * np.cos(2 * np.pi * delta_year / ds.periods_acos))
                    ** 2
                ).sum("periods_acos")
                + (
                    (ds.asin_eslm * np.sin(2 * np.pi * delta_year / ds.periods_asin))
                    ** 2
                ).sum("periods_asin")
            ).values

    # Case of icgem1.0 format: one computation for each given time to use corresponding coefficients
    elif "icgem2." in ds.attrs["format"]:
        clm = np.zeros((ds.sizes["l"], ds.sizes["m"], len(time)))
        slm = np.zeros((ds.sizes["l"], ds.sizes["m"], len(time)))

        if ds.attrs["errors"] != "no":
            eclm = np.zeros((ds.sizes["l"], ds.sizes["m"], len(time)))
            eslm = np.zeros((ds.sizes["l"], ds.sizes["m"], len(time)))

        for i, t in enumerate(time):
            t_ds = ds.isel(time=np.searchsorted(ds.time, t) - 1)

            delta_year = -(t_ds.time - t).dt.days / 365.25

            clm[:, :, i] = (
                t_ds.clm
                + t_ds.trnd_clm * delta_year
                + (
                    t_ds.acos_clm * np.cos(2 * np.pi * delta_year / ds.periods_acos)
                ).sum("periods_acos")
                + (
                    t_ds.asin_slm * np.sin(2 * np.pi * delta_year / ds.periods_asin)
                ).sum("periods_asin")
            )
            slm[:, :, i] = (
                t_ds.slm
                + t_ds.trnd_slm * delta_year
                + (
                    t_ds.acos_slm * np.cos(2 * np.pi * delta_year / ds.periods_acos)
                ).sum("periods_acos")
                + (
                    t_ds.asin_slm * np.sin(2 * np.pi * delta_year / ds.periods_asin)
                ).sum("periods_asin")
            )

            if ds.attrs["errors"] != "no":
                eclm[:, :, i] = np.sqrt(
                    t_ds.eclm**2
                    + (t_ds.trnd_eclm * delta_year) ** 2
                    + (
                        (
                            t_ds.acos_eclm
                            * np.cos(2 * np.pi * delta_year / ds.periods_acos)
                        )
                        ** 2
                    ).sum("periods_acos")
                    + (
                        (
                            t_ds.asin_eslm
                            * np.sin(2 * np.pi * delta_year / ds.periods_asin)
                        )
                        ** 2
                    ).sum("periods_asin")
                )
                eslm[:, :, i] = np.sqrt(
                    t_ds.eslm**2
                    + (t_ds.trnd_eslm * delta_year) ** 2
                    + (
                        (
                            t_ds.acos_eslm
                            * np.cos(2 * np.pi * delta_year / ds.periods_acos)
                        )
                        ** 2
                    ).sum("periods_acos")
                    + (
                        (
                            t_ds.asin_eslm
                            * np.sin(2 * np.pi * delta_year / ds.periods_asin)
                        )
                        ** 2
                    ).sum("periods_asin")
                )

    else:
        raise ValueError(
            "Unknown format, please provide information on "
            "ds.attrs['format'] with either 'icgem1.0' or 'icgem2.0'"
        )

    ds_out = xr.Dataset(
        {
            "clm": (["l", "m", "time"], clm),
            "slm": (["l", "m", "time"], slm),
        },
        coords={
            "l": ds.l,
            "m": ds.m,
            "time": time,
        },
        attrs=ds.attrs,
    )

    if ds.attrs["errors"] != "no":
        ds_out["eclm"] = xr.DataArray(eclm, dims=["l", "m", "time"])
        ds_out["eslm"] = xr.DataArray(eslm, dims=["l", "m", "time"])

    return ds_out
