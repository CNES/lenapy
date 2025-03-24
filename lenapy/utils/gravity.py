"""
The gravity module provides functions for manipulating spherical harmonics datasets with respect to different
reference frames, tide systems and smoothing techniques.

This module includes functions to:
  * Change the reference frame of spherical harmonics datasets.
  * Change the tide system of spherical harmonics datasets.
  * Modify degree 1 Love numbers for different reference frames.
  * Generate Gaussian weights for smoothing spherical harmonics datasets.

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

import numpy as np
import xarray as xr

from lenapy.constants import *


def change_reference(
    ds,
    new_radius=LNPY_A_EARTH_GRS80,
    new_earth_gravity_constant=LNPY_GM_EARTH,
    old_radius=None,
    old_earth_gravity_constant=None,
    apply=False,
):
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
        If True, apply the update to the current dataset without making a deep copy. Default is False.

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

    # if apply = False : Copy the dataset to avoid modifying the input dataset
    ds_out = ds if apply else ds.copy(deep=True)

    # Update the clm and slm values
    ds_out["clm"] *= update_factor
    ds_out["slm"] *= update_factor

    # Update the attributes in the output dataset
    ds_out.attrs["radius"] = new_radius
    ds_out.attrs["earth_gravity_constant"] = new_earth_gravity_constant

    return ds_out


def change_tide_system(ds, new_tide, old_tide=None, k20=None, apply=False):
    """
    Apply a C20 offset to the dataset to change the tide system.
    Follows IERS2010 convention [IERS2010]_ to convert between tide system ('tide_free', 'zero_tide', 'mean_tide').
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
        If True, apply the update to the current dataset without making a deep copy. Default is False.

    Returns
    -------
    ds_out : xr.Dataset
        Updated dataset with the new tidal system. Warning, the dataset is also updated in place if copy=False (default)

    Raises
    ------
    ValueError
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
            raise ValueError(
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

    # if apply = False : Copy the dataset to avoid modifying the input dataset
    ds_out = ds if apply else ds.copy(deep=True)

    ds_out.clm.loc[dict(l=2, m=0)] += conv * A0 * H0
    ds_out.attrs["tide_system"] = new_tide

    # -- return ds
    return ds_out


def change_love_reference_frame(ds, new_frame, old_frame, apply=False):
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
        If True, apply the update to the current dataset without making a deep copy. Default is False.

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
    # if apply = False : Copy the dataset to avoid modifying the input dataset
    ds_love = ds if apply else ds.copy(deep=True)

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


def gauss_weights(radius, lmax, a_earth=LNPY_A_EARTH_GRS80, cutoff=1e-10):
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
