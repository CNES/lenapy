import numpy as np
import xarray as xr

from lenapy.constants import *


def rename_data(data, **kwargs):
    """
    Standardization of coordinates names of a product.
    Looks for different possible names for latitude, longitude, and time, and turn them into a standardized name.
    Definitions are specified in setup.py and are based on standard cf attributes and units : https://cfconventions.org
    Custom names changes can also be performed with kwargs parameter.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        xarray object that has coordinates to rename.
    **kwargs :  {old_name: new_name, ...}, optional
        Dictionary specifying old names to be changed into new names.

    Returns
    -------
    renamed : xr.DataArray | xr.Dataset
        New object containing modified names.

    Example
    -------
    .. code-block:: python

        ds=xr.open_mfdataset('product.nc', preprocess=rename_data)
    """
    data = data.rename(**kwargs)
    for coord in ["latitude", "longitude", "time", "depth"]:
        try:
            old_name = data.cf[coord].name
            data = data.rename({old_name: coord})
        except KeyError:
            pass

    if "longitude" in data.variables and "longitude" not in data.coords:
        lon = data["longitude"]
        del data["longitude"]
        lat = data["latitude"]
        del data["latitude"]
        data = data.assign_coords(longitude=lon, latitude=lat)
    return data


def isosurface(data, target, dim, coord=None, upper=False):
    """
    Linearly interpolate a coordinate isosurface where a field equals a target.
    Compute the isosurface along the specified coordinate at the value defined by the target argument.
    Data is supposed to be monotonic along the chosen dimension. If not, the first fitting value encountered is
    retained, starting from the end (bottom) if upper=False, or from the beggining (top) if upper=True.

    Parameters
    ----------
    data : xr.DataArray
        The field in which to interpolate the target isosurface.
    target : float
        Criterion value to be satisfied at the iso surface.
    dim : str
        Dimension along which to compute the isosurface.
    coord : str (optional)
        The field coordinate to interpolate. If absent, coordinate is supposed to be "dim".
    upper : bool
        Order to perform the research of the criterion value. If True, returns the highest point of the isosurface,
        else the lowest.

    Returns
    -------
    isosurface : xr.DataArray
        DataArray containing the isosurface along the dimension dim on which data=target.

    Examples
    --------
    Calculate the depth of an isotherm with a value of 5.5:

    .. code-block:: python

         temp = xr.DataArray(range(10, 0, -1), coords={"depth": range(10)})
         isosurface(temp, 5.5, dim="depth")
         # Output : <xarray.DataArray ()>
         # array(4.5)
    """
    if coord is None:
        coord = dim

    if dim not in data.dims:
        raise ValueError(f"Dimension '{dim}' not found in data.")
    if coord not in data.coords:
        raise ValueError(f"Coordinate '{coord}' not found in data.")

    slice0 = {dim: slice(None, -1)}
    slice1 = {dim: slice(1, None)}

    field0 = data.isel(slice0).drop(coord)
    field1 = data.isel(slice1).drop(coord)

    crossing_mask_decr = (field0 > target) & (field1 <= target)
    crossing_mask_incr = (field0 < target) & (field1 >= target)
    crossing_mask = xr.where(crossing_mask_decr | crossing_mask_incr, 1, np.nan)

    coords0 = crossing_mask * data[coord].isel(slice0).drop(coord)
    coords1 = crossing_mask * data[coord].isel(slice1).drop(coord)
    field0 = crossing_mask * field0
    field1 = crossing_mask * field1

    iso = coords0 + (target - field0) * (coords1 - coords0) / (field1 - field0)
    if upper:
        return iso.min(dim, skipna=True)
    else:
        return iso.max(dim, skipna=True)


def split_duplicate_coords(data):
    for v in data.data_vars:
        dims = data[v].dims
        dup = {x + "_" for x in dims if dims.count(x) > 1}
        for d in dup:
            data = data.assign_coords({d: data[d[:-1]].data})

        new_dims = tuple(list(set(dims)) + list(dup))

        if len(dup) > 0:
            data[v] = (new_dims, data[v].data)
    return data


def longitude_increase(data):
    if "longitude" in data.coords:
        l = xr.where(
            data.longitude < data.longitude.isel(longitude=0),
            data.longitude + 360,
            data.longitude,
        )
        if l.max() > 360:
            l = l - 360
        data["longitude"] = l
    return data


def reset_longitude(data, orig=-180):
    """
    Rolls the longitude to place the specified longitude at the beginning of the array.

    Returns
    -------
    origin : float
        First longitude in the array.

    Example
    -------
    .. code-block:: python

        data = xr.open_dataset('/home/user/lenapy/data/gohc_2020.nc', engine="lenapyNetcdf")
        surface = data.lngeo.surface_cell()
    """
    i = ((np.mod(data.longitude - orig + 180, 360) - 180) ** 2).argmin().values
    return longitude_increase(data.roll(longitude=-i, roll_coords=True))


def surface_cell(
    data, ellipsoidal_earth=True, a_earth=None, f_earth=LNPY_F_EARTH_GRS80
):
    """
    Returns the Earth surface of each cell defined by a longitude/latitude in a xarray object.
    Cells limits are half the distance between each given coordinate. Given coordinates are not necessary the center of
    each cell. Border cells are supposed to have the same size on each side of the given coordinate.
    Ex : coords=[1,2,4,7,9] ==> cell sizes are [1,1.5,2.5,2.5,2]

    The surface of the cell is given by default for an ellipsoidal earth of radius LNPY_A_EARTH_GRS80 and
    flattening LNPY_F_EARTH_GRS80. The parameter `ellipsoidal_earth` can be set to False for the surface on a
    spherical Earth. It can be set to 'approx' for the surface with an approximation of the ellipsoid on each cell by
    a spherical cell with the radius corresponding to the distance between the ellispoid point and the center.

    If surface_cell() is applied to a complete grid, the sum of all cells is equal to 4πR² for the spherical Earth.
    For the ellipsoidal Earth, the sum is equal to 2πa² + πb²/e*log((1 + e)/(1 - e))


    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        xarray object that must have latitude and longitude coordinates.
    ellipsoidal_earth: bool | str, optional
        Boolean to choose if the surface of the Earth is an ellipsoid or a sphere. Default is True for ellipsoidal Earth
        If ellipsoidal_earth='approx', the given surface is the one of a spherical cell with the radius corresponding to
        the distance between the ellispoid point and the center of the ellipsoid.
    a_earth : float, optional
        Earth semi-major axis [m]. If not provided, use `data.attrs['radius']` and
        if it does not exist, use LNPY_A_EARTH_GRS80.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80.

    Returns
    -------
    surface : xr.DataArray
        DataArray with cell surface.

    Example
    -------
    .. code-block:: python

        xr.open_dataset('/home/user/lenapy/data/gohc_2020.nc', engine="lenapyNetcdf")
        surface = data.lngeo.surface_cell()
    """
    if a_earth is None:
        a_earth = (
            float(data.attrs["radius"])
            if "radius" in data.attrs
            else LNPY_A_EARTH_GRS80
        )

    dlat = ecarts(data, "latitude")
    dlon = ecarts(data, "longitude")

    # case of the cell on an ellipsoid
    if ellipsoidal_earth == "ellipsoidal" or ellipsoidal_earth is True:
        ep = np.sqrt(
            (2 * f_earth - f_earth**2) / (1 - f_earth) ** 2
        )  # eccentricity prime
        # geocentric latitude of the cell border, compute a temporary forced in float64 latitude to reduce numeric error
        tmp_latitude_float64 = data.cf["latitude"].values.astype(np.float64)
        omega_1 = np.arctan(
            (1 - f_earth) ** 2 * np.tan(np.radians(tmp_latitude_float64 - dlat / 2))
        )
        omega_2 = np.arctan(
            (1 - f_earth) ** 2 * np.tan(np.radians(tmp_latitude_float64 + dlat / 2))
        )

        return np.abs(
            a_earth**2
            * (1 - f_earth)
            * np.radians(dlon)
            * (
                (np.arcsinh(ep * np.sin(omega_2)) - np.arcsinh(ep * np.sin(omega_1)))
                / ep
                + (
                    np.sin(omega_2) * np.sqrt(1 + ep**2 * np.sin(omega_2) ** 2)
                    - np.sin(omega_1) * np.sqrt(1 + ep**2 * np.sin(omega_1) ** 2)
                )
            )
            / 2
        )

    # case of the sphere that approximates the ellipsoid
    elif ellipsoidal_earth == "approx":
        return np.abs(
            a_earth**2
            * np.radians(dlon)
            * np.cos(np.radians(data.cf["latitude"]))
            * np.radians(dlat)
            / (1 + f_earth * np.cos(2 * np.radians(data.cf["latitude"]))) ** 2
        )

    # case of the spherical cell with a sphere of radius a_earth
    elif ellipsoidal_earth == "spherical" or ellipsoidal_earth is False:
        return np.abs(
            2
            * a_earth**2
            * np.radians(dlon)
            * np.cos(np.radians(data.cf["latitude"]))
            * np.sin(np.radians(dlat) / 2)
        )

    else:
        raise ValueError(
            'Given argument "ellipsoidal_earth" has to be a boolean '
            'or either "ellispoidal", "spherical" or "approx".'
        )


def ecarts(data, dim):
    """
    Return the width of each cell along specified coordinate.
    Cell limits are half-distance between each given coordinate (that are not necessary the center of each cell).
    Border cells are supposed to have the same size on each side of the given coordinate.
    Ex : coords=[1,2,4,7,9] ==> cells size is [1,1.5,2.5,2.5,2]

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Must have latitude and longitude coordinates.
    dim : str
        Coordinate along which to compute cell width.

    Returns
    -------
    width : xr.DataArray
        xr.DataArray with cell width for each coordinate.
    """

    i0 = data[dim].isel({dim: slice(None, 2)}).diff(dim, label="lower")
    i1 = (data[dim] - data[dim].diff(dim, label="upper") / 2).diff(dim, label="lower")
    i2 = data[dim].isel({dim: slice(-2, None)}).diff(dim, label="upper")
    return xr.concat([i0, i1, i2], dim=dim)


def distance(
    data, pt, ellipsoidal_earth=False, a_earth=None, f_earth=LNPY_F_EARTH_GRS80
):
    """
    Compute the great-circle/geodetic distance between coordinates on a sphere / ellipsoid.
    The computation of the distance for ellipsoidal_earth=True uses the pyproj librairy that implements the
    algorithm given in [Karney2013]_.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        Must have latitude and longitude coordinates.
    pt : xr.DataArray | xr.Dataset
        Object with coordinates that are not latitude or longitude but that contain latitude and longitude.
        For example, a list of points with a coordinate "id".
    ellipsoidal_earth: bool | str, optional
        Boolean to choose if the surface of the Earth is an ellipsoid or a sphere. Default is False for spherical Earth.
    a_earth : float, optional
        Earth semi-major axis [m]. If not provided, use `data.attrs['radius']` and
        if it does not exist, use LNPY_A_EARTH_GRS80.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80.

    Returns
    -------
    distance : xr.DataArray
        DataArray with distance between the latitude and longitude of data and the latitude and longitude of pt.
        The final coordinates are latitude, longitude + coordinates of pt.

    References
    ----------
    .. [Karney2013] C.F.F. Karney,
        "Algorithms for geodesics",
        *Journal of Geodesy*, 87, 43-55, (2013).
        `doi: 10.1007/s00190-012-0578-z <https://doi.org/10.1007/s00190-012-0578-z>`_
    """
    if a_earth is None:
        a_earth = (
            float(data.attrs["radius"])
            if "radius" in data.attrs
            else LNPY_A_EARTH_GRS80
        )

    # Verify the format of 'pt' argument, raise Error with verbose in case of an identified problem
    if "latitude" in pt.coords or "longitude" in pt.coords:
        raise ValueError(
            "Given xr.DataArray to compute distance with must not have latitude and longitude coordinates."
            " Use 'id' as a coordinate for example."
        )
    if pt.latitude.dims != pt.longitude.dims:
        raise ValueError(
            "Given xr.DataArray to compute distance with must have the same dimension along latitude and "
            "longitude. Use 'id' as a dimension for latitude and longitude for example."
        )
    if len(pt.latitude.dims) != 1:
        raise ValueError(
            "Given xr.DataArray to compute distance with must have exactly one dimension."
        )

    if ellipsoidal_earth:
        try:
            import pyproj
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "pyproj module is needed for distance computation on an ellipsoid. "
                "You can still use distance function with the argument ellipsoidal_earth=False."
            )

        # create array input for geod.inv() function, with flat shape (pt.size, data.latitude.size, data.longitude.size)
        lon1 = pt.cf["longitude"].values.repeat(
            data.cf["latitude"].size * data.cf["longitude"].size
        )
        lat1 = pt.cf["latitude"].values.repeat(
            data.cf["latitude"].size * data.cf["longitude"].size
        )
        lon2 = np.tile(
            data.cf["longitude"].values, pt.longitude.size * data.cf["latitude"].size
        )
        lat2 = np.tile(
            data.cf["latitude"].values.repeat(data.cf["longitude"].size),
            pt.latitude.size,
        )

        # call the computation of inverse distance and reshape the output
        geod = pyproj.Geod(a=a_earth, f=f_earth)
        geod_dist = geod.inv(lon1, lat1, lon2, lat2)[2]

        # reshape for DataArray creation
        geod_dist_reshape = geod_dist.reshape(
            (pt.latitude.size, data.cf["latitude"].size, data.cf["longitude"].size)
        )
        return xr.DataArray(
            geod_dist_reshape,
            coords={
                pt.latitude.dims[0]: pt[
                    pt.latitude.dims[0]
                ],  # pt.latitude.dims[0] is the pt dim
                "latitude": data["latitude"],
                "longitude": data["longitude"],
            },
            dims=["id", "latitude", "longitude"],
        )
    else:
        return a_earth * np.real(
            np.arccos(
                np.cos(np.deg2rad(pt.cf["latitude"]))
                * np.cos(np.deg2rad(data.cf["latitude"]))
                * np.cos(np.deg2rad(data.cf["longitude"] - pt.cf["longitude"]))
                + np.sin(np.deg2rad(pt.cf["latitude"]))
                * np.sin(np.deg2rad(data.cf["latitude"]))
            )
        )


def assert_grid(ds):
    """
    Verify if the given xr.Dataset have dimensions 'longitude' and 'latitude'. Raise Assertion error if not.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to verify.

    Returns
    -------
    True : bool
        Returns True if the dataset has dimensions 'longitude' and 'latitude'.

    Raise
    -----
    AssertionError
        This function raises AssertionError is self._obj is not a xr.Dataset corresponding to spherical harmonics.
    """
    if "latitude" not in ds.coords:
        raise AssertionError(
            "The latitude coordinates that should be named 'latitude' does not exist"
        )
    if "longitude" not in ds.coords:
        raise AssertionError(
            "The longitude coordinates that should be named 'longitude' does not exist"
        )
    return True


def surface_grid(da, type="nan", **kwargs):
    if type == "nan":
        sel = da.notnull()
    elif type == "bool":
        sel = da
    return surface_cell(da, **kwargs).where(sel).sum(["latitude", "longitude"])
