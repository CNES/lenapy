"""
This module implements some usuals functions to be applied on gridded data (lat/lon)
"""

# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
import xesmf as xe

from lenapy.utils.geo import *
from lenapy.utils.time import *


@xr.register_dataset_accessor("lngeo")
class GeoSet:
    """
    This class implements an extension of any dataset to add some usefull methods often used on gridded data
    in earth science data handling.
    """

    def __init__(self, xarray_obj):
        assert_grid(xarray_obj)
        self._obj = xarray_obj

    def mean(self, *args, **kwargs):
        """
        Returns the averaged value of all variables in dataset along specified dimension, applying specified weights.

        Parameters
        ----------
        *args : list
            List of the dimensions along which to sum.
        weights : None | list | xr.DataArray
            If None, no weight is applied.
            If 'latitude', a weight is applied as the cos of the latitude.
            If 'latitude_ellipsoid', a weight is applied as the cos of the latitude multiplied by an oblateness factor.
            If 'depth', a weight is applied as the thickness of the layer.
            If xr.DataArray is given : input data are multiplied by this object before summing.
        mask : None | xr.DataArray
            Mask to be applied before averaging.
        na_eq_zero : boolean (default=False)
            Replace NaN values by zeros. The averaging is then applyed on all data, and not only valid ones.
        **kwargs : keyword arguments
            Any keyword arguments given to the native xr.mean function.

        Returns
        -------
        averaged : xr.Dataset
            Dataset with all variables averaged according to specified options.

        Example
        -------
        .. code-block:: python

            data = lngeo.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
            avg = data.lngeo.mean(['latitude', 'longitude'], weights=['latitude'], na_eq_zero=True)
        """
        res = {}
        for var in self._obj.data_vars:
            res[var] = self._obj[var].lngeo.mean(*args, **kwargs)
        return xr.Dataset(res)

    def sum(self, *args, **kwargs):
        """
        Returns the sum for all variables in dataset along specified dimension, applying specified weights.

        Parameters
        ----------
        *args : list
            List of the dimensions along which to sum.
        weights : None | list | xr.DataArray
            If None, no weight is applied.
            If 'latitude', a weight is applied as the cos of the latitude.
            If 'latitude_ellipsoid', a weight is applied as the cos of the latitude multiplied by an oblateness factor.
            If 'depth', a weight is applied as the thickness of the layer.
            If xr.DataArray is given : input data are multiplied by this object before summing.
        mask : None | xr.DataArray
            Mask to be applied before summing up.
        **kwargs : keyword arguments
            Any keyword arguments given to the native xr.sum function.

        Returns
        -------
        averaged : xr.Dataset
            Dataset with all variables summed according to specified options.

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('/home/user/lenapy/data/isas.nc', engine="lenapyNetcdf")
            avg = data.lngeo.sum(['latitude','longitude'],weights=['latitude'])
        """
        res = {}
        for var in self._obj.data_vars:
            res[var] = self._obj[var].lngeo.sum(*args, **kwargs)
        return xr.Dataset(res)

    def isosurface(self, criterion, dim, coord=None, upper=False):
        """
        Compute the isosurface along the specified coordinate at the value defined by the kwarg field=value.
        For example, we want to compute the isosurface defined by a temperature of 10°C along depth dimension.
        All data variables of the Dataset are interpolated on this isosurface.
        Data is supposed to be monotonic along the chosen dimension. If not, the first fitting value encountered is
        retained, starting from the end (bottom) if upper=False, or from the beggining (top) if upper=True.
        Use :func:`lenapy.utils.geo.isosurface` on each DataArray.

        Parameters
        ----------
        criterion : dict
            One-entry dictionary with the key equal to a variable of the dataset, and
            the value equal to the isosurface criterion.
        dim : str
            Dimension along which to compute the isosurface.
        coord : str (optional)
            The field coordinate to interpolate. If absent, coordinate is supposed to be "dim".
        upper : bool (default=False)
            Order to perform the research of the criterion value. If True, returns the highest point of the isosurface,
            else the lowest.

        Returns
        -------
        isosurface : dataset
            Dataset with all the variables interpolated at the criterion value along chosen dimension. The variables
            chosen for a criterion should contain a constant value equal to the criterion. The dimension chosen for the
            isosurface computation is filled with the isosurface itself.

        Returns
        -------
        isosurface : xr.DataArray
            Dataarray containing the isosurface along the dimension dim on which data=target.

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('/home/user/lenapy/data/isas.nc', engine="lenapyNetcdf")
            data.isosurface('depth',dict(temp=3))

            # Output: <xarray.Dataset>
            # Dimensions:    (latitude: 90, longitude: 180)
            # Coordinates:
            #   * latitude   (latitude) float32 -44.5 -43.5 -42.5 -41.5 ... 42.5 43.5 44.5
            #   * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 177.0 178.0 179.0 180.0
            #     time       datetime64[ns] 2005-01-15
            #     depth      (latitude, longitude) float64 918.6 745.8 704.8 ... 912.2 920.0
            # Data variables:
            #     depth_iso  (latitude, longitude) float64 918.6 745.8 704.8 ... 912.2 920.0
            #     temp       (latitude, longitude) float64 3.0 3.0 3.0 3.0 ... 3.0 3.0 3.0 3.0
            #     SA         (latitude, longitude) float64 34.48 34.39 34.39 ... 34.53 34.52

        """
        # Compute the isosurface along coord 'dim' for the field define in the dict **kwargs (ex : temp=10)
        # Return interpolated field on the isosurface (for those with a "dim" coord) and the isosurface itself
        if coord is None:
            coord = dim
        k = list(criterion.keys())[0]
        if k not in self._obj.data_vars:
            raise KeyError("%s not in %s" % (criterion[0], list(self._obj.data_vars)))

        r = isosurface(self._obj[k], criterion[k], dim, coord, upper=upper)
        res = xr.Dataset()
        for var in self._obj.data_vars:
            if coord in self._obj[var].coords:
                res[var] = self._obj[var].interp({coord: r})
            else:
                res[var] = self._obj[var]

        return res

    def regridder(self, gr_out, *args, mask_in=None, mask_out=None, **kwargs):
        """
        Implement a xesmf regridder instance to be used with regrid method to perform regridding from a xarray object
        coordinates to gr_out coordinates.
        The resampling method can be changed (see xesmf documentation).

        Parameters
        ----------
        gr_out : xr.Dataset | xr.DataArray
            Dataset containing the coordinates to regrid on.
        *args :
            Any argument passed to xesmf.Regridder method.
        mask_in : None | xr.DataArray
            Mask to be applied on the data to regrid.
        mask_out : None | xr.DataArray
            Mask of the output Dataset to regrid.
        *kwargs :
            Any keyword argument passed to xesmf.Regridder method.

        Returns
        -------
        regridder : xesmf.Regridder object
            regridder to be used with lngeo.regrid to perform regridding from dataset coordinates to gr_out coordinates.
        """
        assert_grid(gr_out)

        ds = self._obj
        if type(mask_in) is xr.DataArray:
            ds["mask"] = mask_in

        ds_out = xr.Dataset(
            {
                "latitude": gr_out.latitude,
                "longitude": gr_out.longitude,
            }
        )
        if mask_out is not None:
            ds_out["mask"] = mask_out

        return xe.Regridder(ds, ds_out, *args, **kwargs)

    def regrid(self, regridder, *args, **kwargs):
        """
        Implement the xesmf regrid method to perform regridding from Dataset coordinates to gr_out coordinates.

        Parameters
        ----------
        regridder : xesmf.Regridder instance
            regridder set with the lngeo.regridder method.
        *args :
            Any argument passed to xesmf regridder method.
        *kwargs :
            Any keyword argument passed to xesmf regridder method.

        Returns
        -------
        regrid : xr.Dataset
            Dataset regridded to gr_out coordinates.

        Example
        -------
        .. code-block:: python

            ds_out = xr.Dataset({"latitude":(["latitude"], np.arange(-89.5, 90, 1.)),
                                 "longitude":(["longitude"], np.arange(-179.5, 180, 1.))})
            regridder = data.lngeo.regridder(ds_out, "conservative_normed", periodic=True)
            out = data.lngeo.regrid(regridder)
        """
        return regridder(self._obj, *args, **kwargs)

    def surface_cell(
        self, ellipsoidal_earth=True, a_earth=None, f_earth=LNPY_F_EARTH_GRS80
    ):
        """
        Returns the surface of each cell defined by a longitude/latitude in a xarray object.
        For details on the function, see :func:`lenapy.utils.geo.surface_cell` documentation.

        Returns
        -------
        surface : xr.DataArray
            DataArray with cells surface.

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('/home/user/lenapy/data/gohc_2020.nc', engine="lenapyNetcdf")
            surface = data.lngeo.surface_cell()
        """
        return surface_cell(
            self._obj,
            ellipsoidal_earth=ellipsoidal_earth,
            a_earth=a_earth,
            f_earth=f_earth,
        )

    def distance(
        self, pt, ellipsoidal_earth=False, a_earth=None, f_earth=LNPY_F_EARTH_GRS80
    ):
        """
        Returns the great-circle/geodetic distance between coordinates on a sphere/ellipsoid.
        For details on the function, see :func:`lenapy.utils.geo.distance` documentation.

        Parameters
        ----------
        pt : xr.DataArray | xr.Dataset
            Object with coordinates that are not latitude or longitude but that contain latitude and longitude.
            For example, a list of points with a coordinate "id".
        ellipsoidal_earth: bool | str, optional
            Boolean to choose if the surface of the Earth is an ellipsoid or a sphere.
            Default is False for spherical Earth.
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

        """
        return distance(
            self._obj,
            pt,
            ellipsoidal_earth=ellipsoidal_earth,
            a_earth=a_earth,
            f_earth=f_earth,
        )

    def reset_longitude(self, origin=-180):
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
        return reset_longitude(self._obj, origin)


@xr.register_dataarray_accessor("lngeo")
class GeoArray:
    """
    This class implements an extension of any dataArray to add some usefull methods often used on gridded data
    in earth science data handling.
    """

    def __init__(self, xarray_obj):
        assert_grid(xarray_obj)
        self._obj = xarray_obj

    def mean(self, *args, weights=None, mask=True, na_eq_zero=False, **kwargs):
        """
        Returns the averaged value of xr.DataArray along specified dimension, applying specified weights.

        Parameters
        ----------
        *args : list
            List of the dimensions along which to sum.
        weights : None | list | xr.DataArray
            If None, no weight is applied.
            If 'latitude', a weight is applied as the cos of the latitude.
            If 'latitude_ellipsoid', a weight is applied as the cos of the latitude multiplied by an oblateness factor.
            If 'depth', a weight is applied as the thickness of the layer.
            If xr.DataArray is given : input data are multiplied by this object before summing.
        mask : None | xr.DataArray
            Mask to be applied before averaging up.
        na_eq_zero : bool (default=False)
            Replace NaN values by zeros. The averaging is then applied on all data, and not only valid ones.
        **kwargs : keyword arguments
            Any keyword arguments given to the native xr.mean() function.

        Returns
        -------
        averaged : xr.DataArray
            DataArray averaged according to specified options.

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('/home/user/lenapy/data/isas.nc', engine="lenapyNetcdf").temp
            avg = data.lngeo.mean(['latitude','longitude'],weights=['latitude'],na_eq_zero=True)
        """
        argmean = set(np.ravel(*args)).intersection(list(self._obj.coords))
        data = self._obj.where(mask)
        if na_eq_zero:
            data = data.fillna(0.0)

        if len(argmean) == 0:
            argmean = None

        if weights is None:
            return data.mean(argmean, **kwargs)  # simple mean
        elif type(weights) is list or type(weights) is str:
            w = 1
            if (
                "latitude" in weights and "latitude" in self._obj.coords
            ):  # weight = cos(latitude)
                w = np.cos(np.radians(self._obj.latitude))
            if (
                "latitude_ellipsoid" in weights and "latitude" in self._obj.coords
            ):  # weight = cos(latitude)*oblat fact
                w = (
                    np.cos(np.radians(self._obj.latitude))
                    / (
                        1
                        + LNPY_F_EARTH_GRS80
                        * np.cos(2 * np.radians(self._obj.latitude))
                    )
                    ** 2
                )
            if "depth" in weights and "depth" in self._obj.coords:
                # poids *= épaisseur des couches (l'épaisseur de la première couche est la première profondeur)
                w = w * xr.concat(
                    (self._obj.depth.isel(depth=0), self._obj.depth.diff(dim="depth")),
                    dim="depth",
                )
            return data.weighted(w).mean(argmean, **kwargs)
        elif type(weights) is xr.DataArray:
            return data.weighted(weights).mean(
                argmean, **kwargs
            )  # weight DataArray provided
        else:
            raise ValueError(
                "Given weights argument can not be handle, it has to be either str, list or xr.DataArray."
            )

    def sum(self, *args, weights=None, mask=True, **kwargs):
        """
        Returns the sum of the xr.DataArray along specified dimension, applying specified weights.

        Parameters
        ----------
        *args : list
            List of the dimensions along which to sum.
        weights : None | list | xr.DataArray
            If None, no weight is applied.
            If 'latitude', a weight is applied as the cos of the latitude.
            If 'latitude_ellipsoid', a weight is applied as the cos of the latitude multiplied by an oblateness factor.
            If 'depth', a weight is applied as the thickness of the layer.
            If xr.DataArray is given : input data are multiplied by this object before summing.
        mask : None | xr.DataArray
            Mask to be applied before summing up.
        **kwargs : keyword arguments
            Any keyword arguments given to the native xr.sum() function.

        Returns
        -------
        summed : xr.DataArray
            DataArray summed according to specified options.

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('/home/user/lenapy/data/isas.nc', engine="lenapyNetcdf").heat
            avg = data.lngeo.sum(['latitude','longitude'],weights=['latitude'])
        """
        argsum = set(np.ravel(*args)).intersection(list(self._obj.coords))
        data = self._obj.where(mask)

        if weights is None:
            return data.sum(argsum, **kwargs)  # simple sum()
        elif type(weights) is list or type(weights) is str:
            w = 1
            if (
                "latitude" in weights and "latitude" in self._obj.coords
            ):  # weight = cos(latitude)
                w = np.cos(np.radians(self._obj.latitude))
            if (
                "latitude_ellipsoid" in weights and "latitude" in self._obj.coords
            ):  # weight = cos(latitude)*oblat fact
                w = (
                    np.cos(np.radians(self._obj.latitude))
                    / (
                        1
                        + LNPY_F_EARTH_GRS80
                        * np.cos(2 * np.radians(self._obj.latitude))
                    )
                    ** 2
                )
            if "depth" in weights and "depth" in self._obj.coords:
                # poids *= épaisseur des couches (l'épaisseur de la première couche est la première profondeur)
                w = w * xr.concat(
                    (self._obj.depth.isel(depth=0), self._obj.depth.diff(dim="depth")),
                    dim="depth",
                )
            return data.weighted(w).sum(argsum, **kwargs)
        elif type(weights) is xr.DataArray:
            return data.weighted(weights).sum(
                argsum, **kwargs
            )  # weight DataArray provided
        else:
            raise ValueError(
                "Given weights argument can not be handle, it has to be either str, list or xr.DataArray."
            )

    def isosurface(self, target, dim, coord=None, upper=False):
        """
        Linearly interpolate a coordinate isosurface where a field equals a target.
        Compute the isosurface along the specified coordinate at the value defined by the target argument.
        Data is supposed to be monotonic along the chosen dimension. If not, the first fitting value encountered is
        retained, starting from the end (bottom) if upper=False, or from the beggining (top) if upper=True.
        Use :func:`lenapy.utils.geo.isosurface` with an equivalent documentation.

        Parameters
        ----------
        target : float
            Criterion value to be satisfied at the isosurface.
        dim : str
            Dimension along which to compute the isosurface.
        coord : str (optional)
            The field coordinate to interpolate. If absent, coordinate is supposed to be "dim".
        upper : bool (default=False)
            Order to perform the research of the criterion value. If True, returns the highest point of the isosurface,
            else the lowest.

        Returns
        -------
        isosurface : xr.DataArray
            Dataarray containing the isosurface along the dimension dim on which data=target.

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('/home/user/lenapy/data/isas.nc', engine="lenapyNetcdf").temp
            data.isosurface(3,'depth')
            # Output : <xarray.DataArray (latitude: 90, longitude: 180)>
            # dask.array<_nanmax_skip-aggregate, shape=(90, 180), dtype=float64,
            #            chunksize=(90, 180), chunktype=numpy.ndarray>
            # Coordinates:
            #   * latitude   (latitude) float32 -44.5 -43.5 -42.5 -41.5 ... 42.5 43.5 44.5
            #   * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 177.0 178.0 179.0 180.0
            #     time       datetime64[ns] 2005-01-15
        """
        return isosurface(self._obj, target, dim, coord, upper=upper)

    def regridder(self, gr_out, *args, mask_in=None, mask_out=None, **kwargs):
        """
        Implement a xesmf regridder instance to be used with regrid method to perform regridding from a xarray object
        coordinates to gr_out coordinates.
        The resampling method can be changed (see xesmf documentation).

        Parameters
        ----------
        gr_out : xr.Dataset | xr.DataArray
            Dataset containing the coordinates to regrid on.
        *args :
            Any argument passed to xesmf.Regridder method.
        mask_in : None | xr.DataArray
            Mask to be applied on the data to regrid.
        mask_out : None | xr.DataArray
            Mask of the output Dataset to regrid.
        *kwargs :
            Any keyword argument passed to xesmf.Regridder method.

        Returns
        -------
        regridder : xesmf.Regridder object
            regridder to be used with lngeo.regrid to perform regridding from dataset coordinates to gr_out coordinates.
        """
        ds = xr.Dataset({"data": self._obj})
        return ds.lngeo.regridder(
            gr_out, *args, mask_in=mask_in, mask_out=mask_out, **kwargs
        )

    def regrid(self, regridder, *args, **kwargs):
        """
        Implement the xesmf regrid method to perform regridding from DataArray coordinates to gr_out coordinates.

        Parameters
        ----------
        regridder : xesmf.Regridder instance
            regridder set with the lngeo.regridder method.
        *args :
            Any argument passed to xesmf regridder method.
        *kwargs :
            Any keyword argument passed to xesmf regridder method.

        Returns
        -------
        regrid : xr.DataArray
            DataArray regridded to gr_out coordinates.

        Example
        -------
        .. code-block:: python

            ds_out = xr.Dataset({"latitude":(["latitude"], np.arange(-89.5, 90, 1.)),
                                 "longitude":(["longitude"], np.arange(-179.5, 180, 1.))})
            regridder = data.lngeo.regridder(ds_out, "conservative_normed", periodic=True)
            out = data.lngeo.regrid(regridder)
        """
        return regridder(self._obj, *args, **kwargs)

    def surface_cell(
        self, ellipsoidal_earth=True, a_earth=None, f_earth=LNPY_F_EARTH_GRS80
    ):
        """
        Returns the surface of each cell defined by a longitude/latitude in a xarray object.
        For details on the function, see :func:`lenapy.utils.geo.surface_cell` documentation.

        Returns
        -------
        surface : xr.DataArray
            DataArray with cells surface.

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('/home/user/lenapy/data/gohc_2020.nc', engine="lenapyNetcdf")
            surface = data.lngeo.surface_cell()
        """
        return surface_cell(
            self._obj,
            ellipsoidal_earth=ellipsoidal_earth,
            a_earth=a_earth,
            f_earth=f_earth,
        )

    def distance(
        self, pt, ellipsoidal_earth=False, a_earth=None, f_earth=LNPY_F_EARTH_GRS80
    ):
        """
        Returns the great-circle/geodetic distance between coordinates on a sphere/ellipsoid.
        For details on the function, see :func:`lenapy.utils.geo.distance` documentation.

        Parameters
        ----------
        pt : xr.DataArray | xr.Dataset
            Object with coordinates that are not latitude or longitude but that contain latitude and longitude.
            For example, a list of points with a coordinate "id".
        ellipsoidal_earth: bool | str, optional
            Boolean to choose if the surface of the Earth is an ellipsoid or a sphere.
            Default is False for spherical Earth.
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

        """
        return distance(
            self._obj,
            pt,
            ellipsoidal_earth=ellipsoidal_earth,
            a_earth=a_earth,
            f_earth=f_earth,
        )

    def reset_longitude(self, origin=-180):
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
        return reset_longitude(self._obj, origin)

    def geomean(self):
        return self.mean(["latitude", "longitude"], weights="latitude")

    def surface_grid(self, type="nan", **kwargs):
        return surface_grid(self._obj, type=type, **kwargs)
