# -*- coding: utf-8 -*-
import xarray as xr
import operator
import numbers
from .utils_gravi import *


@xr.register_dataset_accessor("xharmo")
class HarmoSet:
    """
    This class implement an extension of any dataset to add some methods related to spherical harmonics decomposition
    The initial dataset must contain the necessary fields to define the spherical harmonics properties
    """

    def __init__(self, xarray_obj):
        """
        Initialise the HarmoSet accessor

        Parameters
        ----------
        xarray_obj : xr.Dataset
            Input dataset
        """
        self._obj = xarray_obj

    def __add__(self, other):
        return self._apply_operator(operator.add, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_operator(operator.sub, other)

    def __rsub__(self, other):
        return (self.__neg__()).__add__(other)

    def __mul__(self, other):
        return self._apply_operator(operator.mul, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        self.ds = self._obj.copy(deep=True)

        try:
            self.assert_sh()
            self.ds['clm'] = self._obj.clm.__neg__()
            self.ds['slm'] = self._obj.slm.__neg__()
        except AssertionError:
            self.assert_grid()
            self.ds = self._obj.__neg__()
        return self

    def __pow__(self, power):
        if isinstance(power, numbers.Number):
            return self._apply_operator(operator.pow, power)
        else:
            raise AssertionError("Cannot raise to power with an object ", power)

    def _apply_operator(self, op, other):
        """
        Generic function to overwrite the operator of HarmoSet object
        Apply the operator to clm and slm variables

        Parameters
        ----------
        op : operator. Operation
            function from operator library to apply
        other : int, float, complex, xr.Dataset or HarmoSet
            second variable for the operation (self being the first variable)

        Returns
        -------
        ds : xr.Dataset | xr.DataArray
            return the object after the operation (that is not self apply)

        Raises
        ------
        AssertionError
            This function cannot operate on a HarmoSet with time dimension to self without time dimension.
        """
        self.ds = self._obj.copy(deep=True)

        try:
            is_sh = self.assert_sh()
        except AssertionError:
            is_sh = not self.assert_grid()

        # case where other is a number (int, float, complex)
        if isinstance(other, numbers.Number):
            if is_sh:
                self.ds['clm'] = op(self._obj.clm, other)
                self.ds['slm'] = op(self._obj.slm, other)
            else:
                self.ds = op(self._obj, other)

        # case where other is another xr.DataSet with spherical harmonics info
        elif isinstance(other, xr.Dataset) or isinstance(other, xr.DataArray):
            if is_sh:
                other.xharmo.assert_sh()

                # change clm and slm size if other.l or other.m are different
                common_l = self._obj.l.where(self._obj.l.isin(other.l)).dropna(dim='l')
                common_m = self._obj.m.where(self._obj.m.isin(other.m)).dropna(dim='m')
                self.ds = self._obj.sel(l=common_l, m=common_m)

                # case where other does not have a time dimension
                if 'time' not in other.coords:
                    self.ds['clm'] = op(self.ds.clm, other.clm.isel(l=common_l, m=common_m))
                    self.ds['slm'] = op(self.ds.slm, other.clm.isel(l=common_l, m=common_m))

                elif 'time' not in self._obj.coords:  # if the previous test, other has time dimension
                    raise AssertionError("Cannot operate on a HarmoSet with time dimension to a Harmoset without it. "
                                         "Inverse the order of the HarmoSet in the operation.")

                # case where both xr.Dataset have a time dimension
                else:
                    common_times = self._obj.time.where(self._obj.time.isin(other.time)).dropna(dim='time')
                    self.ds = self.ds.sel(time=common_times)

                    # change clm and slm on similar time
                    self.ds['clm'] = op(self.ds.clm, other.clm.sel(time=common_times, l=common_l, m=common_m))
                    self.ds['slm'] = op(self.ds.slm, other.slm.sel(time=common_times, l=common_l, m=common_m))

            else:
                other.xharmo.assert_grid()

                # change clm and slm size if other.l or other.m are different
                common_longitude = self._obj.longitude.where(self._obj.longitude.isin(other.longitude)
                                                             ).dropna(dim='longitude')
                common_latitude = self._obj.latitude.where(self._obj.latitude.isin(other.latitude)
                                                           ).dropna(dim='latitude')
                self.ds = self._obj.sel(longitude=common_longitude, latitude=common_latitude)

                # case where other does not have a time dimension
                if 'time' not in other.coords:
                    self.ds = op(self.ds, other.isel(longitude=common_longitude, latitude=common_latitude))

                elif 'time' not in self._obj.coords:  # if the previous test, other has time dimension
                    raise AssertionError("Cannot operate on a HarmoSet with time dimension to a Harmoset without it. "
                                         "Inverse the order of the HarmoSet in the operation.")

                # case where both xr.DataArray have a time dimension
                else:
                    common_times = self._obj.time.where(self._obj.time.isin(other.time)).dropna(dim='time')
                    self.ds = self.ds.sel(time=common_times)

                    # change DataArray on similar time
                    self.ds = op(self.ds, other.sel(time=common_times,
                                                    longitude=common_longitude, latitude=common_latitude))

        else:
            print(other)
            raise AssertionError("Variable used for the operation need to be a number or a xr.Dataset / xr.DataArray.")

        return self.ds

    def assert_sh(self):
        """
        Verify if self._obj have dimensions l and m as well as variables clm and slm
        Raise Assertion error if not

        Returns
        -------
        True : bool
            return True is self._obj have dimensions l and m as well as variables clm and slm

        Raises
        ------
        AssertionError
            This function raise AssertionError is self._obj is not a xr.Dataset corresponding to spherical harmonics
        """
        if 'l' not in self._obj.coords:
            raise AssertionError("The degree coordinates that should be named 'l' does not exist")
        if 'm' not in self._obj.coords:
            raise AssertionError("The order coordinates that should be named 'm' does not exist")
        if 'clm' not in self._obj.keys():
            raise AssertionError("The Dataset have to contain 'clm' variable")
        if 'slm' not in self._obj.keys():
            raise AssertionError("The Dataset have to contain 'slm' variable")
        return True

    def assert_grid(self):
        """
        Verify if self._obj have dimensions longitude and latitude
        Raise Assertion error if not

        Returns
        -------
        True : bool
            return True is self._obj have dimensions longitude and latitude

        Raises
        ------
        AssertionError
            This function raise AssertionError is self._obj is not a xr.Dataset corresponding to spherical harmonics
        """
        if 'latitude' not in self._obj.coords:
            raise AssertionError("The latitude coordinates that should be named 'latitude' does not exist")
        if 'longitude' not in self._obj.coords:
            raise AssertionError("The longitude coordinates that should be named 'longitude' does not exist")
        return True

    def to_grid(self, **kwargs):
        self.assert_sh()
        return sh_to_grid(self._obj, **kwargs)

    def to_sh(self, lmax, **kwargs):
        self.assert_grid()
        return sh_to_grid(self._obj, lmax, **kwargs)

    def change_reference(self, new_radius, new_earth_gravity_constant, old_radius=None, old_earth_gravity_constant=None,
                         apply=False):
        self.assert_sh()
        return change_reference(self._obj, new_radius, new_earth_gravity_constant, old_radius=old_radius,
                                old_earth_gravity_constant=old_earth_gravity_constant, apply=apply)

    def change_tide_system(self, new_tide, old_tide=None, k20=None):
        self.assert_sh()
        return change_tide_system(self._obj, new_tide, old_tide=old_tide, k20=k20)
