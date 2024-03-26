# -*- coding: utf-8 -*-
import operator
import numbers
import xarray as xr
from lenapy.utils.harmo import *
from lenapy.utils.gravity import change_reference, change_tide_system
from lenapy.plots.plotting import plot_hs, plot_power_hs


@xr.register_dataset_accessor("lnharmo")
class HarmoSet:
    """
    This class implement an extension of any xr.Dataset to add some methods related to spherical harmonics decomposition
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
        assert_sh(xarray_obj)
        self._obj = xarray_obj

    def __add__(self, other):
        return self._apply_operator(operator.add, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._apply_operator(operator.sub, other)

    def __rsub__(self, other):
        return (self.__neg__()).xharmo.__add__(other)

    def __mul__(self, other):
        return self._apply_operator(operator.mul, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        self.ds = self._obj.copy(deep=True)

        assert_sh(self._obj)
        self.ds['clm'] = self._obj.clm.__neg__()
        self.ds['slm'] = self._obj.slm.__neg__()

        return self.ds

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
        other : int, float, complex, xr.Dataset
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

        # case where other is a number (int, float, complex)
        if isinstance(other, numbers.Number):
            self.ds['clm'] = op(self._obj.clm, other)
            self.ds['slm'] = op(self._obj.slm, other)

        # case where other is another xr.DataSet with spherical harmonics info
        elif isinstance(other, xr.Dataset) or isinstance(other, xr.DataArray):
            assert_sh(other)

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
            raise AssertionError("Variable used for the operation need to be a number or a xr.Dataset / xr.DataArray.")

        return self.ds

    def to_grid(self, **kwargs):
        return sh_to_grid(self._obj, **kwargs)

    def change_reference(self, new_radius, new_earth_gravity_constant, old_radius=None, old_earth_gravity_constant=None,
                         apply=False):
        return change_reference(self._obj, new_radius, new_earth_gravity_constant, old_radius=old_radius,
                                old_earth_gravity_constant=old_earth_gravity_constant, apply=apply)

    def change_tide_system(self, new_tide, old_tide=None, k20=None):
        return change_tide_system(self._obj, new_tide, old_tide=old_tide, k20=k20)

    def plot_hs(self, **kwargs):
        return plot_hs(self._obj, **kwargs)

    def plot_power_hs(self, **kwargs):
        return plot_power_hs(self._obj, **kwargs)


@xr.register_dataarray_accessor("lnharmo")
class HarmoArray:
    """
    This class implement an extension of any xr.DataArray to add some methods related to spatial grid representing
    spherical harmonics projected onto a grid.
    The initial dataset must contain the necessary fields to define the spherical harmonics properties
    """

    def __init__(self, xarray_obj):
        """
        Initialise the HarmoArray accessor

        Parameters
        ----------
        xarray_obj : xr.DataArray
            Input dataarray
        """
        assert_grid(xarray_obj)
        self._obj = xarray_obj

    def to_sh(self, lmax, **kwargs):
        return grid_to_sh(self._obj, lmax, **kwargs)