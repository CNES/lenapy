"""
The lenapy_harmo module provides functionalities for spherical harmonics dataset (clm/slm) and
projections on latitude/longitude grids.

This module includes two classes:
  * HarmoSet: Provides methods for handling spherical harmonics decompositions, converting them to grid representations
    and performing operations such as changing reference frames and tide systems.
  * HarmoArray: Converts grid representation of a spherical harmonics dataset back to spherical harmonics.

The module is designed to work seamlessly with xarray datasets, enabling efficient manipulation and visualization.

Examples
--------
>>> import xarray as xr
>>> from lenapy import lenapy_harmo
# Load a dataset from a .gfc file
>>> ds = xr.open_dataset('example_file.gfc', engine='lenapyGfc')
# Access HarmoSet methods
>>> grid = ds.lnharmo.to_grid()
# Plot the spherical harmonics
>>> ds.lnharmo.plot_hs()
"""

import numbers
import operator

import xarray as xr

from lenapy.plots.plotting import plot_hs, plot_power
from lenapy.utils.geo import assert_grid
from lenapy.utils.gravity import change_reference, change_tide_system
from lenapy.utils.harmo import *
from lenapy.writers.gravi_writer import dataset_to_gfc


@xr.register_dataset_accessor("lnharmo")
class HarmoSet:
    """
    This class implements an extension of any xr.Dataset to add some methods related to spherical harmonics
    decomposition. The initial dataset must contain the necessary fields to define the spherical harmonics properties.

    Standardized coordinates names:
      * l, m (optional: time)

    Standardized variables names:
      * clm, slm (optional: eclm, eslm, begin_time, end_time, exact_time)
    """

    def __init__(self, xarray_obj):
        """
        Initialize the HarmoSet accessor.

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
        return (self.__neg__()).lnharmo.__add__(other)

    def __mul__(self, other):
        return self._apply_operator(operator.mul, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        self.ds = self._obj.copy(deep=True)

        assert_sh(self._obj)
        self.ds["clm"] = self._obj.clm.__neg__()
        self.ds["slm"] = self._obj.slm.__neg__()

        return self.ds

    def __pow__(self, power):
        if isinstance(power, numbers.Number):
            return self._apply_operator(operator.pow, power)
        else:
            raise AssertionError("Cannot raise to power with an object ", power)

    def _apply_operator(self, op, other):
        """
        Generic function to overwrite the operator of HarmoSet object.
        Apply the operator to clm and slm variables.

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
            self.ds["clm"] = op(self._obj.clm, other)
            self.ds["slm"] = op(self._obj.slm, other)

        # case where other is another xr.Dataset with spherical harmonics info
        elif isinstance(other, xr.Dataset):
            assert_sh(other)

            # change clm and slm size if other.l or other.m are different
            common_l = self._obj.l.where(self._obj.l.isin(other.l)).dropna(dim="l")
            common_m = self._obj.m.where(self._obj.m.isin(other.m)).dropna(dim="m")
            self.ds = self._obj.sel(l=common_l, m=common_m)

            # case where other does not have a time dimension
            if "time" not in other.coords:
                self.ds["clm"] = op(self.ds.clm, other.clm.sel(l=common_l, m=common_m))
                self.ds["slm"] = op(self.ds.slm, other.slm.sel(l=common_l, m=common_m))

            elif (
                "time" not in self._obj.coords
            ):  # if the previous test, other has time dimension
                raise AssertionError(
                    "Cannot operate on a HarmoSet with time dimension to a Harmoset without it. "
                    "Inverse the order of the HarmoSet in the operation."
                )

            # case where both xr.Dataset have a time dimension
            else:
                common_times = self._obj.time.where(
                    self._obj.time.isin(other.time)
                ).dropna(dim="time")
                self.ds = self.ds.sel(time=common_times)

                # change clm and slm on similar time
                self.ds["clm"] = op(
                    self.ds.clm,
                    other.clm.sel(time=common_times, l=common_l, m=common_m),
                )
                self.ds["slm"] = op(
                    self.ds.slm,
                    other.slm.sel(time=common_times, l=common_l, m=common_m),
                )

        else:
            raise AssertionError(
                "Variable used for the operation need to be a number or a xr.Dataset / xr.DataArray."
            )

        return self.ds

    def to_grid(self, **kwargs):
        """
        Transform Spherical Harmonics (SH) dataset into spatial DataArray.
        For details on the function, see :func:`lenapy.utils.harmo.sh_to_grid` documentation.

        Parameters
        ----------
        **kwargs :
            Supplementary parameters used by the function sh_to_grid() for conversion between SH and grid

        Returns
        -------
        xr.DataArray
            The spatial grid representation of the spherical harmonics dataset.
        """
        return sh_to_grid(self._obj, **kwargs)

    def change_reference(
        self,
        new_radius,
        new_earth_gravity_constant,
        old_radius=None,
        old_earth_gravity_constant=None,
        apply=False,
    ):
        """
        Update the reference frame for spherical harmonics.
        For details on the function, see :func:`lenapy.utils.gravity.change_reference` documentation.

        Parameters
        ----------
        new_radius : float
            New Earth radius constant in meters.
        new_earth_gravity_constant : float
            New gravitational constant of the Earth in m³/s².
        old_radius : float | None, optional
            Current Earth radius constant of the dataset in meters. If not provided, uses `ds.attrs['radius']`.
        old_earth_gravity_constant : float | None, optional
            Current gravitational constant of the Earth of the dataset in m³/s².
            If not provided, uses `ds.attrs['earth_gravity_constant']`.
        apply : bool, optional
            If True, apply the update to the current dataset without making a deep copy. Default is False.

        Returns
        -------
        ds_out : xr.Dataset
            Updated dataset with the new constants.
        """
        return change_reference(
            self._obj,
            new_radius,
            new_earth_gravity_constant,
            old_radius=old_radius,
            old_earth_gravity_constant=old_earth_gravity_constant,
            apply=apply,
        )

    def change_tide_system(self, new_tide, old_tide=None, k20=None, apply=False):
        """
        Apply a C20 offset to the dataset to change the tide system.
        For details on the function, see :func:`lenapy.utils.gravity.change_tide_system` documentation.

        Parameters
        ----------
        new_tide : str
            Output tidal system, either 'tide_free', 'zero_tide' or 'mean_tide'.
        old_tide : str | None, optional
            Input tidal system. If not provided, uses `ds.attrs['tide_system']`.
        k20 : float | None, optional
            k20 Earth tide external potential Love number. If not provided, the default value from IERS2010 is used.
        apply : bool, optional
            If True, apply the update to the current dataset without making a deep copy. Default is False.

        Returns
        -------
        ds_out : xr.Dataset
            Updated dataset with the new tidal system.
        """
        return change_tide_system(
            self._obj, new_tide, old_tide=old_tide, k20=k20, apply=apply
        )

    def change_normalization(
        self, new_normalization, old_normalization=None, apply=False
    ):
        """
        Apply a C20 offset to the dataset to change the tide system.
        For details on the function, see :func:`lenapy.utils.gravity.change_tide_system` documentation.

        Parameters
        ----------
        new_normalization : str
            New normalization for the SH dataset, either '4pi', 'ortho', or 'schmidt'.
        old_normalization : str | None, optional
            Current normalization of the SH dataset, either '4pi', 'ortho', 'schmidt', or 'unnorm'.
             If not provided, uses `ds.attrs['norm']`.
        apply : bool, optional
            If True, apply the update to the current dataset without making a deep copy. Default is False.

        Returns
        -------
        ds_out : xr.Dataset
            Updated dataset with the new normalization.
        """
        return change_normalization(
            self._obj,
            new_normalization,
            old_normalization=old_normalization,
            apply=apply,
        )

    def plot_hs(self, **kwargs):
        """
        Plot time series of spherical harmonics.
        For details on the function, see :func:`lenapy.plots.plotting.plot_hs` documentation.

        Parameters
        ----------
        **kwargs :
            Additional parameters to customize the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot of the spherical harmonics.
        """
        return plot_hs(self._obj, **kwargs)

    def plot_power(self, **kwargs):
        """
        Plot the power of the spherical harmonics.
        For details on the function, see :func:`lenapy.plots.plotting.plot_power_hs` documentation.

        Parameters
        ----------
        **kwargs :
            Additional parameters to customize the plot.

        Returns
        -------
        matplotlib.figure.Figure
            The generated plot of the power of the spherical harmonics.
        """
        return plot_power(self._obj, **kwargs)

    def to_gfc(self, filename, **kwargs):
        """
        Save the dataset to a .gfc file.
        For details on the function, see :func:`lenapy.writers.gravi_writer.dataset_to_gfc` documentation.

        Parameters
        ----------
        filename : str | os.PathLike
            The file path where to save the dataset.
        **kwargs :
            Additional parameters for the .gfc file.
        """
        dataset_to_gfc(self._obj, filename, **kwargs)


@xr.register_dataarray_accessor("lnharmo")
class HarmoArray:
    """
    This class implements an extension of any xr.DataArray to add some methods related to spatial grids representing
    spherical harmonics projected onto a grid.
    The initial dataset must contain the necessary fields to define the spherical harmonics properties.
    """

    def __init__(self, xarray_obj):
        """
        Initialize the HarmoArray accessor

        Parameters
        ----------
        xarray_obj : xr.DataArray
            Input dataarray
        """
        assert_grid(xarray_obj)
        self._obj = xarray_obj

    def to_sh(self, lmax, **kwargs):
        """
        Transform gravity field spatial representation DataArray into Spherical Harmonics (SH) dataset.
        For details on the function, see :func:`lenapy.utils.harmo.grid_to_sh` documentation.

        Parameters
        ----------
        lmax : int
            Maximal degree of the SH coefficients to be computed.
        **kwargs :
            Supplementary parameters used by the function grid_to_sh() for conversion between grid and SH
        """
        return grid_to_sh(self._obj, lmax, **kwargs)
