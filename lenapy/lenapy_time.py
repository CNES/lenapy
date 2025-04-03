"""This module implements some usuals functions to be applied on timeseries

"""

# -*- coding: utf-8 -*-

import os.path

import numpy as np
import xarray as xr
import xesmf as xe

from lenapy.plots.plotting import *
from lenapy.utils.climato import *
from lenapy.utils.covariance import *
from lenapy.utils.eof import *
from lenapy.utils.time import *


@xr.register_dataset_accessor("lntime")
class TimeSet:
    """This class implements an extension of any dataset to add some usefull methods often used on timeseries in earth science data handling"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if "time" not in list(xarray_obj.keys()) + list(xarray_obj.coords):
            raise AssertionError("The time coordinates does not exist")

    def Coeffs_climato(self, **kwargs):
        return Coeffs_climato(self._obj, **kwargs)

    def climato(self, **kwargs):
        """Perform climato analysis on all the variables in a dataset
        Input data are decomposed into :
        * annual cycle
        * semi-annual cycle
        * trend
        * mean
        * residual signal
        The returned data are a combination of these elements depending on passed arguments (signal, mean, trend, cycle)
        If return_coeffs=True, the coefficients of the decompositions are returned

        Parameters
        ----------
        signal : Bool (default=True)
            returns residual signal
        mean : Bool (default=True)
            returns mean signal
        trend : Bool (default=True)
            returns trend (unit=day**-1)
        cycle : Bool (default=False)
            return annual and semi-annual cycles (cos and sin)
        return_coeffs : Bool (default=False)
            returns cycle coefficient, mean and trend
        time_period : slice (default=slice(None,None), ie the whole time period of the data)
            Reference time period when climatology has to be computed
        fillna : Bool (default=False)
            if fillna=True and signal=True, Nan in signal is replaced by the other selected components
            Only for 1D signal, for higher dimensions any NaN in the signal will produce a NaN in the output

        Returns
        -------
        climato : dataset
            a dataset with the same structure as the input, with modified data according to the chosen options
        if return_coeffs=True, an extra dataset is provided with the coefficients of the decomposition

        Example
        -------
        .. code-block:: python

            data = lntime.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
            output,coeffs = data.lntime.climato(mean=True, trend=True, signal=True,return_coeffs=True)

        """

        # Pour toutes les données dépendant du temps, retourne l'analyse de la climato
        res = {}
        for var in self._obj.data_vars:
            if "time" in self._obj[var].coords:
                res[var] = climato(self._obj[var], **kwargs)
            else:
                res[var] = self._obj[var]
        return xr.Dataset(res)

    def generate_climato(self, coeffs, **kwargs):
        """
        Returns a signal based on a given climatology (mean, trend, cycles)

        Parameters
        ----------
        coeffs : xr.DataArray
            returned by the climato method with return_climato=True
        mean: Bool (default=True)
            returns mean signal
        trend: Bool (default=True)
            returns trend
        cycle: Bool (default=False)
            return annual and semi-annual cycles
        """
        return generate_climato(self._obj.time, coeffs, **kwargs)

    def filter(self, filter_name="lanczos", q=3, **kwargs):
        """
        Apply a specified filter on all the time-dependent data in the dataset.
        Boundaries are handled by operating a mirror operation on the residual data after removing a q-order polyfit
        from the data. Available filters are in the .utils python file

        Parameters
        ----------
        filter_name : function or str
            if string, filter function name, from the .filters file
            if function, external function defined by user, returning a kernel
        q : int
            order of the polyfit to handle boundary effects
        **kwargs :
            Keyword arguments for the chosen filter

        Returns
        -------
        filtered : xr.Dataset
            Filtered dataset

        Example
        -------
        >>>data = lntime.open_geodata('/home/user/lenapy/data/isas.nc')
        >>>data.lntime.filter(lanczos,q=3,coupure=12,order=2)

        """
        res = {}
        for var in self._obj.data_vars:
            if "time" in self._obj[var].coords:
                res[var] = self._obj[var].lntime.filter(
                    filter_name=filter_name, q=q, **kwargs
                )
            else:
                res[var] = self._obj[var]
        return xr.Dataset(res)

    def interp_time(self, other, **kwargs):
        """
        Interpolate DataArray at the same dates than other

        Parameters
        ----------
        other : xr.DataArray
            must have a time dimension

        Return
        ------
        interpolated : xr.DataArray
            new DataArray interpolated
        """
        res = {}
        for var in self._obj.data_vars:
            if "time" in self._obj[var].coords:
                res[var] = self._obj[var].lntime.interp_time(other, **kwargs)
            else:
                res[var] = self._obj[var]
        return xr.Dataset(res)

    def to_datetime(self, time_type):
        """
        Convert dataset time format to standard pandas time format

        Parameters
        ----------
        time_type : string
            Can be 'frac_year' or '360_day'

        Return
        ------
        converted : dataset
            new dataset with the time dimension in a standard pandas format
        """
        return to_datetime(self._obj, time_type)

    def fill_time(self):
        """
        Fill missing values in a timeseries in adding some new points, by respecting the time sampling. Missing values are not NaN
        but real absent points in the timeseries. A linear interpolation is performed at the missing points.
        """
        return fill_time(self._obj)


@xr.register_dataarray_accessor("lntime")
class TimeArray:
    """
    This class implements an extension of any dataArray to add some usefull methods often used on timeseries in
    earth science data handling.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if "time" not in xarray_obj.coords:
            raise AssertionError("The time coordinates does not exist")

    def Coeffs_climato(self, **kwargs):
        return Coeffs_climato(self._obj, **kwargs)

    def climato(self, **kwargs):
        """
        Perform climato analysis on a dataarray
        Input data are decomposed into :
        * annual cycle
        * semi-annual cycle
        * trend
        * mean
        * residual signal
        The returned data are a combination of these elements depending on passed arguments (signal, mean, trend, cycle)
        If return_coeffs=True, the coefficients of the decompositions are returned

        Parameters
        ----------
        signal: Bool (default=True)
            returns residual signal
        mean: Bool (default=True)
            returns mean signal
        trend: Bool (default=True)
            returns trend (unit=day**-1)
        cycle: Bool (default=False)
            return annual and semi-annual cycles (cos and sin)
        return_coeffs: Bool (default=False)
            returns cycle coefficient, mean and trend
        time_period: slice (default=slice(None,None), ie the whole time period of the data)
            Reference time period when climatology has to be computed
        fillna: Bool (default=False)
            if fillna=True and signal=True, Nan in signal is replaced by the other selected components
            Only for 1D signal, for higher dimensions any NaN in the signal will produce a NaN in the output

        Returns
        -------
        climato : dataset
            a dataset with the same structure as the input, with modified data according to the chosen options
        if return_coeffs=True, an extra dataset is provided with the coefficients of the decomposition

        Example
        -------
        .. code-block:: python

            data = lntime.open_geodata('/home/user/lenapy/data/gohc_2020.nc').ohc
            output,coeffs = data.lntime.climato(mean=True, trend=True, signal=True,return_coeffs=True)

        """
        return climato(self._obj, **kwargs)

    def generate_climato(self, coeffs, **kwargs):
        """
        Returns a signal based on a given climatology (mean, trend, cycles)

        Parameters
        ----------
        coeffs: DataArray
            returned by the climato method with return_climato=True
        mean: Bool (default=True)
            returns mean signal
        trend: Bool (default=True)
            returns trend
        cycle: Bool (default=False)
            return annual and semi-annual cycles
        """
        return generate_climato(self._obj.time, coeffs, **kwargs)

    def filter(self, filter_name="lanczos", q=3, **kwargs):
        """
        Apply a specified filter on all the time-dependent datarray
        Boundaries are handled by operating a mirror operation on the residual data after removing a q-order polyfit
        from the data. Available filters are in the .utils python file

        Parameters
        ----------
        filter_name : function or string
            if string, filter function name, from the .filters file
            if function, external function defined by user, returning a kernel
        q : int
            order of the polyfit to handle boundary effects
        **kwargs :
            keyword arguments for the chosen filter

        Returns
        -------
        filtered : filtered dataset

        Example
        -------
        .. code-block:: python

            data = lntime.open_geodata('/home/user/lenapy/data/isas.nc').temp
            data.lntime.filter(lanczos,q=3,coupure=12,order=2)

        """
        return filter(self._obj, filter_name=filter_name, q=q, **kwargs)

    def interp_time(self, other, **kwargs):
        """
        Interpolate DataArray at the same dates than other

        Parameters
        ----------
        other : xr.DataArray
            must have a time dimension

        Return
        ------
        interpolated : xr.DataArray
            new DataArray interpolated
        """
        return interp_time(self._obj, other, **kwargs)

    def plot(self, **kwargs):
        """
        Plots the timeseries of the data in the TimeArray, including an uncertainty.
        Computes the uncertainty on all dimensions that are not time.

        Parameters
        ----------
        thick_line: String (default='median')
            How to aggregate the data to plot the main thick line. Can be:
            * `median`: computes the median
            * `mean`: computes the mean
            * None: does not plot a main thick line
        shaded_area: String (default='auto')
            How to aggregate the data to plot the uncertainty around the thick line. Can be:
            * `auto`: plots 1.645 standard deviation if thick_line is `mean` and quantiles 5-95 if thick_line is `median`.
            * `auto-multiple`: plots 1,2 and 3 standard deviations if thick_line is `mean` and quantiles 5-95, 17-83 and 25-75 if thick_line is `median`.
            * `std`: plots a multiple of the standard deviation based on kwarg `standard_deviation_multiple`
            * `quantiles`: plots quantiles based on the kwargs `quantile_min` and `quantile_max`
            * None: does not plot uncertainty
        hue: String (default=None)
            Similar to hue in xarray.DataArray.plot(hue=...), group data by the dimension before aggregating and computing uncertainties.
            Has to be a dimension other than time in the dataarray.
        standard_deviation_multiple: Float > 0 (default=1.65)
            The multiple of standard deviations to use for the uncertainty with `shaded_area=std`
        quantile_min: Float between 0 and 1 (default=0.05)
            lower quantile to compute uncertainty with `shaded_area=quantiles`
        quantile_max: Float between 0 and 1 (default=0.95)
            upper quantile to compute uncertainty with `shaded_area=quantiles`
        color: String or List (default=None)
            color of the main thick line and the shaded area. Must be a string
        thick_line_color: String or List (default=None)
            color of the main thick line. Must be a string
            If hue and one color are provided, the single color is used for all line plots.
            If hue and a list of colors are provided, the colors are cycled.
        shaded_area_color: String or List (default=None)
            color of the shaded area. Must be a string.
            If not provided, defaults to the thick_line_color value.
            If hue and one color are provided, the single color is used for all area plots.
            If hue and a list of colors are provided, the colors are cycled.
        shaded_area_alpha: Float between 0 and 1 (default=0.2)
            Transparency of the uncertainty plots
        ax: matplotlib.pyplot.Axes instance (default=None)
            If not provided, plots on the current axes.
        label: String (default=None)
            If provided, label that is provided to ax.plot.
            Does not work if hue is provided.
        line_kwargs: kwargs
            Additional arguments provided to the plot function for the main thick line
        area_kwargs: kwargs
            Additional arguments provided to the plot function for the uncertainty
        add_legend: Bool (default=True)
            if True, adds matplotlib legend to the current ax after plotting the data.
        """
        plot_timeseries_uncertainty(self._obj, **kwargs)

    def to_datetime(self, time_type):
        """
        Convert DataArray time format to standard pandas time format

        Parameters
        ----------
        time_type : string
            Can be 'frac_year' or '360_day'

        Return
        ------
        converted : xr.DataArray
            new DataArray with the time dimension in a standard pandas format
        """
        return to_datetime(self._obj, time_type)

    def diff_3pts(self, dim, **kw):
        """
        Derivative formula along the selected dimension, returning on each point the linear regression on the three points
        defined by the selected point and its two neighbours
        """
        return diff_3pts(self._obj, dim, **kw)

    def diff_2pts(self, dim, **kw):
        """
        Derivative formula along the selected dimension, returning for each pair of points the slope, set at the middle coordinates of
        these two points
        """
        return diff_2pts(self._obj, dim, **kw)

    def trend(self, time_unit="1s"):
        """
        Perform a linear regression on the data, and returns the slope coefficient
        """
        return trend(self._obj, time_unit=time_unit)

    def detrend(self):
        """
        remove the trend from a dataarray
        """
        return detrend(self._obj)

    def fill_time(self):
        """
        Fill missing values in a timeseries in adding some new points, by respecting the time sampling. Missing values are not NaN
        but real absent points in the timeseries. A linear interpolation is performed at the missing points.
        """
        return fill_time(self._obj)

    def covariance_analysis(self):
        """
        Returns an instance of the *covariance* class based on the dataArray
        """
        return covariance(self._obj.time)

    def OLS(self, degree, tref=None, sigma=None, datetime_unit="s"):
        """
        Returns the OLS estimator performed with a degree "degree" regression
        """
        est = estimator(
            self._obj, degree, tref=tref, sigma=sigma, datetime_unit=datetime_unit
        )
        est.OLS()
        return est

    def GLS(self, degree, tref=None, sigma=None, datetime_unit="s"):
        """
        Returns the GLS estimator performed with a degree "degree" regression and a covariance matrix "sigma"
        """
        est = estimator(
            self._obj, degree, tref=tref, sigma=sigma, datetime_unit=datetime_unit
        )
        est.GLS()
        return est

    def corr(self, other, remove_trend=False, **kwargs):
        """
        Returns the Pearson correlation coefficient between the timeseries and another one. The other one is
        interpolated at the dates of the calling timeseries.
        If remove_trend=True, the two timeseries are detrended before correlation.
        """
        if remove_trend:
            r1 = detrend(interp_time(other, self._obj))
            r2 = detrend(self._obj)
        else:
            r1 = interp_time(other, self._obj)
            r2 = self._obj

        return xr.corr(r1, r2, **kwargs)

    def fillna_climato(self, time_period=slice(None, None)):
        """
        Returns a DataArray with all NaN values replaced by climatology and trend
        Climatology is computed over the optional time_period slice
        """
        return fillna_climato(self._obj, time_period=time_period)

    def EOF(self, dim, k):
        """
        Return an instance of the *eof* class based on the data array and the dimension names of the eof
        """
        return EOF(self._obj, dim, k)
