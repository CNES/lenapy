# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import xesmf as xe
import os.path
from .utils import *
from .covariance import *
from .plotting import *
from .sandbox import *
from .produits import rename_data

@xr.register_dataset_accessor("xtime")
class TimeSet:
    """
    This class implements an extension of any dataset to add some usefull methods often used in earth science data handling
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if not 'time' in xarray_obj.coords: raise AssertionError('The time coordinates does not exist')
        
    def climato(self,**kwargs):
        """
        Perform climato analysis on all the variables in a dataset
        Input data are decomposed into :
            annual cycle
            semi-annual cycle
            trend
            mean
            residual signal
        The returned data are a combination of these elements depending on passed arguments (signal, mean, trend, cycle)
        If return_coeffs=True, the coefficients of the decompositions are returned
        
        Parameters
        ----------
        signal : Bool (default=True)
            returns residual signal
        mean : Bool (default=True)
            returns mean signal
        trend : Bool (default=True)
            returns trend
        cycle : Bool (default=False)
            return annual and semi-annual cycles
        return_coeffs : Bool (default=False)
            returns cycle coefficient, mean and trend
        time_period : slice (defalut=slice(None,None), ie the whole time period of the data)
            Reference time period when climatology has to be computed
            
        Returns
        -------
        climato : dataset
            a dataset with the same structure as the input, with modified data according to the chosen options
        if return_coeffs=True, an extra dataset is provided with the coefficients of the decomposition
        
        Example
        -------
        >>>data = xtime.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
        >>>output,coeffs = data.xtime.climato(mean=True, trend=True, signal=True,return_coeffs=True)
        """

        # Pour toutes les données dépendant du temps, retourne l'analyse de la climato
        res={}
        for var in self._obj.data_vars:
            if 'time' in self._obj[var].coords:
                res[var]=climato(self._obj[var],**kwargs)
            else:
                res[var]=self._obj[var]
        return xr.Dataset(res)
    
    def filter(self, filter_name='lanczos',q=3, **kwargs):
        """
        Apply a specified filter on all the time-dependent data in the dataset
        Boundaries are handled by operating a mirror operation on the residual data after removing a q-order polyfit from the data
        Available filters are in the .utils python file
        
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
        >>>data = xtime.open_geodata('/home/user/lenapy/data/isas.nc')
        >>>data.xtime.filter(lanczos,q=3,coupure=12,order=2)
        
        """
        res={}
        for var in self._obj.data_vars:
            if 'time' in self._obj[var].coords:
                res[var]=self._obj[var].xtime.filter(filter_name=filter_name,q=q,**kwargs)
            else:
                res[var]=self._obj[var]
        return xr.Dataset(res)

    def interp_time(self,other,**kwargs):
        """
        Interpolate dataarray at the same dates than other
        
        Parameter
        ---------
        other : dataarray
            must have a time dimension
            
        Return
        ------
        interpolated : dataarray
            new dataarray interpolated
        """        
        res={}
        for var in self._obj.data_vars:
            if 'time' in self._obj[var].coords:
                res[var]=self._obj[var].xtime.interp_time(other,**kwargs)
            else:
                res[var]=self._obj[var]
        return xr.Dataset(res)

    def to_datetime(self,input_type):
        """
        Convert dataset time format to standard pandas time format
        
        Parameter
        ---------
        input_type : string
            Can be 'frac_year' or '360_day'
            
        Return
        ------
        converted : dataset
            new dataset with the time dimension in a standard pandas format
        """
        return to_datetime(self._obj,input_type)        

    def fill_time(self):
        """
        Fill missing values in a timeseries in adding some new points, by respecting the time sampling. Missing values are not NaN
        but real absent points in the timeseries. A linear interpolation is performed at the missing points.
        """
        return fill_time(self._obj)
    
@xr.register_dataarray_accessor("xtime")
class TimeArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if not 'time' in xarray_obj.coords: raise AssertionError('The time coordinates does not exist')
        
    def climato(self,**kwargs):
        """
        Perform climato analysis on a dataarray
        Input data are decomposed into :
            annual cycle
            semi-annual cycle
            trend
            mean
            residual signal
        The returned data are a combination of these elements depending on passed arguments (signal, mean, trend, cycle)
        If return_coeffs=True, the coefficients of the decompositions are returned
        
        Parameters
        ----------
        signal : Bool (default=True)
            returns residual signal
        mean : Bool (default=True)
            returns mean signal
        trend : Bool (default=True)
            returns trend
        cycle : Bool (default=False)
            return annual and semi-annual cycles
        return_coeffs : Bool (default=False)
            returns cycle coefficient, mean and trend
        time_period : slice (defalut=slice(None,None), ie the whole time period of the data)
            Reference time period when climatology has to be computed
            
        Returns
        -------
        climato : dataarray
            a dataarray with the same structure as the input, with modified data according to the chosen options
        if return_coeffs=True, an extra dataset is provided with the coefficients of the decomposition
        
        Example
        -------
        >>>data = xtime.open_geodata('/home/user/lenapy/data/gohc_2020.nc').ohc
        >>>output,coeffs = data.xtime.climato(mean=True, trend=True, signal=True,return_coeffs=True)
        """
        return climato(self._obj,**kwargs)
    
    def filter(self, filter_name='lanczos',q=3, **kwargs):
        """
        Apply a specified filter on all the time-dependent datarray
        Boundaries are handled by operating a mirror operation on the residual data after removing a q-order polyfit from the data
        Available filters are in the .utils python file
        
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
        >>>data = xtime.open_geodata('/home/user/lenapy/data/isas.nc').temp
        >>>data.xtime.filter(lanczos,q=3,coupure=12,order=2)
        
        """        
        return filter(self._obj,filter_name=filter_name,q=q,**kwargs)
    
    def interp_time(self,other,**kwargs):
        """
        Interpolate dataarray at the same dates than other
        
        Parameter
        ---------
        other : dataarray
            must have a time dimension
            
        Return
        ------
        interpolated : dataarray
            new dataarray interpolated
        """        
        return interp_time(self._obj,other,**kwargs)
    
    def plot_timeseries_uncertainty(self, **kwargs):
        plot_timeseries_uncertainty(self._obj, **kwargs)
        
    def to_datetime(self,input_type):
        """
        Convert dataarray time format to standard pandas time format
        
        Parameter
        ---------
        input_type : string
            Can be 'frac_year' or '360_day'
            
        Return
        ------
        converted : dataarray
            new dataarray with the time dimension in a standard pandas format
        """
        return to_datetime(self._obj,input_type)        

    def diff_3pts(self,dim):
        """
        Derivative formula along the selected dimension, returning on each point the linear regression on the three points
        defined by the selected point and its two neighbours
        """
        return diff_3pts(self._obj,dim)    

    def diff_2pts(self,dim,**kw):
        """
        Derivative formula along the selected dimension, returning for each pair of points the slope, set at the middle coordinates of
        these two points
        """
        return diff_2pts(self._obj,dim,**kw)    
    
    def trend(self):
        """
        Perform a linear regression on the data, and returns the slope coefficient
        """
        return trend(self._obj)
    
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
        
        return covariance(self._obj)