# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import xesmf as xe
import os.path
from .utils import *
from .plotting import *
from .sandbox import *
from .produits import rename_data

def open_geodata(file,*args,rename={},nan=None,chunks=None,**kwargs):
    """
    Open a dataset base on xr.open_dataset method, while normalizing coordinates names and choosing NaN values
    
    Parameters
    ----------
    file : path
        pathname of the file to open
    *args : optional
        any arguments passed to open_dataset method
    rename : dict, optional
        dictionnary {old_name:new_name,...}
    nan : optional
        value to be replaced by NaN
    chunks : dict, optional
        dictionnaty to perform chunks on data
    **kwargs : optional
        any keyword arguments passed to open_dataset method
        
    Returns
    -------
    data : Dataset
        Dataset loaded from file
    
    Example
    -------
    data = xgeo.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
    """
    res=rename_data(xr.open_dataset(file,*args,**kwargs),**rename)
    return res.where(res!=nan).chunk(chunks=chunks)

def open_mfgeodata(fic,*args,rename={},nan=None,chunks=None,**kwargs):
    """
    Open a dataset base on xr.open_mfdataset method, while normalizing coordinates names and choosing NaN values
    
    Parameters
    ----------
    file : path
        pattern fitting the file to open
    *args : optional
        any arguments passed to open_dataset method
    rename : dict, optional
        dictionnary {old_name:new_name,...}
    nan : optional
        value to be replaced by NaN
    chunks : dict, optional
        dictionnaty to perform chunks on data
    **kwargs : optional
        any keyword arguments passed to open_dataset method
        
    Returns
    -------
    data : Dataset
        Dataset loaded from file
    
    Example
    -------
    data = xgeo.open_mfgeodata('/home/user/lenapy/data/gohc_*.nc')
    """
    res=rename_data(xr.open_mfdataset(fic,*args,**kwargs),**rename)
    return res.where(res!=nan).chunk(chunks=chunks)
    
@xr.register_dataset_accessor("xgeo")
class GeoSet:
    """
    This class implements an extension of any dataset to add some usefull methods often used in earth science data handling
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
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
            retourne en plus les coefficients des cycles et de la tendance linéaire
            
        Returns
        -------
        climato : dataset
            a dataset with the same structure as the input, with modified data according to the chosen options
        if return_coeffs=True, an extra dataset is provided with the coefficients of the decomposition
        
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
        >>>output,coeffs = data.xgeo.climato(mean=True, trend=True, signal=True,return_coeffs=True)
        """

        # Pour toutes les données dépendant du temps, retourne l'analyse de la climato
        res={}
        for var in self._obj.data_vars:
            if 'time' in self._obj[var].coords:
                res[var]=climato(self._obj[var],**kwargs)
            else:
                res[var]=self._obj[var]
        return xr.Dataset(res)

    def mean(self,*args,**kwargs):
        """
        Returns the averaged value of all variables in dataset along specified dimension, applying specified weights
        
        Parameters
        ----------
        *args : list
            list of the dimensions along which to average
        
        weights : None or list or dataarray
            if None, no weight is applyed
            if 'latitude' or 'depth', a weight is applyed as the cosine of the latitude or 
                    the thickness of the layer
            if dataarray :
                    input data are multiplied by this dataarray before averaging
        mask : None or dataarray
            mask to be applyed befire averaging
        na_eq_zero : boolean (default=False)
            replace NaN values by zeros. The averaging is then applyed on all data, and not only valid ones
        **kwargs : keyword arguments
            any keyword arguments passe to the native xarray.mean function
            
        Returns
        -------
        averaged : dataset
            dataset with all variables averaged according to specified options
            
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
        >>>avg = data.xgeo.mean(['latitude','longitude'],weights=['latitude'],na_eq_zero=True)
            
        """
        res={}
        for var in self._obj.data_vars:
            res[var]=self._obj[var].xgeo.mean(*args,**kwargs)
        return xr.Dataset(res)

    def sum(self,*args,**kwargs):
        """
        Returns the sum for all variables in dataset along specified dimension, applying specified weights
        
        Parameters
        ----------
        *args : list
            list of the dimensions along which to sum
        
        weights : None or list or dataarray
            if None, no weight is applyed
            if 'latitude' or 'depth', a weight is applyed as the cosine of the latitude or 
                    the thickness of the layer
            if dataarray :
                    input data are multiplied by this dataarray before summing
        mask : None or dataarray
            mask to be applyed before summing
        **kwargs : keyword arguments
            any keyword arguments passe to the native xarray.sum function
            
        Returns
        -------
        averaged : dataset
            dataset with all variablessummed according to specified options
            
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/isas.nc')
        >>>avg = data.xgeo.sum(['latitude','longitude'],weights=['latitude'])
            
        """             
        res={}
        for var in self._obj.data_vars:
            res[var]=self._obj[var].xgeo.sum(*args,**kwargs)
        return xr.Dataset(res)
    
    
    def isosurface(self, dim, criterion, upper=False):
        """
        Compute the isosurface along the specified coordinate at the value defined  by the kwarg field=value.
        For example, we want to compute the isosurface defined by a temperature of 10°C along depth dimension.
        All data variables of the data set are interpolated on this iso surface
        Data is supposed to be monotonic along the chosen dimension. If not, the first fitting value encountered is retained,
        starting from the end (bottom) if upper=False, or from the beggining (top) if upper=True
        
        Parameters
        ----------
        dim : string
            dimension along which to compute the isosurface
        criterion : dict
            one-entry dictionnary with the key equal to a variable of the dataset, and the value equal to the isosurface criterion
        upper : boolean (default=False)
            order to perform the research of the criterion value. If False, from the end, if True, form the beggining
            
        Returns
        -------
        isosurface : dataset
            Dataset with all the variables interpolated at the criterion value along chosen dimension. The variables chosen for
                criterion should contain a constant value equal to the criterion. the dimension chosen for the isosurface computation
                is filled with the isosurface itself.
                
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/isas.nc')
        >>>data.isosurface('depth',dict(temp=3))
        <xarray.Dataset>
        Dimensions:    (latitude: 90, longitude: 180)
        Coordinates:
          * latitude   (latitude) float32 -44.5 -43.5 -42.5 -41.5 ... 42.5 43.5 44.5
          * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 177.0 178.0 179.0 180.0
            time       datetime64[ns] 2005-01-15
            depth      (latitude, longitude) float64 918.6 745.8 704.8 ... 912.2 920.0
        Data variables:
            depth_iso  (latitude, longitude) float64 918.6 745.8 704.8 ... 912.2 920.0
            temp       (latitude, longitude) float64 3.0 3.0 3.0 3.0 ... 3.0 3.0 3.0 3.0
            SA         (latitude, longitude) float64 34.48 34.39 34.39 ... 34.53 34.52

        
        """
        # Calcule l'isosurface selon la coordonnée 'dim' pour le champ défini par le dictionnaire **kwargs (ex : temp=10)
        # Retourne tous les champs interpolés sur cette isosurface (pour ceux ayant "dim" en coordonnée), ainsi que l'isosurface elle-même
        k=list(criterion.keys())[0]
        if not(k in self._obj.data_vars):
            raise KeyError("%s not in %s"%(criterion[0],list(data_vars)))
            
        r=isosurface(self._obj[k],criterion[k],dim, upper=upper)
        res=xr.Dataset()
        for var in self._obj.data_vars:
            if dim in self._obj[var].coords:
                res[var]=self._obj[var].interp({dim:r})
            else:
                res[var]=self._obj[var]

        return res

    def regridder(self,gr_out,*args,mask_in=None,**kwargs):
        """
        Implement a xesmf regridder instance to be used with regrid method to perform regridding from dataset 
        coordinates to gr_out coordinates

        Parameters
        ----------
        gr_out : dataset
            dataset containing the coordinates to regrid on
        *args : 
            any argument passed to xesmf.Regridder method
        mask_in : None or dataarray
            mask to be applied on the data to regrid
        method : str
            resampling method (see xesmf documentation)            
        *kwargs : 
            any keyword argument passed to xesmf.Regridder method

        Returns
        -------
        regridder : xesmf.Regridder instance
            regridder to be used with regrid method to perform regridding from dataset coordinates to gr_out coordinates
        """
        if not 'latitude' in gr_out.coords: raise AssertionError('The latitude coordinates does not exist')
        if not 'longitude' in gr_out.coords: raise AssertionError('The longitude coordinates does not exist')

        ds=self._obj
        if type(mask_in)==xr.DataArray:
            ds['mask']=mask_in
            
        ds_out=xr.Dataset({
        "latitude":gr_out.latitude,
        "longitude":gr_out.longitude
        })

        return xe.Regridder(ds,ds_out,*args,**kwargs)
    
    def regrid(self,regridder,*args,**kwargs):
        """
        Implement the xesmf regrid method to perform regridding from dataset coordinates to gr_out coordinates

        Parameters
        ----------
        regridder : xesmf.Regridder instance
            regridder set with the xgeo.regridder method
        *args : 
            any argument passed to xesmf regridder method
        *kwargs : 
            any keyword argument passed to xesmf regridder method

        Returns
        -------
        regrid : dataset
            dataset regridded to gr_out coordinates

        Example
        -------
        >>>ds_out = xr.Dataset({"latitude":(["latitude"],np.arange(-89.5,90,1.)),
        >>>                     "longitude":(["longitude"],np.arange(-179.5,180,1.))})
        >>>regridder = data.xgeo.regridder(ds_out,"conservative_normed",periodic=True)
        >>>out = data.xgeo.regrid(regridder)
        """
        return regridder(self._obj,*args,**kwargs)
    
    def filter(self, filter_name=lanczos,q=3, **kwargs):
        """
        Apply a specified filter on all the time-dependent data in the dataset
        Boundaries are handled by operating a mirror operation on the residual data after removing a q-order polyfit from the data
        Available filters are in the .utils python file
        
        Parameters
        ----------
        filter_name : function
            filter function name, from the .utils file
        q : int
            order of the polyfit to handle boundary effects
        **kwargs :
            keyword arguments for the chosen filter
            
        Returns
        -------
        filtered : filtered dataset
        
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/isas.nc')
        >>>data.xgeo.filter(lanczos,q=3,coupure=12,order=2)
        
        """
        res={}
        for var in self._obj.data_vars:
            if 'time' in self._obj[var].coords:
                res[var]=self._obj[var].xgeo.filter(filter_name=filter_name,q=q,**kwargs)
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
                res[var]=self._obj[var].xgeo.interp_time(other,**kwargs)
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
        
@xr.register_dataarray_accessor("xgeo")
class GeoArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
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
            retourne en plus les coefficients des cycles et de la tendance linéaire
            
        Returns
        -------
        climato : dataarray
            a dataarray with the same structure as the input, with modified data according to the chosen options
        if return_coeffs=True, an extra dataset is provided with the coefficients of the decomposition
        
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/gohc_2020.nc').ohc
        >>>output,coeffs = data.xgeo.climato(mean=True, trend=True, signal=True,return_coeffs=True)
        """
        return climato(self._obj,**kwargs)
        
    def mean(self,*args,weights=None,mask=True,na_eq_zero=False,**kwargs):
        """
        Returns the averaged value of dataarray along specified dimension, applying specified weights
        
        Parameters
        ----------
        *args : list
            list of the dimensions along which to average
        
        weights : None or list or dataarray
            if None, no weight is applyed
            if 'latitude' or 'depth', a weight is applyed as the cosine of the latitude or 
                    the thickness of the layer
            if dataarray :
                    input data are multiplied by this dataarray before averaging
        mask : None or dataarray
            mask to be applyed before averaging
        na_eq_zero : boolean (default=False)
            replace NaN values by zeros. The averaging is then applyed on all data, and not only valid ones
        **kwargs : keyword arguments
            any keyword arguments passe to the native xarray.mean function
            
        Returns
        -------
        averaged : dataarray
            dataarray averaged according to specified options
            
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/isas.nc').temp
        >>>avg = data.xgeo.mean(['latitude','longitude'],weights=['latitude'],na_eq_zero=True)
            
        """
        argmean=set(np.ravel(*args)).intersection(list(self._obj.coords))
        data=self._obj.where(mask)
        if na_eq_zero:
            data=data.fillna(0.)
            
        if len(argmean)==0:
            argmean=None
        if type(weights)==type(None):
            # Moyenne simple
            return data.mean(argmean,**kwargs)
        elif type(weights)==list or type(weights)==str:
            w=1
            if 'latitude' in weights and 'latitude' in self._obj.coords:
                # poids = cos(latitude)
                w=np.cos(np.radians(self._obj.latitude))
            if 'depth' in weights and 'depth' in self._obj.coords:
                # poids *= épaisseur des couches (l'épaisseur de la première couche est la première profondeur)
                w=w*xr.concat((self._obj.depth.isel(depth=0),self._obj.depth.diff(dim='depth')),dim='depth')
            return data.weighted(w).mean(argmean,**kwargs)
        else:
            # matrice de poids définie par l'utilisateur
            return data.weighted(weights).mean(argmean,**kwargs)
    
    def sum(self,*args,weights=None,mask=True,**kwargs):
        """
        Returns the sum of dataarray along specified dimension, applying specified weights
        
        Parameters
        ----------
        *args : list
            list of the dimensions along which to sum
        
        weights : None or list or dataarray
            if None, no weight is applyed
            if 'latitude' or 'depth', a weight is applyed as the cosine of the latitude or 
                    the thickness of the layer
            if dataarray :
                    input data are multiplied by this dataarray before summing
        mask : None or dataarray
            mask to be applyed before summing
        **kwargs : keyword arguments
            any keyword arguments passe to the native xarray.sum function
            
        Returns
        -------
        averaged : dataarray
            dataarray summed according to specified options
            
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/isas.nc').heat
        >>>avg = data.xgeo.sum(['latitude','longitude'],weights=['latitude'])
            
        """       
        argsum=set(np.ravel(*args)).intersection(list(self._obj.coords))
        data=self._obj.where(mask)        
        if type(weights)==type(None):
            # Somme simple
            return data.sum(argsum,**kwargs)
        elif type(weights)==list or type(weights)==str:
            w=1
            if 'latitude' in weights and 'latitude' in self._obj.coords:
                # poids = cos(latitude)
                w=np.cos(np.radians(self._obj.latitude))
            if 'depth' in weights and 'depth' in self._obj.coords:
                # poids *= épaisseur des couches (l'épaisseur de la première couche est la première profondeur)
                w=w*xr.concat((self._obj.depth.isel(depth=0),self._obj.depth.diff(dim='depth')),dim='depth')
            return data.weighted(w).sum(argsum,**kwargs)
        else:
            # matrice de poids définie par l'utilisateur
            return data.weighted(weights).sum(argsum,**kwargs)
    

    def isosurface(self, target, dim, upper=False):   
        """
        Compute the isosurface along the specified coordinate at the value defined  by the target.
        Data is supposed to be monotonic along the chosen dimension. If not, the first fitting value encountered is retained,
        starting from the end (bottom) if upper=False, or from the beggining (top) if upper=True
        
        Parameters
        ----------
        target : float
            criterion value to be satisfied at the iso surface
        dim : string
            dimension along which to compute the isosurface
        upper : boolean (default=False)
            order to perform the research of the criterion value. If False, from the end, if True, form the beggining
            
        Returns
        -------
        isosurface : dataarray
            Dataarray containing the isosurface along the dimension dim on which data=target.
                
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/isas.nc').temp
        >>>data.isosurface(3,'depth')
        <xarray.DataArray (latitude: 90, longitude: 180)>
        dask.array<_nanmax_skip-aggregate, shape=(90, 180), dtype=float64, chunksize=(90, 180), chunktype=numpy.ndarray>
        Coordinates:
          * latitude   (latitude) float32 -44.5 -43.5 -42.5 -41.5 ... 42.5 43.5 44.5
          * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 177.0 178.0 179.0 180.0
            time       datetime64[ns] 2005-01-15
        """        
        return isosurface(self._obj,target,dim,upper=upper)

    def regridder(self,gr_out,*args,mask_in=None,**kwargs):
        """
        Implement a xesmf regridder instance to be used with regrid method to perform regridding from dataarray 
        coordinates to gr_out coordinates

        Parameters
        ----------
        gr_out : dataset
            dataset containing the coordinates to regrid on
        *args : 
            any argument passed to xesmf.Regridder method
        mask_in : None or dataarray
            mask to be applied on the data to regrid
        method : str
            resampling method (see xesmf documentation)            
        *kwargs : 
            any keyword argument passed to xesmf.Regridder method

        Returns
        -------
        regridder : xesmf.Regridder instance
            regridder to be used with regrid method to perform regridding from dataset coordinates to gr_out coordinates
        """

        ds=xr.Dataset({'data':self._obj})
        return ds.xgeo.regridder(gr_out,*args,mask_in,**kwargs)
    
    def regrid(self,regridder,*args,**kwargs):
        """
        Implement the xesmf regrid method to perform regridding from dataset coordinates to gr_out coordinates

        Parameters
        ----------
        regridder : xesmf.Regridder instance
            regridder set with the xgeo.regridder method

        *args : 
            any argument passed to xesmf regridder method
        *kwargs : 
            any keyword argument passed to xesmf regridder method

        Returns
        -------
        regrid : dataset
            dataset regridded to gr_out coordinates

        Example
        -------
        >>>ds_out = xr.Dataset({"latitude":(["latitude"],np.arange(-89.5,90,1.)),
        >>>                     "longitude":(["longitude"],np.arange(-179.5,180,1.))})
        >>>regridder = data.xgeo.regridder(ds_out,"conservative_normed",periodic=True)
        >>>out = data.xgeo.regrid(regridder)
        """        
        return regridder(self._obj,*args,**kwargs)

    def filter(self, filter_name=lanczos,q=3, **kwargs):
        """
        Apply a specified filter on all the time-dependent datarray
        Boundaries are handled by operating a mirror operation on the residual data after removing a q-order polyfit from the data
        Available filters are in the .utils python file
        
        Parameters
        ----------
        filter_name : function
            filter function name, from the .utils file
        q : int
            order of the polyfit to handle boundary effects
        **kwargs :
            keyword arguments for the chosen filter
            
        Returns
        -------
        filtered : filtered dataset
        
        Example
        -------
        >>>data = xgeo.open_geodata('/home/user/lenapy/data/isas.nc').temp
        >>>data.xgeo.filter(lanczos,q=3,coupure=12,order=2)
        
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

    def to_difgri(self,dir_out,prefix,suffix):
        to_difgri(self._obj,dir_out,prefix,suffix)

    def trend(self):
        """
        Perform a linear regression on the data, and returns the slope coefficient
        """
        return trend(self._obj)
    
    def fill_time(self):
        """
        Fill missing values in a timeseries in adding some new points, by respecting the time sampling. Missing values are not NaN
        but real absent points in the timeseries. A linear interpolation is performed at the missing points.
        """
        return fill_time(self._obj)
    