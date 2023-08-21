Module lenapy.xGeo
==================

Functions
---------

    
`open_geodata(file, *args, rename={}, nan=None, chunks=None, **kwargs)`
:   Open a dataset base on xr.open_dataset method, while normalizing coordinates names and choosing NaN values
    
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

    
`open_mfgeodata(fic, *args, rename={}, nan=None, chunks=None, **kwargs)`
:   Open a dataset base on xr.open_mfdataset method, while normalizing coordinates names and choosing NaN values
    
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

Classes
-------

`GeoArray(xarray_obj)`
:   

    ### Descendants

    * lenapy.xOcean.OceanArray

    ### Methods

    `climato(self, **kwargs)`
    :   Perform climato analysis on a dataarray
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

    `diff_3pts(self, dim)`
    :   Derivative formula along the selected dimension, returning on each point the linear regression on the three points
        defined by the selected point and its two neighbours

    `fill_time(self)`
    :   Fill missing values in a timeseries in adding some new points, by respecting the time sampling. Missing values are not NaN
        but real absent points in the timeseries. A linear interpolation is performed at the missing points.

    `filter(self, filter_name=<function lanczos>, q=3, **kwargs)`
    :   Apply a specified filter on all the time-dependent datarray
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

    `interp_time(self, other, **kwargs)`
    :   Interpolate dataarray at the same dates than other
        
        Parameter
        ---------
        other : dataarray
            must have a time dimension
            
        Return
        ------
        interpolated : dataarray
            new dataarray interpolated

    `isosurface(self, target, dim, upper=False)`
    :   Compute the isosurface along the specified coordinate at the value defined  by the target.
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

    `mean(self, *args, weights=None, mask=True, na_eq_zero=False, **kwargs)`
    :   Returns the averaged value of dataarray along specified dimension, applying specified weights
        
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

    `plot_timeseries_uncertainty(self, **kwargs)`
    :

    `regrid(self, regridder, *args, **kwargs)`
    :   Implement the xesmf regrid method to perform regridding from dataset coordinates to gr_out coordinates
        
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

    `regridder(self, gr_out, *args, mask_in=None, **kwargs)`
    :   Implement a xesmf regridder instance to be used with regrid method to perform regridding from dataarray 
        coordinates to gr_out coordinates
        
        Parameters
        ----------
        gr_out : dataset
            dataset containing the coordinates to regrid on
        *args : 
            any argument passed to xesmf.Regridder method
        mask_in : None or dataarray
            mask to be applied on the data to regrid
        *kwargs : 
            any keyword argument passed to xesmf.Regridder method
        
        Returns
        -------
        regridder : xesmf.Regridder instance
            regridder to be used with regrid method to perform regridding from dataset coordinates to gr_out coordinates

    `sum(self, *args, weights=None, mask=True, **kwargs)`
    :   Returns the sum of dataarray along specified dimension, applying specified weights
        
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

    `to_datetime(self, input_type)`
    :   Convert dataarray time format to standard pandas time format
        
        Parameter
        ---------
        input_type : string
            Can be 'frac_year' or '360_day'
            
        Return
        ------
        converted : dataarray
            new dataarray with the time dimension in a standard pandas format

    `to_difgri(self, dir_out, prefix, suffix)`
    :

    `trend(self)`
    :   Perform a linear regression on the data, and returns the slope coefficient

`GeoSet(xarray_obj)`
:   This class implements an extension of any dataset to add some usefull methods often used in earth science data handling

    ### Descendants

    * lenapy.xOcean.OceanSet

    ### Methods

    `climato(self, **kwargs)`
    :   Perform climato analysis on all the variables in a dataset
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

    `fill_time(self)`
    :   Fill missing values in a timeseries in adding some new points, by respecting the time sampling. Missing values are not NaN
        but real absent points in the timeseries. A linear interpolation is performed at the missing points.

    `filter(self, filter_name=<function lanczos>, q=3, **kwargs)`
    :   Apply a specified filter on all the time-dependent data in the dataset
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

    `interp_time(self, other, **kwargs)`
    :   Interpolate dataarray at the same dates than other
        
        Parameter
        ---------
        other : dataarray
            must have a time dimension
            
        Return
        ------
        interpolated : dataarray
            new dataarray interpolated

    `isosurface(self, dim, criterion, upper=False)`
    :   Compute the isosurface along the specified coordinate at the value defined  by the kwarg field=value.
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

    `mean(self, *args, **kwargs)`
    :   Returns the averaged value of all variables in dataset along specified dimension, applying specified weights
        
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

    `regrid(self, regridder, *args, **kwargs)`
    :   Implement the xesmf regrid method to perform regridding from dataset coordinates to gr_out coordinates
        
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

    `regridder(self, gr_out, *args, mask_in=None, **kwargs)`
    :   Implement a xesmf regridder instance to be used with regrid method to perform regridding from dataset 
        coordinates to gr_out coordinates
        
        Parameters
        ----------
        gr_out : dataset
            dataset containing the coordinates to regrid on
        *args : 
            any argument passed to xesmf.Regridder method
        mask_in : None or dataarray
            mask to be applied on the data to regrid
        *kwargs : 
            any keyword argument passed to xesmf.Regridder method
        
        Returns
        -------
        regridder : xesmf.Regridder instance
            regridder to be used with regrid method to perform regridding from dataset coordinates to gr_out coordinates

    `sum(self, *args, **kwargs)`
    :   Returns the sum for all variables in dataset along specified dimension, applying specified weights
        
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

    `to_datetime(self, input_type)`
    :   Convert dataset time format to standard pandas time format
        
        Parameter
        ---------
        input_type : string
            Can be 'frac_year' or '360_day'
            
        Return
        ------
        converted : dataset
            new dataset with the time dimension in a standard pandas format