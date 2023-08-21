Module lenapy.xOcean
====================

Functions
---------

    
`NoneType(var)`
:   

    
`proprietes(da, nom, label, unite)`
:   

Classes
-------

`OceanArray(xarray_obj)`
:   This class extends any dataarray to a xOcean object, to perform specific operations on structured dataarray

    ### Ancestors (in MRO)

    * lenapy.xGeo.GeoArray

    ### Methods

    `above(self, depth, **kwargs)`
    :   Returns the dataarray integrated above a given depth, by interpolating at this depth the cumulative integrale 
        of the data array
        
        Example
        -------
        >>>data=IAP('/home/usr/lenapy/data/IAP')
        >>>mld=data.xocean.mld_sigma0
        >>>data.xocean.heat.xocean.above(mld)       
        <xarray.DataArray (time: 156, latitude: 180, longitude: 360)>
        dask.array<where, shape=(156, 180, 360), dtype=float64, chunksize=(1, 180, 360), chunktype=numpy.ndarray>
        Coordinates:
            depth      (time, latitude, longitude) float64 dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
          * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
          * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
          * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0

    `add_value_surface(self, value=None)`
    :   Add a surface layer with a specified value, or the previous upper value
        Parameter
        ---------
        value : float or array-like, optional
            values to be added in the surface layer (depth=0). If None, the previous upper value is used to fill the new layer
        
        Returns
        -------
        added : DataArray
            new dataarray with a extra layer at depth 0 filled with required values
            
        Example
        -------
        >>>data=IAP('/home/usr/lenapy/data/IAP')
        >>>heat=data.xocean.heat.xocean.add_value_surface()

    `cum_integ_depth(self)`
    :   Returns a cumulative integrated dataarray integrated over the whole depth. The surface value is supposed equal to the 
        most shallow value. A first integration layer by layer is performed, by multupliying the layer's thickness by the mean
        value of upper and lower bound, then a cumulative sum is computed.
        
        Example
        -------
        >>>data=IAP('/home/usr/lenapy/data/IAP')
        >>>data.xocean.heat.xocean.cum_integ_depth()       
        <xarray.DataArray (time: 156, latitude: 180, longitude: 360, depth: 41)>
        dask.array<where, shape=(156, 180, 360, 41), dtype=float64, chunksize=(1, 180, 360, 10), chunktype=numpy.ndarray>
        Coordinates:
          * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
          * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0
          * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
          * depth      (depth) float64 1.0 5.0 10.0 20.0 ... 1.7e+03 1.8e+03 2e+03

    `integ_depth(self)`
    :   Returns the dataarray integrated over the whole depth. The surface value is supposed equal to the most shallow value.
        In order to deal with NaN values in deep water during integration,, all NaN are firt converted to 0, then in the output 
        array NaN values are applied where initial surface values were NaN.
        
        Example
        -------
        >>>data=IAP('/home/usr/lenapy/data/IAP')
        >>>data.xocean.heat.xocean.integ_depth()    
        <xarray.DataArray 'Heat' (time: 156, latitude: 180, longitude: 360)>
        dask.array<where, shape=(156, 180, 360), dtype=float64, chunksize=(1, 180, 360), chunktype=numpy.ndarray>
        Coordinates:
          * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
          * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0
          * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
            depth      float32 1.0

`OceanSet(xarray_obj)`
:   This class extends any dataset to a xOcean object, that allows to access to any TEOS10 variable simply by calling the name of the variable through the xOcean interface.
    The initial dataset must contains the fields necessary to compute the output variable (ex : temperature and salinity to compute heat, heat to compute ohc,...)
    
    Availabe fields
    ---------------
    Temperatures : one of the three types of temperature must be present in the original dataset to perform derived computation.
        temp  : in-situ temperature        
        PT    : potential temperature     
        CT    : conservative temperature 
    
    Salinities :  one of the three types of salinities must be present in the original dataset to perform derived computation.
        psal  : practical salinity  
        SR    : relative salinity    
        SA    : absolute salinity. If there is no location information (lat,lon), absolute salinity is returned equal to relative salinity
    
    Physical properties :
        P     : pressure. If location information is present, pressure is adjusted with regard to latitude, otherwise latitude is equal to 0
        Cp    : heat capacity
        rho   : density
        sigma0: potential density anomaly at 0 dbar
    
    Heat content :
        heat  : specific heat content (J/m3)
        ohc   : local ocean heat content (J/m²), it is heat integrated over the whole ocean depth
        gohc  : global ocean heat content (J/m²), it is ohc averaged over latitude-longitude, excluding continents
        gohc_TOA: idem gohc, including continents (where ohc=0)
        ohc_above: idem ohc, where heat is integrated above a given depth
        gohc_above: idem gohc, averaging ohc_above instead of ohc
    
    Sea level :
        slh   : steric sea layer height anomaly (-), equal to (1. - rho/rhoref)
        ssl   : steric sea surface level anomaly (m), it is slh integrated over the whole ocean depth
        ieeh  : integrated expansion efficiency oh heat (m/(J/m²)), it is (ssl/ohc)
    
    Layer depth :
        ocean_depth  : maximum depth with non Nan values for temperature
        mld_theta0   : ocean mixed layer depth, defined by a temperature drop from 0.2°C wrt to -10m depth 
        mld_sigma0   : ocean mixed layer depth, defined by a potential density increase of 0.03kg/m3 wrt to -10m depth
        mld_sigma0var: ocean mixed layer depth, defined by a potential density equal to the potential density at -10m depth with a temperature dropped by 0.2°C
    
    Examples
    --------
    >>>data=IAP('/home/usr/lenapy/data/IAP')
    >>>print(data)
    <xarray.Dataset>
    Dimensions:    (latitude: 180, longitude: 360, time: 156, depth: 41)
    Coordinates:
      * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
      * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0
      * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
      * depth      (depth) float32 1.0 5.0 10.0 20.0 ... 1.7e+03 1.8e+03 2e+03
    Data variables:
        temp       (time, latitude, longitude, depth) float32 dask.array<chunksize=(1, 180, 360, 10), meta=np.ndarray>
        SA         (time, latitude, longitude, depth) float32 dask.array<chunksize=(1, 180, 360, 10), meta=np.ndarray>
    >>>mld=data.xocean.mld_sigma0
    >>>print(data.xocean.ohc_above(mld))
    <xarray.DataArray 'ohc_above' (time: 156, latitude: 180, longitude: 360)>
    dask.array<where, shape=(156, 180, 360), dtype=float64, chunksize=(1, 180, 360), chunktype=numpy.ndarray>
    Coordinates:
        depth      (time, latitude, longitude) float64 dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
      * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
      * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
      * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0
    Attributes:
        long_name:  Ocean heat content
        units:      J/m²
    >>>print(data.xocean.gohc)
    <xarray.DataArray 'gohc' (time: 156)>
    dask.array<truediv, shape=(156,), dtype=float64, chunksize=(1,), chunktype=numpy.ndarray>
    Coordinates:
      * time     (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
        depth    float32 1.0
    Attributes:
        long_name:  Global ocean heat content wrt to ocean surface area
        units:      J/m²

    ### Ancestors (in MRO)

    * lenapy.xGeo.GeoSet

    ### Instance variables

    `CT`
    :

    `Cp`
    :

    `P`
    :

    `PT`
    :

    `SA`
    :

    `SR`
    :

    `gohc`
    :

    `gohc_TOA`
    :

    `heat`
    :

    `ieeh`
    :

    `mld_sigma0`
    :

    `mld_sigma0var`
    :

    `mld_theta0`
    :

    `mld_theta0minus_only`
    :

    `ocean_depth`
    :

    `ohc`
    :

    `psal`
    :

    `rho`
    :

    `sigma0`
    :

    `slh`
    :

    `ssl`
    :

    `temp`
    :

    ### Methods

    `gohc_above(self, target, na_eq_zero=False)`
    :

    `ohc_above(self, target)`
    :