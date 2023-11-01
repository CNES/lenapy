# -*- coding: utf-8 -*-
"""
This module allows to load temperature and salinity data from different products and format the data with unified definition for variables and coordinates, compatible with the use of xOcean :
standardized coordinates names : latitude, longitude, depth, time
standardized variables names : temp or PT or CT for temperature, psal, SA, SR for salinity
When loading a product, all the files present in the product directory are parsed. To gain computing time, a first filter on the years to load can be applied, as well as a text filter.
A second date filter can be apply afterwards with the .sel(time=slice('begin_date','end_date') method.
All keyword arguments associated with xr.open_mfdataset can be passed.
Dask is implicitely used when using these interface methods.
It is strongly recommanded to chunk the dataset along depth dimension (chunks=dict(depth=10))

Parameters
----------
rep : string
    path of the product's directory
ymin : int, optional
    lowest bound of the time intervalle to be loaded (year)
ymax : int, optional
    highest bound of the time intervalle to be loaded (year)
filter : string, optionnal
    string pattern to filter datafiles names
**kwargs :  optional
    The keyword arguments form of open_mfdataset

Returns
-------
product : Dataset
    New dataset containing temperature and salinity data from the product
    
Examples
--------
>>> data=ISAS('/home/usr/lenapy/data/ISAS20',ymin=2005,ymax=2007,filter='ARGO',chunks={'depth':10})
>>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()
"""

import os.path
from glob import glob
import numpy as np
import xarray as xr
import gsw


def rename_data(data,**kwargs):
    """
    Standardization of coordinates names of a product
    Looks for different possible names for latitude, longitude, and time, and turn them into a standardized name :
    'lon','LON','Longitude','LONGITUDE' become 'longitude'
    'lat','LAT','Latitude','LATITUDE' become 'latitude'
    'date','dates','TIME','Time' become 'time'
    Custom names changes can also be performed with **kwargs parameter

    Parameters
    ----------
    **kwargs :  {old_name: new_name, ...}, optional
        dictionnary specifying old names to be changed into new names

    Returns
    -------
    renamed : Dataset
        New dataset containing modified names

    Examples
    --------
    >>>ds=xr.open_mfdataset('product.nc',preprocess=rename_data)
    """
    data=data.rename(**kwargs)
        
    for l in ['lon','LON','Longitude','LONGITUDE']:
        if l in data.variables:
            data=data.rename({l:'longitude'})

    for l in ['lat','LAT','Latitude','LATITUDE']:
        if l in data.variables:
            data=data.rename({l:'latitude'})

    for l in ['date','dates','TIME','Time']:
        if l in data.variables:
            data=data.rename({l:'time'})
            
    for l in ['Depth','depth_std','LEVEL','level','Time']:
        if l in data.variables:
            data=data.rename({l:'depth'})
            
    if 'longitude' in data.variables and not('longitude' in data.coords):
        lon=data['longitude']
        del data['longitude']
        lat=data['latitude']
        del data['latitude']    
        data=data.assign_coords(longitude=lon,latitude=lat)        
    return data

def filtre_liste(files,year,ymin,ymax,pattern):
    """
    Returns a filtered list of files fitting year range and pattern

    Parameters
    ----------
    files : Array of strings
        Array with filenames to be filtered
        
    year : function
        Function to be applied on each filename, returning the data year
        
    ymin : integer
        Lowest bound of the time range

    ymax : integer
        Highest bound of the time range
        
    pattern : string
        Pattern that must fit the filenames
        
    Return
    ------
    filtered : Array
        Extract of th input array fitting year and pattern conditions
        
    Example
    -------
    >>> def year(f):
    >>>   return f.split('_')[1] 
    >>> fics=filtre_liste(glob(os.path.join(rep,'**','*.nc')),year,2005,2006,'ARGO')

    """
    r=[]
    for u in files:
        d=int(year(os.path.basename(os.path.splitext(u)[0])))
        if d>=ymin and d<=ymax and pattern in u:
            r.append(u)
    return r


#--------------------- ISAS --------------------
def ISAS(rep,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from ISAS product
    Product's directory must have the following format:
    -base directory (ex: ISAS20_ARGO)
       |-year (ex: 2020)
           |-release_type_timestamp_xxx_data.nc (ex: ISAS20_ARGO_20200915_fld_TEMP.nc)
      
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing in-situ temperature and practical salinity data from the product

    Examples
    --------
    >>> data=ISAS('/home/usr/lenapy/data/ISAS20',ymin=2005,ymax=2007,filter='ARGO',chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """

    def year(f):
        return f.split('_')[2][0:4]

    fics=filtre_liste(glob(os.path.join(rep,'**','*.nc')),year,ymin,ymax,filtre)
    
    data=xr.open_mfdataset(fics,**kwargs)
    
    return xr.Dataset({
        'temp':data.TEMP,
        'psal':data.PSAL
    })

#--------------------- NCEI --------------------
def NCEI(rep,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from NCEI product
    Temperature and salinity are reconstructed from anomaly and trimestrial climatology
    
    Product's directory must have the following format:
    -base directory (ex: NCEI)
       |-salinity
           |-sanom_timestamp.nc (ex: sanom_C1C107-09.nc)
       |-temperature
           |-tanom_timestamp.nc (ex: tanom_C1C107-09.nc)
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing in-situ temperature and practical salinity data from the product

    Examples
    --------
    >>> data=NCEI('/home/usr/lenapy/data/NCEI',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """
    def year(f):
        a=f.split('_')[1][0]
        b=f.split('_')[1][1]
        return "%04d"%(1900+10*int('0x%s'%a,16)+int(b))
    
    #Salinite
    fics_sal=filtre_liste(glob(os.path.join(rep,'**','sanom*.nc')),year,ymin,ymax,filtre)
    sal=xr.open_mfdataset(fics_sal,preprocess=rename_data,decode_times=False,**kwargs).xtime.to_datetime("360_day").s_an
    fics_climsal=glob(os.path.join(rep,'climato','*s1[3-6]_01.nc'))
    climsal=xr.open_mfdataset(fics_climsal,preprocess=rename_data,decode_times=False,**kwargs).xtime.to_datetime("360_day").s_an
    
    #Temperature
    fics_temp=filtre_liste(glob(os.path.join(rep,'**','tanom*.nc')),year,ymin,ymax,filtre)
    temp=xr.open_mfdataset(fics_temp,preprocess=rename_data,decode_times=False,**kwargs).xtime.to_datetime("360_day").t_an
    fics_climtemp=glob(os.path.join(rep,'climato','*t1[3-6]_01.nc'))
    climtemp=xr.open_mfdataset(fics_climtemp,preprocess=rename_data,decode_times=False,**kwargs).xtime.to_datetime("360_day").t_an
    
    return xr.Dataset({\
                       'temp':(temp.groupby('time.month')+climtemp.groupby('time.month').mean('time')).drop('month'),
                       'psal':(sal.groupby('time.month')+climsal.groupby('time.month').mean('time')).drop('month')
                      })

#--------------------- SIO ---------------------
def SIO(rep,chunks=None,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from SIO product.
    Temperature and salinity are reconstructed from anomaly and annual climatology, and depth coordinate is derived from pressure.
    
    Product's directory must have the following format:
    -base directory (ex: SIO)
       |-climato
           |-RG_ArgoClim_climato_release.nc (ex: RG_ArgoClim_climato_2004_2018.nc)
       |-monthly
           |-RG_ArgoClim_timestamp_release.nc (ex: RG_ArgoClim_202210_2019.nc)
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing in-situ temperature and practical salinity data from the product

    Examples
    --------
    >>> data=SIO('/home/usr/lenapy/data/SIO',ymin=2005,ymax=2007,filter='ARGO',chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """    
    def year(f):
        return f.split('_')[-2][0:4]

    fics_anom=filtre_liste(glob(os.path.join(rep,'**','RG_ArgoClim_20*.nc')),year,ymin,ymax,filtre)
    fics_clim=glob(os.path.join(rep,'**','RG_ArgoClim_cl*.nc'))
    
    anom=xr.open_mfdataset(fics_anom,preprocess=rename_data,decode_times=False,chunks=chunks,**kwargs).xtime.to_datetime("360_day")
    clim=xr.open_mfdataset(fics_clim,preprocess=rename_data,**kwargs)

    temp = anom.ARGO_TEMPERATURE_ANOMALY + clim.ARGO_TEMPERATURE_MEAN
    psal = anom.ARGO_SALINITY_ANOMALY + clim.ARGO_SALINITY_MEAN

    depth=-gsw.z_from_p(anom.PRESSURE,0)
    pressure=gsw.p_from_z(-depth,anom.latitude)
    pressure=xr.where(pressure<anom.PRESSURE,pressure,anom.PRESSURE)

    return xr.Dataset({'temp':temp.interp(PRESSURE=pressure).assign_coords(PRESSURE=depth).rename(PRESSURE='depth').chunk(chunks=chunks),
                       'psal':psal.interp(PRESSURE=pressure).assign_coords(PRESSURE=depth).rename(PRESSURE='depth').chunk(chunks=chunks)
                      })

#--------------------- IPRC ---------------------
def IPRC(rep,chunks=None,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from IPRC product.
    
    Product's directory must have the following format:
    -base directory (ex: IPRC)
       |-monthly
           |-ArgoData_year_month.nc (ex: ArgoData_2020_01.nc)
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing in-situ temperature and practical salinity data from the product

    Examples
    --------
    >>> data=IPRC('/home/usr/lenapy/data/IPRC',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """     
    def preproc(ds):
        date=os.path.basename(os.path.splitext(ds.encoding["source"])[0]).split('_')
        return rename_data(ds).assign_coords(time=np.datetime64('%s-%s-15'%(date[1],date[2]),'ns')).expand_dims(dim='time')
        

    def year(f):
        return f.split('_')[1]
    
    fics=filtre_liste(glob(os.path.join(rep,'**','ArgoData*.nc')),year,ymin,ymax,filtre)
    
    data=xr.open_mfdataset(fics,preprocess=preproc,chunks=chunks,**kwargs)

    return xr.Dataset({'temp':data.TEMP.chunk(chunks=chunks),
                       'psal':data.SALT.chunk(chunks=chunks)
                          })        
        
#--------------------- ISHII ---------------------
def ISHII(rep,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from ISHII product.
    
    Product's directory must have the following format:
    -base directory (ex: ISHII)
       |-monthly
           |-xxx.year_month.nc (ex: sal.2022_08.nc)
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing in-situ temperature and practical salinity data from the product

    Examples
    --------
    >>> data=ISHII('/home/usr/lenapy/data/ISHII',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """      
    def preproc(ds):
        return rename_data(ds.drop(['VAR_10_4_201_P0_L160_GLL0','VAR_10_4_202_P0_L160_GLL0']))
    
    def year(f):
        return f.split('_')[0].split('.')[1]

    fics=filtre_liste(glob(os.path.join(rep,'**','*.*.nc')),year,ymin,ymax,filtre)

    data=xr.open_mfdataset(fics,preprocess=preproc,**kwargs)

    return xr.Dataset({'temp':data.temperature,
                        'psal':data.salinity
                          })        

#--------------------- IAP ---------------------
def IAP(rep,chunks=None,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from IAP product.
    
    Product's directory must have the following format:
    -base directory (ex: IAP)
       |-sal
           |-CZ16_depth_range_data_year_yyyy_month_mm.year_month.nc (ex: CZ16_1_2000m_salinity_year_2020_month_01.nc)
       |-temp
           |-CZ16_depth_range_data_year_yyyy_month_mm.year_month.nc (ex: CZ16_1_2000m_temperature_year_2020_month_01.nc)
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing in-situ temperature and absolute salinity data from the product

    Examples
    --------
    >>> data=IAP('/home/usr/lenapy/data/IAP',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """     
    def preproc(ds):
        date=os.path.basename(os.path.splitext(ds.encoding["source"])[0]).split('_')
        return rename_data(ds).assign_coords(time=np.datetime64('%s-%s-15'%(date[-3],date[-1]),'ns')).expand_dims(dim='time')

    def year(f):
        return f.split('_')[-3]
    
    fics=filtre_liste(glob(os.path.join(rep,'**','CZ16*.nc')),year,ymin,ymax,filtre)
    
    data=xr.open_mfdataset(fics,preprocess=preproc,chunks=chunks,**kwargs)

    return xr.Dataset({'temp':data.temp.chunk(chunks=chunks),
                        'SA':data.salinity.chunk(chunks=chunks)
                          })        

#--------------------- EN_422 --------------------
def EN_422(rep,corr,chunks=None,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from EN product.
    
    Product's directory must have the following format:
    -base directory (ex: EN)
       |-4.2.2.corr (ex : 4.2.2.g10)
           |-EN.4.2.2.xxx.corr.timestamp.nc (ex: EN.4.2.2.f.analysis.g10.202108.nc)
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    corr : {"g10","l09","c13","c14"}
        correction applyed
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing potential temperature and practical salinity data from the product

    Examples
    --------
    >>> data=EN_422('/home/usr/lenapy/data/EN_422','c13',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """  
    def year(f):
        return f.split('.')[-1][0:4]
   
    if not(corr in ['g10','l09','c13','c14']):
        sys.exit(-1)

    fics=filtre_liste(glob(os.path.join(rep,"4.2.2.%s"%corr,'EN.4.2.2*.nc')),year,ymin,ymax,filtre)
       
    data=xr.open_mfdataset(fics,preprocess=rename_data,chunks=chunks,**kwargs)

    return xr.Dataset({'PT':data.temperature - 273.15,
                       'psal':data.salinity
                       })        

#--------------------- JAMSTEC --------------------
def JAMSTEC(rep,chunks=None,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from JAMSTEC product.
    Depth coordinate is derived from pressure.
    
    Product's directory must have the following format:
    -base directory (ex: JAMSTEC)
       |-monthly
           |-TS_timestamp_xxx.nc (ex: TS_202105_GLB.nc)
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing in-situ temperature and practical salinity data from the product

    Examples
    --------
    >>> data=JAMSTEC('/home/usr/lenapy/data/JAMSTEC',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """       
    def preproc(ds):
        date=os.path.basename(os.path.splitext(ds.encoding["source"])[0]).split('_')[1]
        ds=rename_data(ds).assign_coords(time=np.datetime64('%s-%s-15'%(date[0:4],date[4:6]),'ns')).expand_dims(dim='time')
        ds['longitude']=np.mod(ds.longitude,360)
        return ds

    def year(f):
        return f.split('_')[1][0:4]
    
    fics=filtre_liste(glob(os.path.join(rep,'**','*GLB.nc')),year,ymin,ymax,filtre)
    
    data=xr.open_mfdataset(fics,preprocess=preproc,chunks=chunks,**kwargs)

    depth=-gsw.z_from_p(data.PRES,0)
    pressure=gsw.p_from_z(-depth,data.latitude)
    pressure=xr.where(pressure<data.PRES,pressure,data.PRES)

    return xr.Dataset({'temp':data.TOI.interp(PRES=pressure).assign_coords(PRES=depth).rename(PRES='depth').chunk(chunks=chunks),
                       'psal':data.SOI.interp(PRES=pressure).assign_coords(PRES=depth).rename(PRES='depth').chunk(chunks=chunks)
                      })

#--------------------- ECCO --------------------
def ECCO(rep,chunks=None,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from ECCO product.
    
    Product's directory must have the following format:
    -base directory (ex: ECCO)
       |-SALT
           |-SALT_year_month.nc (ex: SALT_2016_12.nc)
       |-THETA
           |-THETA_year_month.nc (ex: THETA_2016_12.nc)
        
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing potential temperature and practical salinity data from the product

    Examples
    --------
    >>> data=ECCO('/home/usr/lenapy/data/ECCO',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """      
    def preproc(ds):
        ds=ds.set_index(i='longitude',j='latitude',k='Z').\
            rename(i='longitude',j='latitude',k='depth').drop('timestep')
        return ds.assign_coords(depth=-ds.depth)
    
    def year(f):
        return f.split('_')[1]
  
    fics=filtre_liste(glob(os.path.join(rep,'**','*.nc')),year,ymin,ymax,filtre)
    
    data=xr.open_mfdataset(fics,preprocess=preproc,chunks=chunks,**kwargs).chunk(chunks=chunks)
    data=data.where(data!=0.)
                   
    return xr.Dataset({'PT':data.THETA,
                        'psal':data.SALT
                          })        

#--------------------- IAP ---------------------
def Lyman(rep,chunks=None,ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from Lyman product.
    
    Product's directory must have the following format:
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing ohc data from the product

    Examples
    --------
    >>> data=IAP('/home/usr/lenapy/data/IAP',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """     

    def preproc(ds):
        return rename_data(ds).xgeo.reset_longitude(-180)

    def year(f):
        return f.split('_')[-2]
    
    fics=filtre_liste(glob(os.path.join(rep,'**','RFROM_OHCA*.nc')),year,ymin,ymax,filtre)
    
    data=xr.open_mfdataset(fics,preprocess=preproc,chunks=chunks,**kwargs)

    return xr.Dataset({'ohc':data.ocean_heat_content_anomaly.sum('mean_depth').chunk(chunks=chunks),
                          })        

#--------------------- NOC OI --------------------
def NOC_OI(rep,chunks={},ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from NOC ARGO OI product.
    Depth coordinate is derived from pressure.
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing in-situ temperature and practical salinity data from the product

    Examples
    --------
    >>> data=JAMSTEC('/home/usr/lenapy/data/JAMSTEC',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """       
    def preproc(ds):
        ds['time']=ds.indexes['time'].to_datetimeindex()
        return ds
    
    fics=glob(os.path.join(rep,'*.nc'))
    
    data=xr.open_mfdataset(fics,preprocess=preproc,chunks=chunks,**kwargs).sel(time=slice(str(ymin),str(ymax)))

    depth=-gsw.z_from_p(data.pressure,0)
    pressure=gsw.p_from_z(-depth,data.latitude)
    pressure=xr.where(pressure<data.pressure,pressure,data.pressure)

    return xr.Dataset({'temp':data.temperature.interp(pressure=pressure).assign_coords(pressure=depth).rename(pressure='depth').chunk(chunks=chunks),
                       'psal':data.practical_salinity.interp(pressure=pressure).assign_coords(pressure=depth).rename(pressure='depth').chunk(chunks=chunks),
                      })
def SODA(rep,chunks={},ymin=0,ymax=9999,filtre='',**kwargs):
    """
    Load data from SODA product.
    
    Parameters
    ----------
    rep : string
        path of the product's directory
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    **kwargs :  optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing potential temperature and practical salinity data from the product

    Examples
    --------
    >>> data=SODA('/home/usr/lenapy/data/SODA',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    """       
    def preproc(ds):
        return ds.rename(dict(xt_ocean='longitude',yt_ocean='latitude',st_ocean='depth'))
    
    def year(f):
        return f.split('_')[-1]
  
    fics=filtre_liste(glob(os.path.join(rep,'*.nc')),year,ymin,ymax,filtre)
    
    data=xr.open_mfdataset(fics,preprocess=preproc,chunks=chunks,**kwargs).chunk(chunks=chunks)
                   
    return xr.Dataset({'PT':data.temp,
                        'psal':data.salt
                          })        

