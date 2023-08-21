Module lenapy.produits
======================
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

Functions
---------

    
`ECCO(rep, chunks=None, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from ECCO product.
    
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
    >>> data=ECCO('/home/usr/lenapy/data/ECCO',ymin=2005,ymax=2007,chunks={'depth':10})
    >>> data.sel(time=slice('2005-06','2007-06')).xocean.gohc.plot()

    
`EN_422(rep, corr, chunks=None, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from EN product.
    
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

    
`IAP(rep, chunks=None, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from IAP product.
    
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

    
`IPRC(rep, chunks=None, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from IPRC product.
    
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

    
`ISAS(rep, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from ISAS product
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

    
`ISHII(rep, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from ISHII product.
    
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

    
`JAMSTEC(rep, chunks=None, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from JAMSTEC product.
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

    
`NCEI(rep, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from NCEI product
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

    
`SIO(rep, chunks=None, ymin=0, ymax=9999, filtre='', **kwargs)`
:   Load data from SIO product.
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

    
`filtre_liste(files, year, ymin, ymax, pattern)`
:   Returns a filtered list of files fitting year range and pattern
    
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

    
`rename_data(data, **kwargs)`
:   Standardization of coordinates names of a product
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