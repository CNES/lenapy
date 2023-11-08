import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from .constants import *
from . import filters
                   

def filter(data,filter_name='lanczos',annual_cycle=False,q=3,**kwargs):
    """
    Filtre les données en appliquant sur data le filtre filter_name, avec les paramètres définis dans **kwargs
    Effectue un miroir des données au début et à la fin pour éviter les effets de bords. Ce miroir est réalisé
    après avoir retiré un un polynome d'ordre q fittant au mieux les données.

    Parameters
    ----------
    data : xarray DataArray
        Données à filtrer
    filter_name : func (default=Lanczos)
        nom de la fonction de filtrage
    annual_cycle : Bool (deafult=False) : retire le cycle annuel avant de filtrer, et le rajoute à la fin
    q : integer (default=3)
        ordre du polynome pour l'effet miroir (gestion des bords)
    **kwargs :
        paramètres de la fonction de filtrage demandée
    """

    if not 'time' in data.coords: raise AssertionError('The time coordinates does not exist')
    try:
        f = getattr(filters,filter_name)
    except:
        f = filter_name
    # Noyau de convolution
    data_noyau=f(**kwargs)
    k=len(data_noyau)

    noyau=xr.DataArray(data_noyau,dims=['time_win'],coords={'time_win':np.arange(k)})

    if annual_cycle==True:
        # On vire la climato
        data0,coeffs=climato(data,mean=False,trend=False,return_coeffs=True)
    else:
        data0=data

    # On fait un miroir sans la climato
    pf=data0.polyfit('time',q)
    v0=xr.polyval(data0.time,pf).polyfit_coefficients
    # Retrait de ce polynome aux données brutes
    v1=data0-v0
    v1['time']=v1['time'].astype('float')
    # Complète les données par effet miroir au début et à la fin
    v2=v1.pad({'time':(k,k)},mode='reflect',reflect_type='even')
    v2['time']=v1['time'].pad({'time':(k,k)},mode='reflect',reflect_type='odd')

    if annual_cycle==True:
        v3=generate_climato(v2['time'], coeffs, mean=True, trend=True, cycle=True)
    else:
        v3=0.

    # Convolution par le noyau
    v4=((v3+v2).rolling(time=k,center=True).construct(time='time_win')).weighted(noyau).mean('time_win').isel(time=slice(k,-k))
    v4['time']=data['time']
    
    # Ajout du polynome aux données filtrées
    return v0+v4
 
    
def isosurface(data, target, dim, coord=None, upper=False):
    """
    Linearly interpolate a coordinate isosurface where a field
    equals a target

    Parameters
    ----------
    field : xarray DataArray
        The field in which to interpolate the target isosurface
    target : float
        The target isosurface value
    dim : str
        The field dimension to interpolate
    coord : str (optional)
        The field coordinate to interpolate. If absent, coordinate is supposed to be "dim"
    upper : bool
        if True, returns the highest point of the isosurface, else the lowest

    Examples
    --------
    Calculate the depth of an isotherm with a value of 5.5:

    >>> temp = xr.DataArray(
    ...     range(10,0,-1),
    ...     coords={"depth": range(10)}
    ... )
    >>> isosurface(temp, 5.5, dim="depth")
    <xarray.DataArray ()>
    array(4.5)
    """
    if coord==None:
        coord=dim
        
    slice0 = {dim: slice(None, -1)}
    slice1 = {dim: slice(1, None)}

    field0 = data.isel(slice0).drop(coord)
    field1 = data.isel(slice1).drop(coord)

    crossing_mask_decr = (field0 > target) & (field1 <= target)
    crossing_mask_incr = (field0 < target) & (field1 >= target)
    crossing_mask = xr.where(
        crossing_mask_decr | crossing_mask_incr, 1, np.nan
    )

    coords0 = crossing_mask * data[coord].isel(slice0).drop(coord)
    coords1 = crossing_mask * data[coord].isel(slice1).drop(coord)
    field0 = crossing_mask * field0
    field1 = crossing_mask * field1

    iso = (
        coords0 + (target - field0) * 
        (coords1 - coords0) / (field1 - field0)
    )
    if upper:
        return iso.min(dim, skipna=True)
    else:
        return iso.max(dim, skipna=True)
        
def function_climato(t,a,b,c,d,e,f):
        l=2.*np.pi/(DAY_YEAR*SECONDS_DAY*1.e9)
        return a*np.cos(l*t)+b*np.sin(l*t)+c*np.cos(2.*l*t)+d*np.sin(2.*l*t)+e+f*t
        
def climato(data, signal=True, mean=True, trend=True, cycle=False, return_coeffs=False,time_period=slice(None,None)):
    """
    Analyse du cycle annuel, bi-annuel et de la tendance
    Decompose les données en entrée en :
     Un cycle annuel
     Un cycle bi-annuel
     Une tendance
     Une moyenne
     Un signal résiduel
    Retourne la combinaison voulue de ces éléments en fonction des arguments choisis (signal, mean, trend, cycle)
    Si return_coeffs=True, retourne les coefficients des cycles et tendances

    Parameters
    ----------
    signal : Bool (default=True)
        Renvoie le signal résiduel après retrait de la climato, de la tendance, et de la moyenne
    mean : Bool (default=True)
        renvoie la valeur moyenne des données d'entrée
    trend : Bool (default=True)
        renvoie la tendance
    cycle : Bool (default=False)
        renvoie le cycle annuel et bi-annuel
    return_coeffs : Bool (default=False)
        retourne en plus les coefficients des cycles et de la tendance linéaire
    time_period : slice (default=slice(None,None==
        Periode de reference sur laquelle est calculee la climato
    """

    if not 'time' in data.coords: raise AssertionError('The time coordinates does not exist')

    # Donnees de la periode de reference
    data_ref=data.sel(time=time_period)
    
    # Mise à l'échelle pour éviter les pb de précision machine
    d_mean  = data_ref.mean(['time'])
    d_sig = data_ref.std('time')
    
    # Eliminer les séries où il y a moins de 6 points (pas de climato possible)
    data_valid=(data_ref-d_mean).where(data_ref.count(dim='time')>5,0)/d_sig
    
    fit=data_valid.curvefit('time',function_climato).curvefit_coefficients*d_sig
    [a,b,c,d,e,f] = [fit.sel(param=u).drop('param') for u in ['a','b','c','d','e','f']]
    
    time=data.time.astype('float')

    d_cycle = function_climato(time,a,b,c,d,0,0)
    d_trend = function_climato(time,0,0,0,0,e,f)
    d_signal= data - d_cycle - d_trend - d_mean

    res=0
    if cycle:
        res+=d_cycle
    if trend:
        res+=d_trend
    if signal:
        res+=d_signal
    if mean:
        res+=d_mean
    if return_coeffs:
        return res,xr.Dataset({'Year_amplitude':np.sqrt(a**2+b**2),
                       'Day0_YearCycle':np.mod(np.arctan2(b,a)/2./np.pi*DAY_YEAR,DAY_YEAR),
                       'HalfYear_Amplitude':np.sqrt(c**2+d**2),
                       'Day0_HalfYearCycle':np.mod(np.arctan2(d,c)/2./np.pi*DAY_YEAR,DAY_YEAR/2.),
                       'Origin':e+d_mean,
                       'Trend':f*DAY_YEAR*SECONDS_DAY*1.e9
                      })
    else:
        return res
    
def generate_climato(time, coefficients, mean=True, trend=False, cycle=True):
    a=b=c=d=e=f=0.
    if mean:
        e=coefficients.Origin
    if trend:
        f=coefficients.Trend/(DAY_YEAR*SECONDS_DAY*1.e9)
    if cycle:
        a=coefficients.Year_amplitude*np.cos(coefficients.Day0_YearCycle/DAY_YEAR*2*np.pi)
        b=coefficients.Year_amplitude*np.sin(coefficients.Day0_YearCycle/DAY_YEAR*2*np.pi)
        c=coefficients.HalfYear_Amplitude*np.cos(coefficients.Day0_HalfYearCycle/DAY_YEAR*2*np.pi)
        d=coefficients.HalfYear_Amplitude*np.sin(coefficients.Day0_HalfYearCycle/DAY_YEAR*2*np.pi)
    
    return function_climato(time,a,b,c,d,e,f)
    
def trend(data):
    return data.polyfit(dim='time',deg=1).polyfit_coefficients[0]*1.e9

def interp_time(data,other,**kwargs):

    if not 'time' in data.coords: raise AssertionError('The time coordinate does not exist')

    return data.interp(time=other.time,**kwargs)

def to_datetime(data,input_type,format=None):
    if not 'time' in data.coords: raise AssertionError('The time coordinate does not exist')
    if data['time'].dtype=='<M8[ns]':
        return data

    
    if input_type=='frac_year':
        data['time']=[ 
            pd.to_datetime(f'{int(np.floor(i))}')+pd.to_timedelta(float((i-np.floor(i))*DAY_YEAR),unit='D') 
            for i in data.time]
    elif input_type=='360_day':
        data.time.attrs['calendar']='360_day'
        data = xr.decode_cf(data).convert_calendar("standard",align_on="year")
    elif input_type=='cftime':
        data['time']=data.indexes['time'].to_datetimeindex()
    elif input_type=='custom':
        data['time']=[ pd.to_datetime(i,format=format) for i in data.time]
    elif input_type=='gregorian':
        data['time']=netCDF4.num2date(data.time, data.time.Units, data.time.calendar)
    else:
        raise ValueError(f'Format {input_type} not yet considered, please convert manually to datatime')
      
    return data

def split_duplicate_coords(data):
    for u in data.var():
        if (len(set(data[u].dims))!=len(data[u].dims)):
            new_coords={}
            for c in data[u].dims:
                if (c in new_coords.keys()):
                    new_coords[c+"_"]=data[u].coords[c]
                else:
                    new_coords[c]=data[u].coords[c]
            data[u]=xr.DataArray(data=data[u].values,dims=new_coords.keys(),coords=new_coords)
    return data

def longitude_increase(data):
    if 'longitude' in data.coords:
        l=xr.where(data.longitude<data.longitude.isel(longitude=0),data.longitude+360,data.longitude)
        if l.max()>360:
            l=l-360
        data['longitude']=l
    return data
        
def reset_longitude(data, orig=-180):
    i=((np.mod(data.longitude-orig+180,360)-180)**2).argmin().values
    return longitude_increase(data.roll(longitude=-i,roll_coords=True))
    
def diff_3pts(data,dim):
    y=data.where(~data.isnull()).rolling({dim:3},center=True,min_periods=3).construct('win')
    x=data[dim].where(~data.isnull()).rolling({dim:3},center=True,min_periods=3).construct('win').astype('float')

    return ((x*y).mean('win')-x.mean('win')*y.mean('win'))/((x**2).mean('win')-(x.mean('win'))**2)

def diff_2pts(data,dim,interp_na=True,**kw):
    if interp_na:
        d=data.interpolate_na(dim=dim,**kw)
    else:
        d=data
    y=d.diff(dim)
    x=d[dim].rolling({dim:2}).mean()
    return y.assign_coords({dim:x})
    

def fill_time(data):
    # Complète les trous de données dans une série temporelle en respectant approximativement l'échantillonnage, 
    #  et en faisant une interpolation linéaire de la donnée là où il y a des trous.
    
    if not 'time' in data.coords: raise AssertionError('The time coordinates does not exist')
    
    dt=data.time.diff('time')
    # Recherche du pas d'echantillonnage temporel le plus régulier
    tau0=dt.median()
    tau1=(dt[np.where(np.abs((dt-tau0)/tau0)<0.2)]).mean()
    
    # Parours des index temporels, et ajoute des index là où il y a des trous
    nt=data.time[0].values
    for k in range(len(data.time)-1):
        for i in np.arange(1,np.round(dt[k]/tau1)):
            nt=np.append(nt,data.time[k].values+i*tau1)
        nt=np.append(nt,data.time[k+1].values)
    
    # Génération du nouvel index temporel
    newtime=(xr.DataArray(nt,dims='time',coords={'time':nt}))
    
    # Retourne la donnée interpolée
    return data.interp(time=newtime)

def surface_cell(data):
    """
    Returns the earth surface of each cell defined by a longitude/latitude in a array
    Cells limits are half distance between each given coordinate. That means that given coordinates are not necessary the center of each cell.
    Border cells are supposed to have the same size on each side of the given coordinate.
    Ex : coords=[1,2,4,7,9] ==> cells size are [1,1.5,2.5,2.5,2]


    Parameters
    ----------
    data : dataarray or dataset
        Must have latitude and longitude coordinates

    Returns
    -------
    surface : dataarray
        dataarray with cells surface

    Example
    -------
    >>>data = xgeo.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
    >>>surface = surface_cell(data)

    """

    dlat=ecarts(data,'latitude')
    dlon=ecarts(data,'longitude')
    
    return np.radians(dlat)*np.radians(dlon)*LNPY_RTER**2*np.cos(np.radians(data.latitude))/(1+LNPY_f*np.cos(2*np.radians(data.latitude)))**2

def ecarts(data,dim):
    """
    Return the width of each cells along specified coordinate.
    Cells limits are half distance between each given coordinate. That means that given coordinates are not necessary the center of each cell.
    Border cells are supposed to have the same size on each side of the given coordinate.
    Ex : coords=[1,2,4,7,9] ==> cells size are [1,1.5,2.5,2.5,2]
    
    Parameters
    ----------
    data : dataarray or dataset
        Must have latitude and longitude coordinates
    
    dim : str
        Coordinate along which to compute cell width

    Returns
    -------
    width : dataarray
        dataarray with cell width for each coordinate
    
    """
        
    i0=data[dim].isel({dim:slice(None,2)}).diff(dim,label='lower')
    i1=(data[dim]-data[dim].diff(dim,label='upper')/2).diff(dim,label='lower')
    i2=data[dim].isel({dim:slice(-2,None)}).diff(dim,label='upper')
    return xr.concat([i0,i1,i2],dim=dim)

def JJ_to_date(jj):
    """
    Turns a date in format 'Jours Julien CNES' into a standard datetime64 format
    """
    if jj==None:
        return jj
    dt=np.timedelta64(jj,'D')
    t0=np.datetime64('1950-01-01','ns')
    return t0+dt