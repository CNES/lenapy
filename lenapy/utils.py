import xarray as xr
import numpy as np
import pandas as pd

def lanczos(coupure,ordre):
    """ 
    Filtrage de Lanczos
    Implémente un filtre dont la réponse fréquentielle est une porte de largeur spécifiée par "coupure", 
    convoluée à une autre porte dont la largeur est plus étroite d'un facteur "ordre". Temporellement,
    le filtre est tronqué à +/- ordre * coupure / 2
    Plus "ordre" est grand, plus on se rapproche d'un filtre parfait (sinus cardinal)

    Parameters
    ----------
    coupure : integer
    
    ordre : integer
        ordre du filtre
    """
    
    c=coupure/2.
    x = np.arange(-ordre*c,ordre*c+1,1)
    y = np.sinc(x/c)*np.sinc(x/c/ordre)/c
    y[np.abs(x)>ordre*c]=0.
    y=y/np.sum(y)
    return xr.DataArray(y, dims=('x',), coords={'x':x})


def filter(data,filter_name=lanczos,q=3,**kwargs):
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
    q : integer (default=3)
        ordre du polynome pour l'effet miroir (gestion des bords)
    **kwargs :
        paramètres de la fonction de filtrage demandée
    """
    
    # Noyau de convolution
    data_noyau=filter_name(**kwargs)
    k=len(data_noyau)

    noyau=xr.DataArray(data_noyau,dims=['time_win'],coords={'time_win':np.arange(k)})

    # Fit avec un polynome d'ordre q
    pf=data.polyfit('time',q)
    v0=xr.polyval(data.time,pf).polyfit_coefficients
    # Retrait de ce polynome aux données brutes
    v1=data-v0
    v1['time']=v1['time'].astype('float')
    # Complète les données par effet miroir au début et à la fin
    v2=v1.pad({'time':(k,k)},mode='reflect',reflect_type='even')
    v2['time']=v1['time'].pad({'time':(k,k)},mode='reflect',reflect_type='odd')
    # Convolution par le noyau
    v3=(v2.rolling(time=k,center=True).construct(time='time_win')*noyau).sum('time_win').isel(time=slice(k,-k))
    v3['time']=data['time']
    # Ajout du polynome aux données filtrées
    return v3+v0

    
def isosurface(data, target, dim):
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
    slice0 = {dim: slice(None, -1)}
    slice1 = {dim: slice(1, None)}

    field0 = data.isel(slice0).drop(dim)
    field1 = data.isel(slice1).drop(dim)

    crossing_mask_decr = (field0 > target) & (field1 <= target)
    crossing_mask_incr = (field0 < target) & (field1 >= target)
    crossing_mask = xr.where(
        crossing_mask_decr | crossing_mask_incr, 1, np.nan
    )

    coords0 = crossing_mask * data[dim].isel(slice0).drop(dim)
    coords1 = crossing_mask * data[dim].isel(slice1).drop(dim)
    field0 = crossing_mask * field0
    field1 = crossing_mask * field1

    iso = (
        coords0 + (target - field0) * 
        (coords1 - coords0) / (field1 - field0)
    )

    return iso.max(dim, skipna=True)
        
def climato(data, signal=True, mean=True, trend=True, cycle=False, return_coeffs=False):
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
    """

    def func(t,a,b,c,d,e,f):
        l=2.*np.pi/(365.24219*86400.e9)
        return a*np.cos(l*t)+b*np.sin(l*t)+c*np.cos(2.*l*t)+d*np.sin(2.*l*t)+e+f*t*l

    fit=data.curvefit('time',func).curvefit_coefficients
    a = fit.sel(param='a')
    b = fit.sel(param='b')
    c = fit.sel(param='c')
    d = fit.sel(param='d')
    e = fit.sel(param='e')
    f = fit.sel(param='f')
    time=data.time.astype('float')

    d_mean  = data.mean(['time'])
    d_cycle = func(time,a,b,c,d,0,0)
    d_trend = func(time,0,0,0,0,e,f) - d_mean
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
        return res,[a,b,c,d,e,f]
    else:
        return res

def coords_rename(data,**kwargs):
    
    data=data.rename(**kwargs)
        
    for l in ['lon','LON','Longitude','LONGITUDE']:
        if l in data.variables:
            data=data.rename({l:'longitude'})

    for l in ['lat','LAT','Latitude','LATITUDE']:
        if l in data.variables:
            data=data.rename({l:'latitude'})

    for l in ['TIME','Time']:
        if l in data.variables:
            data=data.rename({l:'time'})
            
    if 'longitude' in data.variables and not('longitude' in data.coords):
        lon=data['longitude']
        del data['longitude']
        lat=data['latitude']
        del data['latitude']    
        data=data.assign_coords(longitude=lon,latitude=lat)        
    return data

def interp_time(data,other,**kwargs):
    return data.interp(time=other.time,**kwargs)

def to_datetime(data,input_type):
    if input_type=='frac_year':
        data=data.rename({'dates':'time'})
        data['time']=[ 
            pd.to_datetime(f'{int(np.floor(i))}')+pd.to_timedelta(float((i-np.floor(i))*365.25),unit='D') 
            for i in data.time]
    
    if input_type=='360_day':
        data.time.attrs['calendar']='360_day'
        data = xr.decode_cf(data).convert_calendar("standard",align_on="year")
    
    return data

def diff_3pts(data,dim):
    y=data.where(~data.isnull()).rolling({dim:3},center=True,min_periods=3).construct('win')
    x=data[dim].where(~data.isnull()).rolling({dim:3},center=True,min_periods=3).construct('win').astype('float')

    return ((x*y).mean('win')-x.mean('win')*y.mean('win'))/((x**2).mean('win')-(x.mean('win'))**2)