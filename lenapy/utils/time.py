import xarray as xr
import numpy as np
import dask.array as da
import pandas as pd
import netCDF4
from . import filters
from ..constants import *


def filter(data,filter_name='lanczos',time_coord='time',annual_cycle=False,q=3,**kwargs):
    """Filtre les données en appliquant sur data le filtre filter_name, avec les paramètres définis dans kwargs
    Effectue un miroir des données au début et à la fin pour éviter les effets de bords. Ce miroir est réalisé après avoir retiré un un polynome d'ordre q fittant au mieux les données.

    Parameters
    ----------
    data : xarray DataArray
        Données à filtrer
    filter_name : func (default=Lanczos)
        nom de la fonction de filtrage
    time_coord : str (default='time')
        dimension name along which to apply the filter
    annual_cycle : Bool (default=False)
        retire le cycle annuel avant de filtrer, et le rajoute à la fin
    q : integer (default=3)
        ordre du polynome pour l'effet miroir (gestion des bords)
    **kwargs :
        paramètres de la fonction de filtrage demandée
    """

    if not time_coord in data.coords: raise AssertionError('The time coordinates does not exist')
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
    pf=data0.polyfit(time_coord,q)
    v0=xr.polyval(data0[time_coord],pf).polyfit_coefficients
    # Retrait de ce polynome aux données brutes
    v1=data0-v0
    v1[time_coord]=v1[time_coord].astype('float')
    # Complète les données par effet miroir au début et à la fin
    v2=v1.pad({time_coord:(k,k)},mode='reflect',reflect_type='even')
    v2[time_coord]=v1[time_coord].pad({time_coord:(k,k)},mode='reflect',reflect_type='odd')

    if annual_cycle==True:
        v3=generate_climato(v2[time_coord], coeffs, mean=True, trend=True, cycle=True)
    else:
        v3=0.

    # Convolution par le noyau
    v4=((v3+v2).rolling({time_coord:k},center=True).construct({time_coord:'time_win'})).weighted(noyau).mean('time_win').isel({time_coord:slice(k,-k)})
    v4[time_coord]=data[time_coord]
    
    # Ajout du polynome aux données filtrées
    return v0+v4


def climato(data, signal=True, mean=True, trend=True, cycle=False, return_coeffs=False,time_period=slice(None,None),fillna=False):
    """
    Analyse du cycle annuel, bi-annuel et de la tendance
    Decompose les données en entrée en :
    * Un cycle annuel
    * Un cycle semi-annuel
    * Une tendance
    * Une moyenne
    * Un signal résiduel
    Retourne la combinaison voulue de ces éléments en fonction des arguments choisis (signal, mean, trend, cycle)
    Si return_coeffs=True, retourne les coefficients des cycles et tendances

    Parameters
    ----------
    signal : Bool (default=True)
        Renvoie le signal résiduel après retrait de la climato, de la tendance, et de la moyenne
    mean : Bool (default=True)
        renvoie la valeur moyenne des données d'entrée
    trend : Bool (default=True)
        renvoie la tendance (en jour-1)
    cycle : Bool (default=False)
        renvoie le cycle annuel et semi-annuel
    return_coeffs : Bool (default=False)
        retourne en plus les coefficients des cycles et de la tendance linéaire
    time_period : slice (default=slice(None,None==
        Periode de reference sur laquelle est calculee la climato
    """
    use_dask = True if isinstance(data.data, da.Array) else False
    
    if not 'time' in data.coords: raise AssertionError('The time coordinates does not exist')
    
    # Reference temporelle = milieu de la période
    tmin=data.time.sel(time=time_period).min()
    tmax=data.time.sel(time=time_period).max()
    tref=tmin+(tmax-tmin)/2.
    
    # Construction de la matrice des mesures
    t1=(data.time-tref)/pd.to_timedelta("1D").asm8
    omega=2*np.pi/LNPY_DAYS_YEAR
    X=xr.concat((t1**0,t1,np.cos(omega*t1),np.sin(omega*t1),np.cos(2*omega*t1),np.sin(2*omega*t1)),
                dim=pd.Index(['mean','trend','cosAnnual','sinAnnual','cosSemiAnnual','sinSemiAnnual'], name="coeffs"))
    if use_dask:
        X=X.chunk(time=-1)

    # Vecteur temps a utiliser pour calcul de la climato
    time_vector=data.time.sel(time=time_period)
    
    # Détermination des coefficients par résolution des moindres carrés
    time_vector_in = time_vector.values
    X_in = X.values

    def solve_least_square(data_in):
        """
        For a given 1d time series, returns the coefficients of the fitted climatology
        """
        Y_in_nona = data_in[~np.isnan(data_in)]
        # If less than 6 non-na elements, climato is not computable
        if len(Y_in_nona) <= 6:
            return np.full(X_in.shape[0], np.nan)
        time_in_nona = time_vector_in[~np.isnan(data_in)]
        X_in_nona = X_in[:,~np.isnan(data_in)]
        (coeffs,residus,rank,eig)=np.linalg.lstsq(X_in_nona.T,Y_in_nona,rcond=None)
        return coeffs
    
    # Application de ufunc
    coeffs = xr.apply_ufunc(
        solve_least_square, 
        data, 
        input_core_dims=[['time']],
        output_core_dims=[['coeffs']],
        exclude_dims=set(('time',)),
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'coeffs': X_in.shape[0]}}
    )
    
    coeffs = coeffs.assign_coords(coeffs=X.coeffs.values)
    
    # Calcul des résidus
    data_climato = coeffs*X
    residus = (data-data_climato.sum('coeffs')).assign_coords(coeffs='residu').expand_dims('coeffs')

    # Toutes les composantes de la climatologie
    results = xr.concat((residus,data_climato),dim='coeffs')    

    # Sélection des composantes de la climato à retourner
    composants = np.where([signal,mean,trend,cycle,cycle,cycle,cycle])[0]
    
    if return_coeffs:
        return results.isel(coeffs=composants).sum('coeffs',skipna=fillna), coeffs
    else:
        return results.isel(coeffs=composants).sum('coeffs',skipna=fillna)
    

def generate_climato(time, coeffs, mean=True, trend=False, cycle=True):

    tref=coeffs.time_ref
    t1=(time-tref)/pd.to_timedelta("1D").asm8
    omega=2*np.pi/LNPY_DAYS_YEAR
    X=xr.concat((t1**0,t1,np.cos(omega*t1),np.sin(omega*t1),np.cos(2*omega*t1),np.sin(2*omega*t1)),dim='coeffs').chunk(time=-1)

    # Sélection des composantes de la climato à retourner
    composants=np.where([mean,trend,cycle,cycle,cycle,cycle])[0]

    return (coeffs*X).isel(coeffs=composants).sum('coeffs')

   

def trend(data,time_unit='1s'):
    return data.polyfit(dim='time',deg=1).polyfit_coefficients[0]*pd.to_timedelta(time_unit).asm8.astype('int')

def detrend(data):
    return data - xr.polyval(data.time,data.polyfit(dim='time',deg=1)).polyfit_coefficients

def interp_time(data,other,**kwargs):

    if not 'time' in data.coords: raise AssertionError('The time coordinate does not exist')

    return data.interp(time=other.time,**kwargs)

def to_datetime(data,time_type,format=None):
    if not 'time' in data.coords: raise AssertionError('The time coordinate does not exist')
    if data['time'].dtype=='<M8[ns]':
        return data

    
    if time_type=='frac_year':
        data['time']=[ 
            pd.to_datetime(f'{int(np.floor(i))}')+pd.to_timedelta(float((i-np.floor(i))*LNPY_DAYS_YEAR),unit='D')
            for i in data.time]
    elif time_type=='360_day':
        data.time.attrs['calendar']='360_day'
        data = xr.decode_cf(data).convert_calendar("standard",align_on="year")
    elif time_type=='cftime':
        data['time']=data.indexes['time'].to_datetimeindex()
    elif time_type=='custom':
        data['time']=[ pd.to_datetime(i,format=format) for i in data.time]
    elif time_type=='gregorian':
        data['time']=netCDF4.num2date(data.time, data.time.Units, data.time.calendar)
    else:
        raise ValueError(f'Format {time_type} not yet considered, please convert manually to datatime')

    return data

def diff_3pts(data,dim,time_unit='1s'):
    y=data.where(~data.isnull()).rolling({dim:3},center=True,min_periods=3).construct('win')
    x=data[dim].astype('float').where(~data.isnull()).rolling({dim:3},center=True,min_periods=3).construct('win')

    res=((x*y).mean('win')-x.mean('win')*y.mean('win'))/((x**2).mean('win')-(x.mean('win'))**2)
    return res*(pd.to_timedelta(time_unit).asm8.astype('float'))

def diff_2pts(data,dim,interp_na=True,time_unit='1s',**kw):
    if interp_na:
        d=data.interpolate_na(dim=dim,**kw)
    else:
        d=data
    dy=d.diff(dim)
    dt=d[dim].diff(dim)
    x=d[dim]-dt/2.
    res=dy*(pd.to_timedelta(time_unit).asm8/dt)

    return res.assign_coords({dim:x})


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

def JJ_to_date(jj):
    """
    Turns a date in format 'Jours Julien CNES' into a standard datetime64 format
    """
    if type(jj)==type(None):
        return jj
    dt=pd.Timedelta(jj,'D')
    t0=np.datetime64('1950-01-01','ns')
    return t0+dt

def fillna_climato(data,time_period=slice(None,None)):
    val=climato(data,signal=False,mean=True,trend=True,cycle=True,time_period=time_period)
    return xr.where(data.isnull(),val,data)