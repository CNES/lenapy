import xarray as xr
import numpy as np
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


def function_climato(t,a,b,c,d,e,f):
        l=2.*np.pi/(LNPY_DAYS_YEAR*LNPY_SECONDS_DAY*1.e9)
        return a*np.cos(l*t)+b*np.sin(l*t)+c*np.cos(2.*l*t)+d*np.sin(2.*l*t)+e+f*t

def climato(data, signal=True, mean=True, trend=True, cycle=False, return_coeffs=False,time_period=slice(None,None)):
    """
    Analyse du cycle annuel, bi-annuel et de la tendance
    Decompose les données en entrée en :
    * Un cycle annuel
    * Un cycle bi-annuel
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
    
    fit=data_valid.chunk(dict(time=-1)).curvefit('time',function_climato).curvefit_coefficients*d_sig
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
                       'Day0_YearCycle':np.mod(np.arctan2(b,a)/2./np.pi*LNPY_DAYS_YEAR,LNPY_DAYS_YEAR),
                       'HalfYear_Amplitude':np.sqrt(c**2+d**2),
                       'Day0_HalfYearCycle':np.mod(np.arctan2(d,c)/2./np.pi*LNPY_DAYS_YEAR,LNPY_DAYS_YEAR/2.),
                       'Origin':e+d_mean,
                       'Trend':f*LNPY_DAYS_YEAR*LNPY_SECONDS_DAY*1.e9
                      })
    else:
        return res

def generate_climato(time, coefficients, mean=True, trend=False, cycle=True):
    a=b=c=d=e=f=0.
    if mean:
        e=coefficients.Origin
    if trend:
        f=coefficients.Trend/(LNPY_DAYS_YEAR*LNPY_SECONDS_DAY*1.e9)
    if cycle:
        a=coefficients.Year_amplitude*np.cos(coefficients.Day0_YearCycle/LNPY_DAYS_YEAR*2*np.pi)
        b=coefficients.Year_amplitude*np.sin(coefficients.Day0_YearCycle/LNPY_DAYS_YEAR*2*np.pi)
        c=coefficients.HalfYear_Amplitude*np.cos(coefficients.Day0_HalfYearCycle/LNPY_DAYS_YEAR*2*np.pi)
        d=coefficients.HalfYear_Amplitude*np.sin(coefficients.Day0_HalfYearCycle/LNPY_DAYS_YEAR*2*np.pi)
    
    return function_climato(time,a,b,c,d,e,f)

def trend(data,time_unit='s'):
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
    if jj==None:
        return jj
    dt=np.timedelta64(jj,'D')
    t0=np.datetime64('1950-01-01','ns')
    return t0+dt

def fillna_climato(data,time_period=slice(None,None)):
    val=climato(data,signal=False,mean=True,trend=True,cycle=True,time_period=time_period)
    return xr.where(data.isnull(),val,data)