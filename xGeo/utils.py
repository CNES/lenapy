import xarray as xr
import numpy as np

XGEO_RTER=6371000.

def concat(g,*args,**kwargs):
    t=type(g[0])
    if len(set([type(u) for u in g]))!=1:
        raise TypeError("Incompatible types")
        
    return t(xr.concat(g,*args,**kwargs))

def normalize_coords_names(gd):
    for l in ['lon','LON','Longitude','LONGITUDE']:
        if l in gd.variables:
            gd=gd.rename({l:'longitude'})


    for l in ['lat','LAT','Latitude','LATITUDE']:
        if l in gd.variables:
            gd=gd.rename({l:'latitude'})
            
    if not('longitude' in gd.coords) or  not('latitude' in gd.coords):
        lat=gd['latitude']
        lon=gd['longitude']
        del(gd['latitude'],gd['longitude'])
        gd=gd.assign_coords(latitude=lat,longitude=lon)

    return gd

def lanczos(coupure,a):
    c=coupure/2.
    x = np.arange(-a*c,a*c+1,1)
    y = np.sinc(x/c)*np.sinc(x/c/a)/c
    y[np.abs(x)>a*c]=0.
    y=y/np.sum(y)
    return xr.DataArray(y, dims=('x',), coords={'x':x})


def filtre(data,coupure, ordre,q=3):
    k=int(coupure*ordre+1)
    noyau=xr.DataArray(lanczos(coupure,ordre),dims=['time_win'],coords={'time_win':np.arange(k)})
    pf=data.polyfit('time',q)
    v0=xr.polyval(data.time,pf).polyfit_coefficients
    v1=data-v0
    v1['time']=v1['time'].astype('float')
    v2=v1.pad({'time':(k,k)},mode='reflect',reflect_type='even')
    v2['time']=v1['time'].pad({'time':(k,k)},mode='reflect',reflect_type='odd')
    v3=(v2.rolling(time=k,center=True).construct(time='time_win')*noyau).sum('time_win')
    v3['time']=v3['time'].astype('datetime64[ns]')
    
    return v3+v0
"""
def surface_cell(data):
    lon1 = data.longitude.isel(longitude=slice(None,-1)).drop('longitude')
    lon2 = data.longitude.isel(longitude=slice(1,None)).drop('longitude')
    lat1 = data.latitude.isel(latitude=slice(None,-1)).drop('latitude')
    lat2 = data.latitude.isel(latitude=slice(1,None)).drop('latitude')
    lat = (lat1+lat2)/2.
    lon = (lon1+lon2)/2.
    res = xr.DataArray(XGEO_RTER**2*(np.sin(np.radians(lat2))-np.sin(np.radians(lat1)))*(((lon2-lon1)+360) % 360),
                       dims=['latitude','longitude'],
                       coords={'latitude':lat,'longitude':lon})
    res = res.interp(latitude=data.latitude,longitude=data.longitude,kwargs={"fill_value": "extrapolate"})
    return res    
"""   
    
def compute_auto_weights(data):
    res=1.
    if 'longitude' in data.coords and 'latitude' in data.coords:
        lon1 = data.longitude.isel(longitude=slice(None,-1)).drop('longitude')
        lon2 = data.longitude.isel(longitude=slice(1,None)).drop('longitude')
        lat1 = data.latitude.isel(latitude=slice(None,-1)).drop('latitude')
        lat2 = data.latitude.isel(latitude=slice(1,None)).drop('latitude')
        lat = (lat1+lat2)/2.
        lon = (lon1+lon2)/2.
        res = xr.DataArray(XGEO_RTER**2*(np.sin(np.radians(lat2))-np.sin(np.radians(lat1)))*(((lon2-lon1)+360) % 360),
                           dims=['latitude','longitude'],
                           coords={'latitude':lat,'longitude':lon})
        res = res.interp(latitude=data.latitude,longitude=data.longitude,kwargs={"fill_value": "extrapolate"})
    if 'depth' in data.coords:
        res=res*xr.concat((data.depth.isel(depth=0),data.depth.diff(dim='depth')),dim='depth')
    return res
    
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
        
def climato(data, remove_mean=False, remove_trend=False):
    
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
        
        if not(remove_trend):
            e=0
            f=0

        res=data - func(time,a,b,c,d,e,f)
        if remove_mean:
            return res-res.mean(['time'])
        else:
            return res
        return 
