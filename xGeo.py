import gsw
import numpy as np
import xarray as xr
import xesmf as xe
import os.path

XGEO_DATAPATH=None
XGEO_MASKFILE=None
XGEO_RTER=6378000.

def open_geoset(fic,*args,**kwargs):
    return GeoSet(coords_rename(xr.open_dataset(fic,*args,**kwargs)))

def config(path=None,mask=None):
    global XGEO_DATAPATH, XGEO_MASKFILE
    if path!=None:
        XGEO_DATAPATH=path
        if not(os.path.exists(path)):
            raise NameError("%s does not exist"%XGEO_DATAPATH)

    if mask!=None:
        XGEO_MASKFILE=os.path.join(XGEO_DATAPATH,mask)
        if not(os.path.exists(XGEO_MASKFILE)):
            raise NameError("%s does not exist"%XGEO_MASKFILE)

def concat(g,*args,**kwargs):
    t=type(g[0])
    if len(set([type(u) for u in g]))!=1:
        raise TypeError("Incompatible types")
        
    return t(xr.concat(g,*args,**kwargs))

def coords_rename(gs):
    for l in ['lon','LON','Longitude','LONGITUDE']:
        if l in gs.variables:
            gs=gs.rename({l:'longitude'})


    for l in ['lat','LAT','Latitude','LATITUDE']:
        if l in gs.variables:
            gs=gs.rename({l:'latitude'})
            
    if not('longitude' in gs.coords) or  not('latitude' in gs.coords):
        lat=gs['latitude']
        lon=gs['longitude']
        del(gs['latitude'],gs['longitude'])
        gs=gs.assign_coords(latitude=lat,longitude=lon)

    return gs

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

def cellule(data):
    lon1 = data.longitude.isel(longitude=slice(None,-1)).drop('longitude')
    lon2 = data.longitude.isel(longitude=slice(1,None)).drop('longitude')
    lat1 = data.latitude.isel(latitude=slice(None,-1)).drop('latitude')
    lat2 = data.latitude.isel(latitude=slice(1,None)).drop('latitude')
    lat = (lat1+lat2)/2.
    lon = (lon1+lon2)/2.
    res = GeoArray(XGEO_RTER**1*(np.sin(np.radians(lat2))-np.sin(np.radians(lat1)))*(((lon2-lon1)+360) % 360),
                       dims=['latitude','longitude'],
                       coords={'latitude':lat,'longitude':lon})
    res = res.interp(latitude=data.latitude,longitude=data.longitude,kwargs={"fill_value": "extrapolate"})
    return res    
    
    

# Classe GeoArray
class GeoArray(xr.DataArray):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

    def __add__(self,other):
        return GeoArray(super().__add__(other))

    def __sub__(self,other):
        return GeoArray(super().__sub__(other))

    def __mul__(self,other):
        return GeoArray(super().__mul__(other))

    def __truediv__(self,other):
        return GeoArray(super().__truediv__(other))
    

    def climato(self, remove_mean=False, remove_trend=False):
    
        def func(t,a,b,c,d,e,f):
            l=2.*np.pi/(365.24219*86400.e9)
            return a*np.cos(l*t)+b*np.sin(l*t)+c*np.cos(2.*l*t)+d*np.sin(2.*l*t)+e+f*t*l
    
        fit=self.curvefit('time',func).curvefit_coefficients
        a = fit.sel(param='a')
        b = fit.sel(param='b')
        c = fit.sel(param='c')
        d = fit.sel(param='d')
        e = fit.sel(param='e')
        f = fit.sel(param='f')
        time=self.time.astype('float')
        
        if not(remove_trend):
            e=0
            f=0

        res=GeoArray(self - func(time,a,b,c,d,e,f))
        if remove_mean:
            return res-res.mean(['time'])
        else:
            return res
        return 

    def isosurface(self, target, dim):
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

        field0 = self.isel(slice0).drop(dim)
        field1 = self.isel(slice1).drop(dim)

        crossing_mask_decr = (field0 > target) & (field1 <= target)
        crossing_mask_incr = (field0 < target) & (field1 >= target)
        crossing_mask = xr.where(
            crossing_mask_decr | crossing_mask_incr, 1, np.nan
        )

        coords0 = crossing_mask * self[dim].isel(slice0).drop(dim)
        coords1 = crossing_mask * self[dim].isel(slice1).drop(dim)
        field0 = crossing_mask * field0
        field1 = crossing_mask * field1

        iso = (
            coords0 + (target - field0) * 
            (coords1 - coords0) / (field1 - field0)
        )

        return GeoArray(iso.max(dim, skipna=True))
    
    def filtre(self, coupure, ordre,q=3):
        return GeoArray(filtre(self, coupure, ordre, q))
    
    def interp_time(self,gr_out):
        return self.interp(time=gr_out.time)
    
    def where(self,*args,**kwargs):
        return GeoArray(super().where(*args,**kwargs))
    
    def mean(self,*args,ponderate=False,**kwargs):
        msk=kwargs.pop('mask',True)
        res=xr.DataArray(self.where(msk))
            
        if ponderate:
            return GeoArray(res.weighted(np.cos(np.deg2rad(res.latitude))).mean(*args,**kwargs))
        else:
            return GeoArray(res.mean(*args,**kwargs))

    def mean2(self,*args,**kwargs):
        msk=kwargs.pop('mask',self*0.+1).where(~self.isnull())

        return GeoArray(self.sum(*args,mask=msk,**kwargs)/msk.sum(*args,**kwargs))

    def sum(self,*args,**kwargs):
        cels=cellule(self)
        msk=kwargs.pop('mask',self*0.+1).where(~self.isnull())

        return GeoArray(xr.DataArray.sum((self*cels).where(msk),*args,**kwargs))
                         
    def cumsum(self,*args,**kwargs):
        return GeoArray(super().cumsum(*args,**kwargs))
                         
    def sum_above(self,target,dim,**kwargs):
        return self.cumsum(dim).interp({dim:target},**kwargs)

        
# Classe GeoSet    
class GeoSet(xr.Dataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if 'longitude' in self.coords and 'latitude' in self.coords:
            self['cells']=cellule(self)
      
        
    def __add__(self,other):
        return GeoSet(super().__add__(other))

    def __sub__(self,other):
        return GeoSet(super().__sub__(other))

    def __mul__(self,other):
        return GeoSet(super().__mul__(other))

    def __truediv__(self,other):
        return GeoSet(super().__truediv__(other))
    
    def __getitem__(self,item):
        if item in self.data_vars:
            return GeoArray(super().__getitem__(item))
        else:
            return super().__getitem__(item)
    
    def climato(self, **kwargs):
        res={}
        for var in self.data_vars:
            if 'time' in self[var].coords:
                res[var]=self[var].climato(**kwargs)
            else:
                res[var]=self[var]
        return GeoSet(res)

    def regridder(self,gr_out,*args,**kwargs):
        return xe.Regridder(self,gr_out,*args,**kwargs)
    
    def regrid(self,regridder,*args,**kwargs):
        return GeoSet(regridder(self,*args,**kwargs))

    def interp_time(self,gr_out):
        return self.interp(time=gr_out.time)

    def where(self,*args,**kwargs):
        return GeoSet(super().where(*args,**kwargs))
    
    def mean(self,*args,ponderate=False,mask=False,**kwargs):
        if mask:
            res=self.where(self.mask)
        else:
            res=self
        if ponderate:
            return GeoSet(res.weighted(np.cos(np.deg2rad(self.latitude))).mean(*args,**kwargs))
        else:
            return GeoSet(res.mean(*args,**kwargs))

    def mean2(self,*args,**kwargs):
        msk=self.mask.where(~self.isnull())
        return GeoSet((self*self.cells).where(msk).sum(*args,**kwargs)/self.cells.where(msk).sum(*args,**kwargs))
        
    def isosurface(self, **kwargs):
        dim=kwargs.pop('dim')
        k=list(kwargs.keys())[0]
        if not(k in self.data_vars):
            raise KeyError("%s not in %s"%(kwargs[0],list(data_vars)))
            
        r=self[k].isosurface(kwargs[k],dim)
        res={}
        for u in self.data_vars:
            if dim in self[u].coords:
                res[u]=self[u].interp({dim:r})
            else:
                res[u]=self[u]

        return GeoSet(res)
        
    def sum(self,*args,**kwargs):
        return GeoSet(super().sum(*args,**kwargs))
                       
    def cumsum(self,*args,**kwargs):
        return GeoSet(super().cumsum(*args,**kwargs))
    
    def setmask(self,mask,val=1):
        try:
            del(self['mask'])
        except:
            pass
        gs=open_geoset(XGEO_MASKFILE).rename({mask:'msk_choice'})
        regridder=xe.Regridder(gs,xr.Dataset(self),method='nearest_s2d')
        self['mask']=GeoArray(regridder(gs['msk_choice']==val))
        
