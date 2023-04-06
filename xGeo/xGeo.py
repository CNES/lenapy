import gsw
import numpy as np
import xarray as xr
import xesmf as xe
import os.path
from xMask import xmask
from utils import *

XGEO_DATAPATH=None
XGEO_MASKFILE=None
XGEO_RTER=6378000.

def open_geodata(fic,*args,**kwargs):
    return GeoData(xr.open_dataset(fic,*args,**kwargs))
        
# Classe GeoData    
class GeoData(xr.Dataset):
    def __init__(self,*args,weights='auto',mask=None,**kwargs):
        super().__init__(normalize_coords_names(xr.Dataset(*args,**kwargs)))
        
        # Gestion de la surface des cellules : 
        #   automatique si grille lat/lon separable
        if type(weights)==type('auto'):  
            if not('weights' in self.data_vars):
                self['weights'] = compute_auto_weights(self)
        #  toutes egales a 1 
        elif type(weights)==type(None):
            self['weights'] = 1
        #   ou fournie par l'utilisateur (DataArray)
        elif type(weights)==type(xr.DataArray()):
            self['weights'] = weights
        else:
            raise TypeError('Unknown type for weights : %s'%type(weights))
      
        if type(mask)==type(None):
            if not('mask' in self.data_vars):
                self['mask']=1
        elif type(mask)==type(xr.DataArray()):
            self['mask']=mask
        else:
            raise TypeError('Unknown type for mask : %s'%type(mask))
            
    def __add__(self,other):
        return GeoData(super().__add__(other))

    def __sub__(self,other):
        return GeoData(super().__sub__(other))

    def __mul__(self,other):
        return GeoData(super().__mul__(other))

    def __truediv__(self,other):
        return GeoData(super().__truediv__(other))

    def extract(self,item):
        return GeoData(xr.Dataset({item:self[item],
                                       'mask':self['mask'],
                                       'weights':self['weights']}))
    
    def climato(self, **kwargs):
        res={}
        for var in self.data_vars:
            if 'time' in self[var].coords:
                res[var]=self[var].climato(**kwargs)
            else:
                res[var]=self[var]
        return GeoData(res,weights=self.weights,mask=self.mask)

    def regridder(self,gr_out,*args,**kwargs):
        return xe.Regridder(self,gr_out,*args,**kwargs)
    
    def regrid(self,regridder,*args,cells_surface='auto',**kwargs):
        return GeoData(regridder(self,*args,**kwargs),weights=self.weights)

    def interp_time(self,gr_out):
        return self.interp(time=gr_out.time)

    def where(self,*args,**kwargs):
        return GeoData(super().where(*args,**kwargs),weights=self.weights)
    
    def mean(self,*args,mask=False,**kwargs):
        if mask:
            res=self.where(self.mask)
        else:
            res=self
        return GeoData(res.weighted(self.weights).mean(*args,**kwargs),weights=self.weights.sum(*args,**kwargs),mask=self.mask)

    def sum(self,*args,mask=False,**kwargs):
        if mask:
            res=self.where(self.mask)
        else:
            res=self
        return GeoData(res.weighted(self.weights).sum(*args,**kwargs),weights=self.weights.mean(*args,**kwargs),mask=self.mask)
                       
    def cumsum(self,*args,**kwargs):
        return GeoData(super().cumsum(*args,**kwargs))
    
    
    def isosurface(self, **kwargs):
        dim=kwargs.pop('dim')
        k=list(kwargs.keys())[0]
        if not(k in self.data_vars):
            raise KeyError("%s not in %s"%(kwargs[0],list(data_vars)))
            
        r=isosurface(self[k],kwargs[k],dim)
        res=xr.Dataset({"%s_iso"%(dim):r})
        for u in self.data_vars:
            if dim in self[u].coords:
                res[u]=self[u].interp({dim:r})
            else:
                res[u]=self[u]

        return GeoData(res,mask=self.mask)
        
