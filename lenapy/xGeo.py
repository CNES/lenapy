import numpy as np
import xarray as xr
import xesmf as xe
import os.path
from .utils import *
from .plotting import *
from .sandbox import *

def open_geodata(fic,*args,rename={},nan=None,**kwargs):
    
    res=coords_rename(xr.open_dataset(fic,*args,**kwargs),**rename)
    return res.where(res!=nan)

@xr.register_dataset_accessor("xgeo")
class GeoSet:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def climato(self,**kwargs):
        # Pour toutes les données dépendant du temps, retourne l'analyse de la climato
        res={}
        for var in self._obj.data_vars:
            if 'time' in self._obj[var].coords:
                res[var]=climato(self._obj[var],**kwargs)
            else:
                res[var]=self._obj[var]
        return xr.Dataset(res)

    def mean(self,*args,**kwargs):
        res={}
        for var in self._obj.data_vars:
            res[var]=self._obj[var].xgeo.mean(*args,**kwargs)
        return xr.Dataset(res)

    def sum(self,*args,**kwargs):
        res={}
        for var in self._obj.data_vars:
            res[var]=self._obj[var].xgeo.sum(*args,**kwargs)
        return xr.Dataset(res)
    
    
    def isosurface(self, dim, **kwargs):
        # Calcule l'isosurface selon la coordonnée 'dim' pour le champ défini par le dictionnaire **kwargs (ex : temp=10)
        # Retourne tous les champs interpolés sur cette isosurface (pour ceux ayant "dim" en coordonnée), ainsi que l'isosurface elle-même
        k=list(kwargs.keys())[0]
        if not(k in self._obj.data_vars):
            raise KeyError("%s not in %s"%(kwargs[0],list(data_vars)))
            
        r=isosurface(self._obj[k],kwargs[k],dim)
        res=xr.Dataset({"%s_iso"%(dim):r})
        for var in self._obj.data_vars:
            if dim in self._obj[var].coords:
                res[var]=self._obj[var].interp({dim:r})
            else:
                res[var]=self._obj[var]

        return res

    def regridder(self,gr_out,*args,**kwargs):
        return xe.Regridder(self._obj,gr_out,*args,**kwargs)
    
    def regrid(self,regridder,*args,**kwargs):
        return regridder(self._obj,*args,**kwargs)
    
    def filtre(self, filter_name=lanczos,q=3, **kwargs):
        res={}
        for var in self._obj.data_vars:
            if 'time' in self._obj[var].coords:
                res[var]=self._obj[var].xgeo.filtre(filter_name=filter_name,q=q,**kwargs)
            else:
                res[var]=self._obj[var]
        return xr.Dataset(res)

    def interp_time(self,other,**kwargs):
        res={}
        for var in self._obj.data_vars:
            if 'time' in self._obj[var].coords:
                res[var]=self._obj[var].xgeo.interp_time(other,**kwargs)
            else:
                res[var]=self._obj[var]
        return xr.Dataset(res)

    def to_datetime(self,input_type):
        return to_datetime(self._obj,input_type)        
    def coords_rename(self):
        res=coords_rename(self._obj)
        return xr.Dataset(res)

@xr.register_dataarray_accessor("xgeo")
class GeoArray:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        
    def climato(self,**kwargs):
        # Retourne les données après analyse de la climato
        return climato(self._obj,**kwargs)
        
    def mean(self,*args,weights=None,mask=True,na_eq_zero=False,**kwargs):
        # Retourne la moyenne pondérée des données.
        #  Si weights n'est pas renseigné, retourne la moyenne sur les dimensions spécifiées (*args)
        #  Si weights='latitude' et/ou 'depth', retourne la moyenne pondérée par le cosinus de la latitude et/ou l'épaisseur des couches en profondeur
        #  Si weight est un array, il est utilisé comme matrice de poids 
        
        # La moyenne est effectuée sur les arguments spécifiés s'ils existent dans les coordonnées
        argmean=set(*args).intersection(list(self._obj.coords))
        data=self._obj.where(mask)
        if na_eq_zero:
            data=data.fillna(0.)
            
        if len(argmean)==0:
            argmean=None
        if type(weights)==type(None):
            # Moyenne simple
            return data.mean(argmean,**kwargs)
        elif type(weights)==list:
            w=1
            if 'latitude' in weights and 'latitude' in self._obj.coords:
                # poids = cos(latitude)
                w=np.cos(np.radians(self._obj.latitude))
            if 'depth' in weights and 'depth' in self._obj.coords:
                # poids *= épaisseur des couches (l'épaisseur de la première couche est la première profondeur)
                w=w*xr.concat((self._obj.depth.isel(depth=0),self._obj.depth.diff(dim='depth')),dim='depth')
            return data.weighted(w).mean(argmean,**kwargs)
        else:
            # matrice de poids définie par l'utilisateur
            return data.weighted(weights).mean(argmean,**kwargs)
    
    def sum(self,*args,weights=None,mask=True,**kwargs):
        # Retourne la somme pondérée des données.
        #  Si weights n'est pas renseigné, retourne la somme sur les dimensions spécifiées (*args)
        #  Si weights='latitude' et/ou 'depth', retourne la somme pondérée par le cosinus de la latitude et/ou l'épaisseur des couches en profondeur
        #  Si weight est un array, il est utilisé comme matrice de poids 
        argsum=set(*args).intersection(list(self._obj.coords))
        data=self._obj.where(mask)        
        if type(weights)==type(None):
            # Somme simple
            return data.sum(argsum,**kwargs)
        elif type(weights)==list:
            w=1
            if 'latitude' in weights and 'latitude' in self._obj.coords:
                # poids = cos(latitude)
                w=np.cos(np.radians(self._obj.latitude))
            if 'depth' in weights and 'depth' in self._obj.coords:
                # poids *= épaisseur des couches (l'épaisseur de la première couche est la première profondeur)
                w=w*xr.concat((self._obj.depth.isel(depth=0),self._obj.depth.diff(dim='depth')),dim='depth')
            return data.weighted(w).sum(argsum,**kwargs)
        else:
            # matrice de poids définie par l'utilisateur
            return data.weighted(weights).sum(argsum,**kwargs)
    

    def isosurface(self, target, dim):   
        return isosurface(self._obj,target,dim)

    def regridder(self,gr_out,*args,mask_in=None,**kwargs):
        ds=xr.Dataset({'data':self._obj})
        if type(mask_in)==xr.DataArray:
            ds['mask']=mask_in
        return xe.Regridder(ds,gr_out,*args,**kwargs)
    
    def regrid(self,regridder,*args,**kwargs):
        return regridder(self._obj,*args,**kwargs)

    def filtre(self, filter_name=lanczos,q=3, **kwargs):
        return filter(self._obj,filter_name=filter_name,q=q,**kwargs)
    
    def interp_time(self,other,**kwargs):
        return interp_time(self._obj,other,**kwargs)
    
    def plot_timeseries_uncertainty(self, **kwargs):
        plot_timeseries_uncertainty(self._obj, **kwargs)
        
    def to_datetime(self,input_type):
        return to_datetime(self._obj,input_type)        
    
    def to_difgri(self,dir_out,prefix,suffix):
        to_difgri(self._obj,dir_out,prefix,suffix)