import gsw
import numpy as np
import xarray as xr

from . import xGeo as xg
from .utils import *

def proprietes(da,nom,label,unite):
    out = da.rename(nom)
    out.attrs['long_name'] = label
    out.attrs['units'] = unite
    return out

def NoneType(var):
    return type(var)==type(None)

@xr.register_dataset_accessor("xocean")
class OceanSet(xg.GeoSet):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)
        fields=['temp','PT','CT','psal','SA','SR','P','rho','Cp','heat','slh','ohc','ssl','ieeh','gohc','eeh']
        for f in fields:
            if hasattr(xarray_obj,f):
                setattr(self,f+"_",xarray_obj[f])
            else:
                setattr(self,f+"_",None)

    @property
    def temp(self):
        if NoneType(self.temp_):
            self.temp_ = proprietes(gsw.t_from_CT(self.SA,self.CT,self.P),
                          'temp','In-situ temperature','dgeC')
        return self.temp_

    @property
    def PT(self):
        if NoneType(self.PT_):
            self.PT_ = proprietes(gsw.pt0_from_t(self.SA,self.temp,self.P),
                          'pt','Potential temperature','degC')
        return self.PT_

    @property
    # Temperature conservative en fonction de la salinité absolue, temperature in-situ, et pression
    def CT(self):
        if NoneType(self.CT_):
            self.CT_ = proprietes(gsw.CT_from_pt(self.SA, self.PT),
                          'CT','Conservative temperature','degC')  # degC
        return self.CT_
    
    @property
    # Salinité pratique
    def psal(self):
        return self._obj.psal
    
    @property
    # Salinité relative, en fonction de la salinité pratique
    def SR(self):
        if NoneType(self.SR_):
            self.SR_ = proprietes(gsw.SR_from_SP(self.psal),
                          'SR','Relative salinity','g/kg') # [g/kg]
        return self.SR_
    
    # Salinité absolue, en fonction de la salinité relative, la pression, et la position
    @property
    def SA(self):
        if NoneType(self.SA_):
            if 'latitude' in self._obj.coords and 'longitude' in self._obj.coords:
                self.SA_ = proprietes(gsw.SA_from_SP(self.psal,self.P,self._obj.longitude, self._obj.latitude),
                          'SA','Absolute salinity','g/kg') # [g/kg]
            else:
                self.SA_ = self.SR
        return self.SA_
    
    @property
    # Pression en fonction de la profondeur et de la latitude
    def P(self):
        if NoneType(self.P_):
            if 'latitude' in self._obj.coords:
                self.P_ = proprietes(gsw.p_from_z(self._obj.depth*-1, self._obj.latitude),
                                  'p_db','Pressure','dbar')
            else:
                self.P_ = proprietes(gsw.p_from_z(self._obj.depth*-1, 0),
                                  'p_db','Pressure','dbar')
        return self.P_

    @property
    # Densité en fonction de la salinité absolue, la température conservtaive et la pression
    def rho(self):
        if NoneType(self.rho_):
            self.rho_ = proprietes(gsw.rho(self.SA, self.CT, self.P),
                          'rho','Density','kg/m3') # [kg/m3]
        return self.rho_
    
    @property
    # Capacité calorifique en fonction de la salinité absolue, la température in-situ et la pression
    def Cp(self):
        if NoneType(self.Cp_):
            self.Cp_ = proprietes(gsw.cp_t_exact(self.SA, self.temp, self.P),
                          'Cp','Specific heat capacity', 'J/kg/degC') # [j/kg/degC)]
        return self.Cp_
    
    @property
    # Contenu en chaleur des couches en fonction de la densité, la capacité calorifique, et la température conservative
    def heat(self):
        if NoneType(self.heat_):
            self.heat_ = proprietes((self.rho * self.Cp * self.CT),
                          'Heat','Heat content','J/m3') # [J/m3]
        return self.heat_
    
    @property
    # Anomalie relative de densité par rapport à une référence (0,35)
    def slh(self):
        if NoneType(self.slh_):
            if 'latitude' in self._obj.coords:
                rhoref = gsw.rho(gsw.SA_from_SP(35,self.P,self._obj.longitude, self._obj.latitude),
                                 0, self.P)
            else:
                rhoref = gsw.rho(gsw.SR_from_SP(35), 0, self.P)
        return  proprietes(((1. - self.rho/rhoref)),
                           'slh','Steric sea layer height','m') # [m]
    
    @property
    # Contenu en chaleur de la colonne
    def ohc(self):
        if NoneType(self.ohc_):
            self.ohc_ = proprietes(self.heat.xocean.integ_depth(),
                          'ohc','Ocean heat content','J/m²') # [J/m²]
        return self.ohc_
    
    @property
    # Ecart de hauteur d'eau de la colonne par rapport à une référence (0,35)
    def ssl(self):
        if NoneType(self.ssl_):
            self.ssl_ = proprietes(self.slh.xocean.integ_depth(),
                          'ssl','Steric sea surface level','m') # [m]
        return self.ssl_
    
    @property
    # IEEH de la colonne (grandeur surfacique)
    def ieeh(self):
        if NoneType(self.ieeh_):
            self.ieeh_ = proprietes(self.ssl/self.ohc,
                          'IEEH','Integrated expansion efficiency oh heat','m/(J/m²)') # [m/(J/m²)]
        return self.ieeh_

    @property
    def gohc(self):
        return proprietes(self.ohc.xocean.mean(['latitude','longitude'],weights=['latitude']),
                         'gohc','Global ocean heat content','J/m²')

    @property
    def gohc_TOA(self):
        return proprietes(self.ohc.xocean.mean(['latitude','longitude'],weights=['latitude'],na_eq_zero=True),
                         'gohc','Global ocean heat content','J/m²')
    
    def ohc_above(self,target):
        res=self.heat.xocean.above(target)
        return proprietes(res.where(res!=0),
            'ohc_above','Ocean heat content','J/m²') # [J/m²]
        
    def gohc_above(self,target,na_eq_zero=False):
        return proprietes(self.ohc_above(target).xocean.mean(['latitude','longitude'],weights=['latitude'],na_eq_zero=na_eq_zero),
                         'gohc_above','Global ocean heat content','J/m²')


@xr.register_dataarray_accessor("xocean")
class OceanArray(xg.GeoArray):
    def __init__(self, xarray_obj):
        super().__init__(xarray_obj)

    def add_value_surface(self,value=None):
        if self._obj.depth[0]!=0:
            v0 = self._obj.isel(depth=0)
            v0['depth']=v0['depth']*0.
            if value!=None:
                v0.values=v0.values*0.+value
            return xr.concat([v0,self._obj],dim='depth')
        else:
            return self._obj
                    
    def integ_depth(self):
        return self.add_value_surface().fillna(0).integrate('depth').where(~self._obj.isel(depth=0).isnull())
    
    def cum_integ_depth(self):
        res=self.add_value_surface()
        ep=res.depth.diff('depth')
        vm=res.rolling(depth=2).mean().isel(depth=slice(1,None))
        return (vm*ep).fillna(0).cumsum('depth').where(~self._obj.isel(depth=0).isnull())
    
    def above(self,depth,**kwargs):
        return self.cum_integ_depth().xocean.add_value_surface(0.).interp({'depth':depth},**kwargs)
    
