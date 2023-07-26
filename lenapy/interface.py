import xarray as xr
import numpy as np
import os.path
import sys
import gsw

from . import xGeo as xg

def trouveNaN(data):
    return data.sel(latitude=slice(15,20),longitude=slice(15,20)).values.flatten()[0]

class NCEI:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"%s","%s")
                                        
    def nommage(self, year, season):
        if year<2000:
            a="%02i%02i"%(year-1900,year-1900)
        else:
            d,u=np.divmod(year-2000,10)
            a="%s%i%s%i"%(chr(65+d),u,chr(65+d),u)
            
        b="%02i-%02i"%(3*season-2,3*season)

        self.fictemp=self.rep%('temperature',"tanom_%s%s.nc"%(a,b))
        self.ficsal =self.rep%('salinity',"sanom_%s%s.nc"%(a,b))
        self.ficclimt=self.rep%('climato','woa09_t%02i_01.nc'%(12+season))
        self.ficclims=self.rep%('climato','woa09_s%02i_01.nc'%(12+season))
    
    def ouvre(self,fic):
        v=xg.open_geodata(fic,decode_times=False)
        return v.isel(depth=slice(0,26)).xgeo.to_datetime("360_day")
    
    def charge(self,year,season):
        self.nommage(year,season)
        temp = self.ouvre(self.fictemp).t_an
        climt= self.ouvre(self.ficclimt).t_an
        climt['time']=temp['time']
        
        psal = self.ouvre(self.ficsal).s_an
        clims= self.ouvre(self.ficclims).s_an 
        clims['time']=psal['time']
        
        return xr.Dataset({'temp':temp + climt,
                           'psal':psal + clims
                          })

class SIO:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"%s","%s")
                                        
    def nommage(self, year, month):
        self.fic_data=self.rep%("monthly","RG_ArgoClim_%04d%02d_2019.nc"%(year, month))
        self.fic_clim=self.rep%("climato","RG_ArgoClim_climato_2004_2018.nc")
        
    def charge(self,year,season):
        self.nommage(year,season)
        data=xg.open_geodata(self.fic_data,decode_times=False,rename={'PRESSURE':'pressure'})
        clim=xg.open_geodata(self.fic_clim,rename={'PRESSURE':'pressure'})
        data=data.xgeo.to_datetime("360_day")

        temp = data.ARGO_TEMPERATURE_ANOMALY + clim.ARGO_TEMPERATURE_MEAN
        psal = data.ARGO_SALINITY_ANOMALY + clim.ARGO_SALINITY_MEAN
        
        depth=-gsw.z_from_p(data.pressure,0)
        pressure=gsw.p_from_z(-depth,data.latitude)
        
        return xr.Dataset({'temp':temp.interp(pressure=pressure).assign_coords(pressure=depth).rename(pressure='depth'),
                           'psal':psal.interp(pressure=pressure).assign_coords(pressure=depth).rename(pressure='depth')
                          })
        
        
class ISAS:
    def __init__(self,rep,release,product='ARGO'):
        self.rep = os.path.join(rep,"%04d","%s")
        self.release = release
        self.product=product

    def nommage(self,year, month):
        a='ISAS%02d_%s_%04d%02d15_fld_%s.nc'
        self.fictemp=self.rep%(year,a%(self.release,self.product,year,month,'TEMP'))
        self.ficsal =self.rep%(year,a%(self.release,self.product,year,month,'PSAL'))

    def charge(self,year, month):
        self.nommage(year,month)

        return xr.Dataset({'temp':xg.open_geodata(self.fictemp).TEMP,
                           'psal':xg.open_geodata(self.ficsal).PSAL
                          })        

class ORAS5:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"%s","%s")

    def nommage(self,year, month):
        a='vo%s_control_monthly_highres_3D_%04d%02d_OPER_v0.1.nc'
        self.fictemp=self.rep%('temp',a%('temper',year,month))
        self.ficsal =self.rep%('sal',a%('saline',year,month))

    def charge(self,year, month):
        self.nommage(year,month)

        return xr.Dataset({'temp':xg.open_geodata(self.fictemp).TEMP,
                           'psal':xg.open_geodata(self.ficsal).PSAL
                          })        
        
class IPRC:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"monthly","%s")

    def nommage(self,year, month):
        a='ArgoData_%04d_%02d.nc'
        self.fic=self.rep%(a%(year,month))

    def charge(self,year, month):
        self.nommage(year,month)
        data = xg.open_geodata(self.fic,rename={'LEVEL':'depth'}).\
            assign_coords(time=np.datetime64("%04d-%02d-15"%(year,month))).expand_dims('time')

        return xr.Dataset({'temp':data.TEMP,
                           'psal':data.SALT
                          })        
        
class ISHII:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"%s_monthly","%s")

    def nommage(self,year, month):
        self.fictemp=self.rep%("temp","temp.%04d_%02d.nc"%(year,month))
        self.ficpsal=self.rep%("sal","sal.%04d_%02d.nc"%(year,month))

    def charge(self,year, month):
        self.nommage(year,month)
        return xr.Dataset({'temp':xg.open_geodata(self.fictemp).temperature,
                           'psal':xg.open_geodata(self.ficpsal).salinity
                          })        

        
class IAP:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"%s","%s")

    def nommage(self,year, month):
        self.fictemp=self.rep%("temp","CZ16_1_2000m_Temp_year_%04d_month_%02d.nc"%(year,month))
        self.ficpsal=self.rep%("sal","CZ16_1_2000m_salinity_year_%04d_month_%02d.nc"%(year,month))

    def charge(self,year, month):
        self.nommage(year,month)
        temp = xg.open_geodata(self.fictemp,rename={'depth_std':'depth'}).temp.\
                assign_coords(time=np.datetime64("%04d-%02d-15"%(year,month))).expand_dims('time')
        psal = xg.open_geodata(self.ficpsal,rename={'depth_std':'depth'}).salinity.\
                assign_coords(time=np.datetime64("%04d-%02d-15"%(year,month))).expand_dims('time')
        return xr.Dataset({'temp':temp,
                           'psal':psal
                          })        
                                

class JAMSTEC:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"monthly","%s")

    def nommage(self,year, month):
        a='TS_%04d%02d_GLB.nc'
        self.fic=self.rep%(a%(year,month))

    def charge(self,year, month):
        self.nommage(year,month)
        data = xg.open_geodata(self.fic,rename={'PRES':'pressure'}).\
            assign_coords(time=np.datetime64("%04d-%02d-15"%(year,month))).expand_dims('time')
        data['longitude']=np.mod(data.longitude,360)

        depth=-gsw.z_from_p(data.pressure,0)
        pressure=gsw.p_from_z(-depth,data.latitude)
        
        return xr.Dataset({'temp':data.TOI.interp(pressure=pressure).assign_coords(pressure=depth).rename(pressure='depth'),
                           'psal':data.SOI.interp(pressure=pressure).assign_coords(pressure=depth).rename(pressure='depth')
                          })
    
class EN_422:
    def __init__(self,rep,corr):
        if not(corr in ['g10','l09','c13','c14']):
            sys.exit(-1)
        self.rep = os.path.join(rep,"4.2.2.%s"%corr,"%s")
        self.corr=corr
                                

    def nommage(self,year, month):
        a='EN.4.2.2.f.analysis.%s.%04d%02d.nc'
        self.fic=self.rep%(a%(self.corr,year,month))

    def charge(self,year, month):
        self.nommage(year,month)
        data = xg.open_geodata(self.fic)

        return xr.Dataset({'PT':data.temperature - 273.15,
                           'psal':data.salinity
                          })        

class ECCO:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"%s","%s")

    def charge(self,year, month):
        self.fictemp=self.rep%("THETA","THETA_%04d_%02d.nc"%(year,month))
        self.ficpsal=self.rep%("SALT","SALT_%04d_%02d.nc"%(year,month))

        pt=xg.open_geodata(self.fictemp,nan=0).THETA.set_index(i='longitude',j='latitude',k='Z').\
            rename(i='longitude',j='latitude',k='depth').drop('timestep')
        psal=xg.open_geodata(self.ficpsal,nan=0).SALT.set_index(i='longitude',j='latitude',k='Z').\
            rename(i='longitude',j='latitude',k='depth').drop('timestep')

        return xr.Dataset({'PT':pt.assign_coords(depth=-pt.depth),
                           'psal':psal.assign_coords(depth=-psal.depth)
                          })        

    def charge_mf(self,motif):
        self.fictemp=self.rep%("THETA","THETA_%s.nc"%(motif))
        self.ficpsal=self.rep%("SALT","SALT_%s.nc"%(motif))

        pt=xg.open_mfgeodata(self.fictemp,nan=0).THETA.set_index(i='longitude',j='latitude',k='Z').\
            rename(i='longitude',j='latitude',k='depth').drop('timestep')
        psal=xg.open_mfgeodata(self.ficpsal,nan=0).SALT.set_index(i='longitude',j='latitude',k='Z').\
            rename(i='longitude',j='latitude',k='depth').drop('timestep')

        return xr.Dataset({'PT':pt.assign_coords(depth=-pt.depth),
                           'psal':psal.assign_coords(depth=-psal.depth)
                          })        


    
class LYMAN:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"ohc","%s")

    def nommage(self,year):
        self.fic=self.rep%('RFROM_OHCA_%04d_*.nc'%year)
        
    def charge(self,year):
        self.nommage(year)
        print(self.fic)
        ohc=xr.open_mfdataset(self.fic,data_vars="different")
        ep=ohc.mean_depth_bnds.diff(dim='vertices')
        r=(ohc.ocean_heat_content_anomaly/ep*1.e9).rename(mean_depth='depth')      
        return xr.Dataset({'ohc':r.squeeze()})
    
class LEGOS:
    def __init__(self,rep,nom,version,**kwargs):
        self.fic=os.path.join(rep,nom,"%s_temp_sal_%d.nc"%(nom,version))
        data=xr.open_mfdataset(self.fic).sel(**kwargs)
        data=data.rename(lat='latitude',lon='longitude')

        return xr.Dataset({'temp':data.temperature.where(data.salinity>0),
                           'psal':data.salinity.where(data.salinity>0)
                          })        
    
def charge(fichier,lib):
    data=xr.open_dataset(fichier,use_cftime=False)
    data=data.assign_coords(latitude=data.lat,longitude=data.lon)
    return data[lib]