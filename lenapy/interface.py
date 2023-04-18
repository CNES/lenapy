import xarray as xr
import numpy as np
import os.path

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
        v=xr.open_dataset(fic,decode_times=False)
        v.time.attrs['calendar']='360_day'
        v=v.rename({'lat':'latitude','lon':'longitude'})
        return xr.decode_cf(v).isel(depth=slice(0,26)).convert_calendar("standard",align_on="year")
    
    def charge(self,year,season):
        self.nommage(year,season)
        temp = self.ouvre(self.fictemp).t_an
        climt= self.ouvre(self.ficclimt).t_an
        climt['time']=temp['time']
        self.temp = temp + climt
        
        psal = self.ouvre(self.ficsal).s_an
        clims= self.ouvre(self.ficclims).s_an 
        clims['time']=psal['time']
        self.psal = psal + clims
        
        
class ISAS:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"%04d","%s")

    def nommage(self,year, month):
        a='ISAS21_DM_%04d%02d15_fld_%s.nc'
        self.fictemp=self.rep%(year,a%(year,month,'TEMP'))
        self.ficsal =self.rep%(year,a%(year,month,'PSAL'))

    def charge(self,year, month):
        self.nommage(year,month)
        self.temp = xr.open_dataset(self.fictemp).TEMP
        self.psal = xr.open_dataset(self.ficsal).PSAL
 

class IPRC:
    def __init__(self,rep):
        self.rep = os.path.join(rep,"monthly","%s")

    def nommage(self,year, month):
        a='ArgoData_%04d_%02d.nc'
        self.fic=self.rep%(a%(year,month))

    def charge(self,year, month):
        self.nommage(year,month)
        data = xr.open_dataset(self.fic).rename(LATITUDE='latitude').rename(LONGITUDE='longitude').\
            rename(LEVEL='depth').assign_coords(time=np.datetime64("%04d-%02d-15"%(year,month))).expand_dims('time')
        self.temp = data.TEMP
        self.psal = data.SALT

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
        self.ohc=r.squeeze()
    
class LEGOS:
    def __init__(self,rep,nom,version,**kwargs):
        self.fic=os.path.join(rep,nom,"%s_temp_sal_%d.nc"%(nom,version))
        data=xr.open_mfdataset(self.fic).sel(**kwargs)
        data=data.rename(lat='latitude',lon='longitude')
        self.temp=data.temperature.where(data.salinity>0)
        self.psal=data.salinity.where(data.salinity>0)
            
def charge(fichier,lib):
    data=xr.open_dataset(fichier,use_cftime=False)
    data=data.assign_coords(latitude=data.lat,longitude=data.lon)
    return data[lib]