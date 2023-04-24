import xarray as xr
import numpy as np
import pandas as pd

def to_difgri(data,dir_out,prefix,suffix):
    '''
    difgri format use in gins tool is a non binary format with a specific number of columns and format should be %+13.6e
    for example for 64800 values ( 1deg. x 1deg. ) there are 6480 lines sorted by row from left to right,
    starting from longitude -179.5, latitude 89.5.
    (The first 360 values concern latitude 89.5, the next 360 latitude 88.5, etc.)
    '''
    
    data_tmp=data
    list_lon=data_tmp['longitude'].values
    list_lon[list_lon>180]=list_lon[list_lon>180]-360
    data_tmp['longitude']=list_lon

    for ii,year in enumerate(data_tmp.time):
        if year.dt.month in [1,3,5,7,8,10,12]:
            days=31
        elif year.dt.month in [2]:
            days=28
        else:
            days=30
            
        file_out=(f"{dir_out}/{prefix}_{year.dt.strftime('%Y%m').values}01_{year.dt.strftime('%Y%m').values}{days}{suffix}.txt")
        
        df_monthly=data_tmp.isel(time=ii).to_dataframe().drop(columns='time')
        df_sorted =df_monthly.sort_index(level=['latitude',"longitude"],ascending=[False,True]).reset_index()
        df_sorted['line'] = df_sorted.index%10
        df_sorted['col'] = df_sorted.index//10
        df_sorted = df_sorted.set_index(['col','line']).drop(columns=['latitude','longitude'])
        
        df_sorted.unstack().to_csv(file_out,index=None,header=False,float_format='%+13.6e',sep=' ')
    
    
