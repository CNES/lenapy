# xGeo
## Name
LENAPY

## Description
LENAPY is a set of unctions which are an extension of the xarray DataSet and DataArray. All funcionts can be used by adding the extension .xgeo or .xocean after the object's name. Three main modules are implemented :
 [] xGeo : for all classical operations on lat/lon grids used in geodesy, altimetry
 [] xOcean : implementation of the gsw library


## Installation
clone the git repository in a dedicated environment, and simply type "pip install ." in the lenapy directory

## Usage
Just import the lenapy module at the beggining of the code :
'''
import lenapy
'''

To use any implemented function, add .xgeo or .xocean after the data name. Repeat this prefix each time you want to use a lenapy function (important for one-liner commands)
Ex:
data.xgeo.mean(['latitude','longitude'],weights=['latitude']).xgeo.climato()

## Implemented functions :
### xgeo :
#### For DataArrays:
[] xgeo.climato()
[] xgeo.mean()
[] xgeo.sum()
[] xgeo.isosurface()
[] xgeo.regridder()
[] xgeo.regrid()
[] xgeo.filtre()
[] xgeo.interp_time()
[] xgeo.diff_3pts()
[] xgeo.trend()
[] xgeo.fill_time()
[] xgeo.to_datetime()
[] xgeo.to_difgri()
[] xgeo.plot_timeseries_uncertainties()

####For DataSets:
[] xgeo.climato()
[] xgeo.mean()
[] xgeo.sum()
[] xgeo.isosurface()
[] xgeo.regridder()
[] xgeo.regrid()
[] xgeo.filtre()
[] xgeo.interp_time()
[] xgeo.fill_time()
[] xgeo.to_datetime()
[] xgeo.coord_renames()

The function xgeo.open_geodata is equivalent to xarra.open_dataset, with an automatic renaming to ensure that coordinates names are "time", "depth", "latitude" and "longitude". A user-defined renaming is also possible. xgeo functions work only if the coordinates are named by these labels.
xgeo.open_mfgeodata is also implemented for distributed computing (equivalent to xr.open_mfdataset)

### xocean
GSW functions apply to any dateset whose variables names are in :
[] xocean.temp (in-situ temperature)
[] xocean.CT (conservative temperature)
[] xocean.PT (potential temperature)
[] xocean.psal (in-situ salinity)
[] xocean.SR (reference salinity)
[] xocean.SA (absolute salinity)
[] xocean.P (pressure)
[] xocean.rho (density)
[] xocean.Cp (calorific capacity)
[] xocean.heat (volumic heat (J/m3))
[] xocean.slh (density anomaly wrt (T=0,S=35))
[] xocean.ohc (ocean heat content in a grid column (J/m²))
[] xocean.ssl (sea surface leval anomaly wrt (T=0,S=35))
[] xocean.gohc (global ocean heat content, mean over ocean surface (J/m2))
[] xocean.gohc_TOA (global ocean heat content, mean over TOA surface (J/m2))
[] xocean.ieeh (Integrated expansion efficiency oh heat (m/(J/m2))

Any variable in computed from data already available  in the dataset
Two functions are also available :
[] ohc_above(target) : return the ohc in the grid column above a given depth (can be multivariate)
[] gohc_above(target) : idem for gohc

### xMask
This class allows to deal with mask. Input is a DataArray, a DataSet, or a file name containing a DataSet
###Methods :
[] set : specify the name of the field to use in a DataSet, the list of values to be consideerd as unmasked data, and a flag to use the complementary of the mask
[] regrid : specify the grid to resample the mask
[] val : return the mask itself, according to parameters used in set and regrid.


## Support
sebastien.fourest@legos.obs-mip.fr

