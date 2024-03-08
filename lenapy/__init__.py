"""
LENAPY
======

LENAPY is a set of functions which are an extension of the xarray DataSet and DataArray.
All functions can be used by adding the extension .xgeo or .xocean or xharmo after the object's name.
Three main modules are implemented :

xGeo : for all classical operations on lat/lon grids used in geodesy, altimetry
xOcean : implementation of the gsw library
xHarmo : set of operations on spherical harmonics used in gravimetry


"""
from . import xGeo, xOcean, xTime, xHarmo
