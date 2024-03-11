import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from .constants import *
from . import filters
                   
    
def isosurface(data, target, dim, coord=None, upper=False):
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
    coord : str (optional)
        The field coordinate to interpolate. If absent, coordinate is supposed to be "dim"
    upper : bool
        if True, returns the highest point of the isosurface, else the lowest

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
    if coord==None:
        coord=dim
        
    slice0 = {dim: slice(None, -1)}
    slice1 = {dim: slice(1, None)}

    field0 = data.isel(slice0).drop(coord)
    field1 = data.isel(slice1).drop(coord)

    crossing_mask_decr = (field0 > target) & (field1 <= target)
    crossing_mask_incr = (field0 < target) & (field1 >= target)
    crossing_mask = xr.where(
        crossing_mask_decr | crossing_mask_incr, 1, np.nan
    )

    coords0 = crossing_mask * data[coord].isel(slice0).drop(coord)
    coords1 = crossing_mask * data[coord].isel(slice1).drop(coord)
    field0 = crossing_mask * field0
    field1 = crossing_mask * field1

    iso = (
        coords0 + (target - field0) * 
        (coords1 - coords0) / (field1 - field0)
    )
    if upper:
        return iso.min(dim, skipna=True)
    else:
        return iso.max(dim, skipna=True)

def split_duplicate_coords(data):
    for u in data.var():
        if (len(set(data[u].dims))!=len(data[u].dims)):
            new_coords={}
            for c in data[u].dims:
                if (c in new_coords.keys()):
                    new_coords[c+"_"]=data[u].coords[c].values
                else:
                    new_coords[c]=data[u].coords[c].values
            data[u]=xr.DataArray(data=data[u].values,dims=new_coords.keys(),coords=new_coords)
    return data

def longitude_increase(data):
    if 'longitude' in data.coords:
        l=xr.where(data.longitude<data.longitude.isel(longitude=0),data.longitude+360,data.longitude)
        if l.max()>360:
            l=l-360
        data['longitude']=l
    return data
        
def reset_longitude(data, orig=-180):
    i=((np.mod(data.longitude-orig+180,360)-180)**2).argmin().values
    return longitude_increase(data.roll(longitude=-i,roll_coords=True))


def surface_cell(data):
    """
    Returns the earth surface of each cell defined by a longitude/latitude in a array
    Cells limits are half distance between each given coordinate. That means that given coordinates are not necessary the center of each cell.
    Border cells are supposed to have the same size on each side of the given coordinate.
    Ex : coords=[1,2,4,7,9] ==> cells size are [1,1.5,2.5,2.5,2]


    Parameters
    ----------
    data : dataarray or dataset
        Must have latitude and longitude coordinates

    Returns
    -------
    surface : dataarray
        dataarray with cells surface

    Example
    -------
    >>>data = xgeo.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
    >>>surface = surface_cell(data)

    """

    dlat=ecarts(data,'latitude')
    dlon=ecarts(data,'longitude')
    
    return np.radians(dlat)*np.radians(dlon)*LNPY_MEAN_EARTH_RADIUS**2*np.cos(np.radians(data.latitude))/(1+LNPY_EARTH_FLATTENING*np.cos(2*np.radians(data.latitude)))**2

def ecarts(data,dim):
    """
    Return the width of each cells along specified coordinate.
    Cells limits are half distance between each given coordinate. That means that given coordinates are not necessary the center of each cell.
    Border cells are supposed to have the same size on each side of the given coordinate.
    Ex : coords=[1,2,4,7,9] ==> cells size are [1,1.5,2.5,2.5,2]
    
    Parameters
    ----------
    data : dataarray or dataset
        Must have latitude and longitude coordinates
    
    dim : str
        Coordinate along which to compute cell width

    Returns
    -------
    width : dataarray
        dataarray with cell width for each coordinate
    
    """
        
    i0=data[dim].isel({dim:slice(None,2)}).diff(dim,label='lower')
    i1=(data[dim]-data[dim].diff(dim,label='upper')/2).diff(dim,label='lower')
    i2=data[dim].isel({dim:slice(-2,None)}).diff(dim,label='upper')
    return xr.concat([i0,i1,i2],dim=dim)
