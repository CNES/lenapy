import xarray as xr
import numpy as np
import pandas as pd
import netCDF4
from ..constants import *
                   
def rename_data(data,**kwargs):
    """Standardization of coordinates names of a product
    Looks for different possible names for latitude, longitude, and time, and turn them into a standardized name.
    Definitions are specified in setup.py, and are also based on standard cf attributes and units
    Custom names changes can also be performed with kwargs parameter

    Parameters
    ----------
    **kwargs :  {old_name: new_name, ...}, optional
        dictionnary specifying old names to be changed into new names

    Returns
    -------
    renamed : Dataset
        New dataset containing modified names

    Example
    -------
    .. code-block:: python

        ds=xr.open_mfdataset('product.nc',preprocess=rename_data)
    """
    data=data.rename(**kwargs)
    for coord in ['latitude','longitude','time','depth']:
        try:
            nom=data.cf[coord].name
            data=data.rename({nom:coord})
        except:
            pass
            

    if 'longitude' in data.variables and not('longitude' in data.coords):
        lon=data['longitude']
        del data['longitude']
        lat=data['latitude']
        del data['latitude']    
        data=data.assign_coords(longitude=lon,latitude=lat)        
    return data


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

    .. code-block:: python

         temp = xr.DataArray(
             range(10,0,-1),
             coords={"depth": range(10)}
             )
         isosurface(temp, 5.5, dim="depth")
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

    for v in data.data_vars:
            dims=data[v].dims
            dup = {x+'_' for x in dims if dims.count(x) > 1}
            for d in dup:
                data=data.assign_coords({d:data[d[:-1]].data})

            new_dims=tuple(list(set(dims))+list(dup))

            if len(dup)>0:
                data[v]=(new_dims,data[v].data)
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


def surface_cell(data, ellipsoidal_earth=True, a_earth=None, f_earth=LNPY_F_EARTH_GRS80):
    """
    Returns the earth surface of each cell defined by a longitude/latitude in an array
    Cells limits are half the distance between each given coordinate. Given coordinates are not necessary the center of
    each cell. Border cells are supposed to have the same size on each side of the given coordinate.
    Ex : coords=[1,2,4,7,9] ==> cell sizes are [1,1.5,2.5,2.5,2]

    The surface of the cell is given by default for an ellipsoidal earth of radius LNPY_A_EARTH_GRS80 and
    flattening LNPY_F_EARTH_GRS80. The parameter `ellipsoidal_earth` can be set to False for the surface on a
    spherical Earth. It can be set to 'approx' for the surface with an approximation of the ellipsoid on each cell by
    a spherical cell with the radius corresponding to the distance between the ellispoid point and the center.

    If surface_cell() is applied to a complete grid, the sum of all cells is equal to 4πR² for the spherical Earth.
    For the ellipsoidal Earth, the sum is equal to 2πa² + πb²/e*log((1 + e)/(1 - e))


    Parameters
    ----------
    data : dataarray or dataset
        Must have latitude and longitude coordinates
    ellipsoidal_earth: bool | str, optional
        Boolean to choose if the surface of the Earth is an ellipsoid or a sphere. Default is True for ellipsoidal Earth
        If ellipsoidal_earth='approx', the given surface is the one of a spherical cell with the radius corresponding to
        the distance between the ellispoid point and the center of the ellipsoid.
    a_earth : float, optional
        Earth semi-major axis [m]. If not provided, use `data.attrs['radius']` and
        if it does not exist, use LNPY_A_EARTH_GRS80.
    f_earth : float, optional
        Earth flattening. Default is LNPY_F_EARTH_GRS80.

    Returns
    -------
    surface : xr.DataArray
        DataArray with cell surface

    Example
    -------
    .. code-block:: python
    
        data = xgeo.open_geodata('/home/user/lenapy/data/gohc_2020.nc')
        surface = surface_cell(data)

    """
    if a_earth is None:
        a_earth = float(data.attrs['radius']) if 'radius' in data.attrs else LNPY_A_EARTH_GRS80

    dlat = ecarts(data, 'latitude').astype('float128')
    dlon = ecarts(data, 'longitude').astype('float128')

    # case of the cell on an ellipsoid
    if ellipsoidal_earth == 'ellipsoidal' or ellipsoidal_earth is True:
        ep = np.sqrt((2 * f_earth - f_earth ** 2) / (1 - f_earth) ** 2)  # eccentricity prime
        # geocentric latitude of the cell border
        omega_1 = np.arctan((1 - f_earth)**2 * np.tan(np.radians(data.latitude - dlat/2)))
        omega_2 = np.arctan((1 - f_earth)**2 * np.tan(np.radians(data.latitude + dlat/2)))

        return np.abs(a_earth**2 * (1 - f_earth) * np.radians(dlon) * (
                (np.arcsinh(ep * np.sin(omega_2)) - np.arcsinh(ep * np.sin(omega_1))) / ep +
                (np.sin(omega_2) * np.sqrt(1 + ep**2 * np.sin(omega_2) ** 2) -
                 np.sin(omega_1) * np.sqrt(1 + ep**2 * np.sin(omega_1) ** 2))) / 2)

    # case of the sphere that approximates the ellipsoid
    elif ellipsoidal_earth == 'approx':
        return np.abs(a_earth**2 * np.radians(dlon) * np.cos(np.radians(data.latitude)) * np.radians(dlat) /
                      (1 + f_earth * np.cos(2 * np.radians(data.latitude)))**2)

    # case of the spherical cell with a sphere of radius a_earth
    elif ellipsoidal_earth == 'spherical' or ellipsoidal_earth is False:
        return np.abs(2 * a_earth**2 * np.radians(dlon) * np.cos(np.radians(data.latitude)) *
                      np.sin(np.radians(dlat) / 2))

    else:
        raise ValueError('Given argument "ellipsoidal_earth" has to be a boolean '
                         'or either "ellispoidal", "spherical" or "approx".')
    

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
