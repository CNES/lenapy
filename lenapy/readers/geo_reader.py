"""Open netcdf dataset
"""

import numbers
import os

import numpy as np
import xarray as xr
import xesmf as xe
from xarray.backends import BackendEntrypoint

from lenapy.utils.geo import longitude_increase, rename_data, split_duplicate_coords
from lenapy.utils.time import to_datetime


class lenapyNetcdf(BackendEntrypoint):
    """Open netcdf file

    Open a dataset base on xr.open_dataset method, while normalizing coordinates names and choosing NaN values

    Parameters
    ----------
    file : path
        pathname of the file to open
    rename : dict, optional
        dictionnary {old_name:new_name,...}
    nan : optional
        value to be replaced by NaN
    time_type : str, optional
        type used for time coordinate, used to convert to datetime64
        possible values : 'frac_year' or '360_day' or 'cftime' or 'custom'

        if 'custom', "decode_times=False" and "format" must be specified
    format : str, optional
        strftime to parse time (if time_type=='custom')
    duplicate_coords : boolean, optional
        rename duplicate coordinates, by adding a "_" at the end of duplicate ones
    decode_times : boolean, optional
        try to decode time coordinate to a datetime64 type
    set_time : None or func, optional
        if set to a function name, apply the result of the function passed to the file basename to set the time coordinate
    open_dataset_kwargs : dict, optional
        Additional keyword arguments passed on to native open_dataset function

    Returns
    -------
    data : Dataset
        Dataset loaded from file

    Example
    -------
    .. code-block:: python

        data = xr.open_dataset('/home/user/lenapy/data/gohc_2020.nc', engine="lenapyNetcdf")
    """

    def open_dataset(
        self,
        filename,
        rename={},
        nan=None,
        time_type=None,
        format=None,
        duplicate_coords=False,
        drop_variables=None,
        decode_times=True,
        set_time=None,
        open_dataset_kwargs={},
    ):

        res = rename_data(
            xr.open_dataset(
                filename,
                drop_variables=drop_variables,
                decode_times=decode_times,
                **open_dataset_kwargs,
            ),
            **rename,
        )

        if type(set_time) != type(None):
            res = res.assign_coords(
                time=set_time(
                    os.path.basename(os.path.splitext(res.encoding["source"])[0])
                )
            ).expand_dims(dim="time")
        if time_type != None:
            res = to_datetime(res, time_type=time_type, format=format)
        if duplicate_coords:
            res = split_duplicate_coords(res)
        res = longitude_increase(res)
        if type(nan) != type(None):
            return res.where(res != nan)
        else:
            return res


class lenapyMask(BackendEntrypoint):
    """Open mask file

    Open a mask in a dataset file, choose a given field, and regrid according to a given geometry.
    The returned mask contains an extra dimension named 'zone', corresponding to the different values
    of the mask contained in the opened dataset.
    Required format for the dataset to be opened :
        * contains several dataarrays, each one being a mask (identified as 'field')
        * each mask is defined by values, whose signification is given in the attributes of the dataarray : {'value1' : 'label1',...}
    If there is no valid attribute, the mask is returned with False where NaN, False or 0 are found, True for any other value.

    Parameters
    ----------
    filename : path
        path and filename of the mask file to be opened
    field : string or array of strings
        name of the data to be used as a mask in the dataset
    grid : dataset, optional
        dataset to be regridded on. If None, no regridding is performed

    Returns
    -------
    mask : DataArray
        DataArray with regridded mask

    Example
    -------
    .. code-block:: python

        mask = xr.open_dataset('/home/user/lenapy/data/mask/GEO_mask_1deg_20210406.nc',engine="lenapyMask",field='mask_continents').mask
        mask.zone
        <xarray.DataArray (latitude: 360, longitude: 720, zone: 9)>
        array([[[False,  True, False, ..., False, False, False],
        ...
        [False, False, False, ..., False, False, False]]])
        Coordinates:
          * longitude  (longitude) float64 -179.8 -179.2 -178.8 ... 178.8 179.2 179.8
          * latitude   (latitude) float64 -89.75 -89.25 -88.75 ... 88.75 89.25 89.75
          * zone       (zone) <U18 'Greenland' 'Antarctica' ... 'Maritime continent'
    """

    def open_dataset(self, filename, field=None, grid=None, drop_variables=None):

        res = []
        # Opening
        file = xr.open_dataset(filename)
        if field == None:
            field = file.data_vars

        for f in np.ravel(field):
            mask = file[f]
            if not (
                isinstance(mask.values.ravel()[0], numbers.Number)
                and "latitude" in mask.coords
            ):
                continue

            # Labels reading
            c = []
            d = []
            for k in mask.attrs.keys():
                if k.isnumeric():
                    d.append(int(k))
                    c.append(mask.attrs[k])
                if isinstance(mask.attrs[k], numbers.Number):
                    d.append(mask.attrs[k])
                    c.append(k)
            if len(d) == 0:
                d = [False]
                c = [f]
                mask = mask.where(mask.notnull(), 0).astype("int") == 0

            labels = xr.DataArray(data=d, dims="zone", coords=dict(zone=c))

            # Resampling
            if type(grid) != type(None):
                reg = xe.Regridder(mask, grid, method="nearest_s2d")
                mask = reg(mask)

            # Returning
            res.append(xr.Dataset({"mask": xr.where(mask == labels, True, False)}))

        return xr.concat(res, dim="zone")
