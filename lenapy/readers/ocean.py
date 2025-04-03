# -*- coding: utf-8 -*-
"""Load temperature and salinity data from ocean products

"""
import os.path
import sys
from glob import glob

import gsw
import numpy as np
import xarray as xr
import xesmf as xe
from xarray.backends import BackendEntrypoint


def filtered_list(files, year, ymin, ymax, pattern):
    """
    Returns a filtered list of files fitting year range and pattern

    Parameters
    ----------
    files : Array of strings
        Array with filenames to be filtered

    year : function
        Function to be applied on each filename, returning the data year

    ymin : integer
        Lowest bound of the time range

    ymax : integer
        Highest bound of the time range

    pattern : string
        Pattern that must fit the filenames

    Return
    ------
    filtered : Array
        Extract of th input array fitting year and pattern conditions

    Example
    -------
    .. code-block:: python

       def year(f):
       return f.split('_')[1]
       fics=filtre_liste(glob(os.path.join(rep,'**','*.nc')),year,2005,2006,'ARGO')

    """
    r = []
    for u in files:
        d = int(year(os.path.basename(os.path.splitext(u)[0])))
        if d >= ymin and d <= ymax and pattern in u:
            r.append(u)
    return r


class lenapyOceanProducts(BackendEntrypoint):
    """Open netcdf ocean product

    This module allows to load temperature and salinity data from different products and format the data with unified definition for variables and coordinates, compatible with the use of lenapy_ocean :
        * standardized coordinates names : latitude, longitude, depth, time
        * standardized variables names : temp or PT or CT for temperature, psal, SA, SR for salinity
    When loading a product, all the files present in the product directory are parsed. To gain computing time, a first filter on the years to load can be applied, as well as a text filter.
    A second date filter can be apply afterwards with the .sel(time=slice('begin_date','end_date') method.
    All keyword arguments associated with xr.open_dataset can be passed.
    Dask is implicitely used when using these interface methods.

    It is imperative to chunk the dataset along time dimension (chunks=dict(depth=10)), and recommanded to chunk the dataset along depth dimension (chunks=dict(depth=10))

    Parameters
    ----------
    directory : string
        path of the product's directory
    product : string
        identifier of the product to be opened. Can be one of :
        "ISAS","NCEI","SIO","IPRC","ISHII","IAP","EN4","JAMSTEC","ECCO","NOC_OI","SODA"
    corr : string, optional
        additional parameter for EN4 prodcut ensemble. Can be one of:
        'g10','l09','c13','c14'
    ymin : int, optional
        lowest bound of the time intervalle to be loaded (year)
    ymax : int, optional
        highest bound of the time intervalle to be loaded (year)
    filter : string, optionnal
        string pattern to filter datafiles names
    chunks : dict, optionnal
        dictionnary {dim:chunks}
    open_dataset_kwargs : dict, optional
        The keyword arguments form of open_mfdataset

    Returns
    -------
    product : Dataset
        New dataset containing temperature and salinity data from the product

    Examples
    --------

    .. code-block:: python

     data=xr.open_dataset('/home/usr/lenapy/data/ISAS20',engine='lenapyOceanProduct',product='ISAS",ymin=2005,ymax=2007,filter='ARGO',chunks={'time':1,'depth':10})

    """

    def open_dataset(
        self,
        directory,
        product=None,
        corr=None,
        ymin=0,
        ymax=9999,
        filter="",
        drop_variables=None,
        open_dataset_kwargs={},
    ):

        # --------------------- ISAS ---------------------
        if product == "ISAS":
            """
            Load data from ISAS product
            Product's directory must have the following format:
            -base directory (ex: ISAS20_ARGO)
               |-year (ex: 2020)
                   |-release_type_timestamp_xxx_data.nc (ex: ISAS20_ARGO_20200915_fld_TEMP.nc)
            """

            def year(f):
                return f.split("_")[2][0:4]

            fics = filtered_list(
                glob(os.path.join(directory, "**", "*.nc")), year, ymin, ymax, filter
            )

            data = xr.open_mfdataset(fics, **open_dataset_kwargs)

            return xr.Dataset({"temp": data.TEMP, "psal": data.PSAL})

        # --------------------- NCEI ---------------------
        elif product == "NCEI":
            """
            Load data from NCEI product
            Temperature and salinity are reconstructed from anomaly and trimestrial climatology

            Product's directory must have the following format:
            -base directory (ex: NCEI)
               |-salinity
                   |-sanom_timestamp.nc (ex: sanom_C1C107-09.nc)
               |-temperature
                   |-tanom_timestamp.nc (ex: tanom_C1C107-09.nc)
            """

            def year(f):
                a = f.split("_")[1][0]
                b = f.split("_")[1][1]
                return "%04d" % (1900 + 10 * int("0x%s" % a, 16) + int(b))

            # Salinite
            fics_sal = filtered_list(
                glob(os.path.join(directory, "**", "sanom*.nc")),
                year,
                ymin,
                ymax,
                filter,
            )
            sal = xr.open_mfdataset(
                fics_sal,
                engine="lenapyNetcdf",
                decode_times=False,
                time_type="360_day",
                **open_dataset_kwargs,
            ).s_an
            fics_climsal = glob(os.path.join(directory, "climato", "*s1[3-6]_01.nc"))
            climsal = xr.open_mfdataset(
                fics_climsal,
                engine="lenapyNetcdf",
                decode_times=False,
                time_type="360_day",
                **open_dataset_kwargs,
            ).s_an

            # Temperature
            fics_temp = filtered_list(
                glob(os.path.join(directory, "**", "tanom*.nc")),
                year,
                ymin,
                ymax,
                filter,
            )
            temp = xr.open_mfdataset(
                fics_temp,
                engine="lenapyNetcdf",
                decode_times=False,
                time_type="360_day",
                **open_dataset_kwargs,
            ).t_an
            fics_climtemp = glob(os.path.join(directory, "climato", "*t1[3-6]_01.nc"))
            climtemp = xr.open_mfdataset(
                fics_climtemp,
                engine="lenapyNetcdf",
                decode_times=False,
                time_type="360_day",
                **open_dataset_kwargs,
            ).t_an

            return xr.Dataset(
                {
                    "temp": (
                        temp.groupby("time.month")
                        + climtemp.groupby("time.month").mean("time")
                    ).drop("month"),
                    "psal": (
                        sal.groupby("time.month")
                        + climsal.groupby("time.month").mean("time")
                    ).drop("month"),
                }
            )

        # --------------------- SIO ---------------------
        elif product == "SIO":
            """
            Load data from SIO product.
            Temperature and salinity are reconstructed from anomaly and annual climatology, and depth coordinate is derived from pressure.

            Product's directory must have the following format:
            -base directory (ex: SIO)
               |-climato
                   |-RG_ArgoClim_climato_release.nc (ex: RG_ArgoClim_climato_2004_2018.nc)
               |-monthly
                   |-RG_ArgoClim_timestamp_release.nc (ex: RG_ArgoClim_202210_2019.nc)
            """

            def year(f):
                return f.split("_")[-2][0:4]

            fics_anom = filtered_list(
                glob(os.path.join(directory, "**", "RG_ArgoClim_20*.nc")),
                year,
                ymin,
                ymax,
                filter,
            )
            fics_clim = glob(os.path.join(directory, "**", "RG_ArgoClim_cl*.nc"))

            anom = xr.open_mfdataset(
                fics_anom,
                engine="lenapyNetcdf",
                decode_times=False,
                time_type="360_day",
                **open_dataset_kwargs,
            )
            clim = xr.open_mfdataset(
                fics_clim, engine="lenapyNetcdf", **open_dataset_kwargs
            )

            temp = anom.ARGO_TEMPERATURE_ANOMALY + clim.ARGO_TEMPERATURE_MEAN
            psal = anom.ARGO_SALINITY_ANOMALY + clim.ARGO_SALINITY_MEAN

            depth = -gsw.z_from_p(anom.PRESSURE, 0)
            pressure = gsw.p_from_z(-depth, anom.latitude)
            pressure = xr.where(pressure < anom.PRESSURE, pressure, anom.PRESSURE)

            return xr.Dataset(
                {
                    "temp": temp.interp(PRESSURE=pressure)
                    .assign_coords(PRESSURE=depth)
                    .rename(PRESSURE="depth"),
                    "psal": psal.interp(PRESSURE=pressure)
                    .assign_coords(PRESSURE=depth)
                    .rename(PRESSURE="depth"),
                }
            )

        # --------------------- IPRC ---------------------
        elif product == "IPRC":
            """
            Load data from IPRC product.

            Product's directory must have the following format:
            -base directory (ex: IPRC)
               |-monthly
                   |-ArgoData_year_month.nc (ex: ArgoData_2020_01.nc)
            """

            def set_time(ds):
                file_basename = os.path.basename(
                    os.path.splitext(ds.encoding["source"])[0]
                )
                date = file_basename.split("_")
                ts = np.datetime64("%s-%s-15" % (date[1], date[2]), "ns")
                return ds.assign_coords(time=ts).expand_dims(dim="time")

            def year(f):
                return f.split("_")[1]

            fics = filtered_list(
                glob(os.path.join(directory, "**", "ArgoData*.nc")),
                year,
                ymin,
                ymax,
                filter,
            )
            data = xr.open_mfdataset(
                fics, engine="lenapyNetcdf", preprocess=set_time, **open_dataset_kwargs
            )

            return xr.Dataset({"temp": data.TEMP, "psal": data.SALT})

        # --------------------- ISHII ---------------------
        elif product == "ISHII":
            """
            Load data from ISHII product.

            Product's directory must have the following format:
            -base directory (ex: ISHII)
               |-monthly
                   |-xxx.year_month.nc (ex: sal.2022_08.nc)

            """

            def year(f):
                return f.split("_")[0].split(".")[1]

            fics = filtered_list(
                glob(os.path.join(directory, "**", "*.*.nc")), year, ymin, ymax, filter
            )

            data = xr.open_mfdataset(
                fics,
                engine="lenapyNetcdf",
                drop_variables=[
                    "VAR_10_4_201_P0_L160_GLL0",
                    "VAR_10_4_202_P0_L160_GLL0",
                ],
                **open_dataset_kwargs,
            )

            return xr.Dataset({"temp": data.temperature, "psal": data.salinity})

        # --------------------- IAP ---------------------
        elif product == "IAP":
            """
            Load data from IAP product.

            Product's directory must have the following format:
            -base directory (ex: IAP)
               |-sal
                   |-CZ16_depth_range_data_year_yyyy_month_mm.year_month.nc (ex: CZ16_1_2000m_salinity_year_2020_month_01.nc)
               |-temp
                   |-CZ16_depth_range_data_year_yyyy_month_mm.year_month.nc (ex: CZ16_1_2000m_temperature_year_2020_month_01.nc)
            """

            def set_time(ds):
                file_basename = os.path.basename(
                    os.path.splitext(ds.encoding["source"])[0]
                )
                date = file_basename.split("_")
                ts = np.datetime64("%s-%s-15" % (date[-3], date[-1]), "ns")
                return ds.assign_coords(time=ts).expand_dims(dim="time")

            def year(f):
                return f.split("_")[-3]

            fics = filtered_list(
                glob(os.path.join(directory, "**", "CZ16*.nc")),
                year,
                ymin,
                ymax,
                filter,
            )

            data = xr.open_mfdataset(
                fics, engine="lenapyNetcdf", preprocess=set_time, **open_dataset_kwargs
            )

            return xr.Dataset({"temp": data.temp, "SA": data.salinity})

        # --------------------- EN_422 --------------------
        elif product == "EN4":
            """
            Load data from EN product.

            Product's directory must have the following format:
            -base directory (ex: EN)
               |-4.2.2.corr (ex : 4.2.2.g10)
                   |-EN.4.2.2.xxx.corr.timestamp.nc (ex: EN.4.2.2.f.analysis.g10.202108.nc)
            """

            def year(f):
                return f.split(".")[-1][0:4]

            if not (corr in ["g10", "l09", "c13", "c14"]):
                sys.exit(-1)

            fics = filtered_list(
                glob(os.path.join(directory, "4.2.2.%s" % corr, "EN.4.2.2*.nc")),
                year,
                ymin,
                ymax,
                filter,
            )

            data = xr.open_mfdataset(fics, engine="lenapyNetcdf", **open_dataset_kwargs)

            return xr.Dataset({"PT": data.temperature - 273.15, "psal": data.salinity})

        # --------------------- JAMSTEC --------------------
        elif product == "JAMSTEC":
            """
            Load data from JAMSTEC product.
            Depth coordinate is derived from pressure.

            Product's directory must have the following format:
            -base directory (ex: JAMSTEC)
               |-monthly
                   |-TS_timestamp_xxx.nc (ex: TS_202105_GLB.nc)
            """

            def set_time(ds):
                file_basename = os.path.basename(
                    os.path.splitext(ds.encoding["source"])[0]
                )
                date = file_basename.split("_")[1]
                ts = np.datetime64("%s-%s-15" % (date[0:4], date[4:6]), "ns")
                return ds.assign_coords(time=ts).expand_dims(dim="time")

            def year(f):
                return f.split("_")[1][0:4]

            fics = filtered_list(
                glob(os.path.join(directory, "**", "*GLB.nc")), year, ymin, ymax, filter
            )

            data = xr.open_mfdataset(
                fics, engine="lenapyNetcdf", preprocess=set_time, **open_dataset_kwargs
            )

            depth = -gsw.z_from_p(data.PRES, 90)
            pressure = gsw.p_from_z(-depth, data.latitude)
            v0 = data.isel(PRES=0)
            v0["PRES"] = xr.zeros_like(v0.PRES)
            data2 = xr.concat([v0, data], dim="PRES")

            return xr.Dataset(
                {
                    "temp": data2.TOI.interp(PRES=pressure)
                    .assign_coords(PRES=depth)
                    .rename(PRES="depth"),
                    "psal": data2.SOI.interp(PRES=pressure)
                    .assign_coords(PRES=depth)
                    .rename(PRES="depth"),
                }
            )

        # --------------------- ECCO --------------------
        elif product == "ECCO":
            """
            Load data from ECCO product.

            Product's directory must have the following format:
            -base directory (ex: ECCO)
               |-SALT
                   |-SALT_year_month.nc (ex: SALT_2016_12.nc)
               |-THETA
                   |-THETA_year_month.nc (ex: THETA_2016_12.nc)
            """

            def preproc(ds):
                ds = (
                    ds.set_index(i="longitude", j="latitude", k="Z")
                    .rename(i="longitude", j="latitude", k="depth")
                    .drop("timestep")
                )
                return ds.assign_coords(depth=-ds.depth)

            def year(f):
                return f.split("_")[1]

            fics = filtered_list(
                glob(os.path.join(directory, "**", "*.nc")), year, ymin, ymax, filter
            )

            data = xr.open_mfdataset(fics, preprocess=preproc, **open_dataset_kwargs)
            data = data.where(data != 0.0)

            return xr.Dataset({"PT": data.THETA, "psal": data.SALT})

        # --------------------- NOC OI --------------------
        elif product == "NOC_OI":
            """
            Load data from NOC ARGO OI product.
            Depth coordinate is derived from pressure.
            """

            def preproc(ds):
                ds["time"] = ds.indexes["time"].to_datetimeindex()
                return ds

            fics = glob(os.path.join(directory, "*.nc"))

            data = xr.open_mfdataset(
                fics, engine="lenapyNetcdf", preprocess=preproc, **open_dataset_kwargs
            ).sel(time=slice(str(ymin), str(ymax)))

            depth = -gsw.z_from_p(data.pressure, 0)
            pressure = gsw.p_from_z(-depth, data.latitude)
            pressure = xr.where(pressure < data.pressure, pressure, data.pressure)

            return xr.Dataset(
                {
                    "temp": data.temperature.interp(pressure=pressure)
                    .assign_coords(pressure=depth)
                    .rename(pressure="depth"),
                    "psal": data.practical_salinity.interp(pressure=pressure)
                    .assign_coords(pressure=depth)
                    .rename(pressure="depth"),
                }
            )

        # --------------------- SODA --------------------
        elif product == "SODA":
            """
            Load data from SODA product.
            """

            def preproc(ds):
                return ds.rename(
                    dict(xt_ocean="longitude", yt_ocean="latitude", st_ocean="depth")
                )

            def year(f):
                return f.split("_")[-1]

            fics = filtered_list(
                glob(os.path.join(directory, "*.nc")), year, ymin, ymax, filter
            )
            print(fics)
            data = xr.open_mfdataset(fics, preprocess=preproc, **open_dataset_kwargs)

            return xr.Dataset({"PT": data.temp, "psal": data.salt})
