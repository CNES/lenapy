import cftime
import dask.array as da
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

from lenapy.constants import *
from lenapy.utils import filters
from lenapy.utils.climato import Coeffs_climato


def filter(
    data, filter_name="lanczos", time_coord="time", annual_cycle=False, q=3, **kwargs
):
    """
    Apply a filtering on 'data' with a certain filter specified by 'filter_name'.
    Filter parameters can be given in kwargs. Continue the data with a mirror replication at the beginning and
    the end to reduce edge effects. This mirror is applied after removing a polynomial of order q fitted on the data.

    Parameters
    ----------
    data : xr.DataArray
        Dataset with 'time' dimension to filter.
    filter_name : str, optional
        Name of the filtering function. Default is 'lanczos".
    time_coord : str, optional
        Dimension name along which to apply the filter. Default is 'time'.
    annual_cycle : bool, optional
        Remove the annual cycle before filtering and add it back at the end. Default is False.
    q : int, optional
        Order of the polynomial for the mirror replication (deals with edge-effects). Default is 3.
    **kwargs :
        Parameters for the given filtering function.

    Returns
    -------
    data_filtering : xr.DataArray
        Filtered DataArray.
    """
    if time_coord not in data.coords:
        raise AssertionError("The time coordinates does not exist")
    try:
        f = getattr(filters, filter_name)
    except:
        f = filter_name
    # Kernel of convolution
    data_noyau = f(**kwargs)
    k = len(data_noyau)

    noyau = xr.DataArray(
        data_noyau, dims=["time_win"], coords={"time_win": np.arange(k)}
    )

    if annual_cycle is True:
        # remove the climato
        data0, coeffs = climato(data, mean=False, trend=False, return_coeffs=True)
    else:
        data0 = data

    # Mirror without the climato
    pf = data0.polyfit(time_coord, q)
    v0 = xr.polyval(data0[time_coord], pf).polyfit_coefficients
    # Remove the polynomial to the initial data untreated
    v1 = data0 - v0
    v1[time_coord] = v1[time_coord].astype("float")
    # Fullfil the data with the mirror effect at the beginning and at the end.
    v2 = v1.pad({time_coord: (k, k)}, mode="reflect", reflect_type="even")
    v2[time_coord] = v1[time_coord].pad(
        {time_coord: (k, k)}, mode="reflect", reflect_type="odd"
    )

    if annual_cycle is True:
        v3 = generate_climato(v2[time_coord], coeffs, mean=True, trend=True, cycle=True)
    else:
        v3 = 0.0

    # Convolution with the kernel
    v4 = (
        (
            (v3 + v2)
            .rolling({time_coord: k}, center=True)
            .construct({time_coord: "time_win"})
        )
        .weighted(noyau)
        .mean("time_win")
        .isel({time_coord: slice(k, -k)})
    )
    v4[time_coord] = data[time_coord]

    # Add the polynomial to the filtered data
    data_filtered = v0 + v4

    # Get attributes from the input data
    data_filtered.attrs = data.attrs
    return data_filtered


def climato(
    data,
    signal=True,
    mean=True,
    trend=True,
    cycle=False,
    Nmin=0,
    t_min=None,
    t_max=None,
):
    """
    Annual, Semi-Annual Cycle and Trend Analysis.

    Decomposes the input data into:
    - An annual cycle
    - A semi-annual cycle
    - A trend
    - A mean
    - A residual signal

    Returns the desired combination of these components based on the selected arguments
    (`signal`, `mean`, `trend`, `cycle`).

    If `return_coeffs=True`, also returns the coefficients of the cycles and the trend.

    Parameters
    ----------
    signal : bool, optional
        If True (default), returns the residual signal after removing the climatology
        (annual and semi-annual cycles), the trend, and the mean.
    mean : bool, optional
        If True (default), returns the mean value of the input data.
    trend : bool, optional
        If True (default), returns the trend (in day⁻¹).
    cycle : bool, optional
        If True (default is False), returns the annual and semi-annual cycles.
    return_coeffs : bool, optional
        If True (default is False), also returns the coefficients of the cycles and the
        linear trend.
    time_period : slice, optional
        Reference period over which the climatology is computed.
        Default is `slice(None, None)` (i.e., the entire time range).

    Returns
    -------
    xarray.Dataset or tuple
        The requested components as an xarray.Dataset. If `return_coeffs=True`,
        returns a tuple `(dataset, coeffs)` where `coeffs` contains the cycle and trend coefficients.
    """

    a = Coeffs_climato(xr.Dataset(dict(measure=data)), Nmin=Nmin)
    res = a.solve("measure", t_min=t_min, t_max=t_max)
    ret = []
    if mean:
        ret.append("order_0")
    if trend:
        ret.append("order_1")
    if cycle:
        ret.extend(["cosAnnual", "sinAnnual", "cosSemiAnnual", "sinSemiAnnual"])

    if signal:
        return res.signal(coefficients=ret)
    else:
        return res.climatology(coefficients=ret)

    """
    use_dask = True if isinstance(data.data, da.Array) else False
    if use_dask:
        data = data.chunk(time=-1)

    if not 'time' in data.coords: raise AssertionError('The time coordinates does not exist')
    
    # Reference temporelle = milieu de la période
    tmin=data.time.sel(time=time_period).min().values
    tmax=data.time.sel(time=time_period).max().values
    tref=tmin+(tmax-tmin)/2.
    
    # Construction de la matrice des mesures
    if isinstance(tref, np.datetime64):
        one_day = pd.to_timedelta("1D").asm8
        t1 = t1=(data.time-tref)/one_day
    elif isinstance(tref, 
                        (cftime.Datetime360Day,
                         cftime.DatetimeNoLeap,
                         cftime.DatetimeAllLeap,
                         cftime.DatetimeGregorian,
                         cftime.DatetimeProlepticGregorian,
                         cftime.DatetimeJulian,)
                    ):
        t1 =  xr.DataArray([(date - tref).days for date in data.time.values], 
                           coords=dict(time=data.time),
                           dims=['time'])
    
    omega=2*np.pi/LNPY_DAYS_YEAR
    X=xr.concat((t1**0,t1,np.cos(omega*t1),np.sin(omega*t1),np.cos(2*omega*t1),np.sin(2*omega*t1)),
                dim=pd.Index(['mean','trend','cosAnnual','sinAnnual','cosSemiAnnual','sinSemiAnnual'], name="coeffs"))
    

    # Vecteur temps a utiliser pour calcul de la climato
    time_vector=data.time.sel(time=time_period)
    
    # Détermination des coefficients par résolution des moindres carrés
    time_vector_in = time_vector.values
    X_in = X.values

    def solve_least_square(data_in):
    """
    # For a given 1d time series, returns the coefficients of the fitted climatology
    """
        Y_in_nona = data_in[~np.isnan(data_in)]
        # If less than 6 non-na elements, climato is not computable
        if len(Y_in_nona) <= 6:
            return np.full(X_in.shape[0], np.nan)
        X_in_nona = X_in[:,~np.isnan(data_in)]
        (coeffs,residus,rank,eig)=np.linalg.lstsq(X_in_nona.T,Y_in_nona,rcond=None)
        return coeffs
    
    # Application de ufunc
    coeffs = xr.apply_ufunc(
        solve_least_square, 
        data, 
        input_core_dims=[['time']],
        output_core_dims=[['coeffs']],
        exclude_dims=set(('time',)),
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float],
        dask_gufunc_kwargs={'output_sizes': {'coeffs': X_in.shape[0]}}
    )
    
    coeffs = coeffs.assign_coords(coeffs=X.coeffs.values)
    
    # Calcul des résidus
    data_climato = coeffs*X
    residus = (data-data_climato.sum('coeffs')).assign_coords(coeffs='residu').expand_dims('coeffs')

    # Toutes les composantes de la climatologie
    results = xr.concat((residus,data_climato),dim='coeffs')    

    # Sélection des composantes de la climato à retourner
    composants = np.where([signal,mean,trend,cycle,cycle,cycle,cycle])[0]
    
    results_out = results.isel(coeffs=composants).sum('coeffs',skipna=fillna)

    # Récupérer les attributs des données d'entrées
    results_out.attrs = data.attrs
    
    results_out = results_out.rename(data.name)
    if return_coeffs:
        return results_out, coeffs.assign_attrs(time_ref=tref)
    else:
        return results_out
    
    """


def generate_climato(time, coeffs, mean=True, trend=False, cycle=True):
    tref = coeffs.attrs["time_ref"]
    t1 = (time - tref) / pd.to_timedelta("1D").asm8
    omega = 2 * np.pi / LNPY_DAYS_YEAR
    X = xr.concat(
        (
            t1**0,
            t1,
            np.cos(omega * t1),
            np.sin(omega * t1),
            np.cos(2 * omega * t1),
            np.sin(2 * omega * t1),
        ),
        dim="coeffs",
    ).chunk(time=-1)

    # Sélection des composantes de la climato à retourner
    composants = np.where([mean, trend, cycle, cycle, cycle, cycle])[0]

    return (coeffs * X).isel(coeffs=composants).sum("coeffs")


def trend(data, time_unit="1s"):
    return data.polyfit(dim="time", deg=1).polyfit_coefficients[0] * pd.to_timedelta(
        time_unit
    ).asm8.astype("int")


def detrend(data):
    return (
        data
        - xr.polyval(data.time, data.polyfit(dim="time", deg=1)).polyfit_coefficients
    )


def interp_time(data, other, **kwargs):
    if "time" not in data.coords:
        raise AssertionError("The time coordinate does not exist")

    return data.interp(time=other.time, **kwargs)


def to_datetime(data, time_type, format=None):
    if "time" not in data.coords:
        raise AssertionError("The time coordinate does not exist")
    if data["time"].dtype == "<M8[ns]":
        return data

    if time_type == "frac_year":
        data["time"] = [
            pd.to_datetime(f"{int(np.floor(i))}")
            + pd.to_timedelta(float((i - np.floor(i)) * LNPY_DAYS_YEAR), unit="D")
            for i in data.time
        ]
    elif time_type == "360_day":
        data.time.attrs["calendar"] = "360_day"
        data = xr.decode_cf(data).convert_calendar("standard", align_on="year")
    elif time_type == "cftime":
        data["time"] = data.indexes["time"].to_datetimeindex()
    elif time_type == "custom":
        data["time"] = [pd.to_datetime(i, format=format) for i in data.time]
    elif time_type == "gregorian":
        data["time"] = netCDF4.num2date(data.time, data.time.Units, data.time.calendar)
    else:
        raise ValueError(
            f"Format {time_type} not yet considered, please convert manually to datatime"
        )

    return data


def diff_3pts(data, dim, time_unit="1s"):
    y = (
        data.where(~data.isnull())
        .rolling({dim: 3}, center=True, min_periods=3)
        .construct("win")
    )
    x = (
        data[dim]
        .astype("float")
        .where(~data.isnull())
        .rolling({dim: 3}, center=True, min_periods=3)
        .construct("win")
    )

    res = ((x * y).mean("win") - x.mean("win") * y.mean("win")) / (
        (x**2).mean("win") - (x.mean("win")) ** 2
    )
    return res * (pd.to_timedelta(time_unit).asm8.astype("float"))


def diff_2pts(data, dim, interp_na=True, time_unit="1s", **kwargs):
    if interp_na:
        d = data.interpolate_na(dim=dim, **kwargs)
    else:
        d = data
    dy = d.diff(dim)
    dt = d[dim].diff(dim)
    x = d[dim] - dt / 2.0
    res = dy * (pd.to_timedelta(time_unit).asm8 / dt)

    return res.assign_coords({dim: x})


def fill_time(data):
    # Complète les trous de données dans une série temporelle en respectant approximativement l'échantillonnage.

    if "time" not in data.coords:
        raise AssertionError("The time coordinates does not exist")

    dt = data.time.diff("time")
    # Look for the most regular temporal sampling step
    tau0 = dt.median()
    tau1 = (dt[np.where(np.abs((dt - tau0) / tau0) < 0.2)]).mean()

    # Go through the time dimension and add new index in gaps.
    nt = data.time[0].values
    for k in range(len(data.time) - 1):
        for i in np.arange(1, np.round(dt[k] / tau1)):
            nt = np.append(nt, data.time[k].values + i * tau1)
        nt = np.append(nt, data.time[k + 1].values)

    # Create the new temporal index
    newtime = xr.DataArray(nt, dims="time", coords={"time": nt})

    # Return the interpolated data
    return newtime


def JJ_to_date(jj):
    """
    Turns a date in format 'Jours Julien CNES' into a standard datetime64 format
    """
    if jj is None:
        return jj
    dt = pd.Timedelta(jj, "D").asm8
    t0 = np.datetime64("1950-01-01", "ns")
    return t0 + dt


def fillna_climato(data):
    #
    val = climato(data, signal=False, mean=True, trend=True, cycle=True)
    return xr.where(data.isnull(), val, data)
