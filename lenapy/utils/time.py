from typing import Union

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import factorial as factorial

from lenapy.constants import *
from lenapy.utils import filters
from lenapy.utils.climato import Coeffs_climato


def is_dimension_chunked(da: Union[xr.DataArray, xr.Dataset], dim: str):
    """
    Check if a dimension of an xarray DataArray or Dataset is chunked.

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        The object to check.
    dim : str
        The name of the dimension to check.

    Returns
    -------
    bool
        True if the dimension is chunked, False if it is non-chunked.
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' does not exist in the DataArray/Dataset")

    if da.chunks is None:
        return False

    dim_index = da.get_axis_num(dim)
    chunks_dim = da.chunks[dim_index]
    return len(chunks_dim) > 1 or chunks_dim[0] != da.sizes[dim]


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
        cc = Coeffs_climato(data, cycle=True, order=1).solve()
        data0 = cc.signal(coefficients=[])
    else:
        data0 = data

    # Mirror without the climato
    pf = data0.polyfit(time_coord, q)
    v0 = xr.polyval(data0[time_coord], pf).polyfit_coefficients
    # Remove the polynomial to the initial data untreated
    v1 = data0 - v0
    v1[time_coord] = v1[time_coord].astype(np.int64)
    # Fullfil the data with the mirror effect at the beginning and at the end.
    v2 = v1.pad({time_coord: (k, k)}, mode="reflect", reflect_type="even")
    v2[time_coord] = (
        v1[time_coord]
        .pad({time_coord: (k, k)}, mode="reflect", reflect_type="odd")
        .astype("<M8[ns]")
    )

    if annual_cycle is True:
        v3 = cc.climatology(x=v2[time_coord])
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
    Nmin : int, default=0
        Minimum number of measure to compute climatology
    t_min, t_max : dates, optional
        Reference period over which the climatology is computed.
        Default is `(None, None)` (i.e., the entire time range).

    Returns
    -------
    xarray.Dataset or tuple
        The requested components as an xarray.Dataset. If `return_coeffs=True`,
        returns a tuple `(dataset, coeffs)` where `coeffs` contains the cycle and trend coefficients.
    """
    if is_dimension_chunked(data, "time"):
        raise ValueError("time dimension must be unchunked")

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


def SavitzkyGolay(da, dim="time", window=12, order=1, step=1, sigma=None):
    """
    Perform a Savitzky-Golay filter on a dataArray and return filtered derivatives up to maximal order
    """

    def convolution_matrix(M, C):
        return np.dot(np.linalg.inv(np.dot(M.T, np.dot(C, M))), np.dot(M.T, C))

    def weights(xx):
        if sigma is None:
            return np.diag(xx * 0.0 + 1)
        else:
            return np.diag(np.exp(-(xx**2) / sigma**2))

    if np.mod(window, 2) != 1:
        print("Warning, window is even, set to window-1")
    half_window = int((window - 1) / 2.0)

    # [0,1,...,order]
    o = xr.DataArray(np.int32(np.arange(order + 1)), dims="order")
    # Normalize polynomial coefficients to obtain the corresponding derivative
    norm = xr.DataArray(factorial(o) / step**o)

    # Filter impementation at the edges of the signal (between 0 and half_window, and between n-half_window and n)
    lateral_signal = []
    for i in range(half_window):
        # Signal start
        # X-axis indices
        xm = xr.DataArray(np.arange(-i, half_window + 1, 1), dims=dim)
        # Least-squares obervables matrix
        X = xm**o
        # Weight matrix
        W = weights(xm)
        # SG filter implementation
        filtrem = (
            xr.DataArray(
                convolution_matrix(X, W),
                coords={
                    "order": o,
                    dim: da[dim].isel({dim: slice(None, half_window + i + 1)}),
                },
            )
            * norm
        )
        # Signal end
        # X-axis indices
        xp = xr.DataArray(np.arange(-half_window, half_window - i, 1), dims=dim)
        # Least-squares obervables matrix
        X = xp**o
        # Weight matrix
        W = weights(xp)
        # SG filter implementation
        filtrep = (
            xr.DataArray(
                convolution_matrix(X, W),
                coords={
                    "order": o,
                    dim: da[dim].isel({dim: slice(-2 * half_window + i, None)}),
                },
            )
            * norm
        )

        # Convolution of the signal by the filter
        signalm = da.isel({dim: slice(None, half_window + i + 1)})
        xvm = da[dim].isel({dim: [i]}).values
        signalp = da.isel({dim: slice(-2 * half_window + i, None)})
        xvp = da[dim].isel({dim: [-half_window + i]}).values
        lateral_signal.append(
            (signalm * filtrem).sum(dim).expand_dims({dim: xvm}).rename("ok")
        )
        lateral_signal.append(
            (signalp * filtrep).sum(dim).expand_dims({dim: xvp}).rename("ok")
        )

    # Filter impementation for the central part of the signal (between half_window and n-half_window)
    x = xr.DataArray(np.arange(-half_window, half_window + 1, 1), dims="x_win")
    X = x**o
    # Weight matrix
    W = weights(x)
    filtre = (
        xr.DataArray(convolution_matrix(X, W), coords=dict(order=o, x_win=x)) * norm
    )

    # Fenetres de largeur 2*l+1, convoluées par le filtre (multiplication par le filtre et somme des éléments
    central_signal = (
        da.rolling({dim: 2 * half_window + 1}, center=True)
        .construct("x_win")
        .weighted(filtre)
        .sum("x_win")
        .isel({dim: slice(half_window, -half_window)})
        .rename("ok")
    )

    return xr.merge((xr.merge(lateral_signal), central_signal)).ok
