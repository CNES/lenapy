import numpy as np
import xarray as xr


def lanczos(cutoff, order):
    """
    Lanczos Filter
    Implementation of a filter whose spectral response is a door with a temporal width specified by "cutoff",
    convoluted with anothe door narrower by a factor "order". Typicaly, a = 2 or 3. The filter is
    truncated at +/- order * cutoff / 2.
    With a higher order, the filter becomes close to a perfect sinc filter, but computation takes longer

    Parameters
    ----------
    cutoff : integer
        width of the temporal window (in samples)
    order : integer
        order of the filter

    Returns
    -------
    filter : DataArray
        Lanczos kernel

    """
    c = cutoff / 2.0
    x = np.arange(-order * c, order * c + 1, 1)
    y = np.sinc(x / c) * np.sinc(x / c / order)
    y = y / np.sum(y)
    return xr.DataArray(y, dims=("x",), coords={"x": x})


def moving_average(cutoff):
    """
    Moving average filter
    Implementation of a moving average filter, averaging values over "npoints" samples centered on the current point

    Parameters
    ----------
    cutoff : integer
        width of the temporal window (in samples)

    Returns
    -------
    filter : numpy array
        moving_average kernel

    """

    return np.ones(cutoff) / cutoff


def linear(cutoff):
    x = np.arange(-cutoff, cutoff + 1)
    y = cutoff - np.abs(x)
    y = y / np.sum(y)
    return xr.DataArray(y, dims=("x",), coords={"x": x})
