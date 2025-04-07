import xarray as xr


def subsample_xr(
    obj: xr.DataArray | xr.Dataset, factor: int
) -> xr.DataArray | xr.Dataset:
    """
    Subsample a DataArray or Dataset by taking every `factor`-th element along all dimensions.

    Parameters
    ----------
    obj : Union[xr.DataArray, xr.Dataset]
        The input xarray object (DataArray or Dataset) to subsample.
    factor : int
        The subsampling factor. For example, factor=10 will keep 1 value out of every 10
        along each dimension.

    Returns
    -------
    Union[xr.DataArray, xr.Dataset]
        A new xarray object subsampled along all its dimensions.

    Examples
    --------
    >>> subsample_xr(da, 5)  # for a DataArray
    >>> subsample_xr(ds, 2)  # for a Dataset
    """
    indexers = {dim: slice(None, None, factor) for dim in obj.dims}
    return obj.isel(**indexers)
