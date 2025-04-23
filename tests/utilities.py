import inspect
import textwrap
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image


def lambda_source(func):
    """Return the one‑line source of a lambda (trimmed)."""
    try:
        src = inspect.getsource(func)
        src = textwrap.dedent(src).strip()
        return src.replace("\n", " ")
    except OSError:
        # Fallback if source not found (e.g. in interactive)
        return "<lambda source unavailable>"


def compare_pngs(
    path1: str | Path, path2: str | Path, tol: int = 0
) -> tuple[bool, str]:
    """
    Compare two PNG images pixel by pixel, with an optional tolerance.

    Parameters
    ----------
    path1: str or Path
        Path to the first PNG image.
    path2: str or Path
        Path to the second PNG image.
    tol: int, optional
        Tolerance level (0–255) for pixel differences. Default is 0, which means exact match.

    Returns
    -------
    bool
        True if the images match within the given tolerance, False otherwise.
    str
        An explanation message if the comparison fails; empty string if it succeeds.

    Examples
    --------
    >>> compare_pngs("plot1.png", "plot2.png", tol=5)
    (True, '')
    """
    img1 = np.asarray(Image.open(path1).convert("RGB"))
    img2 = np.asarray(Image.open(path2).convert("RGB"))

    if img1.shape != img2.shape:
        return False, "Shape mismatch"

    diff = np.abs(img1 - img2)
    if np.max(diff) > tol:
        return False, f"Images differ by more than {tol} levels"

    return True, ""


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
