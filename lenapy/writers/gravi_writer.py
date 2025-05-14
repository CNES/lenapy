"""
The gravi_writer.py provides writer functions to save Spherical Harmonics dataset in text file.
"""

import os

from lenapy.utils.harmo import assert_sh


def check_dimensions(var, var_name):
    """Check if a variable has only 'l' and 'm' dimensions or extra dims of size 1."""
    if not (set(var.dims) == {"l", "m"}) and not (
        len(var.dims) > 2 and max(var.shape[2:]) <= 1
    ):
        raise ValueError(
            f"Variable '{var_name}' has extra dimension with a size that exceeds 1."
            "\n You can reduce this extra dimension by using .isel(dim=0) on your dataset."
        )


def prepare_attributes(ds, include_errors: bool, extra_kwargs: dict):
    """Prepare dataset attributes for .gfc file header."""
    attrs = ds.attrs.copy()
    mandatory_attrs_defaults = {
        "product_type": "gravity_field",
        "modelname": "unnamed_model",
        "earth_gravity_constant": "not_set",
        "radius": "not_set",
        "max_degree": str(ds.l.max().values),
    }
    if include_errors and "eclm" in ds and "eslm" in ds:
        mandatory_attrs_defaults["errors"] = "formal"
    else:
        mandatory_attrs_defaults["errors"] = "no"
    for attr, default_value in mandatory_attrs_defaults.items():
        attrs.setdefault(attr, default_value)
    # Keep normalization coherent with .gfc standard
    if "norm" in attrs and attrs["norm"] == "4pi":
        attrs["norm"] = "fully_normalized"
    elif "norm" in attrs and attrs["norm"] == "unnorm":
        attrs["norm"] = "unnormalized"
    # Update dataset attributes with any additional attributes specified by the user
    attrs.update(extra_kwargs)
    return attrs


def dataset_to_gfc(
    ds,
    filename,
    overwrite=True,
    include_errors=False,
    fmt=" .12e",
    fast_save=False,
    **kwargs,
):
    """
    Save a Spherical Harmonics xr.Dataset to a .gfc ASCII file according to the ICGEM format specifications:
    https://icgem.gfz-potsdam.de/docs/ICGEM-Format-2023.pdf
    If the dataset contains additional dimensions beyond 'l' and 'm',
    raise error if the size of these dimensions exceed 1.
    The saving time for degree maximum of 60 is around 10 seconds. If the given l and m dimensions are continuous
    (from 0 to lmax/mmax), this saving time can be reduced under 100ms with argument fast_save=True.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to be saved.
    filename : str | os.PathLike
        The file path where to save the dataset.
    overwrite : bool, optional
        If True, overwrite the existing file. Default is True.
    include_errors : bool, optional
        If True, include error coefficients in the dataset. Default is False.
    fmt : str, optional
        The format specifier for clm, slm and errors values to save. Default is ' .12e'.
    fast_save : bool, optional
        If True, assume that 'l' and 'm' dimensions are continuous (from 0 to lmax/mmax) to reduce saving time.
        Default is False.
    **kwargs : optional
        Additional .gfc attributes to be included in the .gfc file header.

    Returns
    -------
    None

    Examples
    --------
    >>> ds = xr.open_dataset('example_file.nc')
    # Saving a basic dataset without error coefficients and without fast saving:
    >>> dataset_to_gfc(ds, 'output_file.gfc')
    # Saving a dataset with error coefficients included:
    >>> dataset_to_gfc(ds, 'output_file_with_errors.gfc', include_errors=True)
    # Using `fast_save` for a dataset with 'l' and 'm' dimensions that goes continuously from 0 to lmax/mmax:
    >>> dataset_to_gfc(ds, 'fast_save_file.gfc', fast_save=True)
    # Adding custom attributes that are not in ds.attrs to the .gfc file header:
    >>> dataset_to_gfc(ds, 'custom_attrs_file.gfc', modelname='My_Model_Name', tide_system='tide_free')
    """
    assert_sh(ds)

    # Verify dimensions of 'clm', 'slm' and errors array
    list_var = ["clm", "slm"] + (["eclm", "eslm"] if include_errors else [])
    for var in list_var:
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in dataset.")
        check_dimensions(ds[var], var)

    # reduce the dataset to 'l' and 'm' dimensions
    extra_dims = [dim for dim in ds.dims if dim not in ["l", "m"]]
    if extra_dims:
        ds = ds.isel(**{dim: 0 for dim in extra_dims})

    # Set default values for missing mandatory attributes
    attrs = prepare_attributes(ds, include_errors, kwargs)

    # ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # stop if file exists and not overwrite
    if not overwrite and os.path.isfile(filename):
        return None

    with open(filename, "w") as file:
        # -- Write header
        file.write("# File written from a xarray.Dataset with lenapy\n")
        file.write("begin_of_head =============================\n\n")
        for key, value in attrs.items():
            file.write(f"{key:<22} {value}\n")

        if include_errors and "eclm" in ds and "eslm" in ds:
            file.write(
                "\n"
                "key     L    M         C                   S                sigma C             sigma S\n"
            )
        else:
            file.write("\nkey     L    M         C                   S\n")
        file.write("end_of_head ===============================\n")

        # -- Write data section
        if fast_save:
            if ds.l.size != ds.l.max().values + 1 or ds.m.size != ds.m.max().values + 1:
                raise ValueError(
                    "Dimension l or m is not continuous (from 0 to lmax/mmax)"
                )
            clm = ds["clm"].values
            slm = ds["slm"].values
            if include_errors and "eclm" in ds and "eslm" in ds:
                eclm = ds["eclm"].values
                eslm = ds["eslm"].values
            for l in ds["l"].values:
                for m in ds["m"].values:
                    if m <= l:
                        line = (
                            f"gfc{l:>6}{m:>5} " f"{clm[l, m]:{fmt}} {slm[l, m]:{fmt}}"
                        )

                        if include_errors and "eclm" in ds and "eslm" in ds:
                            line += f" {eclm[l, m]:{fmt}} {eslm[l, m]:{fmt}}"

                        file.write(line + "\n")

        else:
            for l in ds["l"].values:
                for m in ds["m"].values:
                    if m <= l:
                        line = (
                            f"gfc{l:>6}{m:>5}"
                            f" {ds['clm'].sel(l=l, m=m).values:{fmt}}"
                            f" {ds['slm'].sel(l=l, m=m).values:{fmt}}"
                        )

                        if include_errors and "eclm" in ds and "eslm" in ds:
                            line += (
                                f" {ds['eclm'].sel(l=l, m=m).values:{fmt}}"
                                f" {ds['eslm'].sel(l=l, m=m).values:{fmt}}"
                            )

                        file.write(line + "\n")
