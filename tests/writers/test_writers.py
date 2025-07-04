import os

import pytest
import xarray as xr

from lenapy.writers.gravi_writer import dataset_to_gfc


def test_gravi_writer(lenapy_paths):
    ds_path = lenapy_paths.data / "COSTG_n12_2002_2022.nc"
    ds = xr.open_dataset(ds_path).isel(time=0)
    ds.attrs["max_degree"] = 12

    output_file = "tmp/test_errors.gfc"
    ds.lnharmo.to_gfc(output_file, include_errors=True)

    # Verify the file existence
    assert os.path.isfile(output_file)
    assert os.path.getsize(output_file) > 0

    # Test no overwriting
    ds.lnharmo.to_gfc(output_file, overwrite=False)

    # Comparison
    ds_written = xr.open_dataset(output_file, engine="lenapyGfc").isel(time=0)
    xr.testing.assert_allclose(ds, ds_written)


def test_gravi_writer_fast(lenapy_paths):
    ds_path = lenapy_paths.data / "COSTG_n12_2002_2022.nc"
    ds = xr.open_dataset(ds_path).isel(time=[0])
    ds.attrs["max_degree"] = 12

    output_file = "tmp/test_fast.gfc"
    ds.lnharmo.to_gfc(output_file, fast_save=True, include_errors=True)

    # Verify the file existance
    assert os.path.isfile(output_file)
    assert os.path.getsize(output_file) > 0

    # Comparison
    ds_written = xr.open_dataset(output_file, engine="lenapyGfc").isel(time=0)
    xr.testing.assert_allclose(ds.isel(time=0), ds_written)


def test_gravi_writer_valueerror(lenapy_paths):
    ds_path = lenapy_paths.data / "COSTG_n12_2002_2022.nc"
    ds = xr.open_dataset(ds_path)

    sub_ds = ds.isel(time=[0, 1, 2])
    with pytest.raises(ValueError):
        sub_ds.lnharmo.to_gfc("tmp/test_errors.gfc")

    sub_ds = ds.isel(time=0).sel(l=[1, 2, 3, 6])
    with pytest.raises(ValueError):
        sub_ds.lnharmo.to_gfc("tmp/test_errors.gfc", fast_save=True)

    sub_ds = ds.isel(time=0).drop_vars(["eclm"])
    with pytest.raises(ValueError):
        dataset_to_gfc(sub_ds, "tmp/test_errors.gfc", include_errors=True)
