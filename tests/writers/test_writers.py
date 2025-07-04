import os

import xarray as xr


def test_gravi_writers(lenapy_paths):
    ds_path = lenapy_paths.data / "COSTG_n12_2002_2022.nc"
    ds = xr.open_dataset(ds_path).isel(time=0)
    ds.attrs["max_degree"] = 12

    output_file = "tmp/test_errors.gfc"
    ds.lnharmo.to_gfc(output_file, include_errors=True)

    # Verify the file existence
    assert os.path.isfile(output_file)
    assert os.path.getsize(output_file) > 0

    # Comparison
    ds_written = xr.open_dataset(output_file, engine="lenapyGfc")
    xr.testing.assert_allclose(ds, ds_written)


def test_gravi_writers_fast(lenapy_paths):
    ds_path = lenapy_paths.data / "COSTG_n12_2002_2022.nc"
    ds = xr.open_dataset(ds_path).isel(time=0)
    ds.attrs["max_degree"] = 12

    output_file = "tmp/test_fast.gfc"
    ds.lnharmo.to_gfc(output_file, fast_save=True, include_errors=True)

    # Verify the file existance
    assert os.path.isfile(output_file)
    assert os.path.getsize(output_file) > 0

    # Comparison
    ds_written = xr.open_dataset(output_file, engine="lenapyGfc")
    xr.testing.assert_allclose(ds, ds_written)
