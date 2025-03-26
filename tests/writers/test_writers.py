import xarray as xr


def test_gravi_writers(lenapy_paths):
    # TODO : asserts
    # TODO : test to_gfc parameters
    ds_path = lenapy_paths.data / "COSTG_n12_2002_2022.nc"
    ds = xr.open_dataset(ds_path)
    ds.isel(time=0).lnharmo.to_gfc("tmp/test.gfc")
