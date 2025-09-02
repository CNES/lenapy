import xarray as xr

from tests.utilities import result_to_dataset, subsample_xr


def test_attributes(overwrite_references, lenapy_paths, ohc_data):
    ref_file = lenapy_paths.ref_data / "lnocean_attributes.nc"
    dataset = result_to_dataset(ohc_data.lnocean)
    dataset = subsample_xr(dataset, 10)
    if overwrite_references:
        dataset.to_netcdf(
            ref_file,
            encoding={var: {"zlib": True, "complevel": 8} for var in dataset.data_vars},
        )
    ref_ocean = xr.open_dataset(ref_file)
    xr.testing.assert_allclose(ref_ocean, dataset)


def test_above(overwrite_references, lenapy_paths, ohc_data):
    ref_file = lenapy_paths.ref_data / "lnocean_above.nc"
    data = result_to_dataset(ohc_data.lnocean)

    mld = data.lnocean.mld_sigma0
    res_above = data.lnocean.heat.lnocean.above(mld).compute()
    if overwrite_references:
        res_above.to_netcdf(
            ref_file,
        )
    ref_above = xr.open_dataarray(ref_file)
    xr.testing.assert_allclose(res_above, ref_above)
