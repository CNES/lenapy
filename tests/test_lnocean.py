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
    xr.testing.assert_equal(ref_ocean, dataset)
