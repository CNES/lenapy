import numpy as np
import pytest
import xarray as xr

from tests.utilities import subsample_xr


@pytest.mark.parametrize("operation", ["mean", "sum"])
def test_mean_or_sum(air_temperature_data, operation):
    raw_values = air_temperature_data.air.values
    ref_result = getattr(np, operation)(raw_values, axis=(1, 2))

    # TODO : Possible refactoring: Extract the weight computations in mean as a utility function and use it as an input.
    #  Currently, both mean and sum share the same weight computations ?
    #  Then it will be possible to test it
    weighted_new_dataset = getattr(air_temperature_data.lngeo, operation)(
        ["latitude", "longitude"], weights=["latitude"]
    )

    dataset_result = getattr(air_temperature_data.lngeo, operation)(
        ["latitude", "longitude"]
    )
    dataarray_result = getattr(air_temperature_data.air.lngeo, operation)(
        ["latitude", "longitude"]
    )

    assert np.allclose(ref_result, dataset_result.air.values)
    assert np.allclose(ref_result, dataarray_result.values)


def test_regridder(
    overwrite_references, air_temperature_data, lenapy_paths, tmp_path, ersstv5_data
):
    ref_filename = "lngeo_regridder.nc"
    ref_file = lenapy_paths.ref_data / ref_filename
    test_file = tmp_path / ref_filename
    result = air_temperature_data.lngeo.regridder(ersstv5_data, method="bilinear")
    result.to_netcdf(test_file)
    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_nc = xr.open_dataset(ref_file)
    result_nc = xr.open_dataset(test_file)
    xr.testing.assert_equal(ref_nc, result_nc)


def test_regrid(overwrite_references, lenapy_paths, air_temperature_data):
    ref_file = lenapy_paths.ref_data / "lngeo_regrid.nc"
    ds_out = xr.Dataset(
        {
            "latitude": (["latitude"], np.arange(-89.5, 90, 1.0)),
            "longitude": (["longitude"], np.arange(-179.5, 180, 1.0)),
        }
    )
    regridder = air_temperature_data.lngeo.regridder(
        ds_out, "conservative_normed", periodic=True
    )
    result = air_temperature_data.lngeo.regrid(regridder)

    result = subsample_xr(result, 20)

    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_regrid = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_regrid, result.air)


def test_surface_cell(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lngeo_surface.nc"
    data = xr.open_dataset(lenapy_paths.data / "ecco.nc", engine="lenapyNetcdf")
    surface = data.lngeo.surface_cell()
    if overwrite_references:
        surface.to_netcdf(ref_file)
    ref_surface = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_surface, surface)


def test_distance(overwrite_references, lenapy_paths, air_temperature_data):
    ref_file = lenapy_paths.ref_data / "lngeo_distance.nc"
    lat_coords = [45]
    lon_coords = [45]
    lat = xr.DataArray(
        lat_coords,
        dims=["N_PROF"],
        name="lat",
        attrs={"standard_name": "lat", "units": "degree_north", "axis": "Y"},
    )
    lon = xr.DataArray(
        lon_coords,
        dims=["N_PROF"],
        name="lon",
        attrs={"standard_name": "lon", "units": "degree_north", "axis": "Y"},
    )
    pt_coords = xr.Dataset(
        {"latitude": lat, "longitude": lon},
        coords={"N_PROF": np.arange(len(lon_coords))},
    )
    results = air_temperature_data.lngeo.distance(pt_coords)
    if overwrite_references:
        results.to_netcdf(ref_file)
    ref_distance = xr.open_dataarray(ref_file)
    xr.testing.assert_allclose(ref_distance, results)


def test_isosurface(overwrite_references, lenapy_paths, ohc_data):
    ref_file = lenapy_paths.ref_data / "lngeo_isosurface.nc"
    depth = ohc_data.lnocean.sigma0.lngeo.isosurface(27, "depth")
    if overwrite_references:
        depth.to_netcdf(ref_file)
    ref_depth = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_depth, depth)
