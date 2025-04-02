import numpy as np
import pytest
import xarray as xr

# TODO add tests about id()


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


def test_regridder(air_temperature_data, ersstv5_data):
    # TODO : asserts
    air_temperature_data.lngeo.regridder(ersstv5_data, method="bilinear")


def test_regrid(air_temperature_data):
    # TODO : asserts
    ds_out = xr.Dataset(
        {
            "latitude": (["latitude"], np.arange(-89.5, 90, 1.0)),
            "longitude": (["longitude"], np.arange(-179.5, 180, 1.0)),
        }
    )
    regridder = air_temperature_data.lngeo.regridder(
        ds_out, "conservative_normed", periodic=True
    )
    out = air_temperature_data.lngeo.regrid(regridder)


def test_surface_cell(lenapy_paths):
    # TODO : asserts
    data = xr.open_dataset(lenapy_paths.data / "ecco.nc", engine="lenapyNetcdf")
    surface = data.lngeo.surface_cell()


def test_distance(air_temperature_data):
    # TODO : asserts
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
    air_temperature_data.lngeo.distance(pt_coords)


@pytest.mark.skip(reason="data isas.nc is missing")
def test_isosurface(lenapy_paths):
    # TODO : asserts
    input_data = lenapy_paths.data / "isas.nc"
    data = xr.open_dataset(input_data, engine="lenapyNetcdf").temp
    data.isosurface(3, "depth")
