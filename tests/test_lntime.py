import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from tests.utilities import compare_pngs


def test_lntime_climato(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_climato.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.climato()
    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_climato = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_climato, result)


def test_lntime_filter(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_filter.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.filter(cutoff=1, order=1)
    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_filter = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_filter, result)


def test_lntime_interp_time(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_interptime.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.interp_time(other=ohc.gohc)
    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_interptime = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_interptime, result)


def test_lntime_plot(overwrite_references, lenapy_paths, tmp_path):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_plot.png"
    test_file = tmp_path / "lntime_plot.png"

    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
    ohc.gohc.lntime.plot()
    fig.savefig(test_file)
    if overwrite_references:
        fig.savefig(ref_file)
    same_pngs, message = compare_pngs(ref_file, test_file)
    assert same_pngs


def test_lntime_to_datetime(overwrite_references, lenapy_paths):
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    ref = ohc.gohc.time.values
    test = ohc.gohc.lntime.to_datetime(time_type="360_day").time.values
    assert np.array_equal(ref, test)


def test_lntime_diff_3pts(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_diff_3pts.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.diff_3pts(dim="time")

    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_diff3pts = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_diff3pts, result)


def test_lntime_diff_2pts(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_diff_2pts.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.diff_2pts(dim="time")

    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_diff2pts = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_diff2pts, result)


def test_lntime_trend(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_trend.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.trend()

    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_trend = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_trend, result)


def test_lntime_detrend(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_detrend.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.detrend()
    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_detrend = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_detrend, result)


def test_lntime_fill_time(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_filltime.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.fill_time()
    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_filltime = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_filltime, result)


def test_lntime_covariance_analysis(overwrite_references, lenapy_paths):
    ref_file = (
        lenapy_paths.ref_data / "lenapy_time" / "lntime_covariance_analysis_time.nc"
    )
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.covariance_analysis().time
    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_covtime = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_covtime, result)


def test_lntime_fillna_climato(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "lenapy_time" / "lntime_fillna_climato.nc"
    ohc = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    result = ohc.gohc.lntime.fillna_climato()
    if overwrite_references:
        result.to_netcdf(ref_file)
    ref_fillna_climato = xr.open_dataarray(ref_file)
    xr.testing.assert_equal(ref_fillna_climato, result)
