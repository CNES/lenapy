import xarray as xr


def test_climato_coeffs(lenapy_paths):
    moheacan = xr.open_dataset(lenapy_paths.data / "ohc.nc")
    clim = moheacan.gohc.lntime.Coeffs_climato()
    ref_coeffs = [
        "cosAnnual",
        "cosSemiAnnual",
        "order_0",
        "order_1",
        "sinAnnual",
        "sinSemiAnnual",
    ]
    assert (
        sorted(clim.coeff_names) == ref_coeffs
    ), "Coeffs_climato outputs do not correspond to the ref."
