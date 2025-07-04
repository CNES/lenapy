from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr
from pre_commit.lang_base import setup_cmd

from lenapy.utils.harmo import compute_plm, l_factor_conv
from tests.utilities import subsample_xr


def test_sh_to_grid(lenapy_paths):
    """
    Test for converting and subsampling a dataset's grid and comparing it to a reference grid.

    Parameters
    ----------
    lenapy_paths : object
        An object that provides paths to reference data and datasets.

    Raises
    ------
    AssertionError
        If the subsampled grid does not match the reference grid exactly.
    """
    ref_grid_file = lenapy_paths.ref_data / "utils" / "costg_grid.nc"
    grid_ref = xr.open_dataarray(ref_grid_file)

    costg_ds = xr.open_dataset(lenapy_paths.data / "COSTG_n12_2002_2022.nc")
    grid = costg_ds.lnharmo.to_grid()
    grid = subsample_xr(grid, 10)
    xr.testing.assert_allclose(grid_ref, grid)


def test_sh_to_grid_errors(lenapy_paths):
    ref_grid_file = lenapy_paths.ref_data / "utils" / "costg_grid_errors.nc"
    grid_ref = xr.open_dataarray(ref_grid_file)

    costg_ds = xr.open_dataset(lenapy_paths.data / "COSTG_n12_2002_2022.nc")
    grid = costg_ds.lnharmo.to_grid(errors=True)
    grid = subsample_xr(grid, 10)
    xr.testing.assert_allclose(grid_ref, grid)


def test_sh_to_grid_mass_conservation(lenapy_paths):
    ref_grid_file = lenapy_paths.ref_data / "utils" / "costg_grid_mass_conservation.nc"
    grid_ref = xr.open_dataarray(ref_grid_file)

    costg_ds = xr.open_dataset(lenapy_paths.data / "COSTG_n12_2002_2022.nc")
    grid = costg_ds.lnharmo.to_grid()
    grid = subsample_xr(grid, 10)
    xr.testing.assert_allclose(grid_ref, grid)


def test_sh_to_grid_mass_conservation_error(lenapy_paths):
    costg_ds = xr.open_dataset(lenapy_paths.data / "COSTG_n12_2002_2022.nc")
    with pytest.raises(ValueError):
        costg_ds.lnharmo.to_grid(force_mass_conservation=True, ellipsoidal_earth=True)


def test_grid_to_sh(lenapy_paths):
    """
    Test for converting a dataset's grid to spherical harmonics and comparing it to a reference dataset.

    Parameters
    ----------
    lenapy_paths : object
        An object that provides paths to reference data and datasets.

    Raises
    ------
    AssertionError
        If the estimated dataset does not match the reference dataset exactly.
    """
    ref_sh_file = lenapy_paths.ref_data / "utils" / "costg_n5_back.nc"
    ds_ref = xr.open_dataset(ref_sh_file)

    grid = xr.open_dataarray(lenapy_paths.ref_data / "utils" / "costg_grid.nc")
    ds_sh = grid.lnharmo.to_sh(5)
    xr.testing.assert_allclose(ds_ref, ds_sh)


@pytest.mark.parametrize(
    "lmax, z, normalization, ref_filename",
    [
        (5, np.linspace(-1, 1, 10), "4pi", "plm_4pi.npy"),
        (5, np.linspace(-1, 1, 10), "ortho", "plm_ortho.npy"),
        (5, np.linspace(-1, 1, 10), "schmidt", "plm_schmidt.npy"),
    ],
)
def test_plm_normalization(
    overwrite_references, lenapy_paths, lmax, z, normalization, ref_filename
):
    """Test compute_plm with different normalization methods"""
    ref_file = lenapy_paths.ref_data / "utils" / ref_filename
    plm = compute_plm(lmax, z, normalization=normalization)
    if overwrite_references:
        np.save(ref_file, plm)
    ref_plm = np.load(ref_file)
    assert np.allclose(ref_plm, plm), f"Failed for normalization {normalization}"


def test_plm_invalid_normalization():
    """Test compute_plm with an invalid normalization type"""
    lmax = 5
    z = np.linspace(-1, 1, 10)
    with pytest.raises(ValueError):
        compute_plm(lmax, z, normalization="invalid")


def test_change_normalization(lenapy_paths):
    ref_ortho_file = lenapy_paths.ref_data / "utils" / "costg_n12_ortho_ntime10.nc"
    ds_ref_ortho = xr.open_dataset(ref_ortho_file)

    ref_schmidt_file = lenapy_paths.ref_data / "utils" / "costg_n12_schmidt_ntime10.nc"
    ds_ref_schmidt = xr.open_dataset(ref_schmidt_file)

    costg_ds_n10 = xr.open_dataset(lenapy_paths.data / "COSTG_n12_2002_2022.nc").isel(
        time=slice(0, 10)
    )

    ortho = costg_ds_n10.lnharmo.change_normalization("4pi", "ortho")
    schmidt = costg_ds_n10.lnharmo.change_normalization("4pi", "schmidt")
    ortho_to_schmidt = ortho.lnharmo.change_normalization("ortho", "schmidt")
    schmidt_to_ortho = schmidt.lnharmo.change_normalization("schmidt", "ortho")
    ortho_to_4pi = ortho.lnharmo.change_normalization("ortho", "4pi")
    schmidt_to_4pi = schmidt.lnharmo.change_normalization("schmidt", "4pi")
    xr.testing.assert_allclose(ortho, ds_ref_ortho)
    xr.testing.assert_allclose(schmidt, ds_ref_schmidt)
    xr.testing.assert_allclose(schmidt_to_ortho, ds_ref_ortho)
    xr.testing.assert_allclose(ortho_to_schmidt, ds_ref_schmidt)
    xr.testing.assert_allclose(ortho_to_4pi, costg_ds_n10)
    xr.testing.assert_allclose(schmidt_to_4pi, costg_ds_n10)


@pytest.mark.parametrize(
    "l, unit",
    [
        (np.array([2, 3, 4]), "mewh"),
        (np.array([1, 2]), "mmgeoid"),
        (np.array([0, 1, 2]), "microGal"),
        (np.array([3, 4]), "norm"),
        (np.array([1, 2]), "pascal"),
        (np.array([1, 2]), "potential"),
        (np.array([1, 2]), "mvcu"),
        (np.array([1, 2]), "mecu"),
        (np.array([1, 2]), "int_radial_mag"),
        (np.array([1, 2]), "ext_radial_mag"),
    ],
)
def test_l_factor_conv(lenapy_paths, l, unit):
    """
    Test l_factor_conv function for different units and options
    """
    ref_file = lenapy_paths.ref_data / "utils" / f"l_factor_conv_{unit}.npy"
    ref_l_factor = np.load(ref_file)

    ds_love = None
    if unit == "mecu":
        ds_love = xr.Dataset(
            {"kl": ("l", [0, 0.3, 0.2, 0.1]), "hl": ("l", [0, 1, 1, 1])},
            coords={"l": np.arange(1, 5)},
        )

    latitude = np.linspace(43, 60, len(l))
    geocentric_colat = xr.DataArray(
        np.arctan2(
            np.cos(np.deg2rad(latitude)),
            (1 - 1 / 300) ** 2 * np.sin(np.deg2rad(latitude)),
        ),
        dims=["latitude"],
        coords={"latitude": latitude},
    )

    l_factor, cst = l_factor_conv(
        l=l,
        unit=unit,
        include_elastic=True,
        ds_love=ds_love,
        ellipsoidal_earth=True,
        geocentric_colat=geocentric_colat,
    )

    assert np.allclose(ref_l_factor, l_factor), f"Failed for scale factor {unit}"
    assert "gm_earth" in cst, "Missing 'gm_earth' in constants"
    assert "a_earth" in cst, "Missing 'a_earth' in constants"


def test_l_factor_conv_invalid_unit():
    """
    Test l_factor_conv function for invalid unit input
    """
    with pytest.raises(ValueError):
        l_factor_conv(l=(np.array([1, 2])), unit="invalid_unit")


def test_ellipsoidal_earth_missing_colatitude():
    with pytest.raises(
        ValueError,
        match="For ellipsoidal Earth, you need to set the parameter 'geocentric_colat'",
    ):
        l_factor_conv(
            l=np.array([1, 2]),
            unit="mewh",
            ellipsoidal_earth=True,
            geocentric_colat=None,
        )
