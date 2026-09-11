import numpy as np
import pytest
import xarray as xr

from lenapy.constants import *
from lenapy.utils.gravity import (
    change_love_reference_frame,
    change_reference,
    change_tide_system,
    gauss_weights,
)
from tests.utilities import subsample_xr

OLD_RADIUS = 6371000.0
OLD_GM = 3.986004418e14
NEW_RADIUS = 6378137.0
NEW_GM = 3.986004418e14 * 1.0001
A0 = 4.4228e-8
H0 = -0.31460
K20_DEFAULT = 0.30190


@pytest.fixture
def dataset_love():
    # Données synthétiques simples pour l=1 uniquement
    kl = xr.DataArray([0.1, 0.2, 0.3], dims="l", coords={"l": [0, 1, 2]})
    hl = xr.DataArray([0.4, 0.5, 0.6], dims="l", coords={"l": [0, 1, 2]})
    ll = xr.DataArray([0.7, 0.8, 0.9], dims="l", coords={"l": [0, 1, 2]})
    return xr.Dataset({"kl": kl, "hl": hl, "ll": ll})


@pytest.mark.parametrize(
    "old_frame,new_frame,expected",
    [
        # From CE to CM → subtract 1
        ("CE", "CM", (-0.8, -0.5, -0.2)),
        # From CM to CE → add 1
        ("CM", "CE", (1.2, 1.5, 1.8)),
        # From CE to CL
        ("CE", "CL", (-0.8, -0.3, 0.0)),
        # From CE to CH
        ("CE", "CH", (-0.5, 0.0, 0.3)),
        (
            "CE",
            "CF",
            (
                -0.5 / 3 - 2 * 0.8 / 3,  # kl
                2 * (0.5 - 0.8) / 3,  # hl
                (0.8 - 0.5) / 3,  # ll
            ),
        ),
    ],
)
def test_reference_conversion(dataset_love, old_frame, new_frame, expected):
    ds_out = change_love_reference_frame(
        dataset_love.copy(deep=True), new_frame=new_frame, old_frame=old_frame
    )
    kl1, hl1, ll1 = (
        ds_out.kl.sel(l=1).item(),
        ds_out.hl.sel(l=1).item(),
        ds_out.ll.sel(l=1).item(),
    )
    assert np.isclose(kl1, expected[0])
    assert np.isclose(hl1, expected[1])
    assert np.isclose(hl1, expected[1])


def test_reference_wrong_frame(dataset_love):
    with pytest.raises(ValueError):
        change_love_reference_frame(
            dataset_love.copy(deep=True), new_frame="AA", old_frame="CM"
        )
    with pytest.raises(ValueError):
        change_love_reference_frame(
            dataset_love.copy(deep=True), new_frame="CM", old_frame="AA"
        )


@pytest.fixture
def base_dataset():
    clm = xr.DataArray(
        np.zeros((5, 5)),
        dims=["l", "m"],
        coords={"l": [0, 1, 2, 3, 4], "m": [0, 1, 2, 3, 4]},
    )
    slm = xr.DataArray(
        np.zeros((5, 5)),
        dims=["l", "m"],
        coords={"l": [0, 1, 2, 3, 4], "m": [0, 1, 2, 3, 4]},
    )
    ds = xr.Dataset({"clm": clm, "slm": slm})
    return ds


@pytest.mark.parametrize(
    "old_tide, new_tide, expected_delta",
    [
        ("mean_tide", "zero_tide", -1 * A0 * H0),
        ("zero_tide", "mean_tide", A0 * H0),
        ("mean_tide", "tide_free", -(1 + K20_DEFAULT) * A0 * H0),
        ("tide_free", "mean_tide", (1 + K20_DEFAULT) * A0 * H0),
        ("zero_tide", "tide_free", -K20_DEFAULT * A0 * H0),
        ("tide_free", "zero_tide", K20_DEFAULT * A0 * H0),
        ("mean_tide", "mean_tide", 0),
    ],
)
def test_tide_conversion_correct(base_dataset, old_tide, new_tide, expected_delta):
    ds = base_dataset.copy(deep=True)
    ds.attrs["tide_system"] = old_tide
    ds_out = change_tide_system(ds, new_tide)
    result = ds_out.clm.sel(l=2, m=0).item()
    assert np.isclose(
        result, expected_delta
    ), f"Conversion {old_tide} → {new_tide} incorrect"
    assert ds_out.attrs["tide_system"] == new_tide


def test_tide_missing(base_dataset):
    """
    Test change_tide_system function for no provided tide_system and for 'missing' value
    """
    with pytest.raises(KeyError):
        base_dataset.lnharmo.change_tide_system("mean_tide")

    base_dataset.attrs["tide_system"] = "missing"
    with pytest.raises(ValueError):
        change_tide_system(base_dataset, "mean_tide")


@pytest.fixture
def dummy_dataset():
    l_values = np.arange(0, 4)
    clm = xr.DataArray(np.ones(4), dims=["l"], coords={"l": l_values})
    slm = xr.DataArray(np.ones(4), dims=["l"], coords={"l": l_values})
    ds = xr.Dataset({"clm": clm, "slm": slm})
    ds.attrs["radius"] = OLD_RADIUS
    ds.attrs["earth_gravity_constant"] = OLD_GM
    return ds


def test_scaling_applied_correctly(dummy_dataset):
    ds_out = change_reference(
        dummy_dataset, new_radius=NEW_RADIUS, new_earth_gravity_constant=NEW_GM
    )
    scale = (OLD_GM / NEW_GM) * (OLD_RADIUS / NEW_RADIUS) ** dummy_dataset.l
    expected = 1.0 * scale
    np.testing.assert_allclose(ds_out.clm.values, expected)
    np.testing.assert_allclose(ds_out.slm.values, expected)


def test_deep_copy_behavior(dummy_dataset):
    ds_copy = change_reference(dummy_dataset, NEW_RADIUS, NEW_GM, apply=False)
    assert not ds_copy.clm.identical(
        dummy_dataset.clm
    ), "Deep copy should return a modified clone"
    assert ds_copy is not dummy_dataset


def test_apply_in_place(dummy_dataset):
    ds_out = change_reference(dummy_dataset, NEW_RADIUS, NEW_GM, apply=True)
    assert ds_out is dummy_dataset, "Should modify in place when apply=True"
    assert not np.allclose(ds_out.clm.values, 1.0), "Values should be updated"


def test_reads_attrs_if_old_constants_not_provided(dummy_dataset):
    ds_out = change_reference(dummy_dataset, NEW_RADIUS, NEW_GM)
    assert "radius" in ds_out.attrs
    assert ds_out.attrs["radius"] == NEW_RADIUS


def test_raises_if_no_attrs_provided():
    ds = xr.Dataset(
        {
            "clm": xr.DataArray([1.0], dims=["l"], coords={"l": [0]}),
            "slm": xr.DataArray([1.0], dims=["l"], coords={"l": [0]}),
        }
    )
    with pytest.raises(KeyError):
        change_reference(ds, NEW_RADIUS, NEW_GM)


def normal_zonal_correction(lenapy_paths, base_dataset):
    ref_file = lenapy_paths.ref_data / "utils" / f"normal_zonal_correction.nc"
    ref_ds = xr.open_dataset(ref_file)

    ds_out = base_dataset.lnharmo.normal_zonal_correction(
        earth_gravity_constant=LNPY_GM_EARTH
    )

    xr.testing.assert_allclose(ds_out, ref_ds)


def test_sh_to_gravity_disturbance(lenapy_paths):
    """
    Test for converting and subsampling a dataset's gravity disturbance grid and comparing it to a reference grid.

    Parameters
    ----------
    lenapy_paths : object
        An object that provides paths to reference data and datasets.

    Raises
    ------
    AssertionError
        If the subsampled grid does not match the reference grid exactly.
    """
    ref_grid_file = lenapy_paths.ref_data / "utils" / "costg_gravi_dist.nc"
    grid_ref = xr.open_dataarray(ref_grid_file)

    costg_ds = xr.open_dataset(lenapy_paths.data / "COSTG_n12_2002_2022.nc")
    grid = costg_ds.lnharmo.to_gravity_disturbance()
    grid = subsample_xr(grid, 10)
    xr.testing.assert_allclose(grid_ref, grid)


@pytest.fixture
def grs80_normal_field():
    """SH dataset holding exactly the GRS80 normal field, up to degree 8."""
    lmax = 8
    degrees = np.arange(lmax + 1)
    zeros = np.zeros((lmax + 1, lmax + 1))
    ds = xr.Dataset(
        {"clm": (("l", "m"), zeros.copy()), "slm": (("l", "m"), zeros.copy())},
        coords={"l": degrees, "m": degrees.copy()},
    )
    ds.attrs["radius"] = LNPY_A_EARTH_GRS80
    ds.attrs["earth_gravity_constant"] = LNPY_GM_EARTH_GRS80

    # the correction removes the normal field, so the normal field is its opposite
    correction = ds.lnharmo.normal_zonal_correction(apply=False).clm.sel(m=0).values
    ds["clm"].data = ds["clm"].data.copy()
    ds.clm.loc[dict(m=0)] = -correction
    return ds


def test_normal_zonal_correction_matches_grs80(grs80_normal_field):
    """
    Check the normal field built by normal_zonal_correction against the published GRS80 zonal
    harmonics. The normalized coefficients relate to the J_l as Clm = -J_l / sqrt(2l + 1).
    """
    j_grs80 = {2: 1.08263e-3, 4: -2.37091222e-6, 6: 6.08347e-9, 8: -1.427e-11}

    assert np.isclose(grs80_normal_field.clm.sel(l=0, m=0), 1.0)
    for l, j_ref in j_grs80.items():
        j = -float(grs80_normal_field.clm.sel(l=l, m=0)) * np.sqrt(2 * l + 1)
        assert np.isclose(j, j_ref, rtol=1e-3), f"J{l} = {j}, expected {j_ref}"


def test_sh_to_gravity_disturbance_geoid_surface(grs80_normal_field):
    """
    Test the surface="geoid" option against an analytic ground truth.

    The geoid of the GRS80 normal field is, by construction, the GRS80 ellipsoid, on which the
    normal gravity is exactly the real gravity. So the gravity disturbance must vanish everywhere.
    A non-zero result means the normal zonal field was not removed before applying Bruns' formula,
    which leaks the real C20 into the undulation and displaces the surface by kilometers.
    """
    latitude = np.array([-89.5, -60.5, -45.5, -30.5, -0.5, 30.5, 45.5, 60.5, 89.5])
    longitude = np.array([0.0, 90.0, 180.0])

    grid = grs80_normal_field.lnharmo.to_gravity_disturbance(
        surface="geoid",
        ellipsoidal_earth=True,
        latitude=latitude,
        longitude=longitude,
        a_earth=LNPY_A_EARTH_GRS80,
        earth_gravity_constant=LNPY_GM_EARTH_GRS80,
    )

    # 1e-9 m.s⁻² is 1e-4 mGal, far below any meaningful gravity signal
    assert np.abs(grid).max() < 1e-9

    # the output must be labelled with the geoid radius, not the reference radius it was built from
    assert "longitude" in grid.coords["radius"].dims


def test_sh_to_gravity_disturbance_geoid_needs_ellipsoid(grs80_normal_field):
    """
    A geoid cannot be located on a spherical Earth: the low degrees of the field would be taken
    for geoid signal and place the surface kilometers away from it, so the case is refused.
    """
    with pytest.raises(ValueError):
        grs80_normal_field.lnharmo.to_gravity_disturbance(
            surface="geoid", ellipsoidal_earth=False
        )


@pytest.mark.parametrize("surface", [None, "geoid", 250.0])
@pytest.mark.parametrize("ellipsoidal_earth", [True, False])
def test_sh_to_gravity_disturbance_honours_grid(
    grs80_normal_field, surface, ellipsoidal_earth
):
    """
    Test that the requested grid is honoured whatever the surface. The radius does not always
    carry the grid by itself, being a float on a spherical Earth and latitude-only on the
    reference ellipsoid, so the grid has to reach the inner conversions another way.
    """
    if surface == "geoid" and not ellipsoidal_earth:
        pytest.skip("geoid surface requires an ellipsoidal Earth")

    grid = grs80_normal_field.lnharmo.to_gravity_disturbance(
        surface=surface, ellipsoidal_earth=ellipsoidal_earth, dlat=10, dlon=20
    )

    assert grid.sizes["latitude"] == 18
    assert grid.sizes["longitude"] == 18


def test_sh_to_potential_partial_derivative_longitude(lenapy_paths):
    """
    Test for converting and subsampling a dataset's potential partial derivative longitude grid and comparing it to
    a reference grid.

    Parameters
    ----------
    lenapy_paths : object
        An object that provides paths to reference data and datasets.

    Raises
    ------
    AssertionError
        If the subsampled grid does not match the reference grid exactly.
    """
    ref_grid_file = (
        lenapy_paths.ref_data
        / "utils"
        / "costg_potential_partial_derivative_longitude.nc"
    )
    grid_ref = xr.open_dataarray(ref_grid_file)

    costg_ds = xr.open_dataset(lenapy_paths.data / "COSTG_n12_2002_2022.nc")
    grid = costg_ds.lnharmo.to_potential_partial_derivative_longitude()
    grid = subsample_xr(grid, 10)
    xr.testing.assert_allclose(grid_ref, grid)


def test_sh_to_deflection_of_vertical(lenapy_paths):
    """
    Test for converting and subsampling a dataset's deflection of vertical grid and comparing it to a reference grid.

    Parameters
    ----------
    lenapy_paths : object
        An object that provides paths to reference data and datasets.

    Raises
    ------
    AssertionError
        If the subsampled grid does not match the reference grid exactly.
    """
    ref_grid_file = lenapy_paths.ref_data / "utils" / "costg_deflection_of_vertical.nc"
    grid_ref = xr.open_dataset(ref_grid_file)

    costg_ds = xr.open_dataset(lenapy_paths.data / "COSTG_n12_2002_2022.nc")
    grid = costg_ds.lnharmo.to_deflection_of_vertical()
    grid = subsample_xr(grid, 10)
    xr.testing.assert_allclose(grid_ref, grid)


def test_returns_dataarray():
    weights = gauss_weights(radius=100_000, lmax=60)
    assert isinstance(weights, xr.DataArray), "Output is not a xarray.DataArray"


def test_length_matches_lmax():
    lmax = 20
    weights = gauss_weights(radius=100_000, lmax=lmax)
    assert len(weights) == lmax + 1, "Length of weights array does not match lmax+1"


def test_first_weight_is_one():
    weights = gauss_weights(radius=100_000, lmax=5)
    assert weights[0].item() == pytest.approx(1.0), "First weight should be exactly 1"


def test_weights_are_non_increasing():
    weights = gauss_weights(radius=300_000, lmax=30)
    diffs = np.diff(weights)
    assert np.all(diffs <= 1e-8), "Weights should be non-increasing"
