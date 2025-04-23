import numpy as np
import pytest
import xarray as xr

from lenapy.utils.gravity import (
    change_love_reference_frame,
    change_reference,
    change_tide_system,
    gauss_weights,
)

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


@pytest.fixture
def base_dataset():
    clm = xr.DataArray(
        np.zeros((3, 3)), dims=["l", "m"], coords={"l": [0, 1, 2], "m": [0, 1, 2]}
    )
    slm = xr.DataArray(
        np.zeros((3, 3)), dims=["l", "m"], coords={"l": [0, 1, 2], "m": [0, 1, 2]}
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
