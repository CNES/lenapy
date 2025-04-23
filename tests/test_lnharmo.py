import numpy as np
import pytest
import xarray as xr

from tests.utilities import lambda_source

cases = [
    (lambda h: h + 10),
    (lambda h: 10 + h),
    (lambda h: h - 10),
    (lambda h: 10 - h),
    (lambda h: h * 2),
    (lambda h: 2 * h),
    (lambda h: -h),
    (lambda h: h**2),
]


@pytest.mark.parametrize("op", cases)
def test_lenapy_gfc(lenapy_paths, op):
    """
    Parametrized regression–test for all arithmetic operators overloaded by
    the **`lnharmo`** object produced with the *lenapyGfc* engine.
    """
    gsm = xr.open_dataset(
        lenapy_paths.data / "GSM-2_2002213-2002243_GRAC_COSTG_BF01_0100.gfc",
        engine="lenapyGfc",
        no_date=False,
    )
    gsm_harmo_1 = gsm.copy(deep=True).lnharmo
    out = op(gsm_harmo_1)
    assert np.allclose(
        op(gsm.clm.values), out.clm.values
    ), f"operator {lambda_source(op)} failed"
