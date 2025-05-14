import pytest
import xarray as xr

from lenapy.readers.gravi_reader import read_tn14


def test_read_tn13(lenapy_paths):
    pass


@pytest.mark.parametrize(
    "rmmean, ref_name",
    [
        (False, "tn14_rmmeanF.nc"),  # F = False
        (True, "tn14_rmmeanT.nc"),  # T = True
    ],
)
def test_read_tn14(overwrite_references, lenapy_paths, rmmean, ref_name):
    """
    Regression test for ``read_tn14`` with and without mean‑removal.

    The test is executed twice:
    * (rmmean=False) -> reference *tn14_rmmeanF.nc*
    * (rmmean=True)  -> reference *tn14_rmmeanT.nc*
    """
    ref_file = lenapy_paths.ref_data / "readers" / ref_name

    ds = read_tn14(
        lenapy_paths.data / "TN-14_C30_C20_GSFC_SLR.txt",
        rmmean=rmmean,
    )

    if overwrite_references:
        ds.to_netcdf(ref_file)

    ref_ds = xr.open_dataset(ref_file)
    xr.testing.assert_allclose(ref_ds, ds)


def test_lenapy_gfc(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "readers" / "lenapyGfc.nc"
    gsm = xr.open_dataset(
        lenapy_paths.data / "GSM-2_2002213-2002243_GRAC_COSTG_BF01_0100.gfc",
        engine="lenapyGfc",
        no_date=False,
    )
    if overwrite_references:
        gsm.to_netcdf(ref_file)
    ref_gsm = xr.open_dataset(ref_file)
    xr.testing.assert_allclose(ref_gsm, gsm)


@pytest.mark.parametrize(
    "input_file, ref_file",
    [
        ("GSM-2_2023001-2023031_GRFO_CNESG_TSVD_0500.txt", "txt_grace_l2.nc"),
        ("GSM-2_2022001-2022031_GRFO_UTCSR_BA01_0601.gz", "gz_grace_l2.nc"),
    ],
)
def test_lenapy_grace(lenapy_paths, input_file, ref_file):
    """
    Test loading GRACE Level-2 files (TXT and GZ) with the lenapyGraceL2 engine.

    Compares parsed datasets to reference NetCDF datasets to ensure consistent structure and values.
    """
    gsm_ref = xr.open_dataset(lenapy_paths.ref_data / ref_file)
    gsm = xr.open_dataset(lenapy_paths.data / input_file, engine="lenapyGraceL2")
    xr.testing.assert_allclose(gsm_ref, gsm)


def test_lenapy_netcdf(overwrite_references, lenapy_paths):
    ref_file = lenapy_paths.ref_data / "readers" / "lenapyNetcdf.nc"
    gmsl = xr.open_dataset(
        lenapy_paths.data / "MSL_wo_seasonal_signal.nc", engine="lenapyNetcdf"
    )
    if overwrite_references:
        gmsl.to_netcdf(ref_file)
    ref_gsm = xr.open_dataset(ref_file)
    xr.testing.assert_allclose(ref_gsm, gmsl)


def test_lenapy_mask(lenapy_paths):
    # TODO test class lenapy.readers.geo_reader.lenapyMask
    pass


def test_read_gfc(lenapy_paths):
    # TODO test class lenapy.readers.gravi_reader.ReadGFC
    pass


def test_read_gracel2(lenapy_paths):
    # TODO test class lenapy.readers.gravi_reader.ReadGRACEL2
    pass


def test_read_sh_loading(lenapy_paths):
    # TODO test class lenapy.readers.gravi_reader.ReadShLoading
    pass


def test_ocean_products(lenapy_paths):
    # TODO test class lenapy.readers.ocean.lenapyOceanProducts
    pass
