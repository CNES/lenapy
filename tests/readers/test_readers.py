import pytest
import xarray as xr

from lenapy.readers.gravi_reader import read_tn14


def test_read_tn13(lenapy_paths):
    pass


def test_read_tn14(lenapy_paths):
    # TODO : asserts
    ds_C20_C30 = read_tn14(
        lenapy_paths.data / "TN-14_C30_C20_GSFC_SLR.txt", rmmean=False
    )
    ds_C20_C30 = read_tn14(
        lenapy_paths.data / "TN-14_C30_C20_GSFC_SLR.txt", rmmean=True
    )


def test_lenapy_gfc(lenapy_paths):

    gsm = xr.open_dataset(
        lenapy_paths.data / "GSM-2_2002213-2002243_GRAC_COSTG_BF01_0100.gfc",
        engine="lenapyGfc",
        no_date=False,
    )


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
    xr.testing.assert_equal(gsm_ref, gsm)


def test_lenapy_netcdf(lenapy_paths):
    # TODO : asserts
    gmsl = xr.open_dataset(
        lenapy_paths.data / "MSL_wo_seasonal_signal.nc", engine="lenapyNetcdf"
    )


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
