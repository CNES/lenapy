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
