from lenapy.readers.gravi_reader import read_tn14


def test_read_tn14(lenapy_paths):
    # TODO : asserts
    ds_C20_C30 = read_tn14(
        lenapy_paths.data / "TN-14_C30_C20_GSFC_SLR.txt", rmmean=False
    )
    ds_C20_C30 = read_tn14(
        lenapy_paths.data / "TN-14_C30_C20_GSFC_SLR.txt", rmmean=True
    )
