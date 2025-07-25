"""
The **gravi_reader** module provides functions to load time-variable gravity field data from different products and format
the data with unified definitions for variables and coordinates.

Standardized coordinates names:
  * l, m, time

Standardized variables names:
  * clm, slm, begin_time, end_time, exact_time (and eclm, eslm if available)

Supported Formats:
  * GRACE Level-2 products from SDS centers (use `engine='lenapyGraceL2'` in
                                             `xr.open_dataset()` or `xr.mfopen_dataset()`)
  * Gravity field products organized as .gfc files (use 'lenapyGfc'` in
                                                    `xr.open_dataset()` or `xr.mfopen_dataset()`)

Dask is implicitly used when using these interface methods.

Examples
--------
>>> import os
>>> import xarray as xr
# Load GRACE Level-2 data
>>> files_csr = [os.path.join(csr_data_dir, f) for f in os.listdir(csr_data_dir)]
>>> ds_csr = xr.open_mfdataset(files_csr, engine='lenapyGraceL2', combine_attrs="drop_conflicts")
# Load gravity field data from .gfc files
>>> files_graz = [os.path.join(graz_data_dir, f) for f in os.listdir(graz_data_dir)]
>>> ds_graz = xr.open_mfdataset(files_graz, engine='lenapyGfc', combine_attrs="drop_conflicts")
"""

import datetime
import gzip
import os
import re
import tarfile
import zipfile
from typing import IO

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from xarray.backends import BackendEntrypoint

from lenapy.constants import *
from lenapy.utils.harmo import mid_month_grace_estimate


def read_tn14(filename: str | os.PathLike, rmmean: bool = False) -> xr.Dataset:
    """
    Read TN14 data to produce a dataset with C20 and C30 information.
    Handles dates in the same way as others GRACE products.
    TN14 data can be downloaded from https://podaac.jpl.nasa.gov/gravity/grace-documentation.

    Parameters
    ----------
    filename : str | os.PathLike[Any]
        Path to the TN14 file.
    rmmean : bool, optional
        If True, use data without mean values. Default is False (use data with mean values).

    Returns
    -------
    ds : xr.Dataset
        Dataset with C20 and C30 information.
    """
    # Based on TN14 header (file from 13 Jul 2023)
    infos_tn14 = {
        "modelname": "TN-14",
        "earth_gravity_constant": 0.3986004415e15,
        "radius": 6378136.3,
        "norm": "fully_normalized",
        "tide_system": "zero_tide",
    }

    with open(filename, "r") as file:
        line = True
        # goes while up to end of header or end of file (corresponding to line "Product:\n")
        while line:
            line = file.readline()
            # test to break when end of header
            if "product:" in line.lower():
                break

        # Read file according to header columns (file from 13 Jul 2023)
        data = np.genfromtxt(
            file,
            dtype=[
                ("begin_MJD", float),
                ("begin_date", float),
                ("C20", float),
                ("C20_rmmean", float),
                ("eC20", float),
                ("C30", float),
                ("C30_rmmean", float),
                ("eC30", float),
                ("end_MJD", float),
                ("end_date", float),
            ],
        )

    clm, slm = np.zeros((2, 1, data.shape[0])), np.zeros((2, 1, data.shape[0]))
    eclm, eslm = np.zeros((2, 1, data.shape[0])), np.zeros((2, 1, data.shape[0]))

    # choose between data with mean value or without
    if not rmmean:
        clm[0, 0] = data["C20"]
        clm[1, 0] = data["C30"]
    else:
        clm[0, 0] = data["C20_rmmean"] * 1e-10
        clm[1, 0] = data["C30_rmmean"] * 1e-10

    eclm[0, 0] = data["eC20"] * 1e-10
    eclm[1, 0] = data["eC30"] * 1e-10

    # date converted to datetime using MJD information
    mjd_origin = datetime.datetime(1858, 11, 17)

    begin_time = [
        mjd_origin + datetime.timedelta(days=beg) for beg in data["begin_MJD"]
    ]
    end_time = [mjd_origin + datetime.timedelta(days=end) for end in data["end_MJD"]]

    exact_time = [begin + (end - begin) / 2 for begin, end in zip(begin_time, end_time)]

    # compute middle of the month for GRACE products
    mid_month = [
        mid_month_grace_estimate(begin, end) for begin, end in zip(begin_time, end_time)
    ]

    ds = xr.Dataset(
        {
            "clm": (["l", "m", "time"], clm),
            "slm": (["l", "m", "time"], slm),
            "eclm": (["l", "m", "time"], eclm),
            "eslm": (["l", "m", "time"], eslm),
        },
        coords={"l": np.array([2, 3]), "m": np.array([0]), "time": mid_month},
        attrs=infos_tn14,
    )

    # -- Add various time information in dataset
    ds["begin_time"] = xr.DataArray(begin_time, dims=["time"])
    ds["end_time"] = xr.DataArray(end_time, dims=["time"])
    ds["exact_time"] = xr.DataArray(exact_time, dims=["time"])

    return ds


def read_tn13(filename: str | os.PathLike) -> xr.Dataset:
    """
    Read TN13 data to produce a dataset with C10, C11 and S11 information.
    Handles dates in the same way as other GRACE products.
    TN13 data can be downloaded from https://podaac.jpl.nasa.gov/gravity/grace-documentation.

    Parameters
    ----------
    filename : str | os.PathLike[Any]
        Path to the TN13 file.

    Returns
    -------
    ds : xr.Dataset
        Dataset with C10, C11, and S11 information.
    """
    infos_tn13 = {"modelname": "TN-13", "norm": "fully_normalized"}

    with open(filename, "r") as file:
        line = True
        # goes while up to end of header
        while line:
            line = file.readline()

            # test to break when end of header
            if "end of header" in line.lower():
                break
            if "GRCOF2   " in line:
                raise ValueError("No 'end_of_head' line in file ", filename)

        # Read file according to header columns (file from 13 Jul 2023)
        data = np.genfromtxt(
            file,
            dtype=[
                ("flag", str),
                ("l", int),
                ("m", int),
                ("Clm", float),
                ("Slm", float),
                ("eClm", float),
                ("eSlm", float),
                ("begin_date", float),
                ("end_date", float),
            ],
        )

    clm, slm = np.zeros((1, 2, data.shape[0] // 2)), np.zeros(
        (1, 2, data.shape[0] // 2)
    )
    eclm, eslm = np.zeros((1, 2, data.shape[0] // 2)), np.zeros(
        (1, 2, data.shape[0] // 2)
    )

    clm[0, 0] = data["Clm"][np.where(data["m"] == 0)]
    clm[0, 1] = data["Clm"][np.where(data["m"] == 1)]
    slm[0, 1] = data["Slm"][np.where(data["m"] == 1)]

    eclm[0, 0] = data["eClm"][np.where(data["m"] == 0)]
    eclm[0, 1] = data["eClm"][np.where(data["m"] == 1)]
    eslm[0, 1] = data["eSlm"][np.where(data["m"] == 1)]

    # date converted to datetime from float YYYYMMDD
    begin_time = [
        datetime.datetime(
            int(beg // 10000), int(beg // 100 - (beg // 10000) * 100), int(beg % 100)
        )
        for beg in data["begin_date"][::2]
    ]
    end_time = [
        datetime.datetime(
            int(end // 10000), int(end // 100 - (end // 10000) * 100), int(end % 100)
        )
        for end in data["end_date"][::2]
    ]

    exact_time = [begin + (end - begin) / 2 for begin, end in zip(begin_time, end_time)]

    # compute middle of the month for GRACE products
    mid_month = [
        mid_month_grace_estimate(begin, end) for begin, end in zip(begin_time, end_time)
    ]

    ds = xr.Dataset(
        {
            "clm": (["l", "m", "time"], clm),
            "slm": (["l", "m", "time"], slm),
            "eclm": (["l", "m", "time"], eclm),
            "eslm": (["l", "m", "time"], eslm),
        },
        coords={"l": np.array([1]), "m": np.array([0, 1]), "time": mid_month},
        attrs=infos_tn13,
    )

    # -- Add various time information in dataset
    ds["begin_time"] = xr.DataArray(begin_time, dims=["time"])
    ds["end_time"] = xr.DataArray(end_time, dims=["time"])
    ds["exact_time"] = xr.DataArray(exact_time, dims=["time"])

    return ds


class ReadGFC(BackendEntrypoint):
    open_dataset_parameters = [
        "filename_or_obj",
        "drop_variables",
        "no_date",
        "date_regex",
        "date_format",
    ]

    @staticmethod
    def _parse_header(file_io: IO, ext: str) -> tuple[dict, str]:
        """
        Parse the header of a .gfc file and extract relevant metadata.

        Parameters
        ----------
        file_io : IO
            File object.
        ext : str
            File extension.

        Returns
        -------
        header : dict
            Parsed header metadata.
        legend_line : str
            Last header line before 'end_of_head'.

        Raises
        ------
        ValueError
            If required header keys are missing or malformed.
        """
        header_parameters = [
            "modelname",
            "product_name",
            "earth_gravity_constant",
            "radius",
            "max_degree",
            "errors",
            "norm",
            "tide_system",
            "format",
        ]
        regex = "(" + "|".join(header_parameters) + ")"
        header = {}
        legend_before_end_header = ""
        while True:
            line = file_io.readline()
            if ext.lower() in [".gz", ".gzip", ".zip", ".tar", ".ZIP"]:
                line = line.decode()
            if re.match(regex, line):
                header[line.split()[0]] = line.split()[1]
            if "end_of_head" in line:
                break
            elif "0    0" in line:
                raise ValueError(f"Missing 'end_of_head' in header of file {file_io}")
            if line:
                legend_before_end_header = line

        # case for COSTG header where 'modelname' is created as 'product_name'
        header["modelname"] = header.get("product_name", header.get("modelname"))
        # default norm is fully_normalized, change it to 4pi if needed to be coherent with lenapy functions
        header["norm"] = "4pi" if "norm" not in header else header["norm"]
        header["norm"] = (
            "4pi" if header["norm"] == "fully_normalized" else header["norm"]
        )
        header["norm"] = (
            "unnorm" if header["norm"] == "unnormalized" else header["norm"]
        )
        header["tide_system"] = header.get("tide_system", "missing")

        # test for mandatory keywords (https://icgem.gfz-potsdam.de/docs/ICGEM-Format-2023.pdf)
        required = [
            "modelname",
            "earth_gravity_constant",
            "radius",
            "max_degree",
            "errors",
        ]
        if not all(k in header for k in required):
            raise ValueError(
                (
                    "File header does not contains mandatory keywords"
                    " (https://icgem.gfz-potsdam.de/docs/ICGEM-Format-2023.pdf)"
                )
            )

        header["max_degree"] = int(header["max_degree"])
        header["earth_gravity_constant"] = float(header["earth_gravity_constant"])
        header["radius"] = float(header["radius"])
        return header, legend_before_end_header

    @staticmethod
    def _open_file(filename: str | os.PathLike) -> tuple[IO, str]:
        """
        Open a .gfc file or a compressed archive containing a .gfc file.

        Parameters
        ----------
        filename : str or os.PathLike
            Path to the .gfc file or archive.

        Returns
        -------
        file : IO
            Opened file object.
        ext : str
            Original file extension.

        Raises
        ------
        ValueError
            If the file extension is unsupported or no .gfc is found in archive.
        """
        ext = os.path.splitext(filename)[-1]

        if ext.lower() == ".gfc":
            return open(filename, "r"), ext

        elif ext in (".gz", ".gzip"):
            return gzip.open(filename, "rb"), ext

        elif ext in (".zip", ".ZIP"):
            zip_file = zipfile.ZipFile(filename, "r")
            gfc_files = [f for f in zip_file.namelist() if f.lower().endswith(".gfc")]
            if not gfc_files:
                raise ValueError("No .gfc file found in ZIP archive.")
            return zip_file.open(gfc_files[0], "r"), ext

        elif ext == ".tar":
            tar_file = tarfile.open(filename, "r")
            gfc_files = [f for f in tar_file.getnames() if f.lower().endswith(".gfc")]
            if not gfc_files:
                raise ValueError("No .gfc file found in TAR archive.")
            return tar_file.extractfile(gfc_files[0]), ext

        raise ValueError("Unsupported file extension.")

    @staticmethod
    def _get_date(
        date_regex: str | None,
        date_format: str | None,
        filename: str | os.PathLike,
        header: dict,
    ) -> tuple[
        datetime.datetime, datetime.datetime, datetime.datetime, datetime.datetime
    ]:
        """
        Extract date from header file information and format dates in datetime objects.

        Parameters
        ----------
        date_regex : str | None
            A regular expression pattern used to search for the date in the modelname header information. It should
            contain at least one capturing group for the begin_time, and optionally a second group for the `end_time`.
        date_format : str | None, optional
            A format string compatible with `datetime.strptime` to parse the extracted date strings.
            Must be provided if `date_regex` is specified.
        filename : str | os.PathLike[Any]
            Name/path of the file to open.
        header : dict
            Parsed header metadata.

        Returns
        -------
        mid_month : datatime.Datetime
            Middle of the month.
        exact_time : datatime.Datetime
            Exact time of the data in the month.
        begin_time : datatime.Datetime
            Begin time of the data in the month.
        end_time : datatime.Datetime
            End time of the data in the month.
        """
        if (
            date_regex
            and date_format
            and bool(re.search(date_regex, header["modelname"]))
        ):
            # Get dates from the given date_regex and date_format
            dates = re.search(date_regex, header["modelname"])

        elif bool(re.search(r"_(\d{7})-(\d{7})", header["modelname"])):
            # For some products, time is stored as YYYYDOY-YYYYDOY in modelname
            # For GRACE products, DOY can not coincide with 1st and last day of month
            dates = re.search(r"_(\d{7})-(\d{7})", header["modelname"])
            date_format = "%Y%j"

        elif bool(re.search(r"(\d{4}-\d{2})", header["modelname"])):
            # For other products, time is stored as YYYY-MM in modelname
            dates = re.search(r"(\d{4}-\d{2})", header["modelname"])
            date_format = "%Y-%m"

        else:
            raise ValueError(
                f"Could not extract date information from modelname in the header "
                f"of {filename}\n Try with the parameter no_date=True or "
                f"check the use of arguments date_regex and date_format."
            )

        # Read begin date and end date if it exists
        begin_time = datetime.datetime.strptime(dates.group(1), date_format)
        end_time_str = dates.group(2) if dates.lastindex >= 2 else None

        if end_time_str:  # Case with a date for the end
            end_time = datetime.datetime.strptime(
                end_time_str, date_format
            ) + datetime.timedelta(days=1)
            exact_time = begin_time + (end_time - begin_time) / 2

            # Compute middle of the month for GRACE products
            mid_month = mid_month_grace_estimate(begin_time, end_time)
        else:  # Case without a date for the end
            if (
                begin_time.day == 1
            ):  # Case where the date contains no day in the month information
                end_time = (begin_time + datetime.timedelta(days=32)).replace(day=1)
                mid_month = begin_time + (end_time - begin_time) / 2
                exact_time = mid_month
            else:  # Case where the date contains day in the month information, we keep the date as reference
                end_time = begin_time
                mid_month = begin_time
                exact_time = begin_time

        return mid_month, exact_time, begin_time, end_time

    @staticmethod
    def _format_icgem1(
        data: pd.DataFrame,
        lmax: int,
        header: dict,
        clm: np.ndarray,
        slm: np.ndarray,
        eclm: np.ndarray | None = None,
        eslm: np.ndarray | None = None,
    ) -> xr.Dataset:
        """
        Subfunction of the gfc reader to read the gfct icgem1.0 format.

        Parameters
        ----------
        data : pd.Dataframe
            pandas Dataframe containing the coefficients information.
        lmax : int
            Maximal degree for the spherical harmonics coefficients.
        header : dict
            Dictionary of header information.
        clm : np.array
            Pre-created clm coefficients array.
        slm : np.array
            Pre-created slm coefficients array.
        eclm : np.array | None
            Pre-created eclm coefficients array.
        eslm : np.array | None
            Pre-created eslm coefficients array.

        Returns
        -------
         ds : xr.Dataset
            Information from the file stored in `xr.Dataset` format.
        """
        periods_acos = data[(data["tag"] == "acos")]["ref_time"].unique()
        periods_asin = data[(data["tag"] == "acos")]["ref_time"].unique()
        periods_acos.sort()
        periods_asin.sort()

        trnd_clm, trnd_slm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros(
            (lmax + 1, lmax + 1, 1)
        )
        acos_clm = np.zeros((lmax + 1, lmax + 1, 1, len(periods_acos)))
        acos_slm = np.zeros((lmax + 1, lmax + 1, 1, len(periods_acos)))
        asin_clm = np.zeros((lmax + 1, lmax + 1, 1, len(periods_asin)))
        asin_slm = np.zeros((lmax + 1, lmax + 1, 1, len(periods_asin)))

        ref_time = np.zeros((lmax + 1, lmax + 1, 1), dtype="datetime64[D]")

        if (
            header["errors"] != "no"
        ):  # (does not deal with calibrated_and_formal error case)
            trnd_eclm = np.zeros((lmax + 1, lmax + 1, 1))
            trnd_eslm = np.zeros((lmax + 1, lmax + 1, 1))
            acos_eclm = np.zeros((lmax + 1, lmax + 1, 1, len(periods_acos)))
            acos_eslm = np.zeros((lmax + 1, lmax + 1, 1, len(periods_acos)))
            asin_eclm = np.zeros((lmax + 1, lmax + 1, 1, len(periods_asin)))
            asin_eslm = np.zeros((lmax + 1, lmax + 1, 1, len(periods_asin)))

        index_gfc_trnd = [
            (data["tag"] == "gfc") | (data["tag"] == "gfct"),
            data["tag"] == "trnd",
        ]
        for ind, c, s in zip(index_gfc_trnd, [clm, trnd_clm], [slm, trnd_slm]):
            c[data[ind]["degree"].values, data[ind]["order"].values] = data[ind][
                "clm"
            ].values[:, np.newaxis]
            s[data[ind]["degree"].values, data[ind]["order"].values] = data[ind][
                "slm"
            ].values[:, np.newaxis]

        for ind, c, s, periods in zip(
            [data["tag"] == "acos", data["tag"] == "asin"],
            [acos_clm, asin_slm],
            [acos_slm, asin_slm],
            [periods_acos, periods_asin],
        ):
            c[
                data[ind]["degree"].values,
                data[ind]["order"].values,
                0,
                np.searchsorted(periods, data[ind]["ref_time"]),
            ] = data[ind]["clm"].values
            s[
                data[ind]["degree"].values,
                data[ind]["order"].values,
                0,
                np.searchsorted(periods, data[ind]["ref_time"]),
            ] = data[ind]["slm"].values

        ind = data["tag"] == "gfct"
        ref_time[data[ind]["degree"].values, data[ind]["order"].values] = (
            pd.to_datetime(
                data[ind]["ref_time"].values.astype(int).astype(str), format="%Y%m%d"
            ).to_numpy()[:, np.newaxis]
        )

        ds = xr.Dataset(
            {
                "clm": (["l", "m", "name"], clm),
                "slm": (["l", "m", "name"], slm),
                "trnd_clm": (["l", "m", "name"], trnd_clm),
                "trnd_slm": (["l", "m", "name"], trnd_slm),
                "acos_clm": (["l", "m", "name", "periods_acos"], acos_clm),
                "acos_slm": (["l", "m", "name", "periods_acos"], acos_slm),
                "asin_clm": (["l", "m", "name", "periods_asin"], asin_clm),
                "asin_slm": (["l", "m", "name", "periods_asin"], asin_slm),
                "ref_time": (["l", "m", "name"], ref_time),
            },
            coords={
                "l": np.arange(lmax + 1),
                "m": np.arange(lmax + 1),
                "name": [header["modelname"]],
                "periods_acos": periods_acos,
                "periods_asin": periods_asin,
            },
            attrs=header,
        )

        # case with error information (does not deal with calibrated_and_formal error case)
        if header["errors"] != "no":
            for ind, ec, es in zip(
                index_gfc_trnd, [eclm, trnd_eclm], [eslm, trnd_eslm]
            ):
                ec[data[ind]["degree"].values, data[ind]["order"].values] = data[ind][
                    "eclm"
                ].values[:, np.newaxis]
                es[data[ind]["degree"].values, data[ind]["order"].values] = data[ind][
                    "eslm"
                ].values[:, np.newaxis]

            for ind, ec, es, periods in zip(
                [data["tag"] == "acos", data["tag"] == "asin"],
                [acos_eclm, asin_eslm],
                [acos_eslm, asin_eslm],
                [periods_acos, periods_asin],
            ):
                ec[
                    data[ind]["degree"].values,
                    data[ind]["order"].values,
                    0,
                    np.searchsorted(periods, data[ind]["ref_time"]),
                ] = data[ind]["eclm"].values
                es[
                    data[ind]["degree"].values,
                    data[ind]["order"].values,
                    0,
                    np.searchsorted(periods, data[ind]["ref_time"]),
                ] = data[ind]["eslm"].values

            ds["eclm"] = xr.DataArray(eclm, dims=["l", "m", "name"])
            ds["eslm"] = xr.DataArray(eslm, dims=["l", "m", "name"])
            ds["trnd_eclm"] = xr.DataArray(trnd_eclm, dims=["l", "m", "name"])
            ds["trnd_eslm"] = xr.DataArray(trnd_eslm, dims=["l", "m", "name"])
            ds["acos_eclm"] = xr.DataArray(
                acos_eclm, dims=["l", "m", "name", "periods_acos"]
            )
            ds["acos_eslm"] = xr.DataArray(
                acos_eslm, dims=["l", "m", "name", "periods_acos"]
            )
            ds["asin_eclm"] = xr.DataArray(
                asin_eclm, dims=["l", "m", "name", "periods_asin"]
            )
            ds["asin_eslm"] = xr.DataArray(
                asin_eslm, dims=["l", "m", "name", "periods_asin"]
            )

        return ds

    @staticmethod
    def _format_icgem2(data: pd.DataFrame, lmax: int, header: dict) -> xr.Dataset:
        """
        Subfunction of the gfc reader to read the gfct icgem2.0 format.

        Parameters
        ----------
        data : pd.Dataframe
            pandas Dataframe containing the coefficients information.
        lmax : int
            Maximal degree for the spherical harmonics coefficients.
        header : dict
            Dictionary of header information.

        Returns
        -------
         ds : xr.Dataset
            Information from the file stored in `xr.Dataset` format.
        """
        periods_acos = data[(data["tag"] == "acos")]["period"].unique()
        periods_asin = data[(data["tag"] == "acos")]["period"].unique()
        periods_acos.sort()
        periods_asin.sort()

        # The epochs are used to create associated array to store the coefficients
        epochs = np.union1d(
            data[data["tag"] == "gfct"]["ref_time"].unique(),
            data[data["tag"] == "gfct"]["ref_time_end"].unique(),
        )
        epochs.sort()

        clm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs)))
        slm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs)))
        trnd_clm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs)))
        trnd_slm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs)))
        acos_clm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs), len(periods_acos)))
        acos_slm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs), len(periods_acos)))
        asin_clm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs), len(periods_asin)))
        asin_slm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs), len(periods_asin)))

        ref_time = pd.to_datetime(
            np.char.mod("%.4f", epochs), format="%Y%m%d.%H%M"
        ).to_numpy()

        clm[
            data[data["tag"] == "gfc"]["degree"].values,
            data[data["tag"] == "gfc"]["order"].values,
        ] = data[data["tag"] == "gfc"]["clm"].values[:, np.newaxis, np.newaxis]
        slm[
            data[data["tag"] == "gfc"]["degree"].values,
            data[data["tag"] == "gfc"]["order"].values,
        ] = data[data["tag"] == "gfc"]["slm"].values[:, np.newaxis, np.newaxis]

        for i in range(len(epochs) - 1):
            for j in range(i + 1, len(epochs)):
                for ind, c, s in zip(
                    [data["tag"] == "gfct", data["tag"] == "trnd"],
                    [clm, trnd_clm],
                    [slm, trnd_slm],
                ):
                    ind_t = (
                        ind
                        & (data["ref_time"] == epochs[i])
                        & (data["ref_time_end"] == epochs[j])
                    )
                    c[
                        data[ind_t]["degree"].values,
                        data[ind_t]["order"].values,
                        0,
                        i:j,
                    ] = data[ind_t]["clm"].values[:, np.newaxis]
                    s[
                        data[ind_t]["degree"].values,
                        data[ind_t]["order"].values,
                        0,
                        i:j,
                    ] = data[ind_t]["slm"].values[:, np.newaxis]

                for ind, c, s, period in zip(
                    [data["tag"] == "acos", data["tag"] == "asin"],
                    [acos_clm, asin_clm],
                    [acos_slm, asin_slm],
                    [periods_acos, periods_asin],
                ):
                    ind_t = (
                        ind
                        & (data["ref_time"] == epochs[i])
                        & (data["ref_time_end"] == epochs[j])
                    )
                    c[
                        data[ind_t]["degree"].values,
                        data[ind_t]["order"].values,
                        0,
                        i:j,
                        np.searchsorted(period, data[ind_t]["period"]),
                    ] = data[ind_t]["clm"].values[:, np.newaxis]
                    s[
                        data[ind_t]["degree"].values,
                        data[ind_t]["order"].values,
                        0,
                        i:j,
                        np.searchsorted(period, data[ind_t]["period"]),
                    ] = data[ind_t]["slm"].values[:, np.newaxis]

        ds = xr.Dataset(
            {
                "clm": (["l", "m", "name", "time"], clm),
                "slm": (["l", "m", "name", "time"], slm),
                "trnd_clm": (["l", "m", "name", "time"], trnd_clm),
                "trnd_slm": (["l", "m", "name", "time"], trnd_slm),
                "acos_clm": (["l", "m", "name", "time", "periods_acos"], acos_clm),
                "acos_slm": (["l", "m", "name", "time", "periods_acos"], acos_slm),
                "asin_clm": (["l", "m", "name", "time", "periods_asin"], asin_clm),
                "asin_slm": (["l", "m", "name", "time", "periods_asin"], asin_slm),
            },
            coords={
                "l": np.arange(lmax + 1),
                "m": np.arange(lmax + 1),
                "name": [header["modelname"]],
                "time": ref_time,
                "periods_acos": periods_acos,
                "periods_asin": periods_asin,
            },
            attrs=header,
        )

        # case with error information (does not deal with calibrated_and_formal error case)
        if header["errors"] != "no":
            eclm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs)))
            eslm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs)))
            trnd_eclm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs)))
            trnd_eslm = np.zeros((lmax + 1, lmax + 1, 1, len(epochs)))
            acos_eclm = np.zeros(
                (lmax + 1, lmax + 1, 1, len(epochs), len(periods_acos))
            )
            acos_eslm = np.zeros(
                (lmax + 1, lmax + 1, 1, len(epochs), len(periods_acos))
            )
            asin_eclm = np.zeros(
                (lmax + 1, lmax + 1, 1, len(epochs), len(periods_asin))
            )
            asin_eslm = np.zeros(
                (lmax + 1, lmax + 1, 1, len(epochs), len(periods_asin))
            )

            eclm[
                data[data["tag"] == "gfc"]["degree"].values,
                data[data["tag"] == "gfc"]["order"].values,
            ] = data[data["tag"] == "gfc"]["eclm"].values[:, np.newaxis, np.newaxis]
            eslm[
                data[data["tag"] == "gfc"]["degree"].values,
                data[data["tag"] == "gfc"]["order"].values,
            ] = data[data["tag"] == "gfc"]["eslm"].values[:, np.newaxis, np.newaxis]

            for i in range(len(epochs) - 1):
                for j in range(i + 1, len(epochs)):
                    for ind, ec, es in zip(
                        [data["tag"] == "gfct", data["tag"] == "trnd"],
                        [eclm, trnd_eclm],
                        [eslm, trnd_eslm],
                    ):
                        ind_t = (
                            ind
                            & (data["ref_time"] == epochs[i])
                            & (data["ref_time_end"] == epochs[j])
                        )
                        ec[
                            data[ind_t]["degree"].values,
                            data[ind_t]["order"].values,
                            0,
                            i:j,
                        ] = data[ind_t]["eclm"].values[:, np.newaxis]
                        es[
                            data[ind_t]["degree"].values,
                            data[ind_t]["order"].values,
                            0,
                            i:j,
                        ] = data[ind_t]["eslm"].values[:, np.newaxis]

                    for ind, ec, es, periods in zip(
                        [data["tag"] == "acos", data["tag"] == "asin"],
                        [acos_eclm, asin_eslm],
                        [acos_eslm, asin_eslm],
                        [periods_acos, periods_asin],
                    ):
                        ind_t = (
                            ind
                            & (data["ref_time"] == epochs[i])
                            & (data["ref_time_end"] == epochs[j])
                        )
                        ec[
                            data[ind_t]["degree"].values,
                            data[ind_t]["order"].values,
                            0,
                            i:j,
                            np.searchsorted(period, data[ind_t]["period"]),
                        ] = data[ind_t]["eclm"].values[:, np.newaxis]
                        es[
                            data[ind_t]["degree"].values,
                            data[ind_t]["order"].values,
                            0,
                            i:j,
                            np.searchsorted(period, data[ind_t]["period"]),
                        ] = data[ind_t]["eslm"].values[:, np.newaxis]

            ds["eclm"] = xr.DataArray(eclm, dims=["l", "m", "name", "time"])
            ds["eslm"] = xr.DataArray(eslm, dims=["l", "m", "name", "time"])
            ds["trnd_eclm"] = xr.DataArray(trnd_eclm, dims=["l", "m", "name", "time"])
            ds["trnd_eslm"] = xr.DataArray(trnd_eslm, dims=["l", "m", "name", "time"])
            ds["acos_eclm"] = xr.DataArray(
                acos_eclm, dims=["l", "m", "name", "time", "periods_acos"]
            )
            ds["acos_eslm"] = xr.DataArray(
                acos_eslm, dims=["l", "m", "name", "time", "periods_acos"]
            )
            ds["asin_eclm"] = xr.DataArray(
                asin_eclm, dims=["l", "m", "name", "time", "periods_asin"]
            )
            ds["asin_eslm"] = xr.DataArray(
                asin_eslm, dims=["l", "m", "name", "time", "periods_asin"]
            )

        return ds

    def open_dataset(
        self,
        filename: str | os.PathLike,
        drop_variables: None = None,
        no_date: bool = False,
        date_regex: str | None = None,
        date_format: str | None = None,
    ) -> xr.Dataset:
        """
        Read a .gfc ASCII file (or compressed) and format it as a xr.Dataset.
        The file needs to follow the ICGEM format: https://icgem.gfz-potsdam.de/docs/ICGEM-Format-2023.pdf

        The header information are stored in ds.attrs. The dataset contains clm and slm arrays
        with error information in eclm and eslm if available.
        For monthly files, time variables are stored as 'begin_time', 'end_time', 'exact_time' and 'mid_month'.
        'mid_month' is used as the time coordinate.

        The time information of the file is read in the mandatory header information 'modelname'. The function search
        by default a date of format 'YYYY-MM' or '_YYYYDOY-YYYYDOY'. The format of the date can be specified with the
        arguments `date_regex` and `date_format`. If the file not associated with a date information, use no_date=True.

        Parameters
        ----------
        filename : str | os.PathLike[Any]
            Name/path of the file to open.
        drop_variables : None
            Not used; included for compatibility with BackendEntrypoint pattern.
        no_date : bool, optional
            True if the data file contains no date information. Default is False.
        date_regex : str | None, optional
            A regular expression pattern used to search for the date in the modelname header information. It should
            contain at least one capturing group for the begin_time, and optionally a second group for the `end_time`.
        date_format : str | None, optional
            A format string compatible with `datetime.strptime` to parse the extracted date strings.
            Must be provided if `date_regex` is specified.

        Returns
        -------
        ds : xr.Dataset
            Information from the file stored in `xr.Dataset` format.

        Examples
        --------
        Default usage with automatic date extraction:
        >>> ds = xr.open_mfdataset('path/to/file.gfc', engine='lenapyGfc')

        Specify custom date pattern and format:
        >>> s = xr.open_mfdataset('path/to/file.gfc', engine='lenapyGfc',
        ...                       date_regex=r'_(\\d{8})-(\\d{8})', date_format='%Y%m%d')

        No date information in the file:
        >>> ds = xr.open_mfdataset('path/to/file.gfc', engine='lenapyGfc', no_date=True)
        """
        file, ext = self._open_file(filename)
        header, legend = self._parse_header(file, ext)

        lmax = header["max_degree"]

        # -- Load clm and slm data
        clm, slm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros((lmax + 1, lmax + 1, 1))

        col_names = ["tag", "degree", "order", "clm", "slm"]
        if (
            header["errors"] != "no"
        ):  # (does not deal with calibrated_and_formal error case)
            col_names.append("eclm")
            col_names.append("eslm")

            eclm, eslm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros(
                (lmax + 1, lmax + 1, 1)
            )

        if "t" in legend:
            col_names.append("ref_time")
            if " period" in legend:
                col_names.append("ref_time_end")
                col_names.append("period")

        # Read file with pandas, delim_whitespace for variable space delimiters
        data = pd.read_csv(
            file, delim_whitespace=True, header=None, names=col_names, engine="python"
        )

        # test if gfct key then have to deal with time
        if "t" not in legend:
            # -- Compute time
            if not no_date:
                mid_month, exact_time, begin_time, end_time = self._get_date(
                    date_regex, date_format, filename, header
                )

            # If no time, time info will be a string with modelname
            else:
                mid_month = header["modelname"]

            clm[data["degree"].values, data["order"].values] = data["clm"].values[
                :, np.newaxis
            ]
            slm[data["degree"].values, data["order"].values] = data["slm"].values[
                :, np.newaxis
            ]

            ds = xr.Dataset(
                {"clm": (["l", "m", "time"], clm), "slm": (["l", "m", "time"], slm)},
                coords={
                    "l": np.arange(lmax + 1),
                    "m": np.arange(lmax + 1),
                    "time": [mid_month],
                },
                attrs=header,
            )

            # case with error information (does not deal with calibrated_and_formal error case)
            if header["errors"] != "no":
                eclm[data["degree"].values, data["order"].values] = data["eclm"].values[
                    :, np.newaxis
                ]
                eslm[data["degree"].values, data["order"].values] = data["eslm"].values[
                    :, np.newaxis
                ]

                ds["eclm"] = xr.DataArray(eclm, dims=["l", "m", "time"])
                ds["eslm"] = xr.DataArray(eslm, dims=["l", "m", "time"])

        elif " period" not in legend:
            if header["errors"] == "no":
                ds = self._format_icgem1(data, lmax, header, clm, slm)
            else:
                ds = self._format_icgem1(data, lmax, header, clm, slm, eclm, eslm)

            if "format" not in ds.attrs:
                ds.attrs["format"] = "icgem1.0"

        else:
            if header["errors"] == "no":
                ds = self._format_icgem2(data, lmax, header)
            else:
                ds = self._format_icgem2(data, lmax, header)

            if "format" not in ds.attrs:
                ds.attrs["format"] = "icgem2.0"

        # -- Add various time information in dataset
        if not no_date and "t" not in legend:
            ds["begin_time"] = xr.DataArray([begin_time], dims=["time"])
            ds["end_time"] = xr.DataArray([end_time], dims=["time"])
            ds["exact_time"] = xr.DataArray([exact_time], dims=["time"])

        # -- Close all file pointers
        file.close()

        return ds

    def guess_can_open(self, filename: str | os.PathLike) -> bool:
        """
        Test the readability of a file with ReadGFC. Test if it is a .gfc file.

        Parameters
        ----------
        filename : str | os.PathLike[Any]
            Path to the file to test.

        Returns
        -------
        can_open : bool
            True if the file can be opened with ReadGFC, False otherwise.
        """
        try:
            ext = os.path.splitext(filename)[-1]
        except TypeError:
            return False

        compress_extensions = [".gz", ".zip", ".tar", ".gzip", ".ZIP"]

        if ext in (".gfc", ".GFC"):
            return True
        elif ext in compress_extensions:
            if ext in (".gz", ".gzip"):
                return (
                    filename.endswith(".gfc.gz")
                    or filename.endswith(".gfc.gzip")
                    or filename.endswith(".GFC.gz")
                    or filename.endswith(".GFC.gzip")
                )
            elif ext in (".zip", ".ZIP"):
                with zipfile.ZipFile(filename, "r") as zip_file:
                    return any(
                        file.endswith(".gfc") for file in zip_file.namelist()
                    ) or any(file.endswith(".GFC") for file in zip_file.namelist())
            elif ext == ".tar":
                with tarfile.open(filename, "r") as tar_file:
                    return any(
                        file.endswith(".gfc") for file in tar_file.getnames()
                    ) or any(file.endswith(".GFC") for file in tar_file.getnames())

        return False

    description = "Use .gfc files in xarray"


class ReadGRACEL2(BackendEntrypoint):
    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    @staticmethod
    def _open_file(filename: str | os.PathLike) -> any:
        """
        Open a file, supporting gzip-compressed files.

        Parameters
        ----------
        filename: str or os.PathLike
            Path to the file to open

        Returns
        -------
        file: file-like
            Opened file object
        """
        ext = os.path.splitext(filename)[-1]
        if ext in (".gz", ".gzip"):
            return gzip.open(filename, "rb")
        return open(filename, "r")

    @staticmethod
    def _read_cnes_header(file: any, ext: str) -> dict:
        """
        Read header from CNES, GRGS, or TUGRZ formatted files.

        Parameters
        ----------
        file: file-like
            Opened file object
        ext: str
            File extension

        Returns
        -------
        header: dict
            Parsed header information
        """
        header = {}
        while True:
            line = file.readline()
            if ext in (".gz", ".gzip"):
                line = line.decode()
            infos = line.split()
            if line[:5] == "EARTH":
                header["earth_gravity_constant"] = float(infos[1])
                header["radius"] = float(infos[2])
            elif line[:3] == "SHM":
                header["max_degree"] = int(infos[1])
                header["norm"] = " ".join(infos[4:6])
                header["tide_system"] = " ".join(infos[6:])
            elif "GRCOF2  " in line:
                break
        return header

    @staticmethod
    def _read_yaml_header(file: any, ext: str, filename: str) -> dict:
        """
        Read header from COST-G, UTCSR, JPL, or GFZ formatted files using YAML.

        Parameters
        ----------
        file: file-like
            Opened file object
        ext: str
            File extension
        filename: str
            File name for error reporting

        Returns
        -------
        header: dict
            Parsed header information
        """
        yaml_header_text = []
        while True:
            line = file.readline()
            if ext in (".gz", ".gzip"):
                line = line.decode()
            if "End of YAML header" in line:
                break
            elif "GRCOF2  " in line:
                raise ValueError(f"No 'End of YAML header' line in file {filename}")
            elif "date_issued" in line or "acknowledgement" in line:
                continue
            yaml_header_text.append(line)
        yaml_header = yaml.safe_load("".join(yaml_header_text))["header"]
        header = {
            "earth_gravity_constant": float(
                yaml_header["non-standard_attributes"]["earth_gravity_param"]["value"]
            ),
            "radius": float(
                yaml_header["non-standard_attributes"]["mean_equator_radius"]["value"]
            ),
            "max_degree": int(yaml_header["dimensions"]["degree"]),
            "norm": yaml_header["non-standard_attributes"]["normalization"],
            "tide_system": yaml_header["non-standard_attributes"].get(
                "permanent_tide_flag", "missing"
            ),
        }
        return header

    def open_dataset(
        self, filename: str | os.PathLike, drop_variables: None = None
    ) -> xr.Dataset:
        """
        Read a GRACE Level-2 gravity field product ASCII file (or compressed) from processing centers and
        format it as a xr.Dataset. The header information are stored in ds.attrs.
        The dataset contains clm and slm array with errors information in eclm and eslm if possible.
        For monthly files, time variables are stored as 'begin_time', 'end_time', 'exact_time' and 'mid_month'.
        'mid_month' is used as the time coordinate.

        Parameters
        ----------
        filename : str | os.PathLike[Any]
            Name/path of the file to open.
        drop_variables : None
            Not used; included for compatibility with BackendEntrypoint pattern.

        Returns
        -------
        ds : xr.Dataset
            Information from the file stored in xr.Dataset format.
        """
        # -- Create a pointer to the file
        ext = os.path.splitext(filename)[-1]
        file = self._open_file(filename)

        # read CNES level 2 products (or GRAZ reprocessed by CNES)
        if any(key in os.path.basename(filename) for key in ["CNES", "GRGS", "TUGRZ"]):
            header = self._read_cnes_header(file, ext)
        elif any(
            key in os.path.basename(filename)
            for key in ["COSTG", "UTCSR", "JPLEM", "GFZOP"]
        ):
            header = self._read_yaml_header(file, ext, filename)
        else:
            raise ValueError(
                "Name of the file does not corresponds to GRACE L2 products (https://archive.podaac."
                "earthdata.nasa.gov/podaac-ops-cumulus-docs/grace/open/L1B/GFZ/AOD1B/RL04/docs/"
                "L2-UserHandbook_v4.0.pdf), it does not contains the name of center key: "
                f"'COSTG', 'UTCSR', 'CNES', 'JPLEM' or 'GFZOP'. Name : {os.path.basename(filename)}"
            )

        header["norm"] = (
            "4pi" if header["norm"] == "fully normalized" else header["norm"]
        )
        lmax = header["max_degree"]

        # Compute time
        # time is stored as YYYYDOY - YYYYDOY in filename
        # For GRACE products, DOY can not coincide with 1st and last day of month
        try:
            dates = re.findall(r"_(\d{7})-(\d{7})_", os.path.basename(filename))[0]
        except IndexError:
            raise ValueError(
                "Name of the file does not corresponds to GRACE L2 products (https://archive.podaac."
                "earthdata.nasa.gov/podaac-ops-cumulus-docs/grace/open/L1B/GFZ/AOD1B/RL04/docs/"
                "L2-UserHandbook_v4.0.pdf), it does not contains date YYYYDOY-YYYYDOY information. "
                f"Name : {os.path.basename(filename)}"
            )

        begin_time = datetime.datetime.strptime(dates[0], "%Y%j")
        end_time = datetime.datetime.strptime(dates[1], "%Y%j") + datetime.timedelta(
            days=1
        )

        exact_time = begin_time + (end_time - begin_time) / 2

        if (end_time - begin_time).days > 11:
            # compute middle of the month for GRACE products
            mid_month = mid_month_grace_estimate(begin_time, end_time)
        else:
            # Exact time for 10 days products
            mid_month = exact_time

        # -- Load clm and slm data and errors
        clm, slm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros((lmax + 1, lmax + 1, 1))
        eclm, eslm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros(
            (lmax + 1, lmax + 1, 1)
        )

        data = np.genfromtxt(
            file,
            dtype=[
                ("tag", "U6"),
                ("degree", int),
                ("order", int),
                ("clm", float),
                ("slm", float),
                ("eclm", float),
                ("eslm", float),
                ("epoch_begin_time", float),
                ("epoch_stop_time", float),
                ("flags", "U4"),
            ],
        )

        clm[data["degree"], data["order"]] = data["clm"][:, np.newaxis]
        slm[data["degree"], data["order"]] = data["slm"][:, np.newaxis]
        eclm[data["degree"], data["order"]] = data["eclm"][:, np.newaxis]
        eslm[data["degree"], data["order"]] = data["eslm"][:, np.newaxis]

        # to deal with the fact that first line with C00 = 1 is passed in CNES files
        if "CNES" in os.path.basename(filename):
            clm[0, 0] = 1

        ds = xr.Dataset(
            {
                "clm": (["l", "m", "time"], clm),
                "slm": (["l", "m", "time"], slm),
                "eclm": (["l", "m", "time"], eclm),
                "eslm": (["l", "m", "time"], eslm),
            },
            coords={
                "l": np.arange(lmax + 1),
                "m": np.arange(lmax + 1),
                "time": [mid_month],
            },
            attrs=header,
        )

        # -- Add time information in dataset
        ds["begin_time"] = xr.DataArray([begin_time], dims=["time"])
        ds["end_time"] = xr.DataArray([end_time], dims=["time"])
        ds["exact_time"] = xr.DataArray([exact_time], dims=["time"])

        file.close()
        return ds

    description = "Use GRACEL2 product files in xarray"


class ReadShLoading(BackendEntrypoint):
    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def open_dataset(
        self, filename: str | os.PathLike, drop_variables=None
    ) -> xr.Dataset:
        """
        Read Loading models in ASCII file (or compressed) from http://loading.u-strasbg.fr and
        format it as a xr.Dataset. The header information are stored in ds.attrs.
        The dataset contains clm and slm array.
        'mid_month' is used as the time coordinate.

        Parameters
        ----------
        filename : str | os.PathLike[Any]
            Name/path of the file to open.
        drop_variables : None
            Not used; included for compatibility with BackendEntrypoint pattern.

        Returns
        -------
        ds : xr.Dataset
            Information from the file stored in `xr.Dataset` format.
        """
        if ".tar.gz" not in filename:
            file = open(filename, "r")
            ds = self._process_file(file, compression=False)

        else:
            tar = tarfile.open(filename)

            ds_list = []
            for member in tar:
                file = tar.extractfile(member.name)
                ds_list.append(self._process_file(file, compression=True))
            ds = xr.concat(ds_list, dim="time")

        return ds

    @staticmethod
    def _process_file(file, compression: bool = False) -> xr.Dataset:
        """
        Process a single file pointer and return it as a xr.Dataset.

        Parameters
        ----------
        file : file-like object
            Opened file to process.
        compression : bool
            Boolean indicating whether to compress the file or not.

        Returns
        -------
        ds : xr.Dataset
            Information from the file stored in `xr.Dataset` format.
        """
        header = {}
        line = True
        while line:
            line = file.readline()
            if compression:
                line = line.decode()
            infos = line.split()

            if "! Maximum degree" in line:
                header["max_degree"] = int(infos[3])
            if "! Epoch:" in line:
                header["epoch"] = " ".join(infos[2:])
            if "! Model:" in line:
                header["modelname"] = " ".join(infos[2:])

            elif "! Comment" in line:
                break

        lmax = header["max_degree"]
        header["norm"] = "4pi"
        # Inforation to confirm with Jean-Paul Boy
        header["earth_gravity_constant"] = LNPY_GM_EARTH
        header["radius"] = LNPY_A_EARTH_GRS80

        # Compute time
        time = datetime.datetime.strptime(header["epoch"], "%Y %m %d %H %M %S.%f")

        # Load clm and slm data
        clm, slm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros((lmax + 1, lmax + 1, 1))

        data = np.genfromtxt(
            file,
            dtype=[("degree", int), ("order", int), ("clm", float), ("slm", float)],
        )

        clm[data["degree"], data["order"]] = data["clm"][:, np.newaxis]
        slm[data["degree"], data["order"]] = data["slm"][:, np.newaxis]

        ds = xr.Dataset(
            {"clm": (["l", "m", "time"], clm), "slm": (["l", "m", "time"], slm)},
            coords={"l": np.arange(lmax + 1), "m": np.arange(lmax + 1), "time": [time]},
            attrs=header,
        )

        file.close()
        return ds

    description = "Use loading models files from http://loading.u-strasbg.fr in xarray"
