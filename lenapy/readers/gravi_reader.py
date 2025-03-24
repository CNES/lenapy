"""
The gravi_reader module provides functions to load time-variable gravity field data from different products and format
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

import numpy as np
import xarray as xr
import yaml
from xarray.backends import BackendEntrypoint

from lenapy.constants import *
from lenapy.utils.harmo import mid_month_grace_estimate


def read_tn14(filename, rmmean=False):
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


def read_tn13(filename):
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

    def open_dataset(
        self,
        filename,
        drop_variables=None,
        no_date=False,
        date_regex=None,
        date_format=None,
    ):
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
        # -- Create a pointer to the '.gfc' file
        ext = os.path.splitext(filename)[-1]
        compress_extensions = [".gz", ".zip", ".tar", ".gzip", ".ZIP"]

        if ext in (".gfc", ".GFC"):
            file = open(filename, "r")

        elif ext in compress_extensions:
            if ext in (".gz", ".gzip"):
                file = gzip.open(filename, "rb")
            elif ext in (".zip", ".ZIP"):
                zip_file = zipfile.ZipFile(filename, "r")
                filenamezip = [
                    file
                    for file in zip_file.namelist()
                    if file.endswith(".gfc") or file.endswith(".GFC")
                ][0]
                file = zip_file.open(filenamezip, "r")
            elif ext == ".tar":
                tar_file = tarfile.open(filename, "r")
                filenametar = [
                    file
                    for file in tar_file.getnames()
                    if file.endswith(".gfc") or file.endswith(".GFC")
                ][0]
                file = tar_file.extractfile(filenametar)

        else:
            raise ValueError(
                "File does not have the good extension. "
                "Should be .gfc or a compress format with a .gfc file in it."
            )

        # -- Extract parameters from '.gfc' header
        header_parameters = [
            "modelname",
            "product_name",
            "earth_gravity_constant",
            "radius",
            "max_degree",
            "errors",
            "norm",
            "tide_system",
        ]
        parameters_regex = "(" + "|".join(header_parameters) + ")"
        header = {}
        line = True

        # goes while up to end of header or end of file
        while line:
            line = file.readline()
            if ext in compress_extensions:
                line = line.decode()
            if re.match(parameters_regex, line):
                header[line.split()[0]] = line.split()[1]

            # test to break when end of header
            if "end_of_head" in line:
                break
            # try to intercept case where no end_of_head to raise Error (if file is starting with degree 0 and order 0)
            elif "0    0" in line:
                raise ValueError("No 'end_of_head' line in file ", filename)
            # keep keys information from the line before end_of_head (to know if there is a time key or not)
            legend_before_end_header = line

        # case for COSTG header where 'modelname' is created as 'product_name'
        header["modelname"] = (
            header["product_name"] if "product_name" in header else header["modelname"]
        )

        # default norm is fully_normalized, change it to 4pi if needed to be coherent with lenapy functions
        header["norm"] = "4pi" if "norm" not in header else header["norm"]
        header["norm"] = (
            "4pi" if header["norm"] == "fully_normalized" else header["norm"]
        )
        header["norm"] = (
            "unnorm" if header["norm"] == "unnormalized" else header["norm"]
        )

        header["tide_system"] = (
            "missing" if "tide_system" not in header else header["tide_system"]
        )

        # test for mandatory keywords (https://icgem.gfz-potsdam.de/docs/ICGEM-Format-2023.pdf)
        if not all(
            key in header
            for key in [
                "modelname",
                "earth_gravity_constant",
                "radius",
                "max_degree",
                "errors",
            ]
        ):
            raise ValueError(
                "File header does not contains mandatory keywords"
                " (https://icgem.gfz-potsdam.de/docs/ICGEM-Format-2023.pdf)"
            )

        # convert str to numbers for adapted header info
        header["max_degree"] = int(header["max_degree"])
        header["earth_gravity_constant"] = float(header["earth_gravity_constant"])
        header["radius"] = float(header["radius"])
        lmax = header["max_degree"]

        # test if gfct key then have to deal with time
        if "t" not in legend_before_end_header:
            # -- Compute time
            if not no_date:
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
                        end_time = (begin_time + datetime.timedelta(days=32)).replace(
                            day=1
                        )
                        mid_month = begin_time + (end_time - begin_time) / 2
                        exact_time = mid_month
                    else:  # Case where the date contains day in the month information, we keep the date as reference
                        end_time = begin_time
                        mid_month = begin_time
                        exact_time = begin_time

            # If no time, time info will be a string with modelname
            else:
                mid_month = header["modelname"]

            # -- Load clm and slm data
            clm, slm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros(
                (lmax + 1, lmax + 1, 1)
            )
            # case with error information (does not deal with calibrated_and_formal error case)
            if header["errors"] != "no":
                eclm, eslm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros(
                    (lmax + 1, lmax + 1, 1)
                )
                data = np.genfromtxt(
                    file,
                    dtype=[
                        ("tag", "U4"),
                        ("degree", int),
                        ("order", int),
                        ("clm", float),
                        ("slm", float),
                        ("eclm", float),
                        ("eslm", float),
                    ],
                )
                eclm[data["degree"], data["order"]] = data["eclm"][:, np.newaxis]
                eslm[data["degree"], data["order"]] = data["eslm"][:, np.newaxis]

            # case for no error in file
            else:
                data = np.genfromtxt(
                    file,
                    dtype=[
                        ("tag", "U4"),
                        ("degree", int),
                        ("order", int),
                        ("clm", float),
                        ("slm", float),
                    ],
                )

            clm[data["degree"], data["order"]] = data["clm"][:, np.newaxis]
            slm[data["degree"], data["order"]] = data["slm"][:, np.newaxis]
            ds = xr.Dataset(
                {"clm": (["l", "m", "time"], clm), "slm": (["l", "m", "time"], slm)},
                coords={
                    "l": np.arange(lmax + 1),
                    "m": np.arange(lmax + 1),
                    "time": [mid_month],
                },
                attrs=header,
            )

            if header["errors"] != "no":
                ds["eclm"] = xr.DataArray(eclm, dims=["l", "m", "time"])
                ds["eslm"] = xr.DataArray(eslm, dims=["l", "m", "time"])

        else:
            raise AssertionError(
                "Reading of .gfc file with time is not implemented yet"
            )

        # -- Add various time information in dataset
        if not no_date:
            ds["begin_time"] = xr.DataArray([begin_time], dims=["time"])
            ds["end_time"] = xr.DataArray([end_time], dims=["time"])
            ds["exact_time"] = xr.DataArray([exact_time], dims=["time"])

        # -- Close all file pointers
        if ext in (".zip", ".ZIP"):
            zip_file.close()
        elif ext == ".tar":
            tar_file.close()
        file.close()

        return ds

    def guess_can_open(self, filename):
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

    def open_dataset(self, filename, drop_variables=None):
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

        if ext in (".gz", ".gzip"):
            file = gzip.open(filename, "rb")
        else:
            file = open(filename, "r")

        line = True
        # read CNES level 2 products (or GRAZ reprocessed by CNES)
        if (
            "CNES" in os.path.basename(filename)
            or "GRGS" in os.path.basename(filename)
            or "TUGRZ" in os.path.basename(filename)
        ):
            header = {}
            while line:
                line = file.readline()
                if ext in (".gz", ".gzip"):
                    line = line.decode()
                infos = line.split()

                if line[:5] == "EARTH":  # line with GM and a
                    header["earth_gravity_constant"] = float(infos[1])
                    header["radius"] = float(infos[2])
                elif line[:3] == "SHM":  # line with lmax, norm and tide
                    header["max_degree"] = int(infos[1])
                    header["norm"] = " ".join(infos[4:6])
                    header["tide_system"] = " ".join(infos[6:])

                # first line with C00 = 1 (because no end of header information to deal with)
                elif "GRCOF2  " in line:
                    break

        # Read other L2 products (COST-G, CSR, JPL or GFZ) where header follows the yaml format
        elif any(
            [
                name in os.path.basename(filename)
                for name in ("COSTG", "UTCSR", "JPLEM", "GFZOP")
            ]
        ):
            yaml_header_text = []
            while line:
                line = file.readline()
                if ext in (".gz", ".gzip"):
                    line = line.decode()
                # test to break when end of header
                if "End of YAML header" in line:
                    break

                # try to intercept case where no end_of_head (starting with degree and order 0)
                elif "GRCOF2  " in line:
                    raise ValueError(f"No 'End of YAML header' line in file {filename}")

                # deal with case where file is weirdly filled with 'date_issued:0000-00-00T00:00:00' to avoid yaml crash
                # deal also with acknowledgement line from GFZ on GRACE-FO periods that crash the yaml parser
                elif "date_issued" in line or "acknowledgement" in line:
                    continue
                yaml_header_text.append(line)

            # read yaml header to create a dict
            yaml_header = yaml.safe_load("".join(yaml_header_text))["header"]

            header = {
                "earth_gravity_constant": float(
                    yaml_header["non-standard_attributes"]["earth_gravity_param"][
                        "value"
                    ]
                ),
                "radius": float(
                    yaml_header["non-standard_attributes"]["mean_equator_radius"][
                        "value"
                    ]
                ),
                "max_degree": int(yaml_header["dimensions"]["degree"]),
                "norm": yaml_header["non-standard_attributes"]["normalization"],
            }
            try:
                header["tide_system"] = yaml_header["non-standard_attributes"][
                    "permanent_tide_flag"
                ]
            except KeyError:
                header["tide_system"] = "missing"

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

        # compute middle of the month for GRACE products
        mid_month = mid_month_grace_estimate(begin_time, end_time)

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

    def open_dataset(self, filename, drop_variables=None):
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
    def _process_file(file, compression=False):
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
