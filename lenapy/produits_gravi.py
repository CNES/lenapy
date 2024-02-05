"""
This module allows to load SH time-variable gravity field data from different products and format the data with unified
definition for variables and coordinates (output dataset are compatible with the use of xHarmo) :
standardized coordinates names : l, m, time
standardized variables names : clm, slm, begin_time, end_time, exact_time and if possible eclm and eslm
GRACE level 2 products from SDS centers can be open with the engine='gracel2' in xr.open_mfdataset()
Gravity field products organised as a .gfc files can be open with the engine='gfc' in xr.open_mfdataset()
Dask is implicitely used when using these interface methods.

Examples
--------
>>> files = [os.path.join(csr_data_dir, f) for f in os.listdir(csr_data_dir)]
>>> ds_csr = xr.open_mfdataset(files, engine='gracel2', combine_attrs="drop_conflicts")

>>> files = [os.path.join(graz_data_dir, f) for f in os.listdir(graz_data_dir)]
>>> ds_graz = xr.open_mfdataset(files, engine='gfc', combine_attrs="drop_conflicts")
"""
import os
import zipfile
import tarfile
import yaml
import datetime
import gzip
import re
import xarray as xr
import numpy as np
from xarray.backends import BackendEntrypoint


class ReadGFC(BackendEntrypoint):
    def open_dataset(self, filename, drop_variables=None):
        """
        Read a .gfc ascii file (or compressed) and format it as a xr.Dataset. The header information are stored in
        ds.attrs. The dataset contains clm and slm array with errors information in eclm and eslm if possible.
        For monthly file, time variable are stored as 'begin_time', 'end_time', 'exact_time' and 'mid_month'.
        mid_month is used to be the time coordinate.

        Parameters
        ----------
        filename : str | os.PathLike[Any]
            Name/path of the file to open
        drop_variables : None
            Variable to align to BackendEntrypoint pattern

        Returns
        -------
        ds : xr.Dataset
            Information of the file stored in xr.Dataset format
        """
        # -- Create a pointer to the '.gfc' file
        ext = os.path.splitext(filename)[-1]
        compress_extensions = ['.gz', '.zip', '.tar', '.gzip', '.ZIP']

        if ext in ('.gfc', '.GFC'):
            file = open(filename, 'r')

        elif ext in compress_extensions:
            if ext in ('.gz', '.gzip'):
                file = gzip.open(filename, 'rb')
            elif ext in ('.zip', '.ZIP'):
                zip_file = zipfile.ZipFile(filename, 'r')
                filenamezip = [file for file in zip_file.namelist() if file.endswith('.gfc')][0]
                file = zip_file.open(filenamezip, 'r')
            elif ext == '.tar':
                tar_file = tarfile.open(filename, 'r')
                filenametar = [file for file in tar_file.getnames() if file.endswith('.gfc')][0]
                file = tar_file.extractfile(filenametar)

        else:
            raise ValueError("File does not have the good extension. "
                             "Should be .gfc or a compress format with a .gfc file in it.")

        # -- Extract parameters from '.gfc' header
        header_parameters = ['modelname', 'product_name', 'earth_gravity_constant', 'radius', 'max_degree', 'errors',
                             'norm', 'tide_system']
        parameters_regex = '(' + '|'.join(header_parameters) + ')'
        header = {}
        while True:
            line = file.readline()
            if ext in compress_extensions:
                line = line.decode()
            if re.match(parameters_regex, line):
                header[line.split()[0]] = line.split()[1]

            # test to break when end of header
            if 'end_of_head' in line:
                break
            # try to intercept case where no end_of_head (if file is starting with degree 0 and order 0)
            elif '0    0' in line:
                raise ValueError("No 'end_of_head' line in file ", filename)
            # keep keys information from the line before end_of_head (to know if there is a time key or not)
            legend_before_end_header = line

        # case for COSTG header where 'modelname' is created as 'product_name'
        if 'product_name' in header:
            header['modelname'] = header['product_name']

        # default norm is fully_normalized
        if 'norm' not in header:
            header['norm'] = 'fully_normalized'
        if 'tide_system' not in header:
            header['tide_system'] = 'missing'

        # test for mandatory keywords (http://icgem.gfz-potsdam.de/ICGEM-Format-2023.pdf)
        if not all(key in header for key in ['modelname', 'earth_gravity_constant', 'radius', 'max_degree', 'errors']):
            raise ValueError("File header does not contains mandatory keywords"
                             " (http://icgem.gfz-potsdam.de/ICGEM-Format-2023.pdf)")

        # convert str to numbers for adapted header info
        header['max_degree'] = int(header['max_degree'])
        header['earth_gravity_constant'] = float(header['earth_gravity_constant'])
        header['radius'] = float(header['radius'])
        lmax = header['max_degree']

        # test if gfct key then have to deal with time
        if 't' not in legend_before_end_header:
            # -- Compute time
            # For some products, time is stored as YYYY-MM in modelname
            if any([name in header['modelname'] for name in ('ITSG', 'IGG', 'SWARM', 'Thongji', 'LUH')]):
                yyyy_mm = re.search(r'(\d{4}-\d{2})', header['modelname']).group(0)
                begin_time = datetime.datetime.strptime(yyyy_mm, '%Y-%m')
                end_time = (begin_time + datetime.timedelta(days=32)).replace(day=1)
                mid_month = begin_time + (end_time - begin_time) / 2
                exact_time = mid_month

            # For others products, time is stored as YYYYDOY-YYYYDOY in modelname
            elif any([name in header['modelname'] for name in ('COSTG', 'UTCSR', 'JPLEM', 'GFZOP', 'CNESG')]):
                dates = re.findall(r'_(\d{7})-(\d{7})_', header['modelname'])[0]
                begin_time = datetime.datetime.strptime(dates[0], '%Y%j')
                end_time = datetime.datetime.strptime(dates[1], '%Y%j') + datetime.timedelta(days=1)

                exact_time = begin_time + (end_time - begin_time) / 2
                # round begin_time to the 1st of the month and deal with May 2015 and December 2011 (JPL)
                # March 2017 and october 2018 cover second half of the month
                if ((begin_time.day <= 15 and begin_time.strftime('%Y%j') != '2015102') or
                        begin_time.strftime('%Y%j') == '2011351' or begin_time.strftime('%Y%j') == '2017076' or
                        begin_time.strftime('%Y%j') == '2018295'):
                    tmp_begin = begin_time.replace(day=1)
                else:
                    tmp_begin = (begin_time.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)

                # round end_time to the 1st of the month after and deal with Janv 2004, Nov 2011 (CSR, GFZ) and May 2015
                if end_time.day <= 15 and end_time.strftime('%Y%j') not in ('2004014', '2011320', '2015132'):
                    tmp_end = end_time.replace(day=1)
                else:
                    tmp_end = (end_time.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)

                mid_month = tmp_begin + (tmp_end - tmp_begin) / 2

            else:
                raise ValueError("Could not extract date information from the header of ", filename)

            # -- Load clm and slm data
            clm, slm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros((lmax + 1, lmax + 1, 1))
            # case with error information (does not deal with calibrated_and_formal error case)
            if header['errors'] != 'no':
                eclm, eslm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros((lmax + 1, lmax + 1, 1))
                data = np.genfromtxt(file, dtype=[('tag', 'U4'), ('degree', int), ('order', int),
                                                  ('clm', float), ('slm', float), ('eclm', float), ('eslm', float)])
                eclm[data['degree'], data['order']] = data['eclm'][:, np.newaxis]
                eslm[data['degree'], data['order']] = data['eslm'][:, np.newaxis]

            # case for no error in file
            else:
                data = np.genfromtxt(file, dtype=[('tag', 'U4'), ('degree', int), ('order', int),
                                                  ('clm', float), ('slm', float)])

            clm[data['degree'], data['order']] = data['clm'][:, np.newaxis]
            slm[data['degree'], data['order']] = data['slm'][:, np.newaxis]
            ds = xr.Dataset({'clm': (['l', 'm', 'time'], clm), 'slm': (['l', 'm', 'time'], slm)},
                            coords={'l': np.arange(lmax + 1), 'm': np.arange(lmax + 1), 'time': [mid_month]},
                            attrs=header)

            if header['errors'] != 'no':
                ds['eclm'] = xr.DataArray(eclm, dims=['l', 'm', 'time'])
                ds['eslm'] = xr.DataArray(eslm, dims=['l', 'm', 'time'])

        else:
            raise AssertionError("Reading of .gfc file with time is not implemented yet")

        # -- Add various time information in dataset
        ds['begin_time'] = xr.DataArray([begin_time], dims=['time'])
        ds['end_time'] = xr.DataArray([end_time], dims=['time'])
        ds['exact_time'] = xr.DataArray([exact_time], dims=['time'])

        # -- Close all file pointers
        if ext in ('.zip', '.ZIP'):
            zip_file.close()
        elif ext == '.tar':
            tar_file.close()
        file.close()

        return ds

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename):
        """
        Test the readability of a file with ReadGFC. Test if it is a .gfc file.

        Parameters
        ----------
        filename : str | os.PathLike[Any]
            Object to try to open

        Returns
        -------
        can_open : bool
            True is the file can be opened with ReadGFC, False otherwise
        """
        try:
            ext = os.path.splitext(filename)[-1]
        except TypeError:
            return False

        compress_extensions = ['.gz', '.zip', '.tar', '.gzip', '.ZIP']

        if ext in ('.gfc', '.GFC'):
            return True
        elif ext in compress_extensions:
            if ext in ('.gz', '.gzip'):
                return (filename.endswith('.gfc.gz') or filename.endswith('.gfc.gzip')
                        or filename.endswith('.GFC.gz') or filename.endswith('.GFC.gzip'))
            elif ext in ('.zip', '.ZIP'):
                with zipfile.ZipFile(filename, 'r') as zip_file:
                    return any(file.endswith('.gfc') for file in zip_file.namelist())
            elif ext == '.tar':
                with tarfile.open(filename, 'r') as tar_file:
                    return any(file.endswith('.gfc') for file in tar_file.getnames())

        return False

    description = "Use .gfc files in xarray"


class ReadGRACEL2(BackendEntrypoint):
    def open_dataset(self, filename, drop_variables=None):
        """
        Read a GRACE Level-2 gravity field product ascii file (or compressed) from centers and
        format it as a xr.Dataset. The header information are stored in ds.attrs.
        The dataset contains clm and slm array with errors information in eclm and eslm if possible.
        For monthly file, time variable are stored as 'begin_time', 'end_time', 'exact_time' and 'mid_month'.
        mid_month is used to be the time coordinate.

        Parameters
        ----------
        filename : str | os.PathLike[Any]
            Name/path of the file to open
        drop_variables : None
            Variable to align to BackendEntrypoint pattern

        Returns
        -------
        ds : xr.Dataset
            Information of the file stored in xr.Dataset format
        """
        # -- Create a pointer to the file
        ext = os.path.splitext(filename)[-1]

        if ext in ('.gz', '.gzip'):
            file = gzip.open(filename, 'rb')
        else:
            file = open(filename, 'r')

        # read GRGS level 2 products
        if 'GRGS' in os.path.basename(filename):
            header = {}
            while True:
                line = file.readline()
                if ext in ('.gz', '.gzip'):
                    line = line.decode()
                infos = line.split()

                if line[:5] == 'EARTH':  # line with GM and a
                    header['earth_gravity_constant'] = float(infos[1])
                    header['radius'] = float(infos[2])
                elif line[:3] == 'SHM':  # line with lmax, norm and tide
                    header['max_degree'] = int(infos[1])
                    header['norm'] = ' '.join(infos[4:6])
                    header['tide_system'] = ' '.join(infos[6:])

                # first line with C00 = 1 (because no end of header information to deal with)
                elif 'GRCOF2  ' in line:
                    break

        # Read other L2 products (COST-G, CSR, JPL or GFZ) where header follows the yaml format
        elif any([name in os.path.basename(filename) for name in ('COSTG', 'UTCSR', 'JPLEM', 'GFZOP')]):
            yaml_header_text = []
            while True:
                line = file.readline()
                if ext in ('.gz', '.gzip'):
                    line = line.decode()
                # test to break when end of header
                if 'End of YAML header' in line:
                    break

                # try to intercept case where no end_of_head (starting with degree and order 0)
                elif 'GRCOF2  ' in line:
                    raise ValueError("No 'End of YAML header' line in file ", filename)

                # deal with case where file is weirdly filled with 'date_issued:0000-00-00T00:00:00' to avoid yaml crash
                # deal also with acknowledgement line from GFZ on GRACE-FO periods that crash the yaml parser
                elif 'date_issued' in line or 'acknowledgement' in line:
                    continue
                yaml_header_text.append(line)

            # read yaml header to create a dict
            yaml_header = yaml.safe_load(''.join(yaml_header_text))['header']

            header = {
                'earth_gravity_constant': float(yaml_header['non-standard_attributes']['earth_gravity_param']['value']),
                'radius': float(yaml_header['non-standard_attributes']['mean_equator_radius']['value']),
                'max_degree': int(yaml_header['dimensions']['degree']),
                'norm': yaml_header['non-standard_attributes']['normalization'],
            }
            try:
                header['tide_system'] = yaml_header['non-standard_attributes']['permanent_tide_flag']
            except KeyError:
                header['tide_system'] = 'missing'

        else:
            raise ValueError("Name of the file does not corresponds to GRACE L2 products (https://archive.podaac."
                             "earthdata.nasa.gov/podaac-ops-cumulus-docs/grace/open/L1B/GFZ/AOD1B/RL04/docs/"
                             "L2-UserHandbook_v4.0.pdf), it does not contains the name of center key: "
                             "'COSTG', 'UTCSR', 'GRGS', 'JPLEM' or 'GFZOP'. Name : ", os.path.basename(filename))

        lmax = header['max_degree']

        # Compute time
        try:
            dates = re.findall(r'_(\d{7})-(\d{7})_', os.path.basename(filename))[0]
        except IndexError:
            raise ValueError("Name of the file does not corresponds to GRACE L2 products (https://archive.podaac."
                             "earthdata.nasa.gov/podaac-ops-cumulus-docs/grace/open/L1B/GFZ/AOD1B/RL04/docs/"
                             "L2-UserHandbook_v4.0.pdf), it does not contains date YYYYDOY-YYYYDOY information. "
                             "Name : ", os.path.basename(filename))

        begin_time = datetime.datetime.strptime(dates[0], '%Y%j')
        end_time = datetime.datetime.strptime(dates[1], '%Y%j') + datetime.timedelta(days=1)

        exact_time = begin_time + (end_time - begin_time) / 2
        # round begin_time to the 1st of the month and deal with May 2015 and December 2011 (JPL)
        # March 2017 and october 2018 cover second half of the month
        if ((begin_time.day <= 15 and begin_time.strftime('%Y%j') != '2015102') or
                begin_time.strftime('%Y%j') == '2011351' or begin_time.strftime('%Y%j') == '2017076' or
                begin_time.strftime('%Y%j') == '2018295'):
            tmp_begin = begin_time.replace(day=1)
        else:
            tmp_begin = (begin_time.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)

        # round end_time to the 1st of the month after and deal with Janv 2004, Nov 2011 (CSR, GFZ) and May 2015
        if end_time.day <= 15 and end_time.strftime('%Y%j') not in ('2004014', '2011320', '2015132'):
            tmp_end = end_time.replace(day=1)
        else:
            tmp_end = (end_time.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)

        mid_month = tmp_begin + (tmp_end - tmp_begin) / 2

        # -- Load clm and slm data and errors
        clm, slm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros((lmax + 1, lmax + 1, 1))
        eclm, eslm = np.zeros((lmax + 1, lmax + 1, 1)), np.zeros((lmax + 1, lmax + 1, 1))

        data = np.genfromtxt(file, dtype=[('tag', 'U6'), ('degree', int), ('order', int),
                                          ('clm', float), ('slm', float), ('eclm', float), ('eslm', float),
                                          ('epoch_begin_time', float), ('epoch_stop_time', float), ('flags', 'U4')])

        clm[data['degree'], data['order']] = data['clm'][:, np.newaxis]
        slm[data['degree'], data['order']] = data['slm'][:, np.newaxis]
        eclm[data['degree'], data['order']] = data['eclm'][:, np.newaxis]
        eslm[data['degree'], data['order']] = data['eslm'][:, np.newaxis]

        # to deal with the fact that first line with C00 = 1 is passed in GRGS files
        if 'GRGS' in os.path.basename(filename):
            clm[0, 0] = 1

        ds = xr.Dataset({'clm': (['l', 'm', 'time'], clm), 'slm': (['l', 'm', 'time'], slm),
                         'eclm': (['l', 'm', 'time'], eclm), 'eslm': (['l', 'm', 'time'], eslm)},
                        coords={'l': np.arange(lmax + 1), 'm': np.arange(lmax + 1), 'time': [mid_month]},
                        attrs=header)

        # -- Add time information in dataset
        ds['begin_time'] = xr.DataArray([begin_time], dims=['time'])
        ds['end_time'] = xr.DataArray([end_time], dims=['time'])
        ds['exact_time'] = xr.DataArray([exact_time], dims=['time'])

        file.close()
        return ds

    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    description = "Use GRACEL2 product files in xarray"
