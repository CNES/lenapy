import warnings
from importlib.metadata import version as _version

from lenapy import constants, lenapy_geo, lenapy_harmo, lenapy_time

try:
    __version__ = _version("lenapy")
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "9999"

try:
    from lenapy import lenapy_ocean
except Exception:
    warnings.warn("To use lenapy_ocean, please install gsw : pip install gsw>=3.6.16")
import cf_xarray as cfxr

from lenapy.readers import geo_reader, ocean

criteria = {
    "depth": {"name": "Depth|depth_std|LEVEL|level"},
    "longitude": {"name": "lon|LON|Longitude|LONGITUDE"},
    "latitude": {"name": "lat|LAT|Latitude|LATITUDE"},
    "time": {"name": "date|dates|TIME|Time"},
}
cfxr.set_options(custom_criteria=criteria)
