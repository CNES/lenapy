"""

Modules
^^^^^^^
Accessor modules

.. autosummary::
    :toctree:

    lenapy_geo
    lenapy_time
    lenapy_ocean
    lenapy_harmo
"""

import warnings

from lenapy import constants, lenapy_geo, lenapy_harmo, lenapy_time

try:
    from lenapy import lenapy_ocean
except:
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
