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
from . import constants, lenapy_geo, lenapy_time, lenapy_harmo
try:
    from . import lenapy_ocean
except:
    warnings.warn("To use lenapy_ocean, please install gsw : pip install gsw>=3.6.16")
from lenapy.readers import geo_reader, ocean

import cf_xarray as cfxr

criteria = {
    "depth": {"name": 'Depth|depth_std|LEVEL|level'},
    "longitude": {"name": 'lon|LON|Longitude|LONGITUDE'},
    "latitude": {"name": 'lat|LAT|Latitude|LATITUDE'},
    "time": {"name":'date|dates|TIME|Time'}
}
cfxr.set_options(custom_criteria=criteria)
