.. lenapy documentation master file, created by
   sphinx-quickstart on Thu Mar 14 16:52:33 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to lenapy's documentation!
==================================

**Lenapy** is a python library based on xarray with several modules:

* lenapy_ocean: encapsulation of the GSW library containing the main ocean thermodynamic calculation routines (TEOS10)
* lenapy_geo: current operations on gridded data (longitude/latitude/depth)
* lenapy_time: common operations on time series (filtering, climatology, regressions, etc.)
* lenapy_harmo: operations related to spherical harmonics decomposition and their projections on latitude/longitude grids

This library is based on the principle of class extension, i.e. functions can be applied directly to Datasets or DataArrays, by adding the extension of the module concerned and the name of the method. Ex: ds.lntime.filter(...), ds.lngeo.mean(...), de.lnocean.gohc, etc.

Reading interfaces are implemented, enabling netcdf files to be opened with a compatible formalism from the **lenapy** library.

.. toctree::
   :maxdepth: 4
   :caption: Contents:
   
   changelog
   gettingStarted
   api
   tutorials



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
