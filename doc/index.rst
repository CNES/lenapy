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

Scientific datasets in Earth observation and geophysics are often complex to process due to their multidimensional structure, diversity of formats, and the physical corrections required for proper interpretation.
While libraries like xarray and netCDF4 offer powerful tools to handle n-dimensional arrays, they do not natively support specific operations such as climatological decomposition, harmonic analysis, geodetic corrections, or thermodynamics computations.

**Lenapy** is designed to serve Earth scientists, oceanographers, geodesists, hydrologists, and climate researchers who work with global or regional observational datasets and require specific processing without reinventing the wheel.
It combines clarity, scientific rigor, and compatibility with the broader PyData ecosystem.

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :hidden:

   gettingStarted
   Changelog <changelog>
   tutorials
   lenapy API Reference <api/index>
   How to contribute <contributing>
   Release procedure <release_procedure>
