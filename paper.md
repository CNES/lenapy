---
title: "Lenapy: Enhancing xarray for multidimensional geophysical data analysis"
tags:
  - Python
  - xarray
  - geophysics
  - oceanography
  - spherical harmonics
  - gravity field
authors:
  - name: Hugo Lecomte
    orcid: 0000-0002-7007-4748
    affiliation: "1, 2"
  - name: Sebastien Fourest
    orcid: 0009-0001-3123-1026
    affiliation: "1, 3"
  - name: Alejandro Blazquez
    orcid: 0000-0002-7719-7468
    affiliation: "1, 3"
affiliations:
 - name: "Université de Toulouse, LEGOS (CNES/CNRS/IRD/UT3), France"
   index: 1
   ror: 02chvqy57
 - name: "Finnish Geospatial Research Institute, National Land Survey of Finland, Finland"
   index: 2
   ror: 05tkycb73
 - name: "Centre national d'Études Spatiales (CNES), France"
   index: 3
   ror: 04h1h0y33
date: 10 July 2025
bibliography: paper.bib
---

# Summary

`Lenapy` is a Python library designed to facilitate the processing and analysis of geophysical and climate datasets, such as those used in oceanography, geodesy, and Earth observation in general. 
Built on top of `xarray` and fully compatible with `Dask`, it enables scalable workflows on multidimensional datasets using community standards for data formats (NetCDF) and metadata conventions (CF).

`Lenapy` provides high-level accessors that extend `xarray.Dataset` and `xarray.DataArray` objects, allowing application of specialized methods.

One of `Lenapy`’s key features is its unified approach for Global Mean Sea Level estimation, enabling consistent computation of its components.

`Lenapy` is intended to support geophysicists, oceanographers, and climate service providers with coherent tools through an extensible interface.



# Statement of need

Geophysical and climate datasets are intrinsically multidimensional, often encompassing spatial (longitude, latitude, depth/height) and temporal dimensions.
Analyzing these data requires specialized operations such as harmonic analysis, climatology, thermodynamics, and geodetic corrections.
For the oceanography part, the accurate computation of seawater properties is essential in physical oceanography.
The Thermodynamic Equation of Seawater (TEOS-10) framework provides consistent definitions and algorithms for quantities such as potential temperature, conservative temperature, and density [@Mcdougall_2011].

`Lenapy` is designed for Earth scientists, oceanographers, climate researchers, and geodesists who routinely manipulate global or regional gridded datasets and require specific processing workflows.
`Lenapy` aims to maintain full compatibility with the PyData ecosystem.

Furthermore, `Lenapy` offers a unified approach for calculating Global Mean Sea Level by integrating both steric and manometric components, as well as relative sea-level changes [@Gregory_2019].
`Lenapy` facilitates this decomposition with a unique Python library that compute these components directly from xarray-based datasets, enabling researchers to analyze sea-level changes comprehensively within a consistent framework.

# State of the field
Existing Python libraries address specific needs in geophysical and climate data analysis, but often in a fragmented or domain-specific manner.
`pyshtools` [@Wieczorek_2018] and `grates` [@Kvas] offer tools for gravity spherical harmonic analysis.
They operate on standalone arrays and lack support for `xarray`. `windspharm` [@Dawson_2016] implements `xarray` spherical harmonics analysis but without the gravity and geodetic related operations.
`gsw-xarray` [@Caneill_2024] wraps the TEOS-10 framework for oceanographic computations within `xarray`, but does not support `Dask`.
While `Lenapy` GSW wrapper is less complete than `gsw-xarray`, it proposes complementary geodetic tools for spatial and spherical harmonics operations.

To our knowledge, no other existing Python library offers a coherent suite of oceanographic and geophysical operations (detailed below) within a unified, xarray-native framework supporting both scalability (via `Dask` [@Dask_2016]) and labeled, multidimensional arrays.
Moreover, critical geospatial utilities (such as surface-aware averaging, weighted statistics, spherical distance computation, and climatology fitting) remain fragmented across ecosystems or require custom implementations.

# Key Features
`Lenapy` addresses this gap by providing a modular Python package built on `xarray` and `Dask`, exposing accessors (.lngeo, .lnharmo, .lnocean, .lntime) for direct and simple application of domain-specific methods to xarray.Dataset or xarray.DataArray objects. 
For example, users can compute area-weighted means via `ds.lngeo.mean()` or extract the global ocean heat content using `ds.lnocean.gohc()`.


## Spatial operations (`.lngeo`)

The `lngeo` accessor provides geodetic tools designed for gridded data on spherical or ellipsoidal Earth models, including:

- Geodetic estimation of grid cell surface areas and distances and geographical weighted operations (e.g., mean or sum);
- Isosurface computation;
- A wrapper to the `xESMF` regridding library [@Zhuang_2025].

## Spherical harmonics operations and gravity field processing (`.lnharmo`)

The `lnharmo` accessor offers dedicated methods for working with spherical harmonic representations, particularly in the context of Earth gravity field modeling, including:

- Reading, handling, and manipulating datasets containing spherical harmonic coefficients (variables `clm`, `slm`), with options to change reference frames;
- Converting spherical harmonic representations into gridded spatial fields and inverse transformation.

## Oceanography (`.lnocean`)

The `lnocean` accessor provides a lightweight wrapper around selected GSW (TEOS-10) routines [@Mcdougall_2011], exposing them as native `xarray` methods for oceanographic datasets. 
Based on any dataset containing any temperature or any salinity with depth coordinate it provides integrated values over all or part of the water column, including:

- Ocean heat content;
- Steric sea levels;
- Density and dynamic height.

## Time series and climatology tools (`.lntime`)

The `lntime` accessor enables common temporal operations on geophysical time series, including:

- Climatological and polynomial signal extraction and fitting (e.g., seasonal decomposition);
- Filtering of time series;
- Detrending, interpolation, derivation, evaluation of missing values, ...

## Input/Output utilities

`Lenapy` includes I/O helpers to support multiple data formats used in Earth observation and geoscience:

- Readers for gridded ocean temperature/salinity products from various sources;
- Readers for spherical harmonics formats, including Gravity Recovery and Climate Experiment (GRACE) and GRACE Follow-On Science Data System files and ICGEM-format files `.gfc`;
- Writers for ICGEM-format, enabling export of custom spherical harmonics to interoperable standards.

# Projects using Lenapy

`Lenapy` is actively used in several international research projects and operational workflows focused on Earth system science and climate monitoring.

Notable projects include:

- **Sea Level Budget Closure CCI+ (SLBC_CCI+)**, funded by the European Space Agency, uses `Lenapy` in several work packages for computing steric and manometric contributions to sea level and closing the global sea level budget.
- **ERC Synergy GRACEFUL**, a European Research Council project dedicated to improving the understanding of Earth’s interior using gravimetric data.

In addition, `Lenapy` has been employed in related research studies [@Bouih_2025] for consistent and reproducible treatment of steric, manometric, or relative sea level.
`Lenapy` gravity field and spherical harmonics operations are used at LEGOS for the processing GRACE (& Follow-On) Level 2 dataset to create a Level 3 gravity solution [@Blazquez_2018].

## Origin and research context

`Lenapy` originated from geophysical tools developed within the LEGOS research laboratory during PhD works focused on the variability of sea level and ocean–continent water exchanges [@Meyssignac_2012; @Dieng_2017; @Blazquez_2020]. 
These early tools have been generalized and integrated into a modern Python framework.

# Acknowledgements

Lenapy benefited of CNES support to improve its maintainability and its documentation.

We thank Robin Guillaume-Castel and Arthur Vincent for their help with the early code.

# References
