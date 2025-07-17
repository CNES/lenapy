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
    affiliation: "1"
  - name: Sebastien Fourest
    orcid: 0009-0001-3123-1026
    affiliation: "1, 2"
  - name: Alejandro Blazquez
    orcid: 0000-0002-7719-7468
    affiliation: "1, 2"
affiliations:
 - name: "Université de Toulouse, LEGOS (CNES/CNRS/IRD/UT3), France"
   index: 1
   ror: 02chvqy57
 - name: "Centre national d'Études Spatiales (CNES), France"
   index: 2
   ror: 04h1h0y33
date: 10 July 2025
bibliography: paper.bib
---

# Summary

`Lenapy` is a Python library designed to facilitate the processing and analysis of geophysical and climate datasets, such as those used in oceanography, geodesy, and Earth observation in general. 
Built on top of `xarray` and fully compatible with `dask`, it enables scalable and reproducible workflows on multidimensional datasets by leveraging community standards for data formats (NetCDF) and metadata conventions (CF).

`Lenapy` provides high-level accessors that extend `xarray.Dataset` and `xarray.DataArray` objects, allowing direct application of specialized methods.

One of `Lenapy`’s key features is its unified approach for Global Mean Sea Level (GMSL) estimation, enabling consistent computation from steric, manometric, or relative sea level components.

`Lenapy` is intended to support Earth system scientists, oceanographers, and climate service providers with unified and coherent tools through an extensible and transparent interface.



# Statement of need

Geophysical and climate datasets are intrinsically multidimensional, often encompassing spatial (longitude, latitude, depth/height) and temporal dimensions.
Analyzing these data requires specialized operations such as harmonic analysis, climatology, thermodynamics, and geodetic corrections.
For the oceanography part, the accurate computation of seawater properties is essential in physical oceanography.
The Thermodynamic Equation of Seawater (TEOS-10) framework provides consistent definitions and algorithms for quantities such as potential temperature, conservative temperature, and density [@Mcdougall_2011].

Existing libraries address specific aspects of these requirements:

- `pyshtools` [@Wieczorek_2018] is a comprehensive package for spherical harmonic transforms and spectral analysis, particularly in geophysics. Yet, it operates on standalone arrays and does not natively support xarray, limiting its compatibility with NetCDF-based workflows.
- `grates` [@Kvas] provides object-oriented tools for spherical harmonics and geodetic computations, but similarly lacks integration with labeled multidimensional data structures like xarray.
- `gsw-xarray` [@Caneill_2024] provides implementations of the TEOS-10, facilitating oceanographic computations. It offers a wrapper around `GSW-Python` for xarray objects, but without Dask support.

While `gsw-xarray` is more complete than our GSW wrapper in `Lenapy`, our library propose complementary geodetic tools for spatial and spherical harmonics operations.

To our knowledge, no other existing Python library offers a coherent suite of oceanographic and geophysical operations (detailed below) within a unified, xarray-native framework supporting both scalability (via Dask [@Dask_2016]) and labeled, multidimensional arrays.
Moreover, critical geospatial utilities (such as surface-aware averaging, weighted statistics, spherical distance computation, and climatology fitting) remain fragmented across ecosystems or require custom implementations.

`Lenapy` addresses this gap by providing a modular Python package built on xarray and Dask, exposing accessors (.lngeo, .lnharmo, .lnocean, .lntime) for direct and simple application of domain-specific methods to xarray.Dataset or xarray.DataArray objects. 
For example, users can compute area-weighted means via ds.lngeo.mean() or extract the global ocean heat content using ds.lnocean.gohc().

`Lenapy` is designed for Earth scientists, oceanographers, climate researchers, and geodesists who routinely manipulate global or regional gridded datasets and require specific processing workflows.
`Lenapy` aims to maintain full compatibility with the PyData ecosystem.

Furthermore, `Lenapy` offers a unified approach for calculating Global Mean Sea Level (GMSL) by integrating both steric and manometric components, as well as relative sea-level changes [@Gregory_2019].
`Lenapy` facilitates this decomposition by providing a unique Python library that compute these components directly from xarray-based datasets, enabling researchers to analyze sea-level changes comprehensively within a consistent framework.


# Key Features
## Spatial operations (`.lngeo`)

The `lngeo` accessor provides geodetic tools designed for gridded data on spherical or ellipsoidal Earth models. It includes:

- Geodetic estimation of grid cell surface areas and distances;
- Support for geographical weighted operations (e.g., mean or sum) based on grid cell surface areas;
- Isosurface computation;
- A wrapper to the `xESMF` regridding library [@Zhuang_2022], enabling seamless interpolation between different spatial grids with xarray compatibility.

## Spherical harmonics operations and gravity field processing (`.lnharmo`)

The `lnharmo` accessor offers dedicated methods for working with spherical harmonic representations, particularly in the context of Earth gravity field modeling. It includes:

- Reading, handling, and manipulating datasets containing spherical harmonic coefficients (variables `clm`, `slm`), with options to change reference frames or tide systems;
- Converting spherical harmonic representations into gridded spatial fields;
- Inverse transformation: estimation of spherical harmonic components from a gridded spatial field.

## Oceanography (`.lnocean`)

The `lnocean` accessor provides a lightweight wrapper around selected GSW (TEOS-10) routines [@Mcdougall_2011], exposing them as native xarray methods for oceanographic datasets. 
Based on any dataset containing :
- any of insitu, conservative or potential temperature
- any of practical, relative or absolute salinity
- depth coordinate

it provides through the .lncoean extension direct access to selected GSW routines, and integrated values over all or part of the water column, including :
- Ocean heat content;
- Steric sea levels;
- Density and dynamic height;

## Time series and climatology tools (`.lntime`)

The `lntime` accessor enables common temporal operations on geophysical time series, including:

- Climatological signal extraction and fitting (e.g., seasonal decomposition);
- Filtering of time series;
- Polynomial or harmonic regressions over specified periods;
- Detrending, interpolation, derivating, evaluation of missing values,...

## Input/Output utilities

`Lenapy` includes I/O helpers to support multiple data formats used in Earth observation and geoscience:

- Readers for gridded ocean temperature/salinity products from various sources;
- Parsers for gravity field spherical harmonics formats, including Gravity Recovery and Climate Experiment (GRACE) and GRACE Follow-On Science Data System files as well as ICGEM-format files `.gfc`;
- Writers for ICGEM-format, enabling export of custom spherical harmonics to interoperable standards.

# Projects using Lenapy

`Lenapy` is actively used in several international research projects and operational workflows focused on Earth system science and climate monitoring. Its modular design and compatibility with the scientific Python ecosystem make it particularly suitable for large-scale, multidimensional data processing.

Notable projects include:

- **Sea Level Budget Closure CCI+ (SLBC_CCI+)**, funded by the European Space Agency (ESA), which uses `Lenapy` in several work packages of the project for computing steric and manometric contributions to sea level and closing the global sea level budget.
- **ERC Synergy GRACEFUL**, a European Research Council project dedicated to improving the understanding of Earth’s interior mass redistribution using gravimetric data.

In addition, `Lenapy` has been employed in related research studies [@Bouih_2025] for consistent and reproducible treatment of steric, manometric, or relative sea level.
`Lenapy` gravity field and spherical harmonics operation are used at LEGOS for the processing GRACE and GRACE Follow-On Level 2 dataset to create a Level 3 time-variable gravity solution [@Blazquez_2018].

## Origin and research context

`Lenapy` originated from a set of geophysical tools developed within the LEGOS research laboratory during PhD works focused on the regional and global variability of sea level and the characterization of ocean–continent water exchanges [@Meyssignac_2012, @Dieng_2017, @Blazquez_2020]. 
These early tools have since been generalized and integrated into a modern Python framework, making them more accessible and reusable by the broader Earth science community.

# Acknowledgements

Lenapy benefited of CNES support to improve its maintainability and its documentation.

We thank the contributors to the early version of `Lenapy`: Robin Guillaume-Castel, Arthur Vincent.

# References
