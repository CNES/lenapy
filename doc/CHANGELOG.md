# Changelog

All notable changes to **LENAPY** will be documented in this file.

## [0.9] - 2024-12-17
Introduction of the lenapyHarmo module

### Enhancements

#### Lenapy Harmo
The lenapy_harmo module provides functionalities for spherical harmonics dataset (clm/slm) and
projections on latitude/longitude grids.

This module includes two classes:

    HarmoSet: Provides methods for handling spherical harmonics decompositions, converting them to grid representations
    and performing operations such as changing reference frames and tide systems.
    HarmoArray: Converts grid representation of a spherical harmonics dataset back to spherical harmonics.

The module is designed to work seamlessly with xarray datasets, enabling efficient manipulation and visualization.

#### lntime.climato
Refactoring of the Climatology computation with new functionnalities 

### Changes

### Bugfixes
