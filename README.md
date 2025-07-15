# Getting started with LENAPY

## Documentaion

Documentation can be found [here](https://lenapy.readthedocs.io/en/latest/)

## Installation


To install Lenapy, follow the steps :


* Create a conda environment and activate it ::
    ```
    conda create --name lenapy python=3.11
    conda activate lenapy
    ```
 
* Install lenapy from conda-forge ::

    ```
    conda install -c conda-forge lenapy
    ```
    
## Use

In your python code, simply import the library with ::

  ```
  import lenapy
  ```
  
You can now use all the functionalities of lenapy by adding the right suffix after your Dataset or DataArray ::
  ```
  import xarray as xr
  import lenapy
  ds = lenapy.utils.geo.rename_data(xr.tutorial.open_dataset('air_temperature'))
  ds.air.lngeo.mean(['latitude','longitude'],weights=['latitude']).lntime.climato().plot()
  ```
  
# Contributors

We appreciate the contributions from our community. A complete list of contributors can be found in the [CONTRIBUTORS.md](https://github.com/CNES/lenapy/blob/main/CONTRIBUTORS.md) file.
