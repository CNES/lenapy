# Getting started with LENAPY

## Installation


To install Lenapy, follow the steps :

* Download lenapy code ::

    ```
    git clone git@github.com:CNES/lenapy.git
    ```

* Create a conda environment and activate it ::
    ```
    conda create --name lenapy python=3.11
    conda activate lenapy
    ```
 
* Manually install esmpy ::

    ```
    conda install -c conda-forge esmpy
    ```

* Go into the lenapy source directory and install it ::

    ```
    pip install . 
    ```
    
## Use

In your python code, simply import the library with ::

  ```
  import lenapy
  ```
  
You can now use all the functionnalities of lenapy by adding the right suffixe after your Dataset or DataArray ::
  ```
  import xarray as xr
  import lenapy
  ds = lenapy.utils.geo.rename_data(xr.tutorial.open_dataset('air_temperature'))
  ds.air.lngeo.mean(['latitude','longitude'],weights=['latitude']).lntime.climato().plot()
  ```
  
  
