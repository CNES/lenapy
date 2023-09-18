# LENAPY

## Description
LENAPY is a set of unctions which are an extension of the xarray DataSet and DataArray. All functions can be used by adding the extension .xgeo or .xocean after the object's name. Two main modules are implemented :
 * xGeo : for all classical operations on lat/lon grids used in geodesy, altimetry
 * xOcean : implementation of the gsw library


## Installation
To install and add lenapy to an existing virtual environment, an external library (esmpy) has to be manually installed :
 * Activate the conda environment :
 ``conda activate my_env``
 * Install esmpy library
 ``conda install -c conda-forge esmpy=8.1.0``
 * Clone lenapy gitlab repository
  - From hal or trex:
 ``git clone git@gitlab.cnes.fr:campus/lenapy.git``
  - From an external environment :
 ``git clone gu=<login>@git@gitlab-ssh.cnes.fr:campus/lenapy.git``
 * Install lenapy from the downloaded lenapy repository
 ``pip intall .``
 
## Usage
Just import the lenapy module at the beggining of the code :
``
import lenapy
``

To use any implemented function, add .xgeo or .xocean after the data name. Repeat this prefix each time you want to use a lenapy function (important for one-liner commands)
Ex:
data.xgeo.mean(['latitude','longitude'],weights=['latitude']).xgeo.climato()

## Full documentation is [here](doc/index.html)


## Support
sebastien.fourest@cnes.fr

