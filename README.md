# LENAPY

## Description
LENAPY is a set of unctions which are an extension of the xarray DataSet and DataArray. All funcionts can be used by adding the extension .xgeo or .xocean after the object's name. Three main modules are implemented :
 * xGeo : for all classical operations on lat/lon grids used in geodesy, altimetry
 * xOcean : implementation of the gsw library


## Installation
Clone the git repository in a dedicated environment, and simply type ``pip install .`` in the lenapy directory.
To clone the repository, execute the following command :
 * From hal :
 ``git clone git@gitlab.cnes.fr:campus/lenapy.git``
 * From an external environment :
 ``git clone gu=<login>@git@gitlab-ssh.cnes.fr:campus/lenapy.git``

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

