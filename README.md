# Getting started with LENAPY

`Lenapy` is a Python library designed to facilitate the processing and analysis of geophysical and climate datasets, such as those used in oceanography, geodesy, and Earth observation in general.
Built on top of `xarray` and compatible with `dask`, it enables scalable workflows on multidimensional datasets using community standards for data formats (NetCDF) and metadata conventions (CF).

A key feature of `lenapy` is its ability to produce consistent computations of the Global Mean Sea Level components from various data sources, including satellite altimetry, GRACE/GRACE-FO, thermo-steric datasets, and climate model outputs.

## Documentation

Documentation can be found at [https://lenapy.readthedocs.io/en/latest/](https://lenapy.readthedocs.io/en/latest/).

## Installation


### Step 1: Install Conda

To use Conda packages like `lenapy`, you first need to install **Miniconda** or **Anaconda**. Miniconda is a lightweight option, while Anaconda includes many preinstalled data science packages.

**To install Miniconda (recommended for minimal setup):**

1. Go to the [Miniconda download page](https://docs.conda.io/en/latest/miniconda.html).
2. Download the installer for your operating system (Windows, macOS, or Linux).
3. Run the installer and follow the prompts. Make sure to:
   - Add Conda to your system’s PATH (if asked).
   - Allow the installer to initialize Conda automatically.

After installation, open a terminal (or Anaconda Prompt on Windows) and verify that Conda is installed by running:

```bash
conda --version
```

### Step 2: Install lenapy Using Conda

Once Conda is installed, you can install `lenapy` by following these steps:

#### 1. Create a new Conda environment (recommended):
```bash
conda create -n lenapy_env python=3.11
```

#### 2. Activate the environment:
```bash
conda activate lenapy_env
```

#### 3. Install lenapy:
```bash
conda install -c conda-forge lenapy
```

This installs `lenapy` and its dependencies from the conda-forge channel.

## Use

In your python code, simply import the library with ::

  ```
  import lenapy
  ```
  
You can now use all the functionalities of `lenapy` by adding the right suffix after your Dataset or DataArray ::
  ```
  import xarray as xr
  import lenapy
  ds = lenapy.utils.geo.rename_data(xr.tutorial.open_dataset('air_temperature'))
  ds.air.lngeo.mean(['latitude','longitude'],weights=['latitude']).lntime.climato().plot()
  ```
More complete notebooks tutorials and functions descriptions can be found in the documentation.

[Notebooks tutorials](https://lenapy.readthedocs.io/en/latest/tutorials.html)

[Functions reference](https://lenapy.readthedocs.io/en/latest/api/index.html)

## Contributing
We welcome contributions to `lenapy`!

Guidelines: [https://lenapy.readthedocs.io/en/latest/contributing.html](https://lenapy.readthedocs.io/en/latest/contributing.html)
