# Getting started with LENAPY

## Documentation

Documentation can be found [here](https://lenapy.readthedocs.io/en/latest/)

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

Once Conda is installed, you can install lenapy by following these steps:

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

This installs lenapy and its dependencies from the conda-forge channel.

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
  
  
