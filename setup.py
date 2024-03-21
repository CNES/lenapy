from setuptools import setup
setup(name='lenapy',
version='0.7',
description='First release of new lenapy architecture',
url='#',
author='Sebastien Fourest & EMC2 team',
author_email='sebastien.fourest@cnes.fr',
license='APACHE2',
packages=['lenapy','lenapy.utils','lenapy.readers','lenapy.plots'],
install_requires=['matplotlib>=3.8.3','esmpy>=8.6.0','xesmf>=0.8.4','xarray>=2024.2','gsw>=3.6.16','netCDF4>=1.6.5','pyyaml>=6.0','dask>=2023.6ccd ..
                  '],      
zip_safe=False,
entry_points={"xarray.backends": ["lenapyNetcdf = lenapy.readers.geo_reader:lenapyNetcdf",
                                  "lenapyMask   = lenapy.readers.geo_reader:lenapyMask",
                                  "lenapyOceanProducts = lenapy.readers.ocean:lenapyOceanProducts"]},
     ) 
