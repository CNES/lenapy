from setuptools import setup
setup(name='lenapy',
version='0.6.0',
description='',
url='#',
author='Sebastien Fourest & EMC2 team',
author_email='',
license='MIT',
packages=['lenapy'],
install_requires=['matplotlib>=3.7.0','esmpy>=8.4.0','xesmf>=0.7','xarray>=2023','gsw==3.6.17','netCDF4>=1.6.4','pyyaml','dask'],      
zip_safe=False) 
