from setuptools import setup

setup(
    name="lenapy",
    version="0.9",
    description="Second beta release of lenapy on github",
    url="#",
    author="Sebastien Fourest & EMC2 team",
    author_email="sebastien.fourest@cnes.fr",
    license="GPL3.0",
    packages=[
        "lenapy",
        "lenapy.utils",
        "lenapy.readers",
        "lenapy.writers",
        "lenapy.plots",
        "lenapy.resources",
    ],
    install_requires=[
        "matplotlib>=3.6",
        "esmpy>=8.4.0",
        "xesmf>=0.8.2",
        "xarray>=2024.2",
        "gsw>=3.6.16",
        "netCDF4>=1.6.5",
        "pyyaml>=6.0",
        "dask>=2023.6",
    ],
    zip_safe=False,
    entry_points={
        "xarray.backends": [
            "lenapyNetcdf = lenapy.readers.geo_reader:lenapyNetcdf",
            "lenapyMask   = lenapy.readers.geo_reader:lenapyMask",
            "lenapyOceanProducts = lenapy.readers.ocean:lenapyOceanProducts",
            "lenapyGfc    = lenapy.readers.gravi_reader:ReadGFC",
            "lenapyGraceL2 = lenapy.readers.gravi_reader:ReadGRACEL2",
            "lenapyShLoading = lenapy.readers.gravi_reader:ReadShLoading",
        ]
    },
    package_data={"lenapy.resources": ["*"]},
    include_package_data=True,
)
