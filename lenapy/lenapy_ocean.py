# -*- coding: utf-8 -*-
"""This module implements some useful TEOS10 gsw functions on oceanic dataset

"""

import gsw
import numpy as np
import xarray as xr

from lenapy.constants import *


def proprietes(da, nom, label, unite):
    out = da.rename(nom)
    out.attrs["long_name"] = label
    out.attrs["units"] = unite
    return out


def NoneType(var):
    return type(var) == type(None)


@xr.register_dataset_accessor("lnocean")
class OceanSet:
    """
    This class extends any dataset to a xOcean object, that allows to access to any TEOS10 variable simply by calling the name of the variable through the xOcean interface.
    The initial dataset must contains the fields necessary to compute the output variable (ex : temperature and salinity to compute heat, heat to compute ohc,...)

    Fields
    ------
    Temperatures : one of the three types of temperature must be present in the original dataset to perform derived computation.
        * temp  : in-situ temperature
        * PT    : potential temperature
        * CT    : conservative temperature

    Salinities :  one of the three types of salinities must be present in the original dataset to perform derived computation.
        * psal  : practical salinity
        * SR    : relative salinity
        * SA    : absolute salinity. If there is no location information (lat,lon), absolute salinity is returned equal to relative salinity

    Physical properties :
        * P     : pressure. If location information is present, pressure is adjusted with regard to latitude, otherwise latitude is equal to 0
        * Cp    : heat capacity
        * rho   : density
        * sigma0: potential density anomaly at 0 dbar

    Heat content :
        * heat  : specific heat content (J/m3)
        * ohc   : local ocean heat content (J/m²), it is heat integrated over the whole ocean depth
        * gohc  : global ocean heat content (J/m²), it is ohc averaged over latitude-longitude, excluding continents
        * gohc_TOA: idem gohc, including continents (where ohc=0)
        * ohc_above: idem ohc, where heat is integrated above a given depth
        * gohc_above: idem gohc, averaging ohc_above instead of ohc

    Sea level :
        * slh   : steric sea layer height anomaly (-), equal to (1. - rho/rhoref)
        * ssl   : steric sea surface level anomaly (m), it is slh integrated over the whole ocean depth
        * ssl_above : idem ssl, where heat is integrated above a given depth
        * tssl  : thermosteric sea surface level (m)
        * hssl  : halosteric sea surface level (m)
        * ieeh  : integrated expansion efficiency oh heat (m/(J/m²)), it is (tssl/ohc)

    Layer depth :
        * ocean_depth  : maximum depth with non Nan values for temperature
        * mld_theta0   : ocean mixed layer depth, defined by a temperature drop from 0.2°C wrt to -10m depth
        * mld_sigma0   : ocean mixed layer depth, defined by a potential density increase of 0.03kg/m3 wrt to -10m depth
        * mld_sigma0var: ocean mixed layer depth, defined by a potential density equal to the potential density at -10m depth with a temperature dropped by 0.2°C

    Example
    -------
    .. code-block:: python

        data=xr.open_dataset('/home/usr/lenapy/data/IAP',engine='lenapyOceanProduct',product='IAP')
        print(data)
        <xarray.Dataset>
        Dimensions:    (latitude: 180, longitude: 360, time: 156, depth: 41)
        Coordinates:
          * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
          * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0
          * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
          * depth      (depth) float32 1.0 5.0 10.0 20.0 ... 1.7e+03 1.8e+03 2e+03
        Data variables:
            temp       (time, latitude, longitude, depth) float32 dask.array<chunksize=(1, 180, 360, 10), meta=np.ndarray>
            SA         (time, latitude, longitude, depth) float32 dask.array<chunksize=(1, 180, 360, 10), meta=np.ndarray>
        mld=data.lnocean.mld_sigma0
        print(data.lnocean.ohc_above(mld))
        <xarray.DataArray 'ohc_above' (time: 156, latitude: 180, longitude: 360)>
        dask.array<where, shape=(156, 180, 360), dtype=float64, chunksize=(1, 180, 360), chunktype=numpy.ndarray>
        Coordinates:
            depth      (time, latitude, longitude) float64 dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
          * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
          * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
          * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0
        Attributes:
            long_name:  Ocean heat content
            units:      J/m^2
        print(data.lnocean.gohc)
        <xarray.DataArray 'gohc' (time: 156)>
        dask.array<truediv, shape=(156,), dtype=float64, chunksize=(1,), chunktype=numpy.ndarray>
        Coordinates:
          * time     (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
            depth    float32 1.0
        Attributes:
            long_name:  Global ocean heat content wrt to ocean surface area
            units:      J/m^2

    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        fields = [
            "temp",
            "PT",
            "CT",
            "psal",
            "SA",
            "SR",
            "P",
            "rho",
            "sigma0",
            "Cp",
            "heat",
            "slh",
            "ohc",
            "ssl",
            "tssl",
            "hssl",
            "ieeh",
            "gohc",
            "eeh",
            "ocean_depth",
            "mld_theta0",
            "mld_sigma0",
            "mld_sigma0var",
        ]
        for f in fields:
            if hasattr(xarray_obj, f):
                setattr(self, f + "_", xarray_obj[f])
            else:
                setattr(self, f + "_", None)

        if NoneType(self.heat_):
            if NoneType(self.temp_) and NoneType(self.CT_) and NoneType(self.PT_):
                raise ValueError(
                    "At least one temperature must be set (temp, CT or PT)"
                )
            if NoneType(self.SA_) and NoneType(self.SR_) and NoneType(self.psal_):
                raise ValueError("At least one salinity must be set (psal, SA or SR)")

        self.oml_theta0_threshold = 0.2
        self.oml_sigma0_threshold = 0.03

    @property
    def temp(self):
        if NoneType(self.temp_):
            self.temp_ = proprietes(
                gsw.t_from_CT(self.SA, self.CT, self.P),
                "temp",
                "In-situ temperature",
                "dgeC",
            )
        return self.temp_

    @property
    def PT(self):
        if NoneType(self.PT_):
            self.PT_ = proprietes(
                gsw.pt0_from_t(self.SA, self.temp, self.P),
                "pt",
                "Potential temperature",
                "degC",
            )
        return self.PT_

    @property
    # Temperature conservative en fonction de la salinité absolue, temperature in-situ, et pression
    def CT(self):
        if NoneType(self.CT_):
            self.CT_ = proprietes(
                gsw.CT_from_pt(self.SA, self.PT),
                "CT",
                "Conservative temperature",
                "degC",
            )  # degC
        return self.CT_

    @property
    # Salinité pratique
    def psal(self):
        if NoneType(self.psal_):
            if NoneType(self.SA_) or not (
                "latitude" in self._obj.variables and "longitude" in self._obj.variables
            ):
                self.psal_ = proprietes(
                    gsw.SP_from_SR(self.SR), "psal", "Practical salinity", "g/kg"
                )  # [g/kg]

            else:
                self.psal_ = proprietes(
                    gsw.SP_from_SA(
                        self.SA, self.P, self._obj.longitude, self._obj.latitude
                    ),
                    "psal",
                    "Practical salinity",
                    "g/kg",
                )  # [g/kg]

        return self.psal_

    @property
    # Salinité relative, en fonction de la salinité pratique
    def SR(self):
        if NoneType(self.SR_):
            self.SR_ = proprietes(
                gsw.SR_from_SP(self.psal), "SR", "Relative salinity", "g/kg"
            )  # [g/kg]
        return self.SR_

    # Salinité absolue, en fonction de la salinité relative, la pression, et la position
    @property
    def SA(self):
        if NoneType(self.SA_):
            if "latitude" in self._obj.variables and "longitude" in self._obj.variables:
                self.SA_ = proprietes(
                    gsw.SA_from_SP(
                        self.psal, self.P, self._obj.longitude, self._obj.latitude
                    ),
                    "SA",
                    "Absolute salinity",
                    "g/kg",
                )  # [g/kg]
            else:
                self.SA_ = self.SR
        return self.SA_

    @property
    # Pression en fonction de la profondeur et de la latitude
    def P(self):
        if NoneType(self.P_):
            if "latitude" in self._obj.variables:
                self.P_ = proprietes(
                    gsw.p_from_z(self._obj.depth * -1, self._obj.latitude),
                    "p_db",
                    "Pressure",
                    "dbar",
                )
            else:
                self.P_ = proprietes(
                    gsw.p_from_z(self._obj.depth * -1, 0), "p_db", "Pressure", "dbar"
                )
        return self.P_

    @property
    # Densité en fonction de la salinité absolue, la température conservative et la pression
    def rho(self):
        if NoneType(self.rho_):
            self.rho_ = proprietes(
                gsw.rho(self.SA, self.CT, self.P), "rho", "Density", "kg/m3"
            )  # [kg/m3]
        return self.rho_

    @property
    # Anomalie de densité potentielle à 0dbar en fonction de la salinité absolue et la température conservative
    def sigma0(self):
        if NoneType(self.sigma0_):
            self.sigma0_ = proprietes(
                gsw.sigma0(self.SA, self.CT),
                "sigma0",
                "Potential Density Anomaly",
                "kg/m3",
            )  # [kg/m3]
        return self.sigma0_

    @property
    # Capacité calorifique en fonction de la salinité absolue, la température in-situ et la pression
    def Cp(self):
        if NoneType(self.Cp_):
            self.Cp_ = proprietes(
                gsw.cp_t_exact(self.SA, self.temp, self.P),
                "Cp",
                "Specific heat capacity",
                "J/kg/degC",
            )  # [j/kg/degC)]
        return self.Cp_

    @property
    # Contenu en chaleur des couches en fonction de la densité, la capacité calorifique, et la température conservative
    def heat(self):
        if NoneType(self.heat_):
            self.heat_ = proprietes(
                (self.rho * self.Cp * self.CT), "Heat", "Heat content", "J/m3"
            )  # [J/m3]
        return self.heat_

    @property
    # Anomalie relative de densité par rapport à une référence (CT=0°C,SA=35.16404psu)
    def slh(self):
        if NoneType(self.slh_):
            rhoref = gsw.rho(LNPY_SSO, 0.0, self.P)
        return proprietes(
            ((1.0 - self.rho / rhoref)), "slh", "Steric sea layer height anomaly", "-"
        )  # [-]

    @property
    # Contenu en chaleur de la colonne
    def ohc(self):
        if NoneType(self.ohc_):
            self.ohc_ = proprietes(
                self.heat.lnocean.integ_depth(), "ohc", "Ocean heat content", "J/m²"
            )  # [J/m²]
        return self.ohc_

    @property
    # Ecart de hauteur d'eau de la colonne par rapport à une référence (CT=0°C,SA=35psu)
    def ssl(self):
        if NoneType(self.ssl_):
            self.ssl_ = proprietes(
                self.slh.lnocean.integ_depth(),
                "ssl",
                "Steric sea surface level anomaly",
                "m",
            )  # [m]
        return self.ssl_

    @property
    # Ecart de hauteur d'eau thermosterique de la colonne par rapport à une référence (CT=0°C)
    def tssl(self):
        if NoneType(self.tssl_):
            rho = gsw.rho(self.SA, self.CT, self.P)
            rhoref = gsw.rho(self.SA, 0.0, self.P)
            tslh = 1.0 - rho / rhoref
            self.tssl_ = proprietes(
                tslh.lnocean.integ_depth(),
                "tssl",
                "Thermosteric sea surface level anomaly",
                "m",
            )  # [m]
        return self.tssl_

    @property
    # Ecart de hauteur d'eau halosterique de la colonne par rapport à une référence (SA=35psu)
    def hssl(self):
        if NoneType(self.hssl_):
            rho = gsw.rho(self.SA, self.CT, self.P)
            rhoref = gsw.rho(LNPY_SSO, self.CT, self.P)
            hslh = 1.0 - self.rho / rhoref
            self.hssl_ = proprietes(
                hslh.lnocean.integ_depth(),
                "hssl",
                "Halosteric sea surface level anomaly",
                "m",
            )  # [m]
        return self.hssl_

    # Ecart de hauteur d'eau de la colonne par rapport à une référence (CT=0°C,SA=35psu)
    def ssl_above(self, target):
        return proprietes(
            self.slh.lnocean.above(target),
            "ssl",
            "Steric sea surface level anomaly above targeted depth",
            "m",
        )  # [m]

    @property
    # EEH local (en
    def eeh(self):
        if NoneType(self.eeh_):
            rho, alpha, beta = gsw.rho_alpha_beta(
                self.SA.load(), self.CT.load(), self.P.load()
            )
            self.eeh_ = proprietes(
                alpha / (rho * self.Cp),
                "EEH",
                "Local expansion efficiency oh heat",
                "m/(J/m²)",
            )  # [m/(J/m²)]
        return self.eeh_

    @property
    # IEEH de la colonne (grandeur surfacique)
    def ieeh(self):
        if NoneType(self.ieeh_):
            self.ieeh_ = proprietes(
                self.tssl / self.ohc,
                "IEEH",
                "Integrated expansion efficiency oh heat",
                "m/(J/m²)",
            )  # [m/(J/m²)]
        return self.ieeh_

    @property
    def gohc(self):
        return proprietes(
            self.ohc.lngeo.mean(["latitude", "longitude"], weights=["latitude"]),
            "gohc",
            "Global ocean heat content wrt to ocean surface area",
            "J/m²",
        )

    @property
    def msl(self):
        return proprietes(
            self.ssl.lngeo.mean(["latitude", "longitude"], weights=["latitude"]),
            "msl",
            "Mean ocean sea level anomaly",
            "m",
        )

    @property
    def tmsl(self):
        return proprietes(
            self.tssl.lngeo.mean(["latitude", "longitude"], weights=["latitude"]),
            "tmsl",
            "Mean thermosteric ocean sea level anomaly",
            "m",
        )

    @property
    def hmsl(self):
        return proprietes(
            self.hssl.lngeo.mean(["latitude", "longitude"], weights=["latitude"]),
            "hmsl",
            "Mean halosteric ocean sea level anomaly",
            "m",
        )

    @property
    def gohc_TOA(self):
        return proprietes(
            self.ohc.lngeo.mean(
                ["latitude", "longitude"], weights=["latitude"], na_eq_zero=True
            ),
            "gohc",
            "Global ocean heat content wrt to TOA area",
            "J/m²",
        )

    def ohc_above(self, target):
        res = self.heat.lnocean.above(target)
        return proprietes(
            res.where(res != 0),
            "ohc_above",
            "Ocean heat content above targeted depth",
            "J/m²",
        )  # [J/m²]

    def gohc_above(self, target, na_eq_zero=False, mask=True):
        return proprietes(
            self.ohc_above(target).lngeo.mean(
                ["latitude", "longitude"],
                weights=["latitude"],
                na_eq_zero=na_eq_zero,
                mask=mask,
            ),
            "gohc_above",
            "Global ocean heat content above target",
            "J/m²",
        )

    @property
    def ocean_depth(self):
        if NoneType(self.ocean_depth_):
            self.ocean_depth_ = xr.where(
                self.temp.isnull(), np.nan, self._obj.depth
            ).max("depth")
        return self.ocean_depth_

    # Profondeur de l'Ocean Mixed Layer definie par une variation de temperature potentielle de 0.2°C par rapport à -10m
    @property
    def mld_theta0(self):

        theta0 = self.PT.interp(depth=10).drop("depth")
        mld1 = self.PT.lngeo.isosurface(
            theta0 - self.oml_theta0_threshold, "depth", upper=True
        )
        mld2 = self.PT.lngeo.isosurface(
            theta0 + self.oml_theta0_threshold, "depth", upper=True
        )
        mld1 = mld1.fillna(mld2)
        mld2 = mld2.fillna(mld1)
        self.mld_theta0_ = (
            xr.where(mld2 < mld1, mld2, mld1)
            .rename("OMLD_theta0")
            .fillna(self.ocean_depth)
        )

        return self.mld_theta0_

    # Profondeur de l'Ocean Mixed Layer definie par une diminution de temperature potentielle de 0.2°C par rapport à -10m
    @property
    def mld_theta0minus_only(self):

        theta0 = self.PT.interp(depth=10).drop("depth")
        self.mld_theta0_ = (
            self.PT.lngeo.isosurface(
                theta0 - self.oml_theta0_threshold, "depth", upper=True
            )
            .rename("OMLD_theta0minus_only")
            .fillna(self.ocean_depth)
        )

        return self.mld_theta0_

    # Profondeur de l'Ocean Mixed Layer definie par une augmentation de densité potentielle de 0.03kg/m3 par rapport à -10m
    @property
    def mld_sigma0(self):

        sigma0 = self.sigma0.interp(depth=10).drop("depth")
        self.mld_sigma0_ = (
            self.sigma0.lngeo.isosurface(
                sigma0 + self.oml_sigma0_threshold, "depth", upper=True
            )
            .rename("OMLD_sigma0")
            .fillna(self.ocean_depth)
        )

        return self.mld_sigma0_

    # Profondeur de l'Ocean Mixed Layer definie par une augmentation de densité potentielle correspondant à -0.2°C par rapport à -10m
    @property
    def mld_sigma0var(self):
        ref = self._obj.interp(depth=10)
        ref["PT"] = ref["PT"] - self.oml_theta0_threshold
        self.mld_sigma0var_ = (
            self.sigma0.lngeo.isosurface(
                ref.lnocean.sigma0.drop("depth"), "depth", upper=True
            )
            .rename("OMLD_sigma0var")
            .fillna(self.ocean_depth)
        )

        return self.mld_sigma0var_


@xr.register_dataarray_accessor("lnocean")
class OceanArray:
    """This class extends any dataarray to a xOcean object, to perform specific operations on structured dataarray"""

    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_value_surface(self, value=None):
        """Add a surface layer with a specified value, or the previous upper value
        Parameters
        ----------
        value : float or array-like, optional
            values to be added in the surface layer (depth=0). If None, the previous upper value is used to fill the new layer

        Returns
        -------
        added : DataArray
            new dataarray with a extra layer at depth 0 filled with required values

        Example
        -------
        .. code-block:: python

            data=IAP('/home/usr/lenapy/data/IAP')
            heat=data.lnocean.heat.lnocean.add_value_surface()

        """
        if self._obj.depth[0] != 0:
            v0 = self._obj.isel(depth=0)
            v0["depth"] = xr.zeros_like(v0.depth)
            if value != None:
                v0 = xr.full_like(v0, fill_value=value)
            return xr.concat([v0, self._obj], dim="depth")
        else:
            return self._obj

    def integ_depth(self):
        """Returns the dataarray integrated over the whole depth. The surface value is supposed equal to the most shallow value.
        In order to deal with NaN values in deep water during integration,, all NaN are firt converted to 0, then in the output
        array NaN values are applied where initial surface values were NaN.

        Example
        -------
        .. code-block:: python

            data=IAP('/home/usr/lenapy/data/IAP')
            data.lnocean.heat.lnocean.integ_depth()
            <xarray.DataArray 'Heat' (time: 156, latitude: 180, longitude: 360)>
            dask.array<where, shape=(156, 180, 360), dtype=float64, chunksize=(1, 180, 360), chunktype=numpy.ndarray>
            Coordinates:
              * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
              * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0
              * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
                depth      float32 1.0

        """
        return (
            self.add_value_surface()
            .fillna(0)
            .integrate("depth")
            .where(~self._obj.isel(depth=0).isnull())
        )

    def cum_integ_depth(self):
        """Returns a cumulative integrated dataarray integrated over the whole depth. The surface value is supposed equal to the
        most shallow value. A first integration layer by layer is performed, by multupliying the layer's thickness by the mean
        value of upper and lower bound, then a cumulative sum is computed.

        Example
        -------
        .. code-block:: python

            data=IAP('/home/usr/lenapy/data/IAP')
            data.lnocean.heat.lnocean.cum_integ_depth()
            <xarray.DataArray (time: 156, latitude: 180, longitude: 360, depth: 41)>
            dask.array<where, shape=(156, 180, 360, 41), dtype=float64, chunksize=(1, 180, 360, 10), chunktype=numpy.ndarray>
            Coordinates:
              * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
              * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0
              * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
              * depth      (depth) float64 1.0 5.0 10.0 20.0 ... 1.7e+03 1.8e+03 2e+03

        """
        res = self.add_value_surface()
        ep = res.depth.diff("depth")
        vm = res.rolling(depth=2).mean().isel(depth=slice(1, None))
        return (
            (vm * ep).fillna(0).cumsum("depth").where(~self._obj.isel(depth=0).isnull())
        )

    def above(self, depth, **kwargs):
        """Returns the dataarray integrated above a given depth, by interpolating at this depth the cumulative integrale
        of the data array

        Example
        -------
        .. code-block:: python

            data=IAP('/home/usr/lenapy/data/IAP')
            mld=data.lnocean.mld_sigma0
            data.lnocean.heat.lnocean.above(mld)
            <xarray.DataArray (time: 156, latitude: 180, longitude: 360)>
            dask.array<where, shape=(156, 180, 360), dtype=float64, chunksize=(1, 180, 360), chunktype=numpy.ndarray>
            Coordinates:
                depth      (time, latitude, longitude) float64 dask.array<chunksize=(1, 180, 360), meta=np.ndarray>
              * time       (time) datetime64[ns] 2005-01-15 2005-02-15 ... 2017-12-15
              * latitude   (latitude) float32 -89.5 -88.5 -87.5 -86.5 ... 87.5 88.5 89.5
              * longitude  (longitude) float32 1.0 2.0 3.0 4.0 ... 357.0 358.0 359.0 360.0

        """
        d = xr.where(self._obj.depth.max() < depth, self._obj.depth.max(), depth)
        return (
            self.cum_integ_depth()
            .lnocean.add_value_surface(0.0)
            .interp({"depth": d}, **kwargs)
        )
