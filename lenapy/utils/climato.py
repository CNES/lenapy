import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

from lenapy.constants import *


def lstsq(X, Y, weight=None):
    if weight is None:
        C = 1
    else:
        C = np.diag(weight)
    H = np.linalg.inv(X.T.dot(np.dot(C, X)))
    return H @ (X.T.dot(np.dot(C, Y)))


# Fonctions de base
def annual(x):
    omega = 2 * np.pi / LNPY_DAYS_YEAR
    return xr.concat((np.cos(omega * x), np.sin(omega * x)), dim="coeffs")


def semiannual(x):
    omega = 2 * np.pi / LNPY_DAYS_YEAR
    return xr.concat((np.cos(2 * omega * x), np.sin(2 * omega * x)), dim="coeffs")


def pol(x, order):
    return x ** xr.DataArray(np.arange(order + 1), dims="coeffs")


class coeffs_clim:
    def __init__(self, coefficients, func, *args, ref=0.0, scale=1.0, ds=None, **param):
        self.func = func
        self.coefficients = coefficients
        self.ds = ds
        self.ref = ref
        self.scale = scale
        self.args = args
        self.param = param

    def compute(self, x=None):
        if x is None:
            x = self.ds
        ref = x[self.args[0]].min() if self.ref is None else self.ref
        var = [(x[self.args[0]] - ref) / self.scale] + [x[u] for u in self.args[1:]]
        return self.func(*var, **self.param).assign_coords(coeffs=self.coefficients)


class Coeffs_climato:
    def __init__(
        self, data, dim="time", var=None, Nmin=0, cycle=True, order=1, ref=None
    ):
        self.data = data
        self.dim = dim
        self.var = dim if var is None else var
        self.Nmin = Nmin
        self.coeffs = []
        self.coeff_names = []

        if cycle:
            self.cycle(ref=ref)
        if order < 0:
            raise ValueError("Order must be >=0")
        self.poly(order, ref=ref)

    def add_coeffs(
        self,
        coefficients,
        func,
        *args,
        ref=None,
        scale=pd.to_timedelta("1D").asm8,
        **kwargs,
    ):
        if ref is None:
            ref = self.data[args[0]].min()
        if hasattr(ref, "values"):
            ref = ref.values
        self.coeffs.append(
            coeffs_clim(
                coefficients, func, *args, ref=ref, scale=scale, ds=self.data, **kwargs
            )
        )
        if type(coefficients) == str:
            coefficients = [coefficients]

        self.coeff_names.extend(coefficients)

    def cycle(self, **kwargs):
        self.add_coeffs(["cosAnnual", "sinAnnual"], annual, self.var, **kwargs)
        self.add_coeffs(
            ["cosSemiAnnual", "sinSemiAnnual"], semiannual, self.var, **kwargs
        )

    def poly(self, order=2, **kwargs):
        self.add_coeffs(
            ["order_%i" % i for i in np.arange(order + 1)],
            pol,
            self.var,
            order=order,
            **kwargs,
        )

    def expl(self, x=None):
        return xr.concat([u.compute(x) for u in self.coeffs], dim="coeffs").transpose(
            ..., "coeffs"
        )

    def solve(self, measure=None, weight=None, t_min=None, t_max=None):

        if type(self.data) is xr.Dataset:
            data_mes = self.data[measure]
        else:
            data_mes = self.data

        ok_time = True
        if t_min is not None:
            if type(t_min) is str:
                t_min = pd.to_datetime(t_min)
            ok_time = ok_time & (self.data[self.var] >= t_min)
        if t_max is not None:
            if type(t_max) is str:
                t_max = pd.to_datetime(t_max)
            ok_time = ok_time & (self.data[self.var] <= t_max)

        self.measure = measure
        X_in = self.expl().values

        def solve_least_square(data_in):
            """
            For a given 1d time series, returns the coefficients of the fitted climatology
            """
            ok = (~np.isnan(data_in)) & ok_time
            Y_in_nona = data_in[ok]
            # If less than minimum number of non-na elements, climato is not computable
            min_elements = (
                np.sum([len(u.coefficients) for u in self.coeffs]) + self.Nmin
            )
            if len(Y_in_nona) <= min_elements:
                return np.full(X_in.shape[1], np.nan)
            X_in_nona = X_in[ok, :]
            return lstsq(X_in_nona, Y_in_nona, weight)

        # Application de ufunc
        coeffs = xr.apply_ufunc(
            solve_least_square,
            data_mes,
            input_core_dims=[[self.dim]],
            output_core_dims=[["coeffs"]],
            exclude_dims=set((self.dim,)),
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"coeffs": X_in.shape[1]}},
        )
        result = coeffs.assign_coords(coeffs=self.coeff_names)
        clim = Signal_climato(
            result,
            dim=self.dim,
            var=self.var,
            cycle=False,
            order=0,
            ds=self.data,
            measure=self.measure,
        )
        clim.coeffs = self.coeffs
        return clim


class Signal_climato:
    def __init__(
        self,
        result,
        dim="time",
        var=None,
        cycle=True,
        order=1,
        ref=None,
        ds=None,
        measure=None,
    ):
        self.result = result
        self.coeffs = []
        self.dim = dim
        self.var = dim if var is None else var
        self.ds = ds
        self.measure = measure
        if cycle:
            self.cycle(ref=ref)
        if order >= 0:
            self.poly(order, ref=ref)

    def add_coeffs(
        self,
        coefficients,
        func,
        *args,
        ref=None,
        scale=pd.to_timedelta("1D").asm8,
        **kwargs,
    ):
        for u in np.ravel(coefficients):
            if u not in self.result.coeffs:
                raise ValueError(
                    "Coefficient %s not in list %s" % (u, self.result.coeffs)
                )

        #        if hasattr(ref,'values'):
        #            ref = ref.values
        self.coeffs.append(
            coeffs_clim(
                coefficients, func, *args, ref=ref, scale=scale, ds=self.ds, **kwargs
            )
        )

    def cycle(self, **kwargs):
        self.add_coeffs(["cosAnnual", "sinAnnual"], annual, self.var, **kwargs)
        self.add_coeffs(
            ["cosSemiAnnual", "sinSemiAnnual"], semiannual, self.var, **kwargs
        )

    def poly(self, order=2, **kwargs):
        self.add_coeffs(
            ["order_%i" % i for i in np.arange(order + 1)],
            pol,
            self.var,
            order=order,
            **kwargs,
        )

    def expl(self, x=None):
        return xr.concat([u.compute(x) for u in self.coeffs], dim="coeffs").transpose(
            ..., "coeffs"
        )

    def climatology(self, coefficients=None, x=None):
        res = self.expl(x) * self.result
        if coefficients is None:
            return res.sum("coeffs")
        else:
            return res.sel(coeffs=np.ravel(coefficients)).sum("coeffs")

    def residuals(self, coefficients=None):
        if type(self.ds) is xr.Dataset:
            return self.ds[self.measure] - self.climatology(coefficients=coefficients)
        else:
            return self.ds - self.climatology(coefficients=coefficients)

    def signal(self, x=None, coefficients=None, method="linear"):
        if x is None:
            x = self.ds[self.var]
        X_in = self.ds[self.var].astype("float").values

        def interp_residuals(data_in):
            ok = ~np.isnan(data_in)
            if sum(ok) > 1:
                Y_in_nona = data_in[ok]
                X_in_nona = X_in[ok]
                return interp1d(
                    X_in_nona,
                    Y_in_nona,
                    kind=method,
                    bounds_error=False,
                    fill_value=(Y_in_nona[0], Y_in_nona[-1]),
                )(x.astype("float").values)
            else:
                return x.astype("float").values + np.nan

        resid_interp = xr.apply_ufunc(
            interp_residuals,
            self.residuals(),
            input_core_dims=[[self.dim]],
            output_core_dims=[[self.dim]],
            exclude_dims=set((self.dim,)),
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {self.dim: x.shape[0]}},
        )

        return resid_interp + self.climatology(coefficients=coefficients, x=x)
