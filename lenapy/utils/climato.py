import xarray as xr
import pandas as pd
import numpy as np
from ..constants import *
from scipy.interpolate import interp1d

def lstsq(X, Y, poids=None):
    if poids is None:
        C = 1
    else:
        C = np.diag(poids)
    H = np.linalg.inv(X.T.dot(np.dot(C, X)))
    return H @ (X.T.dot(np.dot(C, Y)))

 
# Fonctions de base
def annual(x):
    omega = 2*np.pi/LNPY_DAYS_YEAR
    return xr.concat((np.cos(omega*x), np.sin(omega*x)), dim="coeffs")


def semiannual(x):
    omega = 2*np.pi/LNPY_DAYS_YEAR
    return xr.concat((np.cos(2*omega*x), np.sin(2*omega*x)), dim="coeffs")


def pol(x, order):
    return x**xr.DataArray(np.arange(order+1),dims='coeffs')


class coeffs_clim:
    def __init__(self, coefficients, func, *args, ref=0., scale=1., ds=None, **param):
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
        var = [(x[self.args[0]] - self.ref)/self.scale] + [x[u] for u in self.args[1:]]
        return self.func(*var, **self.param).assign_coords(coeffs=self.coefficients)
        

class Climato:
    def __init__(self, data, dim='time', var=None, Nmin=0, cycle=True, order=1):
        self.data = data
        self.dim = dim
        if var is None:
            self.var = dim
        else:
            self.var = var
        self.Nmin = Nmin
        self.coeffs = []
        self.coeff_names = []
        self.func_attrs = []
        
        if cycle:
            self.cycle()
        if order > 0:
            self.poly(order)
        
    def add_coeffs(self, coefficients, func, *args, ref=None, scale=pd.to_timedelta("1D").asm8, **kwargs):
        if ref is None:
            ref = self.data[args[0]].min()
        self.coeffs.append(coeffs_clim(coefficients, func, *args, ref=ref, scale=scale, ds=self.data, **kwargs))
        if type(coefficients) is str:
            self.coeff_names.append(coefficients)
        else:
            self.coeff_names.extend(coefficients)
        self.func_attrs.append([func.__name__, ref, scale, coefficients, args, kwargs])

    def cycle(self, **kwargs):
        self.add_coeffs(['cosAnnual', 'sinAnnual'], annual, self.var, **kwargs)
        self.add_coeffs(['cosSemiAnnual', 'sinSemiAnnual'], semiannual, self.var, **kwargs)
    
    def poly(self, order=2, **kwargs):
        self.add_coeffs(['order_%i' % i for i in np.arange(order+1)], pol, self.var, order=order, **kwargs)
            
    def expl(self, x=None):
        return xr.concat([u.compute(x) for u in self.coeffs], dim='coeffs').transpose(..., 'coeffs')
            
    def solve(self, mesure, chunk={}, poids=None, t_min=None, t_max=None):
        ok_time = True
        if t_min is not None:
            if type(t_min) is str:
                t_min = pd.to_datetime(t_min)
            ok_time = ok_time & (self.data[self.var] >= t_min)
        if t_max is not None:
            if type(t_max) is str:
                t_max = pd.to_datetime(t_max)
            ok_time = ok_time & (self.data[self.var] <= t_max)
                    
        self.mesure = mesure
        X_in = self.expl().values
        
        def solve_least_square(data_in):
            """
            For a given 1d time series, returns the coefficients of the fitted climatology
            """
            ok = (~np.isnan(data_in)) & ok_time
            Y_in_nona = data_in[ok]
            # If less than 6 non-na elements, climato is not computable
            if len(Y_in_nona) <= self.Nmin + len(self.coeffs):
                return np.full(X_in.shape[1], np.nan)
            X_in_nona = X_in[ok, :]
            # (coeffs, residus, rank, eig) = np.linalg.lstsq(X_in_nona, Y_in_nona)
            return lstsq(X_in_nona, Y_in_nona, poids)

        # Application de ufunc
        coeffs = xr.apply_ufunc(
            solve_least_square, 
            self.data[mesure].chunk(chunk), 
            input_core_dims=[[self.dim]],
            output_core_dims=[['coeffs']],
            exclude_dims=set((self.dim,)),
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'coeffs': X_in.shape[1]}}
        )
        self.result = coeffs.assign_coords(coeffs=self.coeff_names).assign_attrs(functions=self.func_attrs)
        return self
        
    def climatology(self, coefficients=None,x=None):
        res = self.expl(x)*self.result
        if coefficients is None:
            return res.sum('coeffs')
        else:
            return res.sel(coeffs=np.ravel(coefficients)).sum('coeffs')
        
    def residuals(self, coefficients=None):
        return self.data[self.mesure]-self.climatology(coefficients=coefficients)
    
    def signal(self, x=None, coefficients=None, method='linear'):
        
        if x is None:
            x = self.data[self.var]
        X_in = self.data[self.var].astype('float').values
        
        def interp_residuals(data_in):
            ok = ~np.isnan(data_in)
            if sum(ok) > 1:
                Y_in_nona = data_in[ok]
                X_in_nona = X_in[ok]
                return interp1d(X_in_nona, Y_in_nona, kind=method)(x.astype('float').values)
            else:
                return x.astype('float').values+np.nan

        resid_interp = xr.apply_ufunc(
            interp_residuals, 
            self.residuals(),
            input_core_dims=[[self.dim]],
            output_core_dims=[[self.dim]],
            exclude_dims=set((self.dim,)),
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {self.dim: x.shape[0]}}
        )
        
        return resid_interp + self.climatology(coefficients=coefficients,x=x)


class Generate_climato:
    def __init__(self, result):
        self.result = result
        self.func = {}
        self.add_function(annual)
        self.add_function(semiannual)
        self.add_function(pol)
        
    def add_function(self, func):
        self.func[func.__name__] = func

    def climatology(self, x, coefficients=None):
        self.coeffs = []
        for name, ref, scale, coeffs, args, kwargs in self.result.attrs['functions']:
            self.coeffs.append(coeffs_clim(coeffs, self.func[name], *args, ref=ref, scale=scale, **kwargs))

        expl = xr.concat([u.compute(x) for u in self.coeffs],dim='coeffs').transpose(...,'coeffs')
        
        res = expl*self.result
        
        if coefficients is None:
            return res.sum('coeffs')
        else:
            return res.sel(coeffs=np.ravel(coefficients)).sum('coeffs')
