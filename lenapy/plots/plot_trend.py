import numpy as np
import pandas as od
import xarray as xr


def plot_trend(data, sigma, out="OLS", freq="1M", save=None, **kwargs):
    # Periode centrale sur toute l'emprise des donnees
    tmin = data.time.min().values
    tmax = data.time.max().values
    t_ = pd.date_range(tmin, tmax, freq=freq)
    # Duree de la periode multiple de l'echantillonnage
    dt_ = t_[4::2] - tmin
    # Vecteurs temps et delta_t
    t = xr.DataArray(
        data=t_,
        dims="time",
        coords=dict(time=t_),
        attrs=dict(long_name="Central date of period"),
    ).chunk(time=100)
    dt = xr.DataArray(
        data=dt_,
        dims="delta",
        coords=dict(delta=dt_.values),
        attrs=dict(long_name="Period length (yr)"),
    ).chunk(delta=100)

    res = xr.DataArray(
        data=cov_trend(
            data=data, sigma=sigma, t1=(t - dt / 2.0), t2=(t + dt / 2.0), out=out
        ),
        dims=["time", "delta"],
        coords=dict(time=t, delta=dt_.days / 365),
    ).T

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    res.plot(**kwargs)
    plt.grid()
    if save != None:
        fig.savefig(save)


def cov_trend_(data, sigma, t1, t2, out="OLS"):
    t1 = np.datetime64(t1, "ns")
    t2 = np.datetime64(t2, "ns")
    if t1 >= data.time.min() and t2 <= data.time.max():
        d = data.sel(time=slice(t1, t2))
        sig = sigma.sel(time=slice(t1, t2), time1=slice(t1, t2))
        est = estimateur(d, 2, sigma=sig)
        if out == "OLS":
            est.OLS()
            return est.params[1].values
        elif out == "GLS":
            est.GLS()
            return est.params[1].values
        elif out == "err":
            return est.std_err()[1].values
    else:
        return np.nan


cov_trend = np.vectorize(cov_trend_, excluded=["data", "sigma", "out"])
