import matplotlib.pyplot as plt
import numpy as np
import yaml

from lenapy.utils.time import *


# Ajouter de la doc !!!
class estimator:
    """This class implements least squares estimators based on GLS or OLS

    Parameters
    ----------
    data : DataArray
        Contains the data on which to apply the regression. Must have a "time" coordinate
    degree : int
        Degree of the polynomial regression (0=constant, 1=trend, 2=quadratic, ...)
    tref : None or datetime64, optional (default=None)
        Time origine in the regression. If None, tref is the first timestep
    sigma : None or numpy array, optional (default=None)
        Covariance matrix for the estimator and its covariance. If None, identity matrix is used. Must have the same size than data
    datetime_unit : string, optional (default='s')
        Time unit
    """

    def __init__(self, data, degree, tref=None, sigma=None, datetime_unit="s"):
        if type(tref) == type(None):
            tref = data.time.mean("time")
        if type(sigma) == type(None):
            sigma = np.diag(np.ones(len(data.time)))

        if sigma.shape != (data.shape + data.shape):
            raise Exception("data and covariance matrix don" "t have the same size")
        t1 = (data.time - tref) / pd.to_timedelta("1%s" % datetime_unit).asm8
        self.deg = xr.DataArray(
            data=np.arange(degree + 1),
            dims="degree",
            coords=dict(degree=np.arange(degree + 1)),
        )
        self.data = data
        self.expl = t1**self.deg
        self.X = self.expl.values
        self.Y = data.values
        self.cov_matrix = sigma
        self.type = None

    def OLS(self):
        """Compute the Ordinaty Least Square regression on the data
        Results are accessible with class properties

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('GMSL.nc')
            estimator = data.estimator(data,2)
            estimator.OLS()

        """

        H = np.linalg.inv(np.dot(self.X.T, self.X))
        self.beta = np.dot(H, np.dot(self.X.T, self.Y))
        self.params = xr.zeros_like(self.deg) + self.beta
        self.type = "OLS"

    def GLS(self):
        """Compute the Generalized Least Square regression on the data, with specified covariance matrix
        Results are accessible with class properties

        Example
        -------
        .. code-block:: python

            data = xr.open_dataset('GMSL.nc')
            covmatrix = xr.open_dataset('covariance.nc')
            estimator = data.estimator(data,2,sigma=covmatrix)
            estimator.GLS()

        """
        C = np.linalg.inv(self.cov_matrix)
        H = np.linalg.inv(np.dot(self.X.T, np.dot(C, self.X)))
        self.beta = np.dot(H, np.dot(self.X.T, np.dot(C, self.Y)))
        self.params = xr.zeros_like(self.deg) + self.beta
        self.type = "GLS"

    @property
    def estimate(self):
        """Result of the least squares estimation"""
        return (self.expl * self.params).sum("degree")

    @property
    def residuals(self):
        """Residuals of the computed least squares estimation"""
        return self.data - self.estimate

    @property
    def residuals_stderr(self):
        """Residuals standard error"""
        return self.residuals.std()

    @property
    def estimator_covariance(self):
        """Covariance matrix of the estimator (OLS or GLS)"""
        if self.type == "OLS":
            H = np.linalg.inv(np.dot(self.X.T, self.X))
            A = np.dot(np.dot(self.X.T, self.cov_matrix), self.X)
            return np.dot(np.dot(H, A), H)
        elif self.type == "GLS":
            C = np.linalg.inv(self.cov_matrix)
            return np.linalg.inv(np.dot(self.X.T, np.dot(C, self.X)))
        else:
            raise ValueError(
                "Type must be OLS or GLS : execute OLS or GLS method before calling"
            )

    @property
    def uncertainties(self):
        """Standard error of the estimator (square root of the estimator covariance diagonal)"""
        return xr.zeros_like(self.deg) + np.sqrt(np.diag(self.estimator_covariance))

    @property
    def coefficients(self):
        return self.params


class cov_element:
    """This class generates a covariance matrix based on a single uncertainty type.
    Available uncertainties are:
    * bias : uncertainty of the connection bias at a given date
    * drift : trend uncertainty
    * noise : correlated noise defined by a span
    * random : white noise
    * random_random : random value on the covariance matrix diagonal

    Parameters
    ----------
    time : DataArray
        timeseries on which to calculate the covariance matrix
    uncertainty_type : string
        uncertainty type. Can be 'bias', 'drif', 'noise', 'random', 'random_random'
    value : float
        value associated with the uncertainty (standard error)
    t0 : datetime64
        application date for the bias uncertainty
    tmin : datetime64
        beginning of the uncertainty period
    tmax :datetime64
        end of the uncertainty period
    dt : timedelta64
        span for the correlated noise
    bias_type : string
        type of the bias to be applied to satisfy specific constraints in the realization of the covariance matrix :
        * 'no_bias' or None : no bias is applied
        * 'begin' or 'right' : all realizations are constrained to zero at the beginning of the period
        * 'end' or 'left' : all realizations are constrained to zero at the end of the period
        * 'centered' : all realizations are constrained to a zero mean
        bias_type do not apply to random or random_random type

    """

    def __init__(
        self,
        time,
        uncertainty_type,
        value,
        t0=None,
        tmin=None,
        tmax=None,
        dt=None,
        bias_type=None,
    ):

        self.time = time
        self.type = uncertainty_type
        if bias_type == "no_bias":
            bias_type = None
        self.bias_type = bias_type
        self.value = value

        t = self.time

        if self.type == "bias":
            t1 = xr.where(self.time < t0, 1.0, 0.0)
            t2 = t1.rename(time="time1")
            self.sigma = value**2 * t1 * t2

        elif self.type == "drift":
            if tmax != None:
                t = xr.where(t <= tmax, t, tmax)
            if tmin != None:
                t = xr.where(t >= tmin, t, tmin)
            t1 = t.astype("float") * 1.0e-9 / 86400.0
            t2 = t1.rename(time="time1")
            self.sigma = value**2 * t1 * t2

        elif self.type == "noise":
            if tmax != None:
                t = xr.where(t <= tmax, t, tmax)
            if tmin != None:
                t = xr.where(t >= tmin, t, tmin)
            t1 = t
            t2 = t1.rename(time="time1")
            self.sigma = value**2 * np.exp(-0.5 * ((t1 - t2) / dt) ** 2)

        elif self.type == "random":
            if tmax == None:
                tmax = self.time.max()
            if tmin == None:
                tmin = self.time.min()
            t1 = self.time.astype("float")
            t2 = t1.rename(time="time1")
            self.sigma = xr.zeros_like(t1 - t2) + np.diag(
                xr.where(
                    np.logical_and(self.time >= tmin, self.time <= tmax), value**2, 0
                )
            )
            # Pour ce type d'erreur, l'ajustement n'est pas pertinent
            self.bias_type = None

        elif self.type == "random_random":
            t1 = self.time.astype("float")
            t2 = t1.rename(time="time1")
            self.sigma = xr.zeros_like(t1 - t2) + np.diag(
                value * np.random.rand(len(self.time))
            )
            # Pour ce type d'erreur, l'ajustement n'est pas pertinent
            self.bias_type = None

        self.ajuste()

    def ajuste(self):
        if self.bias_type != None:
            eigenvalues, eigenvectors = np.linalg.eigh(self.sigma)
            # Une matrice de covariance est semi définie positive, elle a donc toutes ses valeurs propres positives. La précision de calcul peut
            # amener des valeurs popres négatives négligeables, elles sont mises à zéro pour pouvoir prendre leur racine carrée.
            eigenvalues = np.where(eigenvalues < 0, 0.0, eigenvalues)
            A = xr.ones_like(self.sigma)
            A.values = np.sqrt(eigenvalues) * eigenvectors

            if self.bias_type[0:8] == "centered":
                B = A.mean("time")
            elif self.bias_type == "begin" or self.bias_type == "right":
                B = A.isel(time=0)
            elif self.bias_type == "end" or self.bias_type == "left":
                B = A.isel(time=-1)
            else:
                raise Exception("unknown bias type : %s" % self.bias_type)
            Ap = A - xr.ones_like(A.isel(time1=0)) * B
            self.sigma.values = np.dot(Ap, Ap.T)


class covariance:
    """This class generates a covariance matrix from the combination of covariance matrixes baed on single uncertainty type.
    Unertainties can be read from a yaml file or directly woth the 'add_errors' method.

    Parameters
    ----------
    time : DataArray
        timeseries on which to calculate the covariance matrix
    """

    def __init__(self, time):
        self.time = time
        self.sigma = None
        self.errors = []

    def add_errors(
        self,
        uncertainty_type,
        value=1.0,
        t0=None,
        tmin=None,
        tmax=None,
        dt=None,
        bias_type=None,
    ):
        """Add a new covariance matrix based on single uncertainty type

        Parameters
        ----------
        uncertainty_type : string
            uncertainty type. Can be 'bias', 'drif', 'noise', 'random', 'random_random'
        value : float
            value associated with the uncertainty (standard error)
        t0 : datetime64
            application date for the bias uncertainty
        tmin : datetime64
            beginning of the uncertainty period
        tmax :datetime64
            end of the uncertainty period
        dt : timedelta64
            span for the correlated noise
        bias_type : string
            type of the bias to be applied to satisfy specific constraints in the realization of the covariance matrix :
            * 'no_bias' or None : no bias is applied
            * 'begin' or 'right' : all realizations are constrained to zero at the beginning of the period
            * 'end' or 'left' : all realizations are constrained to zero at the end of the period
            * 'centered' : all realizations are constrained to a zero mean
            bias_type do not apply to random or random_random type

        """
        new_err = cov_element(
            self.time, uncertainty_type, value, t0, tmin, tmax, dt, bias_type
        )
        if type(self.sigma) != type(None):
            self.sigma = self.sigma + new_err.sigma
        else:
            self.sigma = new_err.sigma
        self.errors.append(new_err)

    def read_yaml(self, filename):
        """Read a yaml file to populate the covariance matrix

        Parameters
        ----------
        filename : string
            path of the yaml file to be read

        Example
        -------
        Structure of the yaml file (dates are given in CNES julian days)

        .. code-block:: python

            errors:                                 # Necessary at the beggining of the file
            -                                       # each section must start with a -
                type: drift                         # can be bias, drift, noise, random, random_random
                parameters:                         #
                    time:                           # (only for type=bias) : empty or date of the bias uncertainy (CNES Julian day)
                    time_min:                       # empty or beginning of the uncertainty period (CNES Julian day)
                    time_max: 17946                 # empty or end of the uncertainty period (CNES Julian day)
                    value: 0.7                      # value associated with the uncertainty (standard error)
                    span:                           # (only for type=noise)span duration for the correlated noise
                    bias_type: centered             # can be null/no_bias, or begin/right, or end/left, or centered
                    conversion_factor: 0.0000027378 # conversion factor to be applied on the value

        """

        def lit(var, label, func=lambda x: x):
            try:
                return func(var[label])
            except:
                return None

        self.sigma = None
        yaml_file = open(filename, "r")
        yaml_content = yaml.full_load(yaml_file)
        for e in yaml_content["errors"]:

            param = lit(e, "parameters")
            typ = lit(e, "type")
            sigma = lit(param, "value")
            t0 = lit(param, "time", JJ_to_date)
            t1 = lit(param, "time_min", JJ_to_date)
            t2 = lit(param, "time_max", JJ_to_date)
            dt = lit(param, "span", lambda x: np.timedelta64(int(x), "D"))
            bias_type = lit(param, "bias_type")
            conv = lit(param, "conversion_factor")
            if conv == None:
                conv = 1.0

            self.add_errors(typ, conv * sigma, t0, t1, t2, dt, bias_type)

    def visu(self, n=100, vmax=None, save=None, cmap="RdBu_r"):
        """Show a vizualisation of the covariance matrix
        * some random realizations of the matrix
        * the plot of the squared roots diagonal values (standard error)
        * the pcolormesh plot of the covariance matrix

        Parameters
        ----------
        n: int (default=100)
            number of realizations to plot
        vmax : float or None (default=None)
            maximum value of the covariance matrix to be plotted
        save : None or string (default=None)
            if not None, filename to save the plot
        cmap : string (default='RdBu_r')
            colormap to use
        """
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        ax = ax.ravel()

        eigenvalues, eigenvectors = np.linalg.eigh(self.sigma)
        eigenvalues = np.where(eigenvalues < 0, 0.0, eigenvalues)
        A = np.sqrt(eigenvalues) * eigenvectors

        mu = 0
        sig = 1.0
        for i in range(n):
            s = np.random.normal(mu, sig, len(self.time))
            ax[0].plot(self.time, np.dot(A, s))
        ax[0].grid()
        ax[1].plot(self.time, np.sqrt(self.sigma.values.diagonal()))
        ax[1].set_ylim(0)
        ax[1].grid()
        self.sigma.plot(ax=ax[2], vmax=vmax, cmap=cmap, cbar_kwargs={"label": ""})
        if save != None:
            fig.savefig(save)
