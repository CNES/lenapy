import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

def plot_timeseries_uncertainty(xgeo_data, 
                                x_dim='time',
                                y_dim=None,
                                thick_line='median',
                                shaded_area='quantiles',
                                quantile_min=0.05,
                                quantile_max=0.95,
                                thick_line_color=None,
                                shaded_area_color=None,
                                shaded_area_alpha=0.2,
                                ax=None,
                                label=None,
                                line_kwargs = dict(),
                                area_kwargs = dict(),
                                add_legend=True
                                ):
    """
    Affiche une série temporelle avec une enveloppe d'incertitude.
    
    Parameters
    ----------
    data : xr.DataArray, la série temporelle que l'on veut plotter
    x_dim : la dimension qui sera sur l'axe des x. Par defaut on prend 'time'
    y_dim : la dimension qui permet de calculer l'incertitude
    thick_line : la métrique afficher en ligne epaisse: "median" ou "mean"
    shaded_area : la métrique qui permet de calculer l'incertitude: "std" ou "quantiles"
    quantile_min : si shaded_area="quantile", la valeur du quantile inférieur (entre 0 et 1)
    quantile_max : si shaded_area="quantile", la valeur du quantile supérieur (entre 0 et 1)
    thick_line_color : couleur de la ligne épaisse
    shaded_area_color : couleur de l'applat de couleur pour l'incertitude
    shaded_area_alpha : transparence de l'aplat de couleur pour l'incertitude
    add_legend : bool, ajoute une legend
    """
    # Il faut gérer les assert dimensions
    # Il faut gérer la légende
    data = xgeo_data
    variable = data.name
    if ax is None:
        ax = plt.gca()
    data_dims = data.dims
    if y_dim is None and len(data_dims)>=2:
        y_dim = [dim for dim in data_dims if dim!=x_dim][0]

    if thick_line=='median':
        main_metric = data.median(y_dim)
    elif thick_line=='mean':
        main_metric = data.mean(y_dim)
    elif thick_line is None:
        pass
    else:
        raise ValueError("thick_line can only be 'mean', 'median' or None.")
        
    if label is None:
        label=f"{variable} {thick_line}"
    
    if thick_line is not None:
        plot_line = main_metric.plot(ax=ax, color=thick_line_color, **line_kwargs, label=label)
    if shaded_area_color is None:
        shaded_area_color = plot_line[0].get_color()
    if 'std' in shaded_area:
        data_std = data.std(y_dim)
        if thick_line is None:
            main_metric = data.mean(y_dim)
        if shaded_area == 'std':
            ax.fill_between(data.time.values, main_metric-data_std, main_metric+data_std,
                            color=shaded_area_color, alpha=shaded_area_alpha, 
                            linewidth=0, **area_kwargs, label='_nolegend_')
        if shaded_area == '2std':
            ax.fill_between(data.time.values, main_metric-2*data_std, main_metric+2*data_std,
                            color=shaded_area_color, alpha=shaded_area_alpha, 
                            linewidth=0, **area_kwargs, label='_no_legend_')
        if shaded_area == '3std':
            ax.fill_between(data.time.values, main_metric-3*data_std, main_metric+3*data_std,
                            color=shaded_area_color, alpha=shaded_area_alpha, 
                            linewidth=0, **area_kwargs, label='_no_legend_')

    elif shaded_area == 'quantiles':
        data_quantile = data.quantile([quantile_min, quantile_max],
                                      dim = y_dim)
        if thick_line is None:
            main_metric = data.median(y_dim)
        ax.fill_between(data.time.values, data_quantile.isel(quantile=0), data_quantile.isel(quantile=1),
                        color=shaded_area_color, alpha=shaded_area_alpha,
                        linewidth=0, **area_kwargs,
                        label='_no_legend_')
    elif shaded_area is None:
        pass
    else:
        raise ValueError("shaded_area can only be 'std','2std','3std', 'quantiles' or None.")

    if add_legend:
        ax.legend()
        
        
class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False, rlocs=None):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.

        Parameters:

        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        if rlocs==None:
            rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
            
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]
        
        # Add RMS contours, and label them
        contours = self.add_contours(levels=5, colors='0.3')  # 5 levels in grey
        plt.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

        self.add_grid()                                  # Add grid
        self._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward
        
       

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours

    
