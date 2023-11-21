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

    

def watson_graph(trends, uncertainty_min, uncertainty_max, txts, colors_plot=None, stylish_levels=None, show_values=True, ylims_user=None,ax=None, 
                 fontsize=12,ylabel_pos=None):

    if ylims_user is None:
        ylims_user=[min(uncertainty_min),max(uncertainty_max)]
        
    if ylabel_pos==None:
        ylabel_pos=ylims_user[0]

    for ii in range(len(trends)):
        draw_bar(ax, ii, trends[ii], uncertainty_max[ii], uncertainty_min[ii], colors_plot[ii], txt=txts[ii], marker='o', ls='-', lw=3, show_values=show_values, stylish_levels=stylish_levels,fontsize=fontsize,ylabel_pos=ylabel_pos)

    ax.set_xlim([-0.5, len(trends)-0.2])
    ax.set_ylim(ylims_user)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    leg_x0, leg_x1, leg_y0, leg_y1 = xlims[0]+(xlims[1]-xlims[0])*0.88, xlims[0]+(xlims[1]-xlims[0])*0.91, ylims[0]+(ylims[1]-ylims[0])*0.8, ylims[0]+(ylims[1]-ylims[0])*0.9
    if all([el == colors_plot[0] for el in colors_plot]):
        color_loc = colors_plot[0]
    else:
        color_loc = 'black'
    y_step = (leg_y1-leg_y0)/(0.7*len(stylish_levels))
    barwidth_loc = 0.5*(leg_x1-leg_x0)
    leg_xmid = 0.5*(leg_x0+leg_x1)
    zorder = 100 + 30*len(stylish_levels)
    tt=0
    for ii in stylish_levels:
        if tt==0:
            ax.fill_between(np.linspace(leg_xmid-barwidth_loc, leg_xmid+barwidth_loc, 2), np.ones(2)*(leg_y0+(ii)*y_step), np.ones(2)*(leg_y0+(ii+1)*y_step), color=color_loc, zorder=zorder+10)
            ax.text(leg_xmid+barwidth_loc+0.1*(leg_x1-leg_x0), leg_y0+(ii+1)*y_step, '%.0f%% C.L.'%(sigma_to_confidence_interval(ii+1)*100.), fontsize=12, ha="left", va="center", \
                #bbox=dict(facecolor='white', edgecolor='white')
                   )
            ax.text(leg_xmid-barwidth_loc-0.1*(leg_x1-leg_x0), leg_y0+(ii+1)*y_step, '%.2f - $\sigma$'%(ii+1), fontsize=12, ha="right", va="center", \
                #bbox=dict(facecolor='white', edgecolor='white')
                   )
            barwidth_loc *= 0.6
            zorder -= 10
            tt=tt+1
        else:
            ax.fill_between(np.linspace(leg_xmid-barwidth_loc, leg_xmid+barwidth_loc, 2), np.ones(2)*(leg_y0+(ii)*y_step), np.ones(2)*(leg_y0+(ii+1)*y_step), color=color_loc, zorder=zorder+10, alpha=0.5)
            ax.text(leg_xmid+barwidth_loc+0.1*(leg_x1-leg_x0), leg_y0+(ii+1)*y_step, '%.0f%% C.L.'%(sigma_to_confidence_interval(ii+1)*100.), fontsize=12, ha="left", va="center", \
                #bbox=dict(facecolor='white', edgecolor='white')
                   )
            ax.text(leg_xmid-barwidth_loc-0.1*(leg_x1-leg_x0), leg_y0+(ii+1)*y_step, '%.2f - $\sigma$'%(ii+1), fontsize=12, ha="right", va="center", \
                #bbox=dict(facecolor='white', edgecolor='white')
                   )
            barwidth_loc *= 0.6
            zorder -= 10

    ax.set_xticks([])
    ax.set_axisbelow(True)
    ax.grid(True)


    
def draw_bar(ax, x, y, y_max, y_min, color, txt=None, marker=None, ls='-', lw=1, bar_tick_width=0.1, fontsize=12, show_values=True, stylish_levels=None,ylabel_pos=0):
    dy = (y_max - y_min)/2
    y_mean = (y_max+y_min)/2

    color_loc = color
    barwidth_loc = bar_tick_width
    zorder = 100 + 10*len(stylish_levels)
    tt=0
    for ii in stylish_levels:
        if tt==0:
            ax.fill_between(np.array([x-barwidth_loc, x+barwidth_loc]), np.ones(2)*(y_mean-(ii+1)*dy), np.ones(2)*(y_mean+(ii+1)*dy), color=[color_loc], zorder=zorder)
            tt=tt+1
        else:
            ax.fill_between(np.array([x-barwidth_loc, x+barwidth_loc]), np.ones(2)*(y_mean-(ii+1)*dy), np.ones(2)*(y_mean+(ii+1)*dy), color=[color_loc], zorder=zorder, alpha=0.5)
        barwidth_loc *= 0.6
        zorder -= 10

    if marker is not None:
        ax.scatter(x, y, marker=marker, color='white', zorder = 100 + 10*len(stylish_levels)+10)
    if txt is not None:
        ax.text(x+0.05*txt.count('\n'), ylabel_pos, txt, color=color, fontsize=fontsize, fontweight='bold', rotation=45, ha="center", va="top")
    if show_values:
       ax.text(x, y_mean-dy*1.65-0.1, '%.2f\n [%.2f; %.2f]'%(y, y_mean-dy*1.65, y_mean+dy*1.65), fontsize=fontsize-2, rotation=0, ha="center", va="center")

    
def sigma_to_confidence_interval(sigma_in, message=False):
    f_convert = sigma_to_confidence_interval_object(0.0001, 10.,10000)
    if message:
        print('%s sigma = %s %% confidence interval'%(sigma_in, 100.*f_convert(sigma_in)))
    return f_convert(sigma_in)

def sigma_to_confidence_interval_object(min_sigma, max_sigma, number_measures):
    if min_sigma < 0. or max_sigma <= 0. or number_measures < 2:
        raise IOError('min_sigma must be >=0 and max_sigma must be >0 and number_measures must be >=2')
    x = np.concatenate((np.zeros(1).astype(np.float64), np.logspace(np.log10(min_sigma),np.log10(max_sigma),number_measures).astype(np.float64)), axis=0)
    semigauss = np.exp(-0.5*(x**2))
    y = np.zeros(number_measures+1).astype(np.float64)
    tot = 0.5*np.sqrt(2.0*np.pi)
    u = np.float64(0.)
    for ii in range(1,number_measures):
        u += 0.5*(semigauss[ii-1]+semigauss[ii])*(x[ii]-x[ii-1])
        y[ii] = u/tot
    return interpolate.interp1d(x,y,kind='linear', bounds_error=False,fill_value=1.)