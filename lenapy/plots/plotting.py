import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from xarray.plot.dataarray_plot import _infer_line_data

from lenapy.utils.harmo import l_factor_conv


def plot_timeseries_uncertainty(
    xgeo_data,
    thick_line="median",
    shaded_area="auto",
    standard_deviation_multiple=1.645,
    quantile_min=0.05,
    quantile_max=0.95,
    color=None,
    thick_line_color=None,
    shaded_area_color=None,
    shaded_area_alpha=0.2,
    ax=None,
    label=None,
    line_kwargs=dict(),
    area_kwargs=dict(),
    add_legend=True,
    hue=None,
):
    """
    Plots the timeseries of the data in the TimeArray, including an uncertainty.
    Computes the uncertainty on all dimensions that are not time.

    Parameters
    ----------
    thick_line : String (default='median')
        How to aggregate the data to plot the main thick line. Can be:
        * `median`: computes the median
        * `mean`: computes the mean
        * None: does not plot a main thick line
    shaded_area : String (default='auto')
        How to aggregate the data to plot the uncertainty around the thick line. Can be:
        * `auto`: plots 1.645 standard deviation if thick_line is `mean` and quantiles 5-95 if thick_line is `median`.
        * `auto-multiple`: plots 1,2 and 3 standard deviations if thick_line is `mean` and quantiles 5-95, 17-83 and 25-75 if thick_line is `median`.
        * `std`: plots a multiple of the standard deviation based on kwarg `standard_deviation_multiple`
        * `quantiles`: plots quantiles based on the kwargs `quantile_min` and `quantile_max`
        * None: does not plot uncertainty
    hue : String (default=None)
        Similar to hue in xarray.DataArray.plot(hue=...), group data by the dimension before aggregating and computing uncertainties.
        Has to be a dimension other than time in the dataarray.
    standard_deviation_multiple : Float > 0 (default=1.65)
        The multiple of standard deviations to use for the uncertainty with `shaded_area=std`
    quantile_min : Float between 0 and 1 (default=0.05)
        lower quantile to compute uncertainty with `shaded_area=quantiles`
    quantile_max : Float between 0 and 1 (default=0.95)
        upper quantile to compute uncertainty with `shaded_area=quantiles`
    color : String or List (default=None)
        color of the main thick line and the shaded area. Must be a string
    thick_line_color : String or List (default=None)
        color of the main thick line. Must be a string
        If hue and one color are provided, the single color is used for all line plots.
        If hue and a list of colors are provided, the colors are cycled.
    shaded_area_color : String or List (default=None)
        color of the shaded area. Must be a string.
        If not provided, defaults to the thick_line_color value.
        If hue and one color are provided, the single color is used for all area plots.
        If hue and a list of colors are provided, the colors are cycled.
    shaded_area_alpha : Float between 0 and 1 (default=0.2)
        Transparency of the uncertainty plots
    ax : matplotlib.pyplot.Axes instance (default=None)
        If not provided, plots on the current axes.
    label : String (default=None)
        If provided, label that is provided to ax.plot.
        Does not work if hue is provided.
    line_kwargs : kwargs
        Additional arguments provided to the plot function for the main thick line
    area_kwargs : kwargs
        Additional arguments provided to the plot function for the uncertainty
    add_legend : Bool (default=True)
        If True, adds matplotlib legend to the current ax after plotting the data.
    """
    data = xgeo_data
    variable = data.name

    if ax is None:
        ax = plt.gca()
    if label is None:
        label = f"{variable}"

    if color is not None:
        thick_line_color = color

    data_dims = data.dims
    y_dim = [dim for dim in data_dims if dim != "time"]
    if hue is not None and hue not in y_dim:
        raise ValueError(f"hue must be in None or in {y_dim}.")
    elif hue is not None and hue in y_dim:
        hue_values = data[hue].values
        if type(thick_line_color) is str or thick_line_color is None:
            thick_line_colors = [thick_line_color for i in range(hue_values.size)]
        elif type(thick_line_color) is list:
            thick_line_colors = [
                thick_line_color[i % len(thick_line_color)]
                for i in range(hue_values.size)
            ]
        else:
            raise ValueError(
                "thick_line_color must be None, a string or a list of strings"
            )
        if type(shaded_area_color) is str or shaded_area_color is None:
            shaded_area_colors = [shaded_area_color for i in range(hue_values.size)]
        elif type(shaded_area_color) is list:
            shaded_area_colors = [
                shaded_area_color[i % len(thick_line_color)]
                for i in range(hue_values.size)
            ]
        else:
            raise ValueError(
                "shaded_area_color must be None, a string or a list of strings"
            )

        for k, value in enumerate(hue_values):
            data_hue = data.sel({hue: value})

            plot_timeseries_uncertainty(
                data_hue,
                thick_line=thick_line,
                shaded_area=shaded_area,
                quantile_min=quantile_min,
                quantile_max=quantile_max,
                thick_line_color=thick_line_colors[k],
                shaded_area_color=shaded_area_colors[k],
                shaded_area_alpha=shaded_area_alpha,
                ax=ax,
                label=value,
                line_kwargs=line_kwargs,
                area_kwargs=area_kwargs,
                add_legend=add_legend,
                hue=None,
            )

    else:
        if len(y_dim) == 0:
            main_metric = data
        else:
            if thick_line == "median":
                main_metric = data.median(y_dim)
            elif thick_line == "mean":
                main_metric = data.mean(y_dim)
            elif thick_line is None:
                pass
            else:
                raise ValueError("thick_line can only be 'mean', 'median' or None.")

        if thick_line is not None:
            plot_line = main_metric.plot(
                ax=ax, color=thick_line_color, **line_kwargs, label=label
            )
        else:
            ax.plot([], [], color=thick_line_color, **line_kwargs, label=label)

        if len(data_dims) > 0:
            if shaded_area not in [
                "auto",
                "auto-multiple",
                "std",
                "2std",
                "3std",
                "quantiles",
                None,
            ]:
                raise ValueError(
                    "Possible values for shaded_area can only be 'auto','auto-multiple', 'std', '2std', "
                    "'3std', 'quantiles', None"
                )

            if shaded_area_color is None:
                shaded_area_color = plot_line[0].get_color()
            if shaded_area is None:
                pass
            elif shaded_area == "auto-multiple":
                if thick_line == "mean":
                    data_std = data.std(y_dim)
                    ax.fill_between(
                        data.time.values,
                        main_metric - 3 * data_std,
                        main_metric + 3 * data_std,
                        color=shaded_area_color,
                        alpha=shaded_area_alpha,
                        linewidth=0,
                        **area_kwargs,
                        label="_nolegend_",
                    )
                    ax.fill_between(
                        data.time.values,
                        main_metric - 2 * data_std,
                        main_metric + 2 * data_std,
                        color=shaded_area_color,
                        alpha=shaded_area_alpha,
                        linewidth=0,
                        **area_kwargs,
                        label="_nolegend_",
                    )
                    ax.fill_between(
                        data.time.values,
                        main_metric - data_std,
                        main_metric + data_std,
                        color=shaded_area_color,
                        alpha=shaded_area_alpha,
                        linewidth=0,
                        **area_kwargs,
                        label="_nolegend_",
                    )
                if thick_line == "median":
                    quantiles = data.quantile(
                        [0.05, 0.17, 0.25, 0.75, 0.83, 0.95], y_dim
                    )
                    ax.fill_between(
                        quantiles.time.values,
                        quantiles.sel(quantile=0.05),
                        quantiles.sel(quantile=0.95),
                        color=shaded_area_color,
                        alpha=shaded_area_alpha,
                        linewidth=0,
                        **area_kwargs,
                        label="_nolegend_",
                    )
                    ax.fill_between(
                        quantiles.time.values,
                        quantiles.sel(quantile=0.17),
                        quantiles.sel(quantile=0.83),
                        color=shaded_area_color,
                        alpha=shaded_area_alpha,
                        linewidth=0,
                        **area_kwargs,
                        label="_nolegend_",
                    )
                    ax.fill_between(
                        quantiles.time.values,
                        quantiles.sel(quantile=0.25),
                        quantiles.sel(quantile=0.75),
                        color=shaded_area_color,
                        alpha=shaded_area_alpha,
                        linewidth=0,
                        **area_kwargs,
                        label="_nolegend_",
                    )

            elif shaded_area == "auto":
                if thick_line == "mean":
                    data_std = data.std(y_dim)
                    ax.fill_between(
                        data.time.values,
                        main_metric - standard_deviation_multiple * data_std,
                        main_metric + standard_deviation_multiple * data_std,
                        color=shaded_area_color,
                        alpha=shaded_area_alpha,
                        linewidth=0,
                        **area_kwargs,
                        label="_nolegend_",
                    )
                if thick_line == "median":
                    quantiles = data.quantile([0.05, 0.95], y_dim)
                    ax.fill_between(
                        quantiles.time.values,
                        quantiles.sel(quantile=0.05),
                        quantiles.sel(quantile=0.95),
                        color=shaded_area_color,
                        alpha=shaded_area_alpha,
                        linewidth=0,
                        **area_kwargs,
                        label="_nolegend_",
                    )
            elif shaded_area == "std":
                data_std = data.std(y_dim)
                if thick_line is None:
                    main_metric = data.mean(y_dim)
                ax.fill_between(
                    data.time.values,
                    main_metric - standard_deviation_multiple * data_std,
                    main_metric + standard_deviation_multiple * data_std,
                    color=shaded_area_color,
                    alpha=shaded_area_alpha,
                    linewidth=0,
                    **area_kwargs,
                    label="_nolegend_",
                )

            elif shaded_area == "quantiles":
                data_quantile = data.quantile([quantile_min, quantile_max], dim=y_dim)
                if thick_line is None:
                    main_metric = data.median(y_dim)
                ax.fill_between(
                    data.time.values,
                    data_quantile.isel(quantile=0),
                    data_quantile.isel(quantile=1),
                    color=shaded_area_color,
                    alpha=shaded_area_alpha,
                    linewidth=0,
                    **area_kwargs,
                    label="_no_legend_",
                )
        if add_legend:
            ax.legend()


class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(
        self,
        refstd,
        fig=None,
        rect=111,
        label="_",
        srange=(0, 1.5),
        extend=False,
        rlocs=None,
    ):
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

        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF
        from matplotlib.projections import PolarAxes

        self.refstd = refstd  # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        if rlocs == None:
            rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])

        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi / 2
        tlocs = np.arccos(rlocs)  # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)  # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1,
            tick_formatter1=tf1,
        )

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")  # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")  # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left"
        )

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)  # Unused

        self._ax = ax  # Graphical axes
        self.ax = ax.get_aux_axes(tr)  # Polar coordinates

        # Add reference point and stddev contour
        (l,) = self.ax.plot([0], self.refstd, "k*", ls="", ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, "k--", label="_")

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

        # Add RMS contours, and label them
        contours = self.add_contours(levels=5, colors="0.3")  # 5 levels in grey
        plt.clabel(contours, inline=1, fontsize=10, fmt="%.2f")

        self.add_grid()  # Add grid
        self._ax.axis[:].major_ticks.set_tick_out(True)  # Put ticks outward

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        (l,) = self.ax.plot(
            np.arccos(corrcoef), stddev, *args, **kwargs
        )  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(
            np.linspace(self.smin, self.smax), np.linspace(0, self.tmax)
        )
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2 * self.refstd * rs * np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


def plot_hs(
    ds,
    lmin=1,
    lmax=None,
    mmin=0,
    mmax=None,
    reverse=False,
    ax=None,
    cbar_ax=None,
    cbar_kwargs=None,
    **kwargs,
):
    """
    Plot time series of spherical harmonic dataset, with only l and m dimensions, into pyramidal plot.

    Parameters
    ----------
    ds : xr.Dataset
        Spherical harmonics dataset with only l and m dimensions to plot.
    lmin : int, optional
        Minimal degree of the spherical harmonics coefficient to plot, default is 1.
    lmax : int, optional
        Minimal degree of the spherical harmonics coefficient to plot, default is ds.l.max().
    mmin : int, optional
        Minimal order of the spherical harmonics coefficient to plot, default is 1.
    mmax : int, optional
        Minimal order of the spherical harmonics coefficient to plot, default is ds.m.max().
    reverse : bool, optional
        Reverse y-axis, default is False.
    ax : plt.Axes, optional
        Axes on which to plot. By default, use the current axes.
    cbar_ax : plt.Axes, optional
        Axes in which to draw the colorbar.
    cbar_kwargs : dict, optional
        Dictionary of keyword arguments to pass to the colorbar.
    **kwargs : optional
        Additional keyword arguments to plt.matshow() function.


    Returns
    -------
    ax : plt.Axes
        Axes with the plot.
    """
    # -- set default param values
    lmax = ds.l.max().values if lmax is None else lmax
    mmax = lmax if mmax is None else mmax

    ax = plt.gca() if ax is None else ax

    cbar_kwargs = {"shrink": 0.7} if cbar_kwargs is None else cbar_kwargs
    if cbar_ax is None:
        cbar_kwargs.setdefault("ax", ax)
    else:
        cbar_kwargs.setdefault("cax", cbar_ax)

    # -- Creation of the array for matshow with clm and slm
    mat = np.zeros((lmax + 1, 2 * lmax + 1)) * np.NaN
    i, j = np.tril_indices(lmax + 1)

    # set slm before clm to plot order 0 coefficient of clm
    mat[i, lmax - j] = (
        ds.slm.where(ds.l >= lmin, np.NaN)
        .where(np.logical_and(ds.m >= mmin, ds.m <= mmax), np.NaN)
        .isel(l=xr.DataArray(i, dims="tril"), m=xr.DataArray(j, dims="tril"))
    ).values
    mat[i, lmax + j] = (
        ds.clm.where(ds.l >= lmin, np.NaN)
        .where(np.logical_and(ds.m >= mmin, ds.m <= mmax), np.NaN)
        .isel(l=xr.DataArray(i, dims="tril"), m=xr.DataArray(j, dims="tril"))
        .values
    )

    # -- plot
    im = ax.matshow(mat, extent=[-lmax - 0.5, lmax + 0.5, lmax + 0.5, -0.5], **kwargs)

    fig = ax.get_figure()
    cbar = fig.colorbar(im, **cbar_kwargs)

    if reverse:
        ax.invert_yaxis()
    ax.text(
        -lmax / 1.7, lmax / 4, "$S_{l,m}$", fontsize=25, horizontalalignment="center"
    )
    ax.text(
        lmax / 1.7, lmax / 4, "$C_{l,m}$", fontsize=25, horizontalalignment="center"
    )
    ax.set_ylabel("Order", fontsize=17)
    ax.set_xlabel("Degree", fontsize=17)
    ax.xaxis.set_label_position("top")

    def int_abs(tick_value, pos):
        return int(np.abs(tick_value))

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(int_abs))
    ax.tick_params(labelsize=13)
    cbar.ax.tick_params(labelsize=15)

    return ax


def plot_power(
    ds,
    kind="degree",
    unit=None,
    lmin=0,
    lmax=None,
    mmin=0,
    mmax=None,
    unit_kwargs=None,
    hue=None,
    add_legend=True,
    ax=None,
    **kwargs,
):
    """
    Plot degree or order power spectrum of a spherical harmonic dataset, with only l and m dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        Spherical harmonics dataset with only l and m dimensions to plot.
    kind : str
        Type of power spectrum to plot, either "degree" or "order".
    unit : str
        'mewh', 'mmgeoid', 'microGal', 'bar', 'mvcu', or 'norm'
        Unit of the spatial data to use for the conversion. Default is 'mewh' for meters of Equivalent Water Height.
        See utils.harmo.l_factor_conv() doc for details on the units.
    lmin : int, optional
        Minimal degree of the spherical harmonics coefficient to plot, default is 1.
    lmax : int, optional
        Minimal degree of the spherical harmonics coefficient to plot, default is ds.l.max().
    mmin : int, optional
        Minimal order of the spherical harmonics coefficient to plot, default is 1.
    mmax : int, optional
        Minimal order of the spherical harmonics coefficient to plot, default is ds.m.max().
    unit_kwargs : dict | None, optional
        Dictionary of keyword arguments to pass to the l_factor_conv() function.
    hue : str | None, optional
        Dimension or coordinate for which you want multiple lines plotted.
    add_legend : bool
        Add legend with *hue* axis coordinates if given. Default is True.
    ax : plt.Axes, optional
        Axes on which to plot. By default, use the current axes.
    **kwargs : optional
        Additional keyword arguments to plt.plot() function.

    Returns
    -------
    ax : plt.Axes
        Axes with the plot.
    """
    # -- set default param values
    if unit is None and "units" in ds.attrs:
        unit = ds.attrs["units"]
    elif unit is None:
        unit = "mewh"

    lmax = ds.l.max().values if lmax is None else lmax
    mmax = lmax if mmax is None else mmax

    ax = plt.gca() if ax is None else ax
    unit_kwargs = {} if unit_kwargs is None else unit_kwargs

    l_factor = l_factor_conv(ds.l.values, unit=unit, **unit_kwargs)[0]

    if kind == "degree":
        deg_amp = l_factor * np.sqrt(
            (ds.clm**2).sel(m=slice(mmin, mmax)).sum("m")
            + (ds.slm**2).sel(m=slice(mmin, mmax)).sum("m")
        )

        xplt, yplt, hueplt, hue_label = _infer_line_data(
            deg_amp.sel(l=slice(lmin, lmax)), "l", None, hue
        )

        ax.set_xlabel("Degree")

    elif kind == "order":
        deg_amp = np.sqrt(
            ((l_factor * ds.clm) ** 2).sel(l=slice(lmin, lmax)).sum("l")
            + ((l_factor * ds.slm) ** 2).sel(l=slice(lmin, lmax)).sum("l")
        )

        xplt, yplt, hueplt, hue_label = _infer_line_data(
            deg_amp.sel(m=slice(mmin, mmax)), "m", None, hue
        )

        ax.set_xlabel("Order")
    else:
        raise ValueError("Argument 'kind=' must be either 'degree' or 'order'.")

    primitive = ax.plot(xplt, yplt, **kwargs)

    if hue is not None and add_legend and len(deg_amp.dims) == 2:
        ax.legend(handles=primitive, labels=list(hueplt.to_numpy()), title=hue_label)

    ax.set_ylabel(unit)

    return ax
