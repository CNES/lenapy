import matplotlib.pyplot as plt

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
    if thick_line is not None:
        plot_line = main_metric.plot(ax=ax, color=thick_line_color, **line_kwargs, label=f"{variable} {thick_line}")
    if shaded_area_color is None:
        shaded_area_color = plot_line[0].get_color()
    if 'std' in shaded_area:
        data_std = data.std(y_dim)
        if thick_line is None:
            main_metric = data.mean(y_dim)
        if shaded_area == 'std':
            ax.fill_between(data.time.values, main_metric-data_std, main_metric+data_std,
                            color=shaded_area_color, alpha=shaded_area_alpha, 
                            linewidth=0, **area_kwargs, label=variable +r' 1$\sigma$')
        if shaded_area == '2std':
            ax.fill_between(data.time.values, main_metric-2*data_std, main_metric+2*data_std,
                            color=shaded_area_color, alpha=shaded_area_alpha, 
                            linewidth=0, **area_kwargs, label=variable +r' 2$\sigma$')
        if shaded_area == '3std':
            ax.fill_between(data.time.values, main_metric-3*data_std, main_metric+3*data_std,
                            color=shaded_area_color, alpha=shaded_area_alpha, 
                            linewidth=0, **area_kwargs, label=variable +r' 3$\sigma$')

    elif shaded_area == 'quantiles':
        data_quantile = data.quantile([quantile_min, quantile_max],
                                      dim = y_dim)
        if thick_line is None:
            main_metric = data.median(y_dim)
        ax.fill_between(data.time.values, data_quantile.isel(quantile=0), data_quantile.isel(quantile=1),
                        color=shaded_area_color, alpha=shaded_area_alpha,
                        linewidth=0, **area_kwargs,
                        label=f'{variable} ci {int(100*quantile_min)}-{int(100*quantile_max)}'+r'%')
    elif shaded_area is None:
        pass
    else:
        raise ValueError("shaded_area can only be 'std','2std','3std', 'quantiles' or None.")

    if add_legend:
        ax.legend()