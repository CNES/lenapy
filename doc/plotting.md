Module lenapy.plotting
======================

Functions
---------

    
`plot_timeseries_uncertainty(xgeo_data, x_dim='time', y_dim=None, thick_line='median', shaded_area='quantiles', quantile_min=0.05, quantile_max=0.95, thick_line_color=None, shaded_area_color=None, shaded_area_alpha=0.2, ax=None, label=None, line_kwargs={}, area_kwargs={}, add_legend=True)`
:   Affiche une série temporelle avec une enveloppe d'incertitude.
    
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