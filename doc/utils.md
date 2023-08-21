Module lenapy.utils
===================

Functions
---------

    
`climato(data, signal=True, mean=True, trend=True, cycle=False, return_coeffs=False)`
:   Analyse du cycle annuel, bi-annuel et de la tendance
    Decompose les données en entrée en :
     Un cycle annuel
     Un cycle bi-annuel
     Une tendance
     Une moyenne
     Un signal résiduel
    Retourne la combinaison voulue de ces éléments en fonction des arguments choisis (signal, mean, trend, cycle)
    Si return_coeffs=True, retourne les coefficients des cycles et tendances
    
    Parameters
    ----------
    signal : Bool (default=True)
        Renvoie le signal résiduel après retrait de la climato, de la tendance, et de la moyenne
    mean : Bool (default=True)
        renvoie la valeur moyenne des données d'entrée
    trend : Bool (default=True)
        renvoie la tendance
    cycle : Bool (default=False)
        renvoie le cycle annuel et bi-annuel
    return_coeffs : Bool (default=False)
        retourne en plus les coefficients des cycles et de la tendance linéaire

    
`diff_3pts(data, dim)`
:   

    
`fill_time(data)`
:   

    
`filter(data, filter_name=<function lanczos>, q=3, **kwargs)`
:   Filtre les données en appliquant sur data le filtre filter_name, avec les paramètres définis dans **kwargs
    Effectue un miroir des données au début et à la fin pour éviter les effets de bords. Ce miroir est réalisé
    après avoir retiré un un polynome d'ordre q fittant au mieux les données.
    
    Parameters
    ----------
    data : xarray DataArray
        Données à filtrer
    filter_name : func (default=Lanczos)
        nom de la fonction de filtrage
    q : integer (default=3)
        ordre du polynome pour l'effet miroir (gestion des bords)
    **kwargs :
        paramètres de la fonction de filtrage demandée

    
`function_climato(t, a, b, c, d, e, f)`
:   

    
`generate_climato(time, coefficients, mean=True, trend=True, cycle=False)`
:   

    
`interp_time(data, other, **kwargs)`
:   

    
`isosurface(data, target, dim, upper=False)`
:   Linearly interpolate a coordinate isosurface where a field
    equals a target
    
    Parameters
    ----------
    field : xarray DataArray
        The field in which to interpolate the target isosurface
    target : float
        The target isosurface value
    dim : str
        The field dimension to interpolate
    upper : bool
        if True, returns the highest point of the isosurface, else the lowest
    
    Examples
    --------
    Calculate the depth of an isotherm with a value of 5.5:
    
    >>> temp = xr.DataArray(
    ...     range(10,0,-1),
    ...     coords={"depth": range(10)}
    ... )
    >>> isosurface(temp, 5.5, dim="depth")
    <xarray.DataArray ()>
    array(4.5)

    
`lanczos(coupure, ordre)`
:   Filtrage de Lanczos
    Implémente un filtre dont la réponse fréquentielle est une porte de largeur spécifiée par "coupure", 
    convoluée à une autre porte dont la largeur est plus étroite d'un facteur "ordre". Temporellement,
    le filtre est tronqué à +/- ordre * coupure / 2
    Plus "ordre" est grand, plus on se rapproche d'un filtre parfait (sinus cardinal)
    Parameters
    ----------
    coupure : integer
    
    ordre : integer
        ordre du filtre

    
`moving_average(npoints)`
:   

    
`to_datetime(data, input_type)`
:   

    
`trend(data)`
: