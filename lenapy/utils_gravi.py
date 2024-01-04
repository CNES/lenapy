import numpy as np


# TODO see if output in xr.Dataset or in array
def compute_plm(lmax, z, mmax=None, normalisation='4pi'):
    """
    Compute all the associated Legendre functions up to a maximum degree and
    order using the recursion relation from [Holmes2002]
    (Adapted from SHTOOLS/pyshtools tools [Wieczorek2018])

    Parameters
    ----------
    lmax: int
        maximum degree of legrendre functions
    z : np.ndarray
        argument of the associated Legendre functions
    mmax : int or NoneType, default None
        maximum order of associated legrendre functions
    normalisation : str, optional, default = '4pi'
        '4pi', 'ortho', or 'schmidt' for use with geodesy 4pi
        normalized, orthonormalized, or Schmidt semi-normalized
        spherical harmonic functions, respectively.

    Returns
    -------
    plm: np.ndarray
        fully-normalized legendre functions

    References
    ----------
    .. [Holmes2002] S. A. Holmes and W. E. Featherstone,
        "A unified approach to the Clenshaw summation and the
        recursive computation of very high degree and order
        normalised associated Legendre functions",
        *Journal of Geodesy*, 76, 279--299, (2002).
        `doi: 10.1007/s00190-002-0216-2 <https://doi.org/10.1007/s00190-002-0216-2>`_
    .. [Wieczorek2018]  M. A. Wieczorek and M. Meschede,
        "SHTools: Tools for working with spherical harmonics",
        *Geochemistry, Geophysics, Geosystems*, 19, 2574–2592, (2018).
        `doi: 10.1029/2018GC007529 <https://doi.org/10.1029/2018GC007529>`_
    """
    # removing singleton dimensions of x
    z = np.atleast_1d(z).flatten()

    # if default mmax, set mmax to be maximal degree
    if mmax is None:
        mmax = lmax

    # scale factor based on Holmes2002
    scalef = 1e-280

    # create multiplicative factors and p
    f1 = np.zeros(((lmax + 1) * (lmax + 2) // 2))
    f2 = np.zeros(((lmax + 1) * (lmax + 2) // 2))
    p = np.zeros(((lmax + 1) * (lmax + 2) // 2, len(z)))

    k = 2
    if normalisation in ('4pi', 'ortho'):
        norm_p10 = np.sqrt(3)

        for l in range(2, lmax + 1):
            k += 1
            f1[k] = np.sqrt(2*l - 1) * np.sqrt(2*l + 1) / l
            f2[k] = (l - 1) * np.sqrt(2*l + 1) / (np.sqrt(2*l - 3) * l)
            for m in range(1, l - 1):
                k += 1
                f1[k] = np.sqrt(2*l + 1) * np.sqrt(2*l - 1) / (np.sqrt(l + m) * np.sqrt(l - m))
                f2[k] = (np.sqrt(2*l + 1) * np.sqrt(l - m - 1) * np.sqrt(l + m - 1) /
                         (np.sqrt(2*l - 3) * np.sqrt(l + m) * np.sqrt(l - m)))
            k += 2

        if normalisation == '4pi':
            norm_4pi = 1
        else:
            norm_4pi = 4*np.pi

    elif normalisation == 'schmidt':
        norm_p10 = 1
        norm_4pi = 1

        for l in range(2, lmax + 1):
            k += 1
            f1[k] = (2*l - 1) / l
            f2[k] = (l - 1) / l
            for m in range(1, l - 1):
                k += 1
                f1[k] = (2*l - 1) / (np.sqrt(l + m) * np.sqrt(l - m))
                f2[k] = np.sqrt(l - m - 1) * np.sqrt(l + m - 1) / (np.sqrt(l + m) * np.sqrt(l - m))
            k += 2

    else:
        raise AssertionError("Unknown normalisation given: ", normalisation, ". It should be either "
                                                                             "'4pi', 'ortho' or 'schmidt'")

    # u is sine of colatitude (cosine of latitude), for z=cos(th): u=sin(th)
    u = np.sqrt(1 - z**2)
    # update where u==0 to minimal numerical precision different from 0 to prevent invalid divisions
    u[u == 0] = np.finfo(np.float64).eps

    # Calculate P(l,0) (not scaled)
    p[0, :] = 1 / np.sqrt(norm_4pi)
    p[1, :] = norm_p10 * z / np.sqrt(norm_4pi)
    k = 1
    for l in range(2, lmax + 1):
        k += l
        p[k, :] = f1[k] * z * p[k - l, :] - f2[k] * p[k - 2*l + 1, :]

    # Calculate P(m,m), P(m+1,m), and P(l,m)
    pmm = np.sqrt(2) * scalef / np.sqrt(norm_4pi)
    rescalem = 1 / scalef
    kstart = 0

    # case lmax == 1, does not go into the 'for' and need m a value for after
    m = 1
    # elif lmax != 1
    for m in range(1, lmax+1):
        rescalem = rescalem * u
        # Calculate P(m,m)
        kstart += m + 1
        pmm = pmm * np.sqrt(2*m + 1) / np.sqrt(2*m)
        if normalisation in ('4pi', 'ortho'):
            p[kstart, :] = pmm
        else:
            p[kstart, :] = pmm / np.sqrt(2*m + 1)

        if m != lmax:  # test if P(m+1,m) exist
            # Calculate P(m+1,m)
            k = kstart + m + 1
            if normalisation in ('4pi', 'ortho'):
                p[k, :] = z * np.sqrt(2*m + 3) * pmm
            else:
                p[k, :] = z * pmm
        else:
            # set up k for rescale P(lmax,lmax)
            k = kstart

        # Calculate P(l,m)
        for l in range(m+2, lmax+1):
            k += l
            p[k, :] = z * f1[k] * p[k - l, :] - f2[k] * p[k - 2*l + 1, :]
            p[k - 2*l + 1, :] = p[k - 2*l + 1, :] * rescalem

        # rescale
        p[k, :] = p[k, :] * rescalem
        p[k - lmax, :] = p[k - lmax, :] * rescalem

    # reshape Legendre polynomials to output dimensions (lower triangle array)
    plm = np.zeros((lmax + 1, lmax + 1, len(z)))
    ind = np.tril_indices(lmax + 1)
    plm[ind] = p

    # return the legendre polynomials and truncating orders to mmax
    return plm[:, :mmax + 1, :]
