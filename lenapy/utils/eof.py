import numpy as np
import dask.array as da
import xarray as xr

class EOF:
    """This class implements EOF (Empirical Orthogonal Functions) decomposition.
    

    Parameters
    ----------
    data : xarray.DataArray
        Contains the data on which to apply the decomposition. Must have a "time" coordinate.
    dim : str or list of str
        Dimensions of the EOF.
    k : int
        order of the last EOF to compute
    """

    def __init__(self, data, dim, k=6):
        self.data = data
        self.dim = np.ravel(dim)
        
        self.moyenne = self.data.mean('time').persist()

        # Remove the mean value
        M = (self.data - self.moyenne).stack(pos=tuple(self.dim)).fillna(0).persist()
        
        # Create the covariance matrix
        mat = M.transpose(..., 'pos', 'time').chunk(dict(pos=-1,time=-1))
        
        
        def decompose(S):
            u,s,v=da.linalg.svd_compressed(da.array(S),k=k)

            return u,s**2

        res = xr.apply_ufunc(
            decompose, 
            mat, 
            input_core_dims=[['pos','time']],
            output_core_dims=[['pos','order'],['order']],
            exclude_dims=set(('time',)),
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float,float],
            dask_gufunc_kwargs={'output_sizes': {'order': k}}
        )

    
        # Create xarray DataArray for eigenvectors and eigenvalues
        self.vp = res[0].persist()
        self.val = res[1].persist()

        # Project the signal onto the eigenvectors
        self.lbd = (M * self.vp).sum('pos')

    def eof(self, n):
        """
        Returns the n-th EOF.
        
        Parameters
        ----------
        n : int
            Index of the EOF to return.
        
        Returns
        -------
        xarray.DataArray
            The n-th EOF.
        """
        return self.vp.sel(order=n).unstack('pos')
    
    def amplitude(self, n):
        """
        Returns the amplitude timeseries of the n-th EOF.
        
        Parameters
        ----------
        n : int
            Index of the EOF amplitude to return.
        
        Returns
        -------
        xarray.DataArray
            Amplitude timeseries of the n-th EOF.
        """
        return self.lbd.sel(order=n)
    
    def variance(self, n):
        """
        Returns the n-th eigenvalue (variance).
        
        Parameters
        ----------
        n : int
            Index of the variance to return.
        
        Returns
        -------
        xarray.DataArray
            The n-th eigenvalue (variance).
        """
        return self.val.sel(order=n)
    
    def reconstruct(self, n):
        """
        Reconstructs the dataset using the first n EOFs.
        
        Parameters
        ----------
        n : int
            Number of EOFs to use for reconstruction.
        
        Returns
        -------
        xarray.DataArray
            Reconstructed dataset.
        """
        return (self.vp.sel(order=slice(None,n)) * self.lbd.sel(order=slice(None,n))).sum('order').unstack('pos') + self.moyenne
    
    def explained_variance(self, n):
        """
        Returns the explained variance by the first n EOFs.
        
        Parameters
        ----------
        n : int
            Number of EOFs to use for explained variance calculation.
        
        Returns
        -------
        float
            Explained variance by the first n EOFs.
        """
        return (self.val.sel(order=slice(None,n)).sum('order') / self.val.sum('order'))


