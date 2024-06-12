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
    """

    def __init__(self, data, dim):
        self.data = data
        self.dim = np.ravel(dim)
        
        self.moyenne = self.data.mean('time').persist()

        # Remove the mean value
        M = (self.data - self.moyenne).stack(pos=tuple(self.dim)).fillna(0)
        
        # Create the covariance matrix
        mat1 = M.transpose(..., 'pos', 'time').data
        mat2 = M.transpose(..., 'time', 'pos').data
        S = da.matmul(mat1, mat2)
        self.S = S
        self.M = M

        # Decomposition function for Dask
        def decompose(S):
            val, vec = np.linalg.eigh(S)
            return np.append(vec, np.expand_dims(val, S.ndim-1), axis=S.ndim-1)
        
        # Chunk size for the decomposition function
        cs = S.chunksize[:-1] + (S.chunksize[-1] + 1,)
        
        # Eigenvalue decomposition distributed by Dask
        r = S.map_blocks(decompose, chunks=(cs))

        # Create xarray DataArray for eigenvectors and eigenvalues
        c = M.isel(time=0, pos=0).dims + ('pos', 'order')
        self.vp = xr.DataArray(r[...,:-1], dims=c, coords=dict(M.drop(['time']).coords)).persist()

        c = M.isel(time=0, pos=0).dims + ('order',)
        self.val = xr.DataArray(r[...,-1], dims=c, coords=dict(M.drop(['time','pos']).coords)).persist()

        # Project the signal onto the eigenvectors
        self.lbd = (M * self.vp).sum('pos').persist()

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
        return self.vp.sel(order=-n).unstack('pos')
    
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
        return self.lbd.sel(order=-n)
    
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
        return self.val.sel(order=-n)
    
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
        return (self.vp.sel(order=slice(-n, None)) * self.lbd.sel(order=slice(-n, None))).sum('order').unstack('pos') + self.moyenne
    
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
        return (self.val.sel(order=slice(-n, None)).sum('order') / self.val.sum('order'))


