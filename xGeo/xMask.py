import xarray as xr
import xesmf as xe

class xmask():
    __slots__=('_datamask','_mask','_grout')
    def __init__(self,msk_in):
        if type(msk_in)==type(xr.DataArray()):
            self._mask = msk_in
        elif type(msk_in)==type(xr.Dataset()):
            self._datamask = msk_in
        elif type(msk_in)==type('path'):
            self._datamask = xr.open_dataset(msk_in)
        else:
            raise TypeError('mask should be a file, a dataSet or a dataArray')
               
    def set(self,mask):
        self._mask = self._datamask[mask]
    
    def val(self,*args,inverse=False):
        if len(args)==0:
            res = ~self._mask.isnull() ^ inverse
        else:
            res = xr.concat(([self._mask==u for u in args]),dim='_select').any(dim='_select') ^ inverse
        
        if hasattr(self,'_grout'):
            m=xr.Dataset({'mask':res})
            reg=xe.Regridder(m,self._grout,method='conservative_normed')
            return reg(res)
        else:
            return res
        
    def regrid(self,gr_out):
        self._grout=gr_out