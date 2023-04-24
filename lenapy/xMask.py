import xarray as xr
import xesmf as xe

class xmask():
    __slots__=('_datamask','_mask','_grout','_val','value','inverse')
    def __init__(self,msk_in):
        if type(msk_in)==type(xr.DataArray()):
            self._mask = msk_in
        elif type(msk_in)==type(xr.Dataset()):
            self._datamask = msk_in
        elif type(msk_in)==type('path'):
            self._datamask = xr.open_dataset(msk_in)
        else:
            raise TypeError('mask should be a file, a dataSet or a dataArray')
               
        self.value=None
        self._val=None
        self.inverse=False
        
    def set(self,mask=None,value=None,inverse=False):
        if type(mask)!=type(None):
            self._mask = self._datamask[mask]
        if type(value)!=type(None):
            self.value=value
        if type(inverse)!=type(None):
            self.inverse=inverse

    @property
    def val(self):
        if type(self._val)==type(None):
            if type(self.value)==type(None):
                res = ~self._mask.isnull() ^ self.inverse
            else:
                res = xr.concat(([self._mask==u for u in self.value]),dim='_select').any(dim='_select') ^ self.inverse

            if hasattr(self,'_grout'):
                m=xr.Dataset({'mask':res})
                reg=xe.Regridder(m,self._grout,method='conservative_normed')
                self._val = reg(res)
            else:
                self._val = res
        return self._val
        
    def regrid(self,gr_out):
        self._grout=xr.Dataset({
            "latitude":gr_out.latitude,
            "longitude":gr_out.longitude
            })
