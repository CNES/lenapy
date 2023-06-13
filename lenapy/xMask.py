import xarray as xr
import xesmf as xe
import numpy as np

class xmask():
    __slots__=('_datamask','_grout','_val','value','inverse','layer')
    def __init__(self,msk_in):
        if type(msk_in)==type(xr.DataArray()):
            self._datamask = xr.Dataset({'mask':msk_in})
        elif type(msk_in)==type(xr.Dataset()):
            self._datamask = msk_in
        elif type(msk_in)==type('path'):
            self._datamask = xr.open_dataset(msk_in)
        else:
            raise TypeError('mask should be a file, a dataSet or a dataArray')
               
        self.layer='mask'
        self.value=None
        self._val=None
        self.inverse=False
        
    def set(self,mask=None,value=None,inverse=False):
        if type(mask)!=type(None):
            self.layer=mask
        if type(value)!=type(None):
            self.value=np.ravel(value)
        if type(inverse)!=type(None):
            self.inverse=inverse
        self._val=None
            

    @property
    def val(self):
        if type(self._val)==type(None):
            if type(self.value)==type(None):
                self._val = ~self._datamask[self.layer].isnull() ^ self.inverse
            else:
                self._val = xr.concat(([self._datamask[self.layer]==u for u in self.value]),dim='_select').any(dim='_select') ^ self.inverse

        return self._val
        
    def regrid(self,gr_out):
        self._grout=xr.Dataset({
            "latitude":gr_out.latitude,
            "longitude":gr_out.longitude
            })
        reg=xe.Regridder(self._datamask,self._grout,method='nearest_s2d')
        self._datamask = reg(self._datamask)
        
        self._val=None
