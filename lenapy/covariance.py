import numpy as np
import matplotlib.pyplot as plt
import yaml
from .utils import *

class estimateur:
    def __init__(self,data,degree,tref=None,sigma=None,datetime_unit="s"):
        if type(tref)==type(None):
            tref=data.time.isel(time=0)
        if type(sigma)==type(None):
            sigma=np.diag(np.ones(len(data.time)))
        
        t1=(data.time-tref)/pd.to_timedelta("1%s"%datetime_unit).asm8
        self.deg=xr.DataArray(data=np.arange(degree),dims='degree',coords=dict(degree=np.arange(degree)))
        self.data = data
        self.expl = (t1**self.deg)
        self.X = self.expl.values
        self.Y = data.values
        self.cov_matrix=sigma

    def OLS(self):
        H = np.linalg.inv(np.dot(self.X.T,self.X))
        self.beta = np.dot(H,np.dot(self.X.T,self.Y))
        self.params = xr.zeros_like(self.deg)+self.beta
        
    def GLS(self):
        eigenvalues,eigenvectors=np.linalg.eig(self.cov_matrix)
        eigenvalues=np.where(eigenvalues<0,0.,eigenvalues)
        C=np.linalg.inv((np.sqrt(eigenvalues)*eigenvectors).real)
        
        X = np.dot(C,self.X)
        Y = np.dot(C,self.Y)
        H = np.linalg.inv(np.dot(X.T,X))

        self.beta = np.dot(H,np.dot(X.T,Y))
        self.params = xr.zeros_like(self.deg)+self.beta
        
    def residus(self):
        return self.data - self.estimation()
    
    def estimation(self):
        return (self.expl*self.params).sum('degree')
    
    def sigma2(self):
        return (self.residus()**2).mean().values
    
    def sigma(self):
        return np.sqrt(self.sigma2())
    
    def cov_estim(self):
        H = np.linalg.inv(np.dot(self.X.T,self.X))
        A = np.dot(np.dot(self.X.T,self.cov_matrix),self.X)
        return np.dot(np.dot(H,A),H)

    def std_err(self):
        return xr.zeros_like(self.deg)+np.sqrt(np.diag(self.cov_estim()))
    

class cov_element:
    
    def __init__(self,time,type,sig,t0=None,tmin=None,tmax=None,dt=None,bias_type=None):
        self.time = time
        self.type = type
        self.bias_type = bias_type
        
        if self.type=='bias':
            t1 = xr.where(self.time<t0,1.,0.)
            t2 = t1.rename(time='time1')
            self.sigma = sig**2*t1*t2

        elif self.type=='drift':
            t=self.time
            if tmax!=None:
                t = xr.where(t<=tmax,t,tmax)
            if tmin!=None:
                t = xr.where(t>=tmin,t,tmin)

            t1=t.astype('float')*1.e-9/86400.
            t2=t1.rename(time='time1')
            self.sigma = sig**2*t1*t2

        elif self.type=='noise':
            t=self.time
            if tmax!=None:
                t=xr.where(t<=tmax,t,tmax)
            if tmin!=None:
                t=xr.where(t>=tmin,t,tmin)

            t1=t
            t2=t1.rename(time='time1')
            self.sigma = sig**2*np.exp(-0.5*((t1-t2)/dt)**2)

        elif self.type=='random':
            self.sigma=sig**2*np.diag(np.ones(len(self.time)))

        self.ajuste()
            
    def ajuste(self):
        if self.bias_type!=None:
            eigenvalues,eigenvectors=np.linalg.eig(self.sigma)

            A=xr.ones_like(self.sigma)
            A.values=(np.sqrt(eigenvalues)*eigenvectors).real
            if self.bias_type=='centered':
                B=A.mean('time')
            elif self.bias_type=='begin' or self.bias_type=='right':
                B=A.isel(time=0)
            elif self.bias_type=='end' or self.bias_type=='left':
                B=A.isel(time=-1)
            else:
                raise Exception("unknown bias type : %s"%self.bias_type)
            Ap=A-xr.ones_like(A.isel(time1=0))*B
            self.sigma.values = np.dot(Ap,Ap.T) 
            
class covariance:
    
    def __init__(self,data):
        self.time=data.time
        self.data=data
        self.sigma=None
            
            
    def add_errors(self,typ,sigma=1.,t0=None,t1=None,t2=None,dt=None,bias_type=None):

        if type(self.sigma)!=type(None):
            self.sigma=self.sigma + cov_element(self.time, typ, sigma, t0, t1, t2, dt, bias_type).sigma
        else:
            self.sigma=cov_element(self.time, typ, sigma, t0, t1, t2, dt, bias_type).sigma

    def read_yaml(self,fic):
        
        def lit(var,label,func=lambda x:x):
            try:
                return func(var[label])
            except:
                return None
            
        self.sigma=None
        yaml_file = open(fic, 'r')
        yaml_content = yaml.full_load(yaml_file)
        for e in yaml_content['errors']:

            param = lit(e,'parameters')
            typ   = lit(e,'type')
            sigma = lit(param,'value')
            t0    = lit(param,'time',JJ_to_date)
            t1    = lit(param,'time_min',JJ_to_date)
            t2    = lit(param,'time_max',JJ_to_date)
            dt    = lit(param,'span',lambda x:np.timedelta64(int(x),'D'))
            bias_type = lit(param,'bias_type')
            conv = lit(param,'conversion_factor')
            if conv==None:
                conv=1.
            
            self.add_errors(typ,conv*sigma,t0,t1,t2,dt,bias_type)

            
    def visu(self,n=100,sigmax=None,save=None):
        fig, ax = plt.subplots(1,3, figsize=(18, 5))
        ax=ax.ravel()

        eigenvalues,eigenvectors=np.linalg.eig(self.sigma)
        eigenvalues=np.where(eigenvalues<0,0.,eigenvalues)
        A=(np.sqrt(eigenvalues)*eigenvectors).real

        mu=0
        sig=1.
        for i in range(n):
            s=np.random.normal(mu,sig,len(self.time))
            ax[0].plot(self.time,np.dot(A,s))
        ax[0].grid()
        ax[1].plot(self.time,np.sqrt(self.sigma.values.diagonal()))
        ax[1].set_ylim(0)
        ax[1].grid()
        self.sigma.plot(ax=ax[2],vmax=sigmax)
        if save!=None:
            fig.savefig(save)
        
    def plot_trend(self,out='OLS',freq='1M',save=None,**kwargs):
        # Periode centrale sur toute l'emprise des donnees
        tmin=self.time.min().values
        tmax=self.time.max().values
        t_ = pd.date_range(tmin,tmax,freq=freq)
        # Duree de la periode multiple de l'echantillonnage
        dt_ = (t_[4::2]-tmin)
        # Vecteurs temps et delta_t
        t=xr.DataArray(data=t_,dims='time',coords=dict(time=t_),attrs=dict(long_name='Central date of period')).chunk(time=100)
        dt=xr.DataArray(data=dt_,dims='delta',coords=dict(delta=dt_.values),attrs=dict(long_name='Period length (yr)')).chunk(delta=100)


        res = xr.DataArray(data=cov_trend(data=self.data,sigma=self.sigma,t1=(t-dt/2.),t2=(t+dt/2.),out=out),
                         dims=['time','delta'],coords=dict(time=t,delta=dt_.days/365)).T
        
        fig, ax = plt.subplots(1,1, figsize=(12, 8))
        res.plot(**kwargs)
        plt.grid()        
        if save!=None:
            fig.savefig(save)

        
def cov_trend_(data,sigma,t1,t2,out='OLS'):
    t1=np.datetime64(t1,'ns')
    t2=np.datetime64(t2,'ns')
    if t1>=data.time.min() and t2<=data.time.max():
        d=data.sel(time=slice(t1,t2))
        sig=sigma.sel(time=slice(t1,t2),time1=slice(t1,t2))
        est=estimateur(d,2,sigma=sig)
        if out=='OLS':
            est.OLS()
            return est.params[1].values
        elif out=='GLS':
            est.GLS()
            return est.params[1].values
        elif out=='err':
            return est.std_err()[1].values
    else:
        return np.nan

cov_trend = np.vectorize(cov_trend_,excluded=['data','sigma','out'])
                      