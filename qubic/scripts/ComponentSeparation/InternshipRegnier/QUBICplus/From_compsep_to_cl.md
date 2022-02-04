```python
import s4bi
import matplotlib.pyplot as plt
from importlib import reload
from scipy import constants
from astropy.cosmology import Planck15
import qubic
from qubic import mcmc
import healpy as hp
import numpy as np
import os
import qubicplus
#import fgbuster
import fgbuster
from qubic import NamasterLib as nam
center = qubic.equ2gal(0, -57)
from fgbuster.component_model import (CMB, Dust, Dust_2b, Synchrotron, AnalyticComponent)
from fgbuster import basic_comp_sep, get_instrument
# If there is not this command, the kernel shut down every time..
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import pickle
import sys
from qubic import AnalysisMC as amc
from qubic import camb_interface as qc
import scipy

import pickle

def get_coverage(fsky, nside, center_radec=[0., -57.]):
    center = qubic.equ2gal(center_radec[0], center_radec[1])
    uvcenter = np.array(hp.ang2vec(center[0], center[1], lonlat=True))
    uvpix = np.array(hp.pix2vec(nside, np.arange(12*nside**2)))
    ang = np.arccos(np.dot(uvcenter, uvpix))
    indices = np.argsort(ang)
    okpix = ang < -1
    okpix[indices[0:int(fsky * 12*nside**2)]] = True
    mask = np.zeros(12*nside**2)
    mask[okpix] = 1
    return mask

covmap = get_coverage(0.03, nside=256)
N=7
nb_exp=5
cl_noise_s4 = np.zeros(((((N*nb_exp, 1, 16, 4, 1)))))
cl_noise_bi = np.zeros(((((N*nb_exp, 1, 16, 4, 1)))))

for i in range(N):
    with open('results/cl_reshape/cl_ins0_5reals_{}.pkl'.format(i+1), 'rb') as f:
        data = pickle.load(f)
        cl_noise_s4[i*nb_exp:(i+1)*nb_exp, 0]=data['cl_noise']
    with open('results/cl_reshape/cl_ins1_5reals_{}.pkl'.format(i+1), 'rb') as f:
        data = pickle.load(f)
        cl_noise_bi[i*nb_exp:(i+1)*nb_exp, 0]=data['cl_noise']

leff=data['leff']
lmin=40
lmax=2*256-1
delta_ell=30
```


```python
scl_s4 = np.std(cl_noise_s4, axis=0)
covbin_s4, corrbin = amc.get_covcorr_patch(cl_noise_s4[:, 0], stokesjoint=True, doplot=False)
scl_bi = np.std(cl_noise_bi, axis=0)
covbin_bi, corrbin = amc.get_covcorr_patch(cl_noise_bi[:, 0], stokesjoint=True, doplot=False)
```


```python
cl_noise_s4.shape
```




    (35, 1, 16, 4, 1)




```python

```


```python
def plot_errors_lines(leff, err, dl, s, color='r', label=''):
    for i in range(len(leff)):
        if i==0:
            plt.plot([leff[i]-dl/2, leff[i]+dl/2], [err[i,s], err[i,s]],color, label=label)
        else:
            plt.plot([leff[i]-dl/2, leff[i]+dl/2], [err[i,s], err[i,s]],color)
        if i < (len(leff)-1):
            plt.plot([leff[i]+dl/2,leff[i]+dl/2], [err[i,s], err[i+1,s]], color)

def ana_likelihood(rv, leff, fakedata, errors, model, prior, 
                   mylikelihood=mcmc.LogLikelihood, covariance_model_funct=None, otherp=None):
    ll = mylikelihood(xvals=leff, yvals=fakedata, errors=errors, 
                            model = model, flatprior=prior, covariance_model_funct=covariance_model_funct)  

    like = np.zeros_like(rv)
    for i in range(len(rv)):
        like[i] = np.exp(ll([rv[i]]))
    cumint = scipy.integrate.cumtrapz(like, x=rv)
    cumint = cumint / np.max(cumint)
    onesigma = np.interp(0.68, cumint, rv[1:])
    if otherp:
        other = np.interp(otherp, cumint, rv[1:])
        return like, cumint, onesigma, other
    else:
        return like, cumint, onesigma


def explore_like(leff, mcl_noise, errors, lmin, dl, cc, rv, otherp=None,
                 cov=None, plotlike=False, plotcls=False, 
                 verbose=False, sample_variance=True, mytitle='', color=None, mylabel='',my_ylim=None):
    
#     print(lmin, dl, cc)
#     print(leff)
#     print(scl_noise[:,2])
    ### Create Namaster Object
    # Unfortunately we need to recalculate fsky for calculating sample variance
    nside = 256
    lmax = 2 * nside - 1
    if cov is None:
        Namaster = nam.Namaster(None, lmin=lmin, lmax=lmax, delta_ell=dl)
        Namaster.fsky = 0.018
    else:
        okpix = cov > (np.max(cov) * float(cc))
        maskpix = np.zeros(12*nside**2)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl)
    
#     print('Fsky: {}'.format(Namaster.fsky))
    lbinned, b = Namaster.get_binning(nside)

    ### Bibnning CambLib
#     binned_camblib = qc.bin_camblib(Namaster, '../../scripts/QubicGeneralPaper2020/camblib.pickle', 
#                                     nside, verbose=False)
    binned_camblib = qc.bin_camblib(Namaster, 'camblib.pkl', 
                                    nside, verbose=False)


    ### Redefine the function for getting binned Cls
    def myclth(ell,r):
        clth = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=False)[0]
        return clth
    allfakedata = myclth(leff, 0.)
    
    ### And we need a fast one for BB only as well
    def myBBth(ell, r):
        clBB = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=False, specindex=2)[0]
        return clBB

    ### Fake data
    fakedata = myBBth(leff, 0.)        
    
    if sample_variance:
        covariance_model_funct = Namaster.knox_covariance
    else:
        covariance_model_funct = None
    if otherp is None:
        like, cumint, allrlim = ana_likelihood(rv, leff, fakedata, 
                                            errors, 
                                            myBBth, [[0,1]],
                                           covariance_model_funct=covariance_model_funct)
    else:
        like, cumint, allrlim, other = ana_likelihood(rv, leff, fakedata, 
                                            errors, 
                                            myBBth, [[0,1]],
                                           covariance_model_funct=covariance_model_funct, otherp=otherp)
    
    if plotcls:
        if plotlike:
            plt.subplot(1,2,1)
            if np.ndim(BBcov) == 2:
                errorstoplot = np.sqrt(np.diag(errors))
            else:
                errorstoplot = errors
        #plot(inputl, inputcl[:,2], 'k', label='r=0')
        plt.plot(leff, errorstoplot, label=mylabel+' Errors', color=color)
        plt.xlim(0,lmax)
        if my_ylim is None:
            plt.ylim(1e-4,1e0)
        else:
            plt.ylim(my_ylim[0], my_ylim[1])
        plt.yscale('log')
        plt.xlabel('$\\ell$')
        plt.ylabel('$D_\\ell$')
        plt.legend(loc='upper left')
    if plotlike:
        if plotcls:
            plt.subplot(1,2,2)
        p=plt.plot(rv, like/np.max(like), 
               label=mylabel+' $\sigma(r)={0:6.4f}$'.format(allrlim), color=color)
        plt.plot(allrlim+np.zeros(2), [0,1.2], ':', color=p[0].get_color())
        plt.xlabel('r')
        plt.ylabel('posterior')
        plt.legend(fontsize=8, loc='upper right')
        plt.xlim(0,0.1)
        plt.ylim(0,1.2)
        plt.title(mytitle)
    
    if otherp is None:
        return like, cumint, allrlim
    else:
        return like, cumint, allrlim, other
```


```python
### BB Covariance
#BBcov_bi = covbin[0][:, :, 2]
BBcov_s4 = covbin_s4[:, :, 2]
BBcov_bi = covbin_bi[:, :, 2]
### BB sigmas
#sclBB_bi = scl[0][:, 2]
sclBB_s4 = scl_s4[:, 2]
sclBB_bi = scl_bi[:, 2]


method='covariance'

if method=='sigma':
    #to_use_bi = sclBB_bi.copy()
    to_use_s4 = sclBB_s4.copy()
    to_use_bi = sclBB_bi.copy()
elif method=='covariance':
    #to_use_bi = BBcov_bi.copy()
    to_use_s4 = BBcov_s4.copy()
    to_use_bi = BBcov_bi.copy()


### Likelihood
#camblib = qc.read_camblib(global_dir + '/doc/CAMB/camblib.pkl')
rv = np.linspace(0,1,1000)
like_s4, cumint_s4, rlim68_s4, rlim95_s4 = explore_like(leff, sclBB_s4, to_use_s4, lmin, delta_ell, 0.1, rv,
                                 cov=covmap, plotlike=True, plotcls=False,
                                 verbose=True, sample_variance=True, otherp=0.95)

rv = np.linspace(0,1,1000)
like_bi, cumint_bi, rlim68_bi, rlim95_bi = explore_like(leff, sclBB_bi, to_use_bi, lmin, delta_ell, 0.1, rv,
                                 cov=covmap, plotlike=True, plotcls=False,
                                 verbose=True, sample_variance=True, otherp=0.95)
```


    
![png](output_5_0.png)
    



```python
camblib = qc.read_camblib('camblib.pkl')

rv = np.linspace(0,2,1000)

lll = np.arange(512)
cl0 = qc.get_Dl_fromlib(lll, 0, lib=camblib, unlensed=False)[0]   
cl0_01 = qc.get_Dl_fromlib(lll, 0.01, lib=camblib, unlensed=False)[0]   
cl0_06 = qc.get_Dl_fromlib(lll, 0.06, lib=camblib, unlensed=False)[0]   

plt.figure(figsize=(16, 10))
plt.plot(lll, cl0[:, 2])
plt.plot(lll, cl0_01[:, 2])
plt.plot(lll, cl0_06[:, 2])
plot_errors_lines(leff, scl_s4[0, :, :, 0], delta_ell, 2, color='r', label='')
plot_errors_lines(leff, scl_bi[0, :, :, 0], delta_ell, 2, color='b', label='')
plt.yscale('log')
plt.ylim(1e-4, 1)
plt.show()
```


    
![png](output_6_0.png)
    



```python
plt.figure(figsize=(16, 10))
plt.plot(rv, like_s4)
plt.plot(rv, like_bi)
#plt.axvline(rlim68_s4, ls='--', color='black', label='68% C.L : {:.5f}'.format(rlim68_s4))
#plt.axvline(rlim68_bi, ls='--', color='black', label='68% C.L : {:.5f}'.format(rlim68_bi))
plt.xlim(0, 0.1)
plt.legend(fontsize=15)
plt.show()
```

    No handles with labels found to put in legend.



    
![png](output_7_1.png)
    



```python

```


```python
scl_s4.shape
```




    (1, 16, 4, 1)




```python

```


```python

```
