#!/usr/bin/python
from pylab import *
import os
import sys
import time
import pickle
from importlib import reload


# Specific science modules
import healpy as hp
import numpy as np
import scipy

# Specific qubic modules
from qubicpack.utilities import Qubic_DataDir
from pysimulators import FitsArray
#from mpi4py import MPI
# from pyoperators import MPI
import pysm
import qubic
from qubic import QubicSkySim as qss
from qubic import fibtools as ft
from qubic import camb_interface as qc
from qubic import SpectroImLib as si
from qubic import NamasterLib as nam
from qubic import mcmc
from qubic import AnalysisMC as amc

########################################################################################################
#### Now in a function to loop over ell binning, lmin, coverage
def run_mc(nbmc, Namaster, cov, d, configs, verbose=False, clnoise=None, duration=4):
    
    #### Dictionnary for 150 GHz
    dA = d.copy()
    dA['effective_duration'] = duration
    dA['nside'] = 256
    dA['nf_sub']=1
    dA['filter_nu'] = int(configs[0][-3:]) * 1e9
    print(configs[0], dA['filter_nu']/1e9, dA['effective_duration'],'Years')
    
    dB = d.copy()
    dB['nside'] = 256
    dB['effective_duration'] = duration
    dB['nf_sub']=1
    dB['filter_nu'] = int(configs[1][-3:]) * 1e9
    print(configs[1], dB['filter_nu']/1e9, dB['effective_duration'],'Years')    
    
    ell_bins, b = Namaster.get_binning(d['nside'])
    mask_apo = Namaster.mask_apo
    okpix = cov > (np.max(cov)*0.1)


    seed = np.random.randint(1,100000)
    sky_config = {'cmb': seed}
    Qubic_sky_A = qss.Qubic_sky(sky_config, dA)
    Qubic_sky_B = qss.Qubic_sky(sky_config, dB)

    w=None
    cl_noise_qubic = np.zeros((nbmc, 1, len(ell_bins), 4))
    print('        Starting MC')
    for imc in range(nbmc):
        t0 = time.time()
        qubicnoiseA = Qubic_sky_A.get_partial_sky_maps_withnoise(spatial_noise=True, 
                                                                 noise_only=True, 
                                                                 Nyears=dA['effective_duration'],
                                                                 old_config=old_config)[0][0,:,:]
        qubicnoiseB = Qubic_sky_B.get_partial_sky_maps_withnoise(spatial_noise=True, 
                                                                 noise_only=True, 
                                                                 Nyears=dB['effective_duration'],
                                                                 old_config=old_config)[0][0,:,:]
        print(qubicnoiseA.shape)
        
        ### Compute Spectra:
        # Noise Only
        if verbose: print('   - QUBIC Noise maps')
        leff, cl_noise_qubic[imc, 0, :,:], w = Namaster.get_spectra(qubicnoiseA.T, 
                                                                 map2 = qubicnoiseB.T,
                                                                 purify_e=True, 
                                                                 purify_b=False, 
                                                                 w=w, 
                                                                 verbose=False,
                                                                 beam_correction=True,
                                                                 pixwin_correction=True)
        t1 = time.time()
        print('             Monte-Carlo: Iteration {0:} over {1:} done in {2:5.2f} sec'.format(imc, nbmc,t1-t0))
        
    
    # average MC results
    mcl_noise_qubic = np.mean(cl_noise_qubic, axis=0)[0]
    scl_noise_qubic = np.std(cl_noise_qubic, axis=0)[0]
    
    # The shape of cl_noise_qubic is : (#reals, #bands, #bins, 4)
    print('Old shape:', cl_noise_qubic.shape)
    cl_noise_qubic_reshape = np.moveaxis(cl_noise_qubic, [1, 2, 3], [3, 1, 2])
    print('New shape:', cl_noise_qubic_reshape.shape)
    # Covariance and correlation matrices for TT EE BB TE
    covbin, corrbin = amc.get_covcorr_patch(cl_noise_qubic_reshape, stokesjoint=True, doplot=False)


    return leff, mcl_noise_qubic, scl_noise_qubic, covbin


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
    binned_camblib = qc.bin_camblib(Namaster, global_dir + '/doc/CAMB/camblib.pkl', 
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
            subplot(1,2,1)
            if np.ndim(BBcov) == 2:
                errorstoplot = np.sqrt(np.diag(errors))
            else:
                errorstoplot = errors
        #plot(inputl, inputcl[:,2], 'k', label='r=0')
        plot(leff, errorstoplot, label=mylabel+' Errors', color=color)
        xlim(0,lmax)
        if my_ylim is None:
            ylim(1e-4,1e0)
        else:
            ylim(my_ylim[0], my_ylim[1])
        yscale('log')
        xlabel('$\\ell$')
        ylabel('$D_\\ell$')
        legend(loc='upper left')
    if plotlike:
        if plotcls:
            subplot(1,2,2)
        p=plot(rv, like/np.max(like), 
               label=mylabel+' $\sigma(r)={0:6.4f}$'.format(allrlim), color=color)
        plot(allrlim+np.zeros(2), [0,1.2], ':', color=p[0].get_color())
        xlabel('r')
        ylabel('posterior')
        legend(fontsize=8, loc='upper right')
        xlim(0,0.1)
        ylim(0,1.2)
        title(mytitle)
    
    if otherp is None:
        return like, cumint, allrlim
    else:
        return like, cumint, allrlim, other

########################################################################################################



### Decode arguments
nbmc = int(sys.argv[1])
instA = str(sys.argv[2])
instB = str(sys.argv[3])
duration = float(sys.argv[4])
lmin = int(sys.argv[5])
delta_ell = int(sys.argv[6])
covcut = float(sys.argv[7])
method = str(sys.argv[8])
outdir = str(sys.argv[9])

outnameCl = outdir + '/MC_Cls_{}_{}_nbmc_{}_dur_{}_lmin_{}_dl_{}_cc_{}_meth_{}.pkl'
outnameLike = outdir + '/MC_Like_{}_{}_nbmc_{}_dur_{}_lmin_{}_dl_{}_cc_{}_meth_{}.pkl'

### Instrument Configuration
old_config = False
configs = [instA, instB]
print('----------------------------------------------')
print('We use {}x{}'.format(instA,instB))
print('Nb MC = {}'.format(nbmc))
print('Duration {} Years'.format(duration))
print('lmin = {} - delta_ell = {} - cov-cut = {}'.format(lmin, delta_ell, covcut))
print('Likelihood uses {}'.format(method))
print('Output file for Average Cl will be {}'.format(outnameCl))
print('Output file for Likelihood will be {}'.format(outnameLike))
print('----------------------------------------------')


### Initialize
global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])
dictfilename = global_dir + '/dicts/BmodesNoDustNoSystPaper0_2020.dict'
# Read dictionary chosen
d = qubic.qubicdict.qubicDict()
d.read_from_file(dictfilename)
d['nside'] = 256
center = qubic.equ2gal(d['RA_center'], d['DEC_center'])
d['nf_recon'] = 1
d['nf_sub'] = 1    ### this is OK as we use noise-only simulations

### Bands
dA = d.copy()
dA['filter_nu'] = int(configs[0][-3:]) * 1e9
print('Frequency Band for A:',dA['filter_nu']/1e9)
dB = d.copy()
dB['filter_nu'] = int(configs[1][-3:]) * 1e9
print('Frequency Band for B',dB['filter_nu']/1e9)


### Sky
## Make a sky using PYSM: It will have the expected QUBIC beam, the coverage and noise according to this coverage
## This creates a realization of the sky (a realization of the CMB is there is CMB in sly_config) 
seed = np.random.randint(1)
sky_config = {'cmb': seed}
Qubic_sky_A = qss.Qubic_sky(sky_config, dA)
Qubic_sky_B = qss.Qubic_sky(sky_config, dB)

maptest, coverageA = Qubic_sky_A.get_partial_sky_maps_withnoise(spatial_noise=True, 
                                                                noise_only=True, 
                                                                Nyears=duration,
                                                                old_config=old_config)




################################### Flat Weighting #################################################################
### Create a Namaster object
cov = coverageA.copy()
lmax = 2 * d['nside'] - 1
okpix = cov > np.max(cov) * covcut

### We use Flat weighting
maskpix = np.zeros(12*d['nside']**2)
maskpix[okpix] = 1
Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_ell)



### Run the MC
leff, mcl_noise_qubic, scl_noise_qubic, covbin = run_mc(nbmc, Namaster, cov, d, configs, duration=duration)

### BB Covariance
BBcov = covbin[:, :, 2]
### BB sigmas
sclBB = scl_noise_qubic[:, 2]


if method=='sigma':
    to_use = sclBB.copy()
elif method=='covariance':
    to_use = BBcov.copy()

### Likelihood
camblib = qc.read_camblib(global_dir + '/doc/CAMB/camblib.pkl')
rv = np.linspace(0,2,1000)
like, cumint, rlim68, rlim95 = explore_like(leff, sclBB, to_use, lmin, delta_ell, covcut, rv,
                                 cov=cov, plotlike=False, plotcls=False, 
                                 verbose=True, sample_variance=True, otherp=0.95)

### Save Output
pickle.dump([leff, mcl_noise_qubic, scl_noise_qubic, sys.argv], open(outnameCl, "wb"))
print('Leff:')
print(leff)
print('Errors BB:')
print(scl_noise_qubic[:,2])
pickle.dump([rv, like, cumint, rlim68, rlim95, sys.argv], open(outnameLike, "wb"))
####################################################################################################################







   








