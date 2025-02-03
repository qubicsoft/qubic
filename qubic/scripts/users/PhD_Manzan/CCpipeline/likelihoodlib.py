from importlib import reload
import numpy as np
from pylab import *
#import matplotlib.pyplot as plt
import sys
import pysm3.units as u
from pysm3 import utils
import os


import qubic
from scipy import constants
import healpy as hp
import numpy as np
import qubicplus
from qubic import NamasterLib as nam
from qubicpack.utilities import Qubic_DataDir
import scipy
import pysm3
import qubic
import pickle
from qubic import fibtools as ft
from qubic import camb_interface as qc
from qubic import SpectroImLib as si
from qubic import mcmc


global_dir = './'  #'/sps/qubic/Users/emanzan/libraries/qubic/qubic'  #Qubic_DataDir(datafile='instrument.py', datadir='/home/elenia/libraries/qubic/qubic/')
camblib_filename = 'camblib_with_r_from_m0.1_to_p0.1_minstep1e-06.pickle'  #'camblib_with_r_from_0.0001.pickle'


#def covariance matrix
def eval_cov_matrix(ell, sigma_cls):
    cov_matrix=np.zeros((len(ell), len(ell), 4))
    for i in range(4): #for each spectra, fill diagonal
        np.fill_diagonal(cov_matrix[:,:,i], sigma_cls[:,i]**2)

    return cov_matrix


def plot_errors_lines(leff, err, dl, s, color='r', label=''):
    for i in range(len(leff)):
        if i==0:
            plot([leff[i]-dl/2, leff[i]+dl/2], [err[i,s], err[i,s]],color, label=label)
        else:
            plot([leff[i]-dl/2, leff[i]+dl/2], [err[i,s], err[i,s]],color)
        if i < (len(leff)-1):
            plot([leff[i]+dl/2,leff[i]+dl/2], [err[i,s], err[i+1,s]], color)
        yscale('log')
        xlabel('$\\ell$')
        ylabel('$\\Delta D_\\ell$')
        legend(loc='best')


def ana_likelihood(rv, leff, data, errors, model, prior, 
                   mylikelihood=mcmc.LogLikelihood, covariance_model_funct=None, otherp=None):
    
    #def Log_likelihood ll from qubic.mcmc using binned Cls and their errors (sCl)
    ll = mylikelihood(xvals=leff, yvals=data, errors=errors, 
                            model = model, flatprior=prior, covariance_model_funct=covariance_model_funct) 
    like = np.zeros_like(rv)
    
    #for each r value, eval the likelihood L(r) from the Log_likelihood
    for i in range(len(rv)):
        like[i] = np.exp(ll([rv[i]]))
        #here: given a r value, rv[i], eval. model = (binned) ClBB(rv[i]), and compare it to
        #yvals = (binned) ClBB(r=0) (i.e. fakedata) using Ncov from errors = sCl_noise
        #which means evalutating: logLLH = lp - 0.5 * (((yvals - model(rv[i])).T * invcov * (yvals - model(rv[i])))
        #and return Log_likelihood of rv[i]. Then from LogL calc L.
        #print(rv[i],ll([rv[i]]),like[i])
    
    #now eval integral of L(r) and normalized it
    cumint = scipy.integrate.cumtrapz(like, x=rv)
    cumint = cumint / np.max(cumint)
    
    #extrapolate r value at 1sigma (68% CL)
    onesigma = np.interp(0.68, cumint, rv[1:])
    
    #if otherp!=0, eval r value also at the specified sigmas (e.g. 3sigmas)
    if otherp:
        other = np.interp(otherp, cumint, rv[1:])
        return like, cumint, onesigma, other
    else:
        return like, cumint, onesigma


'''This function calls the evaluation of L(r) from ana_likelihood and makes plots of DlBB_qubic_noise and of L(r).
The function takes in:
- leff = ell vector,
- mcl_noise = mean of Cls noise (can be 0), errors = sigma of Cls noise
- lmin, deltal,
- cc = covcut ie. a value to def. if a pixel in the coverage map is good or not
- rv = vec of r values
- otherp = Nsigmas, e.g. 0.68 or 0.95
- cov = coverage map
- other variable for plotting

The function creates a Namaster object based off the coverage (mask), lmin, lmax=2*nside-1
From the Namaster obj. and the file camblib.pkl (containing TT,EE,BB,TE Cls for 100 r in [0,1]) creates binned Dl spectra
Def a function that returns the binned Dl (BB) spectra for any given r
Def "fakedata" as Dl_BB binned spectra for r=0
Through ana_likelihood, eval L(r) btw [0,1] by comparing Dl_BB for given r and Dl_BB(r=0)+errors
''' 
def explore_like(leff, mcl_BB, errors, lmin, dl, cc, rv, otherp=None, cov=None, verbose=False, sample_variance=True):
    
    ### Create Namaster Object: read a mask
    # Unfortunately we need to recalculate fsky for calculating sample variance
    nside = 256
    lmax = 355  #2 * nside - 1
    if cov is None:
        Namaster = nam.Namaster(None, lmin=lmin, lmax=lmax, delta_ell=dl)
        Namaster.fsky = 0.03
    else:
        okpix = cov > (np.max(cov) * float(cc))
        maskpix = np.zeros(12*nside**2)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl)
        #     print('Fsky: {}'.format(Namaster.fsky))
    lbinned, b = Namaster.get_binning(nside)

    ### Bin the spectra in "/doc/CAMB/camblib.pkl" using Namaster through CambLib
    
    #binned_camblib = qc.bin_camblib(Namaster, global_dir + '/doc/CAMB/camblib.pkl', nside, verbose=False)
    
    binned_camblib = qc.bin_camblib(Namaster, global_dir + camblib_filename, nside, verbose=False)

    ### Redefine the function for getting binned Cls given certain r and mask
    def myclth(ell,r):
        clth = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True)[1]
        return clth
    
    ### And we need a fast one for BB only as well
    def myBBth(ell, r):
        clBB = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True, specindex=2)[1]
        return clBB

    ### Def data ie. Cls BB from recon. cmb 
    data = mcl_BB        
    
    #def sample_variance
    if sample_variance:
        covariance_model_funct = Namaster.knox_covariance
    else:
        covariance_model_funct = None
    
    #eval likelihood L(r), integral and r at different sigmas starting from binned Cls at r=0 with sCls
    if otherp is None:
        like, cumint, allrlim = ana_likelihood(rv, leff, data, 
                                            errors, 
                                            myBBth, [[0,1]],
                                           covariance_model_funct=covariance_model_funct)
    else:
        like, cumint, allrlim, other = ana_likelihood(rv, leff, data, 
                                            errors, 
                                            myBBth, [[0,1]],
                                           covariance_model_funct=covariance_model_funct, otherp=otherp)
    
    if otherp is None:
        return like, cumint, allrlim
    else:
        return like, cumint, allrlim, other

'''This function takes in:
The function calls the "explore_like function" to eval: the likelihood L(r), integral of L, r value with 68% CL and 95% CL

The function returns:
- leff= ell,
- scl_noise_qubic*factornoise**2 * = Cls of qubic noise weighted by atm. noise,
- rv = vector of r values (btw 0,1),
- like = likelihood L(r), cumint= integral of L, rlim68 = r value with 68% CL, rlim95 = r value with 95% CL
'''
def get_results(ell, mcl, scl, coverage, 
                method, lmin=40, delta_ell=30, covcut=0.1, rv=None, factornoise=1.):
    leff=ell
    #check atm noise
    if factornoise != 1.:
        print('**** BEWARE ! Using Factornoise = {}'.format(factornoise))
        
    ### def BB mean cl
    mclBB = mcl   
    ### def BB sigmas
    sclBB = scl*factornoise

    #check method to use: sclBB or BBcov
    if method=='sigma':
        to_use = sclBB.copy()
    elif method=='covariance':
        print('Covariance method has not been implemented yet here')
    #print(to_use)
    
    ### Likelihood L(r)
    if rv is None:
        rv = np.linspace(-0.01,0.01,20000)
        #rv = np.linspace(0,1,10000)

    like, cumint, rlim68, rlim95 = explore_like(leff, mclBB, to_use, lmin, delta_ell, covcut, rv,
                                     cov=coverage, verbose=True, sample_variance=False, otherp=0.95)#, delensing_residuals=delensing_residuals)
    
    return leff, scl*factornoise**2, rv, like, cumint, rlim68, rlim95
