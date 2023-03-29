import numpy as np
from pylab import *
import scipy
from scipy import constants
import healpy as hp
import sys
import os
import os.path as op
import pickle
from operator import itemgetter

import pysm3
import pysm3.units as u
from pysm3 import utils

import qubic
from qubic import NamasterLib as nam
from qubic import fibtools as ft
from qubic import camb_interface as qc
from qubic import SpectroImLib as si
from qubic import mcmc
from qubicpack.utilities import Qubic_DataDir

import fgbuster
#############################################################################
# This library contains all things for
# - posterior likelihood evaluation
# ###########################################################################

#define path to files with r array or cmb Cls -----------------------------------
global_dir = './'   # '/sps/qubic/Users/emanzan/libraries/qubic/qubic'
camblib_filename = 'camblib_with_r_from_m0.1_to_p0.1_minstep1e-06.pickle' #'camblib_with_r_from_0.0001.pickle'  # #'/doc/CAMB/camblib.pkl' #'camblib_with_r_from_0.0001.pickle'

fgb_path = fgbuster.__path__ #if you want to use fgb to generate cmb Cls (start from Planck 2018 best-fit)
CMB_CL_FILE = op.join(fgb_path[0]+'/templates/Cls_Planck2018_%s.fits')
# -------------------------------------------------------------------------

def get_cmb_cls_from_r_Alens(r, Alens):
    '''
    This function generates cmb Cls from a given r value and Alens (amplitude of lensing residual) using Planck 2018 best-fit power spectra
    ''' 
    #Define cmb spectrum
    #clBB = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=False, specindex=2)[0]#[1]
    cmb_cls = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000] #This is the Planck 2018 best-fit. Here l=0,1 are set to zero as expected by healpy
    if Alens != 1.:
        #Define lensing residual, Alens
        cmb_cls[2] *= Alens #here Alens = 0 : no lensing. Alens = 1: full lensing 
    if r:
        #Add primordial B-modes with the given r 
        cmb_cls[2] += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[2,:4000]#[:,:4000] #this takes unlensed cmb B-modes with r=1 and scales them to whatever r is given
    return cmb_cls


def Cls_2_Dls(Cls):
    '''
    This function converts Cls (dim = N_spectra, N_ell) to Dls (dim = N_spectra, N_ell)
    ''' 
    num_spectra = Cls.shape[0]
    num_ell = Cls.shape[1]
    ell = np.arange(0,num_ell)
    Dls = np.zeros((num_spectra, num_ell))
    for i in range(num_spectra):
        Dls[i,:] = Cls[i,:]*ell*(ell+1)/(2*np.pi)
    
    return Dls


def ClsXX_2_DlsXX(Cls_XX):
    '''
    This function converts Cls_XX (dim = N_ell) to Dls_XX (dim = N_ell)
    ''' 
    num_ell = Cls_XX.shape[0]
    ell = np.arange(0,num_ell)
    Dls_XX = Cls_XX*ell*(ell+1)/(2*np.pi)
    
    return Dls_XX


def ClsXX_2_DlsXX_binned(ell, Cls_XX):
    '''
    This function converts Cls_XX (dim = N_ell) to Dls_XX (dim = N_ell) using the binning of the input ell
    ''' 
    num_ell = ell.shape[0] #binned ell
    Dls_XX = np.zeros(num_ell) #binned Dls
    for i in range(num_ell): #fill in the Dls bins
        Dls_XX[i] = Cls_XX[i]*ell[i]*(ell[i]+1)/(2*np.pi)
    
    return Dls_XX


def get_DlsBB_from_Planck(ell, r, Alens):
    '''
    Wrapper function that takes ClsBB from Plank2018 and converts them into Dls with a specified binning
    ''' 
    clBB = get_cmb_cls_from_r_Alens(r, Alens)[2]
    DlsBB = ClsXX_2_DlsXX_binned(ell, clBB)
    return DlsBB[:]


def get_cmb_DlsBB_from_binned_CAMBlib_r_Alens(ell, binned_camblib, r, Alens):
    '''
    This function generates Dls_BB from a given CAMB lib and binning, considering a given r and Alens (amplitude of lensing residual)
    '''  
    #Dls
    DlsBB = qc.get_Dl_fromlib(ell, 0, lib=binned_camblib, unlensed=False, specindex=2)[0] #lensed spectra with r=0
    DlsBB *= Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
    DlsBB += qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True, specindex=2)[1] #this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given

    return DlsBB[:]    


def get_DlsBB_from_CAMB(ell, r, Alens, nside=256, coverage=cov, cov_cut=0.1, lmin=21, delta_ell=35, lmax = 335):
    '''
    Wrapper function that defines a NaMaster object with a specific mask and ell binning, then uses it to create a camb library and get Dls_BB
    '''     
    #NaMaster obj
    okpix = coverage > (np.max(coverage) * float(cov_cut))
    maskpix = np.zeros(hp.nside2npix(nside))
    maskpix[okpix] = 1
    Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=delta_ell)#, aposize=2.0)
    
    #binned
    binned_camblib = qc.bin_camblib(Namaster, global_dir + camblib_filename, nside, verbose=False)
    #call Dls generator function
    DlsBB = get_cmb_DlsBB_from_binned_CAMBlib_r_Alens(ell, binned_camblib, r, Alens)

    return DlsBB[:]


def eval_cov_matrix(ell, sigma_cls):
    '''
    This function generates a covariance matrix starting from the input std of Cls
    ''' 
    cov_matrix=np.zeros((len(ell), len(ell), 4))
    for i in range(4): #for each spectra, fill diagonal
        np.fill_diagonal(cov_matrix[:,:,i], sigma_cls[:,i]**2)

    return cov_matrix


def plot_errors_lines(leff, err, dl, s, color='r', label=''):
    '''
    This function plot cls with error bars
    ''' 
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


def get_likelihood(rv, leff, data, errors, model, prior, mylikelihood=mcmc.LogLikelihood, covariance_model_funct=None, otherp=None):
    '''
    Here: given a r value, rv[i], evaluates model = (binned) ClBB(rv[i]), and compare it to
    yvals = (binned) ClBB(r=num) (i.e. data) using Ncov from errors = std_Cls
    which means evalutating: logLLH = lp - 0.5 * (((yvals - model(rv[i])).T * invcov * (yvals - model(rv[i])))
    and returns Log_likelihood of rv[i]. Then from LogL calc: L, 1-sigma and 3-sigmas r value
    '''
    
    #def Log_likelihood (ll) from qubic.mcmc using binned Cls and their errors (sCl)
    ll = mylikelihood(xvals=leff, yvals=data, errors=errors, 
                            model = model, flatprior=prior, covariance_model_funct=covariance_model_funct) 
    like = np.zeros_like(rv)
    
    #for each r value, eval the likelihood L(r) from the Log_likelihood
    for i in range(len(rv)):
        like[i] = np.exp(ll([rv[i]], verbose=False))
    
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


def explore_like(leff, mcl_BB, errors, lmin, dl, lmax, cc, Alens, rv, otherp=None, cov=None, verbose=False, sample_variance=True):
    '''This function calls the evaluation of L(r) from "get_likelihood" 
    The function takes in:
    - leff = ell vector,
    - mcl_noise = mean of Cls
    - errors = sigma of Cls
    - lmin
    - delta ell
    - cc = covcut ie. a value to def. if a pixel in the coverage map is masked or not. This is used only if coverage map is not passed
    - rv = vec of r values
    - otherp = Nsigmas, e.g. 0.68 or 0.95
    - cov = coverage map
    - sample_variance: bool value

    The function creates a Namaster object based off the coverage (mask), lmin, lmax=2*nside-1
    From the Namaster obj. and the file camblib.pkl (containing TT,EE,BB,TE Cls for 100 r in [0,1]) creates binned Dl spectra
    Def a function that returns the binned Dl (BB) spectra for any given r
    Def "data" as average Dl_BB binned spectra from MC simulation
    Through get_likelihood, eval L(r) btw [0,1] by comparing Dl_BB for given r and Dl_BB+errors from MC sims
    ''' 
    
    ### Create Namaster Object: read a mask
    # Unfortunately we need to recalculate fsky for calculating sample variance
    nside = 256
    #lmax = 285 #335  #2 * nside - 1
    if cov is None:
        Namaster = nam.Namaster(None, lmin=lmin, lmax=lmax, delta_ell=dl)
        Namaster.fsky = 0.03
    else:
        okpix = cov > (np.max(cov) * float(cc))
        maskpix = np.zeros(12*nside**2)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, lmin=lmin, lmax=lmax, delta_ell=dl)#, aposize=2.0)
        #     print('Fsky: {}'.format(Namaster.fsky))
        
    lbinned, b = Namaster.get_binning(nside)
    print('Namaster num bin = ', len(lbinned))

    ### Bin the spectra in "/doc/CAMB/camblib.pkl" using Namaster through CambLib    
    binned_camblib = qc.bin_camblib(Namaster, global_dir + camblib_filename, nside, verbose=False)

    ### Redefine the function for getting binned Cls given certain r and mask
    
    class get_cls(object):
        def __init__(self, Alens, binned_camblib):
            self.Alens = Alens
            self.binned_camblib = binned_camblib
 
        def myBBth(self, ell, r):
            #using Planck 2018
            #DlsBB = get_DlsBB_from_Planck(ell, r, self.Alens) 
            
            #using camblib
            DlsBB = get_cmb_DlsBB_from_binned_CAMBlib_r_Alens(ell, self.binned_camblib, r, self.Alens)
            #print('Dls BB shape: ', DlsBB.shape)
            return DlsBB[:] #clBB[:] #clBB[:]
    
    '''
    def myclth(ell,r):
        clth = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True)[1]
        return clth[:] #clth[:]
    
    ### And we need a fast one for BB only as well
    def myBBth(ell, r):
        clBB = qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=False, specindex=2)[0]#[1]
        return clBB[:] #clBB[:]
    '''

    ### Def data ie. Cls BB from recon. cmb 
    data = mcl_BB
    model = get_cls(Alens, binned_camblib).myBBth
    
    #sample and cosmic variance definition: if the sigma(Dls) you pass contains the cosmic variance, then put sample_variance = False
    if sample_variance:
        covariance_model_funct = Namaster.knox_covariance #this is the cosmic variance and it will be added in the noise covariance matrix
    else:
        covariance_model_funct = None #in this case, the noise covariance matrix will be defined from the sigma(Dls) given in input alone
    
    #eval likelihood L(r), integral and r at different sigmas starting from binned Cls at r=0 with sCls
    if otherp is None:
        like, cumint, allrlim = get_likelihood(rv, leff, data, errors, model, [[0,1]], covariance_model_funct=covariance_model_funct)
    else:
        like, cumint, allrlim, other = get_likelihood(rv, leff, data, errors, model, [[0,1]], covariance_model_funct=covariance_model_funct, otherp=otherp)
    
    if otherp is None:
        return like, cumint, allrlim
    else:
        return like, cumint, allrlim, other


def get_results(ell, mcl, scl, coverage, method, lmin=40, delta_ell=30, lmax=256*2-1, covcut=0.1, Alens=0., rv=None, factornoise=1., sample_variance=False):
    '''
    The function calls the "explore_like" function to eval: the likelihood L(r), integral of L, r value with 68% CL and 95% CL

    The function returns:
    - leff= ell,
    - scl_noise_qubic*factornoise**2 * = Cls  weighted by atm. noise,
    - rv = vector of r values (btw 0,1 for example),
    - like = likelihood L(r)
    - cumint = integral of L
    - rlim68 = r value with 68% CL
    - rlim95 = r value with 95% CL
    '''
    leff = ell.copy()
    #check atm noise
    if factornoise != 1.:
        print('**** BEWARE ! Using Factornoise = {}'.format(factornoise))
        
    ### def BB mean cl
    mclBB = mcl.copy()   
    ### def BB sigmas
    sclBB = scl*factornoise

    #check method to use: sclBB or BBcov
    if method=='sigma':
        to_use = sclBB.copy()
    elif method=='covariance':
            to_use = sclBB.copy()
            if np.ndim(to_use) == 1:
                print('ERROR: Covariance passed is not a matrix!')
                exit()
    
    ### Likelihood L(r)
    if rv is None:
        #rv = np.linspace(-0.1,0.1,200000)
        rv = np.linspace(-0.01,0.01,20000)
        print('WARNING: Using internal rv array!')
        #rv = np.linspace(0,1,10000)

    like, cumint, rlim68, rlim95 = explore_like(leff, mclBB, to_use, lmin, delta_ell, lmax, covcut, Alens, rv,
                                     cov=coverage, verbose=True, sample_variance=sample_variance, otherp=0.95)
    
    return leff, scl*factornoise**2, rv, like, cumint, rlim68, rlim95

################# Current version that I'm using (faster, more compact) ########################

class binned_theo_Dls(object):
    def __init__(self, Alens, nside, coverage, cambfilepath, lmin=21, delta_ell=35, lmax_used=335):
        self.Alens = Alens
        self.nside = nside
        self.coverage = coverage
        self.cambfilepath = cambfilepath
        self.lmin = lmin
        self.lmax = 2*nside-1
        self.delta_ell = delta_ell
        self.lmax_used = lmax_used
        binned_camblib, ell_bin_from_coverage = self.get_binned_camblib() #binned_camblib
        self.binned_camblib = binned_camblib
        self.ell_bin_from_coverage = ell_bin_from_coverage
        self.sample_var = self.get_sample_variance()
        
    def get_binned_camblib(self):
        '''
        This function returns the binned CAMB file given a mask, ell range and a pickle file
        containing theo. Dls for different values of r
        '''
        okpix = self.coverage > (np.max(self.coverage) * float(0))
        npix = hp.nside2npix(self.nside)
        maskpix = np.zeros(npix)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, self.lmin, self.lmax, self.delta_ell)#, aposize=2.0)
        ell_bin_from_coverage, b = Namaster.get_binning(self.nside)
        print('Dim of ell used in NaMaster', ell_bin_from_coverage.shape)
        #Namaster = nam.Namaster(maskpix, lmin=40, lmax=2*self.nside-1, delta_ell=30)
        #ell=np.arange(2*nside-1)

        binned_camblib = qc.bin_camblib(Namaster, self.cambfilepath, self.nside, verbose=False)
        
        return binned_camblib, ell_bin_from_coverage
    
    def get_sample_variance(self):
        '''
        This function returns the binned CAMB file given a mask, ell range and a pickle file
        containing theo. Dls for different values of r
        '''
        okpix = self.coverage > (np.max(self.coverage) * float(0))
        npix = hp.nside2npix(self.nside)
        maskpix = np.zeros(npix)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, self.lmin, self.lmax_used, self.delta_ell)#, aposize=2.0) #-2*self.delta_ell
        Namaster.fsky = 0.03
        ell_bin_from_coverage, b = Namaster.get_binning(self.lmax_used)
        print('Ell for sample variance: ', ell_bin_from_coverage, ell_bin_from_coverage.shape)
        print('Namaster fsky = ' , Namaster.fsky)
        
        sample_var = Namaster.knox_covariance
        
        return sample_var
        
    def get_cmb_DlsBB_from_binned_CAMBlib_r_Alens(self, ell, r):
        '''
        This function generates Dls_BB from a given CAMB lib and binning, considering a given r and Alens (amplitude of lensing residual)
        '''  
        #Dls
        DlsBB = qc.get_Dl_fromlib(ell, 0, lib=self.binned_camblib, unlensed=False, specindex=2)[0] #lensed spectra with r=0
        DlsBB *= self.Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
        DlsBB += qc.get_Dl_fromlib(ell, r, lib=self.binned_camblib, unlensed=True, specindex=2)[1] #this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given

        return DlsBB[:]      
    
    
    def planck_Cls_BB(self, r):
        cmb_clsBB = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[2,:lmax] #This is the Planck 2018 best-fit. Here l=0,1 are set to zero as expected by healpy
        if self.Alens != 1.:
            #print('Defining lensing residual, Alens = ', Alens)
            cmb_clsBB *= self.Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
        if r:
            #print('Adding primordial B-modes with r = ', r)
            cmb_clsBB += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[2,:lmax]#[:,:4000] #this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given
        
        return cmb_clsBB
    
    
    def get_DlsBB_from_Planck(self, ell, r):
        ell_to_use = np.round(ell,0).astype(int)
        cls_BB = self.planck_Cls_BB(r)
        cls_BB_to_use = list(itemgetter(*list(ell_to_use))(list(cls_BB)))
        Dls_BB = cls_BB_to_use*ell_to_use*(ell_to_use+1)/(2*np.pi)
        return Dls_BB
        

    def get_DlsBB(self, ell, r):
        #using camblib
        DlsBB = self.get_cmb_DlsBB_from_binned_CAMBlib_r_Alens(ell, r)
        #print('Dls BB shape: ', DlsBB.shape)
        return DlsBB[:] #clBB[:] #clBB[:]
    
    
    def get_DlsBB_Planck(self, ell, r):
        #using Planck 2018
        DlsBB = self.get_DlsBB_from_Planck(ell, r) 
        
        return DlsBB[:] #clBB[:] #clBB[:]