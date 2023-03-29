import numpy as np
import pickle
import glob

import qubic
from qubic import NamasterLib as nam
#from qubic import fibtools as ft
from qubic import camb_interface as qc
import healpy as hp
import scipy
from scipy.optimize import curve_fit

from pylab import *
from operator import itemgetter

import fgbuster
import os.path as op
fgb_path = fgbuster.__path__ #if you want to use fgb (start from Planck 2018 best-fit)
CMB_CL_FILE = op.join(fgb_path[0]+'/templates/Cls_Planck2018_%s.fits')

global_dir = './'   # '/sps/qubic/Users/emanzan/libraries/qubic/qubic'
camblib_filename = 'camblib_with_r_from_m0.1_to_p0.1_minstep1e-06.pickle' #'camblib_with_r_from_0.0001.pickle' #  #'/doc/CAMB/camblib.pkl' 

#--------------------------------------------------------------------------
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


def get_cmb_DlsBB_from_binned_CAMBlib_r_Alens(ell, binned_camblib, r, Alens):
    '''
    This function generates Dls_BB from a given CAMB lib and binning, considering a given r and Alens (amplitude of lensing residual)
    '''  
    
    #Dls
    DlsBB = qc.get_Dl_fromlib(ell, 0, lib=binned_camblib, unlensed=False, specindex=2)[0] #lensed spectra with r=0
    DlsBB *= Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
    DlsBB += qc.get_Dl_fromlib(ell, r, lib=binned_camblib, unlensed=True, specindex=2)[1] #this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given

    return DlsBB[:]
   

def get_DlsBB_from_CAMB(ell, r, Alens, coverage, nside=256,  cov_cut=0.1, lmin=21, delta_ell=35, lmax = 335):
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


def planck_cls(r, Alens):
    cmb_cls = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000] #This is the Planck 2018 best-fit. Here l=0,1 are set to zero as expected by healpy
    if Alens != 1.:
        print('Defining lensing residual, Alens = ', Alens)
        cmb_cls[2] *= Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
    if r:
        print('Adding primordial B-modes with r = ', r)
        cmb_cls[2] += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[2,:4000]#[:,:4000] #this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given

    return cmb_cls


def ClsXX_2_DlsXX_binned(ell, Cls_XX):
    '''
    This function converts Cls_XX (dim = N_ell) to Dls_XX (dim = N_ell) using the binning of the input ell
    ''' 
    num_ell = ell.shape[0] #binned ell
    Dls_XX = np.zeros(num_ell) #binned Dls
    for i in range(num_ell): #fill in the Dls bins
        Dls_XX[i] = Cls_XX[ell[i]-1]*ell[i]*(ell[i]+1)/(2*np.pi)
        #Dls_XX[i] = Cls_XX[i]*ell[i]*(ell[i]+1)/(2*np.pi)
    
    return Dls_XX


#Likelihood Class --------------------------------------------------------------------------
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
        if self.coverage == None:
            maskpix = None
        else:
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
        This function returns the Sample variance given the coverage, ell range
        '''
        if self.coverage == None:
            maskpix = None
            print('Using None coverage mask')
        else:
            okpix = self.coverage > (np.max(self.coverage) * float(0))
            npix = hp.nside2npix(self.nside)
            maskpix = np.zeros(npix)
            maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, self.lmin, self.lmax_used, self.delta_ell)#, aposize=2.0) #-2*self.delta_ell
        print('Namaster.ell_binned at the beginning: ', Namaster.ell_binned)
        
        Namaster.fsky = 0.03
        print('Namaster fsky = ' , Namaster.fsky)
        
        
        ell_bin_from_coverage, b = Namaster.get_binning(self.lmax_used)
        print('Ell for sample variance: ', ell_bin_from_coverage, ell_bin_from_coverage.shape)
        print('Namaster.ell_binned from Nside:', Namaster.ell_binned)
        
        print('Getting Namaster.ell_binned from Nside', )
        lbinned, b = Namaster.get_binning(self.nside)
        print('Namaster.ell_binned from Nside:', Namaster.ell_binned)
        
        print('Removing last bin')
        Namaster.ell_binned = Namaster.ell_binned[:-1]
        print('Namaster.ell_binned at the end: ' , Namaster.ell_binned)
        
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
        cmb_clsBB = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[2,:1000] #self.lmax This is the Planck 2018 best-fit. Here l=0,1 are set to zero as expected by healpy
        if self.Alens != 1.:
            #print('Defining lensing residual, Alens = ', Alens)
            cmb_clsBB *= self.Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
        if r:
            #print('Adding primordial B-modes with r = ', r)
            cmb_clsBB += r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[2,:1000]#[:,:4000] #self.lmax this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given
        
        return cmb_clsBB
    
    
    def get_DlsBB_from_Planck(self, ell, r):
        ell_to_use = np.round(ell,0).astype(int)
        #print('Ell for theo Dls:', ell_to_use)
        cls_BB = self.planck_Cls_BB(r)
        #print('Cls for theo Dls:', cls_BB)
        
        #cls_BB_to_use = list(itemgetter(*list(ell_to_use))(list(cls_BB)))
        #Dls_BB = cls_BB_to_use*ell_to_use*(ell_to_use+1)/(2*np.pi)
        
        Dls_BB = np.zeros(len(ell_to_use))
        for i in range(len(ell_to_use)):
            Dls_BB[i] = cls_BB[ell_to_use[i]-1]*ell_to_use[i]*(ell_to_use[i]+1)/(2*np.pi)
            
        #print('Dls for theo Dls:', Dls_BB)
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