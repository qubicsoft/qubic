import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as op
import random as rd
import string
from pylab import *
from importlib import reload
from scipy import constants
import pickle

import pysm3
import pysm3.units as u
from pysm3 import utils
from pysm3 import bandpass_unit_conversion

import qubic
from qubic import camb_interface as qc
from qubic import NamasterLib as nam

import fgbuster
from fgbuster.component_model import (CMB, Dust, Dust_2b, Synchrotron, AnalyticComponent)

import warnings
warnings.filterwarnings("ignore")
import sys
from datetime import datetime
#############################################################################
# This library contains all things for
# - define instrument and sky configuration
# - generate instrum noise maps
# - generate sky maps (bandpass integration available)
# ###########################################################################

#define path to cmb Cls to generate cmb maps~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

global_dir='/sps/qubic/Users/emanzan/libraries/qubic/qubic' #if you want to use NaMaster

fgb_path = fgbuster.__path__ #if you want to use fgb (start from Planck 2018 best-fit)
CMB_CL_FILE = op.join(fgb_path[0]+'/templates/Cls_Planck2018_%s.fits')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_coverage(fsky, nside, center_radec=[0, -57.]):
    '''
    This function returns a coverage map (0 or 1) with a given nside. The sky patch is circular within a given sky fraction
    Input:
    - fsky, sky fraction (float)
    - nside
    - center_radec, RA/DEC coordinates of the center of the sky patch
    '''
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

def closest_value(input_list, input_value):
    #To be used with Mathias's approach to bp integration
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i]


def where_closest_value(input_list, input_value):
    #To be used with Mathias's approach to bp integration
    i=closest_value(input_list, input_value)
    argi = np.where(input_list == i)[0][0]
    return argi

def eval_sed(freq_maps):
    '''
    Function that evaluates the SED as RMS of frequency maps.
    Input: - freq_maps, an array of shape [Nfreq, Nstk, Npix]
    '''
    return np.sqrt( np.std(freq_maps, axis=2)**2 + np.mean(freq_maps, axis=2)**2 )


def get_freq_maps(freq, sky, npix):
    '''
    Function that loop over frequency array to get the set of frequency maps
    Input:
    - freq, array of frequency
    - sky, pysm3 sky object
    - npix, number of pixels in the maps
    '''
    freq_maps = np.zeros((len(freq),3,npix))
    for f_idx, f in enumerate(freq):
        freq_maps[f_idx,:,:] = sky.get_emission(f*u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(f*u.GHz))
    return freq_maps

def get_sed_and_maps(freq, sky, npix):
    '''
    Wrapper function that generates the frequency maps from a pysm3 sky object and then evaluates the RMS SED and returns
    both the maps and the sed
    '''
    freq_maps = get_freq_maps(freq, sky, npix)
    print('Evaluated maps')
    sed = eval_sed(freq_maps)
    return sed, freq_maps

# Units conversion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _rj2cmb(freqs):
    return (np.ones_like(freqs) * u.K_RJ).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _cmb2rj(freqs):
    return (np.ones_like(freqs) * u.K_CMB).to(
        u.K_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value

def _rj2jysr(freqs):
    return (np.ones_like(freqs) * u.K_RJ).to(
        u.Jy / u.sr, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _jysr2rj(freqs):
    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_RJ, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _cmb2jysr(freqs):
    return (np.ones_like(freqs) * u.K_CMB).to(
        u.Jy / u.sr, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value


def _jysr2cmb(freqs):
    return (np.ones_like(freqs) * u.Jy / u.sr).to(
        u.K_CMB, equivalencies=u.cmb_equivalencies(freqs * u.GHz)).value

# Noise maps creation part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_noisemaps(nus, nside, depth_i, depth_p, npix, spectra_type='cross'):
    """
    Function that creates a noise realization for the given instrum class by converting sensitivity in muK*arcmin to noise/pixel
    Input:
    - nus, freq array
    - nside
    - depth_i and depth_p, depths in muK*arcmin
    - npix
    """
    np.random.seed(None)
    #np.random.seed(42)
    N = np.zeros(((len(nus), 3, npix)))
    print('Generate noise maps to perform {}-spectra'.format(spectra_type))
    for ind_nu, nu in enumerate(nus):
        #noise in muK_CMB per pixel
        sig_i=depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60) 
        sig_p=depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        #generate noise for all pixels from normal distrib with sigma=noise/pixel
        N[ind_nu, 0] = np.random.normal(0, 1., npix)*sig_i
        N[ind_nu, 1] = np.random.normal(0, 1., npix)*sig_p
        N[ind_nu, 2] = np.random.normal(0, 1., npix)*sig_p

    return N
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# FWHM related functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def give_me_nu_fwhm_S4_2_qubic(nu, largeur, Nf, fwhmS4):

    def give_me_fwhm(nu, nuS4, fwhmS4):
        return fwhmS4*nuS4/nu

    largeurq=largeur/Nf
    min=nu*(1-largeur/2)
    max=nu*(1+largeur/2)
    arr = np.linspace(min, max, Nf+1)
    mean_nu = get_multiple_nus(nu, largeur, Nf)

    fwhm = give_me_fwhm(mean_nu, nu, fwhmS4)

    return mean_nu, fwhm


def smoothing(maps, FWHMdeg, Nf, central_nus, verbose=True):
        """Convolve the maps to the FWHM at each sub-frequency or to a common beam if FWHMdeg is given."""
        fwhms = np.zeros(Nf)
        if FWHMdeg is not None:
            fwhms += FWHMdeg
        for i in range(Nf):
            if fwhms[i] != 0:
                maps[i, :, :] = hp.sphtfunc.smoothing(maps[i, :, :].T, fwhm=np.deg2rad(fwhms[i]),
                                                      verbose=verbose).T
        return fwhms, maps
    

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class BImaps(object):

    def __init__(self, skyconfig, sky, coverage, dict, r=0, Alens=1, corr_l = None, nside=256, spectra_type='cross', lmin=21, delta_ell=35, save_figs=True):
        self.save_figs = save_figs
        self.dict = dict
        self.name = self.dict['name']
        self.center_radec = self.dict['center_radec']
        self.nus = self.dict['frequency']
        self.bw = self.dict['bandwidth']
        self.edges = self.dict['edges']
        self.fwhm = self.dict['fwhm']
        self.fwhmdeg = self.fwhm/60
        self.depth_i = self.dict['depth_i']
        self.depth_p = self.dict['depth_p']

        self.fsky = self.dict['fsky']
        self.coverage = coverage
        self.nside = nside
        self.npix = 12*self.nside**2
        self.lmin = lmin
        self.delta_ell = delta_ell
        self.lmax = 2*self.nside-1 #3 * self.nside
        self.r = r
        self.Alens = Alens
        self.spectratype = spectra_type
        
        self.skyconfig = skyconfig  #dict of the form: {'cmb':seed, 'pysm_fg':[''], 'not_pysm_fg':['','','']}
        self.sky = sky #pysm3 sky object
        self.corr_l = corr_l #if there is freq decorr this is not None

        #save cmb seed
        if 'cmb' in skyconfig.keys():
            self.seed = self.skyconfig['cmb']
            print('cmb seed: ', self.seed)
            
        print('r = ', self.r)
        print('correlation lenght = ', self.corr_l)
        print('spectra type = ', self.spectratype)
        print('Number of freq bands: ', len(self.nus))
       
    def get_cmb_from_r_Alens(self):
        '''
        This function generates a cmb map using a given r value and Alens (amplitude of lensing residual) from Planck 2018 best-fit
        ''' 
        print('Defining cmb spectrum')
        cmb_cls = hp.read_cl(CMB_CL_FILE%'lensed_scalar')[:,:4000] #This is the Planck 2018 best-fit. Here l=0,1 are set to zero as expected by healpy
        if self.Alens != 1.:
            print('Defining lensing residual, Alens = ', self.Alens)
            cmb_cls[2] *= self.Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
        if self.r:
            print('Adding primordial B-modes with r = ', self.r)
            cmb_cls[2] += self.r * hp.read_cl(CMB_CL_FILE%'unlensed_scalar_and_tensor_r1')[2,:4000]#[:,:4000] #this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given
            
        # set EE and TE to zero a' la Mathias
        #cmb_cls[1] = np.zeros(4000)
        #cmb_cls[3] = np.zeros(4000)
        
        #gen maps from Cls
        print('CMB seed = ', self.seed)
        np.random.seed(self.seed)
        maps = hp.synfast(cmb_cls, self.nside, verbose=False, new=True)
        if self.save_figs:
            #save map image
            center = qubic.equ2gal(self.center_radec[0], self.center_radec[1])
            stk = ['I','Q','U']
            figure(figsize=(16,10))
            for i_stk in range(3):
                hp.mollview(maps[i_stk,:], coord=['G','C'], title='Input CMB '+stk[i_stk], norm='hist', sub=(1,3,i_stk+1))
            savefig('./input_maps/Input_CMB_map_IQU_instrum{}_from_Planck2018bestfit_r{}_Alens{}_seed{}.png'.format(self.name, self.r, self.Alens, self.seed))
        
        return maps
       

    def get_cmb(self):
        '''
        This function generates a cmb map from a given seed using a given r value and Alens (amplitude of lensing residual) with CAMB
        '''
        okpix = self.coverage > (np.max(self.coverage) * float(0))
        maskpix = np.zeros(self.npix)
        maskpix[okpix] = 1
        #Namaster = nam.Namaster(maskpix, lmin=2, lmax=2*self.nside-1, delta_ell=1)
        Namaster = nam.Namaster(maskpix, lmin=self.lmin, lmax=2*self.nside-1, delta_ell=self.delta_ell)
        #Namaster = nam.Namaster(maskpix, lmin=40, lmax=2*self.nside-1, delta_ell=30)
        ell=np.arange(2*self.nside-1)

        binned_camblib = qc.bin_camblib(Namaster, global_dir + '/doc/CAMB/camblib.pkl', self.nside, verbose=False)

        #Dls = qc.get_Dl_fromlib(ell, self.r, lib=binned_camblib, unlensed=False)[0]#[1]
        print('Defining cmb spectrum')
        Dls = qc.get_Dl_fromlib(ell, 0, lib=binned_camblib, unlensed=False)[0] #lensed spectra with r=0
        if self.Alens != 1.:
            print('Defining lensing residual, Alens = ', self.Alens)
            Dls[:,2] *= self.Alens #this simulates a lensing residual. Alens = 0 : no lensing. Alens = 1: full lensing 
        if self.r:
            print('Adding primordial B-modes with r = ', self.r)
            Dls[:,2] += qc.get_Dl_fromlib(ell, self.r, lib=binned_camblib, unlensed=True, specindex=2)[1] #this takes unlensed cmb B-modes for r=1 and scales them to whatever r is given
        
        #convert Dls to Cls
        mycls = qc.Dl2Cl_without_monopole(ell, Dls)
        
        #create map
        print('CMB seed = ', self.seed)
        np.random.seed(self.seed)
        maps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
        
        if self.save_figs:
            #save map image
            center = qubic.equ2gal(self.center_radec[0], self.center_radec[1])
            stk = ['I','Q','U']
            figure(figsize=(16,10))
            for i_stk in range(3):
                hp.mollview(maps[i_stk,:], coord=['G','C'], title='Input CMB '+stk[i_stk], norm='hist', sub=(1,3,i_stk+1))
            savefig('./input_maps/Input_CMB_map_IQU_instrum{}_from_CAMB_r{}_Alens{}_seed{}.png'.format(self.name, self.r, self.Alens, self.seed))
        
        return maps
    
    
    def get_fg_maps(self, N_SAMPLE_BAND=100, Nside_patch=0):
        '''
        This function generates fg maps integrated over the frequency band
        '''
        #def output maps
        foreg_maps = np.zeros((len(self.nus), 3, self.npix))
        sky = self.sky
        if 'pysm_fg' in self.skyconfig:
            preset_strings = self.skyconfig['pysm_fg']
        else:
            preset_strings = None
        
        #get emission if no bandpass integration is required
        if N_SAMPLE_BAND == 1:
            for f in range(len(self.nus)):
                if preset_strings is not None:
                    print('Doing Pysm3 models...')
                    #get emission
                    foreg_maps[f,:,:] = sky.get_emission(self.nus[f] * u.GHz)*utils.bandpass_unit_conversion(self.nus[f]*u.GHz,None, u.uK_CMB)
            return foreg_maps, sky #exit
        else:
            pass
                

        #apply standard bandpass integration 
        weights_flat = np.ones(N_SAMPLE_BAND)
        for f in range(len(self.nus)):
            print('Integrate band: {0} GHz with {1} steps'.format(self.nus[f],N_SAMPLE_BAND))
            fmin = self.nus[f]-self.bw[f]/2
            fmax = self.nus[f]+self.bw[f]/2        
            freqs = np.linspace(fmin, fmax, N_SAMPLE_BAND)

            if preset_strings is not None:
                print('Doing Pysm3 models...')
                #apply band integration
                foreg_maps[f,:,:] = sky.get_emission(freqs * u.GHz, weights_flat) * bandpass_unit_conversion(freqs * u.GHz, weights_flat, u.uK_CMB)
                #hp.fitsfunc.write_map('./input_maps/{}_map_f{}GHz_bp{}pts_patch{}nside_{}.fits'.format(preset_strings, self.nus[f], N_SAMPLE_BAND,Nside_patch, self.name),foreg_maps[f,:,:], coord='G', overwrite=True)
                print('...Done.')
           
             
        #if there is a pysm3.sky object, return it together with the map, so that one can also access the parameter maps
        if isinstance(sky, pysm3.sky.Sky): 
            return foreg_maps, sky
        else:
            return foreg_maps, 1
        
        
    def get_fg_maps_same_real(self, fg_seed, fg_freqs, N_SAMPLE_BAND=100, Nside_patch=0):
        '''
        This function generates fg maps integrated over the frequency band. In the d6 case, it uses an external realization of the emission (i.e. an external array of seeds). In the d1 case or in general if no random realization is needed, one can use the "get_fg_maps" function above, which saves computational time
        '''
        #def output maps
        foreg_maps = np.zeros((len(self.nus), 3, self.npix))
        sky = self.sky
        if 'pysm_fg' in self.skyconfig:
            preset_strings = self.skyconfig['pysm_fg']
        else:
            preset_strings = None
            
        #set the first seed for each frequency band
        nus_eff=np.zeros(len(self.nus))
        seed_eff = np.zeros(len(self.nus))
        for i in range(len(self.nus)):
            myargs = where_closest_value(fg_freqs, self.nus[i])
            nus_eff[i] = np.round(fg_freqs[myargs], 3).copy()
            seed_eff[i] = fg_seed[myargs].copy()
        
        #get emission using a fixed fg realization
        if N_SAMPLE_BAND == 1: #no bandpass integration is required
            for f in range(len(self.nus)):
                if preset_strings is not None:
                    print('Doing Pysm3 models...')
                    #generate d6 real. from given seed
                    np.random.seed(int(seed_eff[f]))
                    #get emission
                    foreg_maps[f,:,:] = sky.get_emission(nus_eff[f] * u.GHz)*utils.bandpass_unit_conversion(nus_eff[f]*u.GHz,None, u.uK_CMB)
        else:
            #apply bandpass integration
            weights_flat = np.ones(N_SAMPLE_BAND)

            for f in range(len(self.nus)):
                print('Integrate band: {0} GHz with {1} steps'.format(self.nus[f],N_SAMPLE_BAND))
                fmin = self.nus[f]-self.bw[f]/2
                fmax = self.nus[f]+self.bw[f]/2        
                freqs = np.linspace(fmin, fmax, N_SAMPLE_BAND)
                #generate d6 real. from given seed
                np.random.seed(int(seed_eff[f]))

                if preset_strings is not None:
                    print('Doing Pysm3 models...')
                    #apply band integration
                    foreg_maps[f,:,:] = sky.get_emission(freqs * u.GHz, weights_flat) * bandpass_unit_conversion(freqs * u.GHz, weights_flat, u.uK_CMB)
                    #hp.fitsfunc.write_map('./input_maps/{}_map_f{}GHz_bp{}pts_patch{}nside_{}.fits'.format(preset_strings, self.nus[f], N_SAMPLE_BAND,Nside_patch, self.name),foreg_maps[f,:,:], coord='G', overwrite=True)
                    print('...Done.')
        
        pickle.dump({'d6s1map' : foreg_maps, 'freqs': self.nus}, open('./input_maps/{}_map_bp{}pts_patch{}nside_{}.pkl'.format(preset_strings, N_SAMPLE_BAND,Nside_patch, self.name), "wb"))
        return foreg_maps, sky
      
        
    def get_sky_maps(self, N_SAMPLE_BAND=100, Nside_patch=0, same_resol=None, verbose=False, coverage=False, noise=False):
        """
        This function combines cmb and fg maps if both are present, then if required adds: noise, smoothing, masks
        """
        if same_resol is not None:
            self.fwhmdeg = np.ones(len(self.nus))*np.max(same_resol)
        if verbose:
            print("    FWHM : {} deg \n    nus : {} GHz \n    Bandwidth : {} GHz\n\n".format(self.fwhmdeg, self.nus, self.bw))

        allmaps=np.zeros((len(self.nus), 3, self.npix))
        sky = 1 #dummy variable to be returned if there is no pysm3 sky object
        cmbmap = 1 #dummy variable to be returned if there is no cmb
        #generate sky
        for i in self.skyconfig.keys():
            if i == 'cmb':
                cmbmap = self.get_cmb_from_r_Alens() #use either: self.get_cmb() or self.get_cmb_from_r_Alens() 
                #hp.fitsfunc.write_map('./input_maps/{}_map_r{}_Alens{}_{}.fits'.format(i, self.r, self.Alens, self.name), cmbmap, coord='G', overwrite=True)
                for j in range(len(self.nus)):
                    allmaps[j] += cmbmap                    
            elif i == 'pysm_fg':
                fgmaps, sky = self.get_fg_maps(N_SAMPLE_BAND=N_SAMPLE_BAND, Nside_patch=Nside_patch)
                allmaps += fgmaps
            else:
                print('Sky key {} not recognized!'.format(i))
                
        #convolve with fwhm if required
        if same_resol != 0:
            for j in range(len(self.fwhmdeg)):
                if verbose:
                    print('Convolution to {:.2f} deg'.format(self.fwhmdeg[j]))
                allmaps[j] = hp.sphtfunc.smoothing(allmaps[j, :, :], fwhm=np.deg2rad(self.fwhmdeg[j]),verbose=False)
                
        #apply coverage
        if coverage:
            pixok = self.coverage > 0
            allmaps[:, :, ~pixok] = hp.UNSEEN
                
        #add noise if required
        if noise:
            print('Adding noise')
            noisemaps = create_noisemaps(self.nus, self.nside, self.depth_i, self.depth_p, self.npix, spectra_type=self.spectratype)
            maps_noisy = allmaps+noisemaps
            #apply coverage
            if coverage:
                maps_noisy[:, :, ~pixok] = hp.UNSEEN
                noisemaps[:, :, ~pixok] = hp.UNSEEN
            return maps_noisy, sky, cmbmap, allmaps, noisemaps 
        
        return allmaps, sky, cmbmap

    
    def gen_noise_maps(self, coverage=False):
        """
        Wrapper function that creates a noise realization for the given instrum class (from sensitivity in muK*arcmin to noise/pixel), with a mask if a coverage map is required
        - Input: self; coverage = bool value (True if you want to add a coverage map)
        """
        noisemaps = create_noisemaps(self.nus, self.nside, self.depth_i, self.depth_p, self.npix, spectra_type=self.spectratype)
        #apply coverage
        if coverage:
            pixok = self.coverage > 0
            noisemaps[:, :, ~pixok] = hp.UNSEEN
        return noisemaps
