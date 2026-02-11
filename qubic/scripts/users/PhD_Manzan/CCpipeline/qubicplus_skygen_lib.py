# +
from qubic import camb_interface as qc
import healpy as hp
import numpy as np
import qubicplus
import pysm3
import pysm3.units as u
from pysm3 import utils
from pysm3 import bandpass_unit_conversion
from qubic import camb_interface as qc
import matplotlib.pyplot as plt
from qubic import NamasterLib as nam
import os
import random as rd
import string
import qubic
from importlib import reload
from fgbuster.component_model import (CMB, Dust, Dust_2b, Synchrotron, AnalyticComponent)
from scipy import constants
import fgbuster
import warnings
warnings.filterwarnings("ignore")
import sys
from datetime import datetime

global_dir='/sps/qubic/Users/emanzan/libraries/qubic/qubic'

def get_coverage(fsky, nside, center_radec=[0, -57.]):
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

# FG and noise maps creation part ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def create_dustd0_no_bp(nside, nus):
    # Create d0 dust maps at given number of frequencies
    sky = pysm3.Sky(nside=nside, preset_strings=['d0'])
    maps_353GHz = sky.get_emission(353*u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(353*u.GHz))

    comp=[fgbuster.component_model.Dust(nu0=353)]

    A = fgbuster.MixingMatrix(*comp)
    A_ev = A.evaluator(nus)
    A_maxL = A_ev(np.array([1.54, 20]))

    new_dust_map=np.zeros(((len(nus), 3, 12*nside**2)))
    for i in range(len(nus)):
        new_dust_map[i]=A_maxL[i, 0]*maps_353GHz

    return new_dust_map


def create_sync_no_bp(nside, nus):
    # Create synch maps at given number of frequencies
    sky = pysm3.Sky(nside=nside, preset_strings=['s0'])
    maps_70GHz = sky.get_emission(70*u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(70*u.GHz))

    comp=[fgbuster.component_model.Synchrotron(nu0=70)]

    A = fgbuster.MixingMatrix(*comp)
    A_ev = A.evaluator(nus)
    A_maxL = A_ev(np.array([-3]))

    new_sync_map=np.zeros(((len(nus), 3, 12*nside**2)))
    for i in range(len(nus)):
        new_sync_map[i]=A_maxL[i, 0]*maps_70GHz

    return new_sync_map


def create_dust_with_model2beta_no_bp(nside, nus, betad0, betad1, nubreak, temp, break_width):
    # Create d02b dust maps at given number of frequencies
    sky = pysm3.Sky(nside=nside, preset_strings=['d0'])
    maps_353GHz = sky.get_emission(353*u.GHz).to(u.uK_CMB, equivalencies=u.cmb_equivalencies(353*u.GHz))

    comp2b=[fgbuster.component_model.Dust_2b(nu0=353, break_width=break_width)]

    A2b = fgbuster.MixingMatrix(*comp2b)
    A2b_ev = A2b.evaluator(nus)
    A2b_maxL = A2b_ev([betad0, betad1, nubreak, temp])

    new_dust_map=np.zeros(((len(nus), 3, 12*nside**2)))
    for i in range(len(nus)):
        new_dust_map[i]=A2b_maxL[i, 0]*maps_353GHz

    return new_dust_map


def create_noisemaps(signoise, nus, nside, depth_i, depth_p, npix):
    np.random.seed(None)
    N = np.zeros(((len(nus), 3, npix)))
    for ind_nu, nu in enumerate(nus):

        sig_i=signoise*depth_i[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 0] = np.random.normal(0, sig_i, 12*nside**2)

        sig_p=signoise*depth_p[ind_nu]/(np.sqrt(hp.nside2pixarea(nside, degrees=True)) * 60)
        N[ind_nu, 1] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)
        N[ind_nu, 2] = np.random.normal(0, sig_p, 12*nside**2)*np.sqrt(2)

    return N
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Frequency related functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_freqs_inter(edges, N):
    '''
    This function returns intermediate frequency for integration into bands from edges.
    '''
    if N == 1:
        return np.array([np.mean(edges)])
    freqs_inter=np.linspace(edges[0], edges[1], N)
    return freqs_inter

def get_multiple_nus(nu, bw, nf):
    nus=np.zeros(nf)
    edge=np.linspace(nu-(bw/2), nu+(bw/2), nf+1)
    for i in range(len(edge)-1):
        #print(i, i+1)
        nus[i]=np.mean([edge[i], edge[i+1]])
    return nus
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    
def random_string(nchars):
    lst = [rd.choice(string.ascii_letters + string.digits) for n in range(nchars)]
    str = "".join(lst)
    return (str)


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
    
    
def get_fg_notconvolved(model, nu, nside=256):

    sky = pysm3.Sky(nside=nside, preset_strings=[model])
    maps = np.zeros(((len(nu), 3, 12*nside**2)))
    for indi, i in enumerate(nu) :
        maps[indi] = sky.get_emission(i*u.GHz, None)*utils.bandpass_unit_conversion(i*u.GHz,None, u.uK_CMB)

    return maps
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# SEDs and scaling factors ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def scaling_factor(maps, nus, analytic_expr, beta0, beta1, nubreak):
    nb_nus = maps.shape[0]
    newmaps = np.zeros(maps.shape)
    #print(sed1b)
    for i in range(nb_nus):
        _, sed1b = sed(analytic_expr, nus[i], beta1, beta1, nu0=nus[i], nubreak=nubreak)
        _, sed2b = sed(analytic_expr, nus[i], beta0, beta1, nu0=nus[i], nubreak=nubreak)
        print('nu is {} & Scaling factor is {:.8f}'.format(nus[i], sed2b))
        newmaps[i] = maps[i] * sed2b
    return newmaps, sed1b, sed2b


def sed(analytic_expr, nus, beta0, beta1, temp=20, hok=constants.h * 1e9 / constants.k, nubreak=200, nu0=200):
    sed_expr = AnalyticComponent(analytic_expr,
                             nu0=nu0,
                             beta_d0=beta0,
                             beta_d1=beta1,
                             temp=temp,
                             nubreak=nubreak,
                             h_over_k = hok)
    return nus, sed_expr.eval(nus)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



class BImaps(object):

    def __init__(self, skyconfig, dict, r=0, nside=256):
        self.dict = dict
        self.skyconfig = skyconfig
        self.nus = self.dict['frequency']
        self.bw = self.dict['bandwidth']
        self.fwhm = self.dict['fwhm']
        self.fwhmdeg = self.fwhm/60
        self.depth_i = self.dict['depth_i']
        self.depth_p = self.dict['depth_p']
        #self.depth_p_reconstructed = np.ones(self.Nf)*self.depth_i*np.sqrt(self.Nf)
        #self.depth_i_reconstructed = np.ones(self.Nf)*self.depth_p*np.sqrt(self.Nf)

        self.fsky = self.dict['fsky']
        self.edges = self.dict['edges']

        self.nside=nside
        self.npix = 12*self.nside**2
        self.lmax= 3 * self.nside
        self.r=r
        #print(self.edges)#, self.edges.shape)

        for k in skyconfig.keys():
            if k == 'cmb':
                self.seed = self.skyconfig['cmb']


    def get_cmb(self, coverage):

        okpix = coverage > (np.max(coverage) * float(0))
        maskpix = np.zeros(12*self.nside**2)
        maskpix[okpix] = 1
        Namaster = nam.Namaster(maskpix, lmin=21, lmax=2*self.nside-1, delta_ell=35)
        #Namaster = nam.Namaster(maskpix, lmin=40, lmax=2*self.nside-1, delta_ell=30)
        ell=np.arange(2*self.nside-1)

        binned_camblib = qc.bin_camblib(Namaster, global_dir + '/doc/CAMB/camblib.pkl', self.nside, verbose=False)

        cls = qc.get_Dl_fromlib(ell, self.r, lib=binned_camblib, unlensed=True)[1]
        mycls = qc.Dl2Cl_without_monopole(ell, cls)


        np.random.seed(self.seed)
        maps = hp.synfast(mycls.T, self.nside, verbose=False, new=True)
        return maps

        
    def get_sky(self, coverage):
        setting = []
        iscmb=False
        for k in self.skyconfig:
            if k == 'cmb' :
                iscmb=True
                maps = self.get_cmb(coverage)

                rndstr = random_string(10)
                hp.write_map('/tmp/' + rndstr, maps)
                cmbmap = pysm3.CMBMap(self.nside, map_IQU='/tmp/' + rndstr)
                os.remove('/tmp/' + rndstr)
                #setting.append(skyconfig[k])
            elif k=='dust':
                pass
            else:
                setting.append(self.skyconfig[k])

        sky = pysm3.Sky(nside=self.nside, preset_strings=setting)
        if iscmb:
            sky.add_component(cmbmap)

        return sky
    
    
    def get_fg_freq_integrated_maps(self, verbose=False, N_SAMPLE_BAND=100, beta=[], temp=20):
        '''
        This function generates dust and synch maps integrated over the frequency band
        '''
        #def output maps as transpose of (Nfreq,Nstk,Npix)
        allmaps=np.zeros((self.npix, 3,len(self.nus)))
        weights_flat = np.ones(N_SAMPLE_BAND) #weights in power unit
                        
        #apply band integration
        for f in range(len(self.nus)):
            start_time = datetime.now()
            print('Integrate band: {0} GHz with {1} steps'.format(self.nus[f],N_SAMPLE_BAND))
            
            #def freq array
            fmin = self.nus[f]-self.bw[f]/2
            fmax = self.nus[f]+self.bw[f]/2
            #### bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
            freqs = np.linspace(fmin, fmax, N_SAMPLE_BAND)

            #convert weights from power unit to CMB unit
            weights = weights_flat.copy() / _jysr2rj(freqs)
            weights /= _rj2cmb(freqs)
            weights /= np.trapz(weights, freqs * 1e9)
            
            #cycle on the foregrounds
            for i in self.skyconfig.keys():
                
                if i == 'dust':
                    if self.skyconfig[i] == 'd0':
                        if verbose:
                            print('Model : {}'.format(self.skyconfig[i]))
                        #generate a dust map for each sampled freq
                        fg_maps = create_dustd0_no_bp(self.nside, freqs) #fg in CMB unit
                        #compute bandpass integration
                        integrand = (fg_maps.T*weights)
                        allmaps[:,:,f] += np.trapz( integrand[:,:], freqs * 1e9 ) 
                
                    elif self.skyconfig[i] == 'd02b':
                        if verbose:
                            print('Model : d02b -> Twos spectral index beta ({:.2f} and {:.2f}) with nu_break = {:.2f}'.format(beta[0], beta[1], beta[2]))
                        #generate a dust map for each sampled freq
                        fg_maps = create_dust_with_model2beta_no_bp(self.nside, freqs,
                                                              beta[0], beta[1], beta[2],
                                                              temp=temp, break_width=beta[3]) #fg in CMB unit
                        #compute bandpass integration
                        integrand = (fg_maps.T*weights)
                        allmaps[:,:,f] += np.trapz( integrand[:,:], freqs * 1e9 )
                    
                    else: #
                        print('No dust')
                
                elif i == 'synchrotron':
                    if verbose:
                        print('Model : {}'.format(self.skyconfig[i]))  
                    #generate a dust map for each sampled freq
                    fg_maps = create_sync_no_bp(self.nside, freqs) #fg in CMB unit
                    #compute bandpass integration
                    integrand = (fg_maps.T*weights)
                    allmaps[:,:,f] += np.trapz( integrand[:,:], freqs * 1e9 )
        
                else: ##
                    print('No more fg components')
                
            print('Duration: {}'.format(datetime.now()-start_time))
                
        return allmaps.T


    def get_fg_freq_integrated_maps_w_pysm(self, verbose=False, N_SAMPLE_BAND=100, beta=[], temp=20):
        '''
        This function uses pysm3 bandpass integration to generate dust and synch maps integrated over the frequency band
        '''
        #def output maps
        allmaps=np.zeros(((len(self.nus), 3, self.npix)))
        
        for i in self.skyconfig.keys():
            if i == 'dust':
                dustmaps=np.zeros(((len(self.nus), 3, self.npix)))
                
                if self.skyconfig[i] == 'd0':
                    if verbose:
                        print('Model : {}'.format(self.skyconfig[i]))
                    #def sky model
                    sky = pysm3.Sky(nside=self.nside, preset_strings=[self.skyconfig[i]])
                    #apply band integration
                    for i in range(len(self.nus)):
                        fmin = self.nus[i]-self.bw[i]/2
                        fmax = self.nus[i]+self.bw[i]/2
                        #### bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
                        freqs = np.linspace(fmin, fmax, N_SAMPLE_BAND)
                        weights_flat = np.ones(N_SAMPLE_BAND)

                        dustmaps[i,:,:] = sky.get_emission(freqs * u.GHz, weights_flat) * bandpass_unit_conversion(freqs * u.GHz, weights_flat, u.uK_CMB)
                        
                    allmaps+=dustmaps

                elif self.skyconfig[i] == 'd02b':
                    if verbose:
                        print('Model : d02b -> Twos spectral index beta ({:.2f} and {:.2f}) with nu_break = {:.2f}'.format(beta[0], beta[1], beta[2]))
                    print('Error: Model : d02b NOT implemented yet!')
                    sys.exit()
                    
                
                else: #
                    print('No dust')

            elif i == 'synchrotron':
                syncmaps=np.zeros(((len(self.nus), 3, self.npix)))
                if verbose:
                    print('Model : {}'.format(self.skyconfig[i]))
                #def sky model
                sky = pysm3.Sky(nside=self.nside, preset_strings=[self.skyconfig[i]])
                #apply band integration
                for i in range(len(self.nus)):
                    fmin = self.nus[i]-self.bw[i]/2
                    fmax = self.nus[i]+self.bw[i]/2
                    #### bandpass_frequencies = np.linspace(fmin, fmax, fsteps) * u.GHz
                    freqs = np.linspace(fmin, fmax, N_SAMPLE_BAND)
                    weights_flat = np.ones(N_SAMPLE_BAND)

                    syncmaps[i,:,:] = sky.get_emission(freqs * u.GHz, weights_flat) * bandpass_unit_conversion(freqs * u.GHz, weights_flat, u.uK_CMB)
                    
                allmaps+=syncmaps

            else: ##
                print('No more fg components')
                
        return allmaps    

        
    def getskymaps(self, fg_maps=None ,same_resol=None, verbose=False, coverage=None, iib=False, noise=False, signoise=1., nside_index=256):
        """
        This function adds cmb and noise maps to the input fg maps, if any input fg map is given
        """
        if same_resol is not None:
            self.fwhmdeg = np.ones(len(self.nus))*np.max(same_resol)

        if verbose:
            print("    FWHM : {} deg \n    nus : {} GHz \n    Bandwidth : {} GHz\n\n".format(self.fwhmdeg, self.nus, self.bw))

        allmaps=np.zeros(((len(self.nus), 3, self.npix)))
        #add foregrounds if they are passed in input
        if fg_maps is not None:
            print('Adding fg maps...')
            if (allmaps.shape==fg_maps.shape):
                allmaps+=fg_maps
                print('...Addition of fg maps completed.')
            else:
                print('ERROR: fg maps and sky maps have different shapes! Kill.')
                return
        #add cmb
        for i in self.skyconfig.keys():
            if i == 'cmb':
                cmbmap = self.get_cmb(coverage)
                for j in range(len(self.nus)):
                    allmaps[j]+=cmbmap
        #convolve with fwhm if required
        if same_resol != 0:
            for j in range(len(self.fwhmdeg)):
                if verbose:
                    print('Convolution to {:.2f} deg'.format(self.fwhmdeg[j]))
                allmaps[j] = hp.sphtfunc.smoothing(allmaps[j, :, :], fwhm=np.deg2rad(self.fwhmdeg[j]),verbose=False)
        #add noise if required
        if noise:
            noisemaps = create_noisemaps(signoise, self.nus, self.nside, self.depth_i, self.depth_p, self.npix)
            maps_noisy = allmaps+noisemaps
            #apply coverage
            if coverage is not None:
                pixok = coverage > 0
                maps_noisy[:, :, ~pixok] = hp.UNSEEN
                noisemaps[:, :, ~pixok] = hp.UNSEEN
                allmaps[:, :, ~pixok] = hp.UNSEEN

            return maps_noisy, allmaps, noisemaps
        return allmaps
