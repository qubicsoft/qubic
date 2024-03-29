import numpy as np
import qubic
import sys
import os

sys.path.append(os.getcwd())
CMB_FILE = os.getcwd() + '/data/'
#from qubic.data import PATH
import pysm3
import pysm3.units as u
from pysm3 import utils
import healpy as hp
import matplotlib.pyplot as plt


class ExternalData2Timeline:
    
    def __init__(self, sky, nus, nrec, nside=256, corrected_bandpass=True):
        
        self.nus = nus
        self.nside = nside
        self.nrec = nrec
        self.nsub = len(self.nus)
        self.m_nu = np.zeros((len(self.nus), 12*self.nside**2, 3))
        self.sky = sky
        
        for i in sky.keys():
            if i == 'cmb':
                cmb = self.get_cmb(r=0, Alens=1, seed=self.sky['cmb'])
                self.m_nu += cmb.copy()
            elif i == 'dust':
                self.sky_fg = self._separe_cmb_fg()
                self.sky_pysm = pysm3.Sky(self.nside, preset_strings=self.list_fg)
                self.m_nu_fg = self._get_fg_allnu()
                self.m_nu += self.m_nu_fg.copy()
                
                if corrected_bandpass:
                    self.m_nu = self._corrected_maps(self.m_nu, self.m_nu_fg)
            
        self.maps = self.average_within_band(self.m_nu)
                
        
        
    def give_cl_cmb(self, r=0, Alens=1.):
        
        power_spectrum = hp.read_cl(CMB_FILE+'Cls_Planck2018_lensed_scalar.fits')[:,:4000]
        if Alens != 1.:
            power_spectrum[2] *= Alens
        if r:
            power_spectrum += r * hp.read_cl(CMB_FILE+'Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
        return power_spectrum
    
    def get_cmb(self, r=0, Alens=1, seed=None):
        
        mycls = self.give_cl_cmb(r, Alens)
        np.random.seed(seed)
        return hp.synfast(mycls, self.nside, verbose=False, new=True).T
        
        
        
    def _get_fg_1nu(self, nu):
        return np.array(self.sky_pysm.get_emission(nu * u.GHz, None).T * \
                        utils.bandpass_unit_conversion(nu * u.GHz, None, u.uK_CMB))
    
    def _get_fg_allnu(self):
        
        m = np.zeros((len(self.nus), 12*self.nside**2, 3))
        
        for inu, nu in enumerate(self.nus):
            m[inu] = self._get_fg_1nu(nu)
            
        return m
    
    def _separe_cmb_fg(self):
        
        self.list_fg = []
        new_s = {}
        for i in self.sky.keys():
            if i == 'cmb':
                pass
            else:
                new_s[i] = self.sky[i]
                self.list_fg += [self.sky[i]]
            
        return new_s
    
    
    def average_within_band(self, m_nu):
        
        m_mean = np.zeros((self.nrec, 12*self.nside**2, 3))
        f = int(self.nsub / self.nrec)
        for i in range(self.nrec):
            #print(f'Doing average between {np.min(self.nus[i*f:(i+1)*f])} and {np.max(self.nus[i*f:(i+1)*f])} GHz')
            m_mean[i] = np.mean(m_nu[i*f : (i+1)*f], axis=0)
        return m_mean
    
    def _corrected_maps(self, m_nu, m_nu_fg):
        
        f = int(self.nsub / self.nrec)
        
        mean_fg = self.average_within_band(m_nu_fg)
        
        k=0
        for i in range(self.nrec):
            delta = m_nu_fg[i*f : (i+1)*f] - mean_fg[i]
            for j in range(f):
                m_nu[k] -= delta[j]
                k+=1
                
        return m_nu


                
    
    
