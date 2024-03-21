import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pysm3.units as u

import qubic
from analytical_forecast_lib import *

    
class FastNoise(AnalyticalForecast):
    '''
    Class to compute noise maps for the QUBIC instrument using the formula from analytical_forecst_lib.py

    Arguments : - nus : array(float)
                - nside : float
                - NEPdet : array(float), as same lenght than nus
                - NEPpho : array(float), as same lenght than nus
                - sky : str, 'CMB' for only cmb or 'CMB + Dust' for CMB & Dust on the sky
                - correlation : Bool, corelation between frquency bands (not tested yet)
                - fwhm : array(float), as same lenght than nus (only 0 suported now)
    '''
    
    def __init__(self, nus, nside, NEPdet, NEPpho, sky = 'CMB + Dust', correlation = False, fwhm=np.array([0, 0]), Nyrs=3, Nh=400, fsky=0.0182, instr='DB'):
        
        AnalyticalForecast.__init__(self, nus, NEPdet, NEPpho, fwhm=fwhm, Nyrs=Nyrs, Nh=Nh, fsky=fsky, nside=nside, instr=instr)

        self.nside = nside
        self.correlation = correlation
        
        # We compute NET from NEP using analytical_forecast_lib.py
        self.NETs = np.zeros(len(self.nus))
        for i in range(len(self.nus)):
            self.NETs[i] = NoiseEquivalentTemperature(self.NEPs[i], self.nus[i]).NETs

        # We compute depths for frequency maps from NET using analytical_forecast_lib.py
        self.depths_FMM = self._get_effective_depths(self.NETs)
        
        # We define the mixing matrix for both sky configurations
        if sky == 'CMB + Dust':
            self.A = np.array([[1, 1],
                               [1, 2.92]])
            self.ncomp = 2
        if sky == 'CMB':
            self.A = np.array([[1], 
                               [1]])
            self.ncomp = 1
        
    def get_noise_from_depths(self, depths, unit='uK_CMB'):
        '''
        Function to generate sky maps from depths values

        return : - array(len(depths), nstokes, npix)
        '''

        n = np.shape(depths)[0]
        n_pix = hp.nside2npix(self.nside)
        res = np.random.normal(size=(n_pix, 3, n))
        res[:, 0, :] /= np.sqrt(2)
        depths *= u.arcmin * u.uK_CMB
        depths = depths.to(getattr(u, unit) * u.arcmin,
            equivalencies=u.cmb_equivalencies(self.nus * u.GHz))
        res *= depths.value / hp.nside2resol(self.nside, True) # depths / pixel size in radian
        return res.T   
    
    def get_noise_realisation_FMM(self, unit='uK_CMB'):
        '''
        Function to generate noise frequency maps
        '''
    
        return self.get_noise_from_depths(self.depths_FMM)
    
    def get_noise_realisation_CMM(self, unit='uK_CMB'):
        '''
        Function to generate noise components maps
        '''
        
        bl = np.array([hp.gauss_beam(b, lmax=2*self.nside) for b in self.fwhm])
        nl = (bl / (self.depths_FMM)[:, np.newaxis])**2
        AtNA = np.einsum('fi, fl, fj -> lij', self.A, nl, self.A)

        depths_CMM = np.zeros(self.ncomp)
        for i in range(self.ncomp):
            depths_CMM[i] = np.sqrt(np.linalg.pinv(AtNA))[0][i, i] * hp.nside2resol(self.nside, True)
        return self.get_noise_from_depths(depths_CMM)