import numpy as np
import healpy as hp

import qubic

def give_cl_cmb(ell, r=0, Alens=1.):
        
    power_spectrum = hp.read_cl('data/Cls_Planck2018_lensed_scalar.fits')[:,:4000]
    if Alens != 1.:
        power_spectrum[2] *= Alens
    if r:
        power_spectrum += r * hp.read_cl('data/Cls_Planck2018_unlensed_scalar_and_tensor_r1.fits')[:,:4000]
    return np.interp(ell, np.arange(1, 4001, 1), power_spectrum[2])

class NoiseEquivalentTemperature:
    
    def __init__(self, NEPs, band, relative_bandwidth=0.25):
        
        self.band = band
        self.NEPs = NEPs
        self.T = 2.7255
        self.h = 6.62e-34
        self.k = 1.38e-23
        self.c = 3e8
        _, _, _, _, self.bw, _ = qubic.compute_freq(self.band, Nfreq=1, relative_bandwidth=relative_bandwidth)
        
        self.NETs = self._NEP2NET_db(np.sqrt(np.sum(self.NEPs**2)), self.band)
        
    def _get_derivative_Bnu_db(self, band):
        
        dnu = 0.5 * self.bw * 1e9
        nu = band * 1e9
        x = (self.h * nu) / (self.k * self.T)
        dIdT = ((2 * (self.h**2) * nu**4)/((self.c**2) * self.k * self.T**2)) * (np.exp(x)/(np.exp(x) - 1)**2) * dnu
    
        return dIdT
    
    def _NEP2NET_db(self, NEP, band):
    
        dIdT = self._get_derivative_Bnu_db(band)
    
        return np.array([NEP / (np.sqrt(2) * (dIdT * 1e-12))])

class AnalyticalForecast:
    
    
    def __init__(self, nus, NEPdet, NEPpho, Nyrs=3, Nh=400, fsky=0.0182, nside=256):
    
        ### Check length and type of inputs
        if type(NEPdet) is not list or type(NEPpho) is not list:
            raise TypeError("NEP type should be a list")
        
        if len(NEPdet) != len(NEPpho):
            raise TypeError("NEPdet and NEPpho should have the same length")
        
        self.nside = nside
        self.Nyrs = Nyrs
        self.Tobs = 3600 * 24 * 365 * self.Nyrs
        self.Nh = Nh
        self.fsky = fsky
        self.nus = nus
        self.nfreqs = len(NEPdet)
        self.NEPs = np.zeros((self.nfreqs, 2))
        
        
        ### Store NEPs and convert them to NETs
        for i in range(self.nfreqs):
            self.NEPs[i, 0] = NEPdet[i]
            self.NEPs[i, 1] = NEPpho[i]
            #self.NETs[i] = NoiseEquivalentTemperature(self.NEPs[i], self.nus[i]).NETs
        
    def _get_effective_depths(self, NETs):
    
        
        Omega = (4.0 * np.pi * self.fsky) / ((np.pi / (180.0 * 60.0))**2)
        depths = 4 * np.sqrt((Omega * np.power(NETs, 2.)) / (self.Tobs * self.Nh))
        #depths = np.sqrt((Omega * 2. * np.power(NETs, 2.)) / (self.Tobs * self.Nh))# * (10800. / np.pi)

        return depths
    
    def _get_power_spectra(self, depths, A):
        
        fwhm = np.array([0.0041, 0.0041])
        bl = np.array([hp.gauss_beam(b, lmax=2*self.nside) for b in fwhm])
    
        nl = (bl / np.radians(depths/60.)[:, np.newaxis])**2
        AtNA = np.einsum('fi, fl, fj -> lij', A, nl, A)

        sig2_00 =  np.linalg.pinv(AtNA)
        Nl = sig2_00[0, 0, 0]

        return Nl

    def _fisher(self, ell, Nl):
        ClBB = give_cl_cmb(ell, r=1, Alens=0.)
        return np.sum(((2 * ell + 1)/2) * self.fsky * (ClBB / Nl)**2)**(-0.5)
    
    def _get_sigr(self, A, ell):
        
        ### NEPs [W/sqrt(Hz)] -> NETs [muK.sqrt(s)]
        NETs = np.zeros(self.nfreqs)
        for i in range(self.nfreqs):
            NETs[i] = NoiseEquivalentTemperature(self.NEPs[i], self.nus[i]).NETs
        print(NETs)
        
        ### NETs [muK.sqrt(s)] -> depths [muK.arcmin]
        depths = self._get_effective_depths(NETs)
        print(depths)
        
        ### depths [muK.arcmin] -> Nl [muK^2]
        Nl = self._get_power_spectra(depths, A)
        print(Nl)
        
        sigr = self._fisher(ell, Nl)
        print(sigr)
        
        
        
        
        
        
ell = np.array([40.5, 70.5, 100.5, 130.5, 160.5, 190.5, 220.5, 250.5, 280.5, 310.5, 340.5, 370.5, 400.5, 430.5, 460.5])
A = np.array([[1],
              [1]])   
 
af = AnalyticalForecast([150, 220], [4.7e-17, 4.7e-17], [4.55e-17, 1.16e-16], Nyrs=3, Nh=400, fsky=0.0182, nside=256)
af._get_sigr(A, ell)