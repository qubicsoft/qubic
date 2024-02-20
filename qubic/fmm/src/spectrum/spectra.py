import numpy as np
from qubic import NamasterLib as nam
import healpy as hp
import matplotlib.pyplot as plt


class Spectrum:
    
    def __init__(self, params, mapmaking):
    
        self.params = params
        self.mapmaking = mapmaking
        
 
        self.namaster = nam.Namaster(self.mapmaking.seenpix, 
                                lmin = self.params['Spectrum']['lmin'], 
                                lmax = self.params['Spectrum']['lmax'], 
                                delta_ell = self.params['Spectrum']['dl'],
                                aposize = self.params['Spectrum']['aposize'])
        
        self.ell, _ = self.namaster.get_binning(self.params['Sky']['nside'])
        self.N = len(self.mapmaking.nus_Q)
        self.nspec = int(self.N * (self.N + 1) / 2)
        
    def _get_spectra(self, map1, map2=None):
        
        _, Dls, _ = self.namaster.get_spectra(map=map1, map2=map2, pixwin_correction=False, beam_correction=0, verbose=False)
        
        return Dls[:, 2]
    
    def run(self, file):
        
        self.Dl = np.zeros((self.nspec, len(self.ell)))
        
        s = np.zeros((self.N, self.N))
                
        k=0
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    print(f'Computing auto-spectra at {self.mapmaking.nus_Q[i]:.0f} GHz')
                    self.Dl[k] = self._get_spectra(map1=self.mapmaking.s_hat[i].T)
                    k+=1
                else:
                    if s[i, j] == 0:
                        print(f'Computing X-spectra at {self.mapmaking.nus_Q[i]:.0f} and {self.mapmaking.nus_Q[j]:.0f} GHz')
                        self.Dl[k] = self._get_spectra(map1=self.mapmaking.s_hat[i].T, map2=self.mapmaking.s_hat[j].T)
                        s[i, j] = 1
                        s[j, i] = 1
                        k+=1
                    else:
                        s[i, j] = 1
                        s[j, i] = 1
        
        self.mapmaking.save_data(file, {'nus':self.mapmaking.nus_Q,
                                                                'ell':self.ell,
                                                                'Dls':self.Dl})