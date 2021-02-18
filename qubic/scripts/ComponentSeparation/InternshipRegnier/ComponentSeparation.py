import random
import healpy as hp
import glob
from scipy.optimize import curve_fit
import pickle
from importlib import reload
import time
import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import pylab
from pylab import arange, show, cm
from astropy import units as uq
import gc
from qubic import QubicSkySim as qss


### FGBuster functions module
from fgbuster import get_instrument, get_sky, get_observation, ilc, basic_comp_sep, harmonic_ilc, weighted_comp_sep, multi_res_comp_sep  # Predefined instrumental and sky-creation configurations
from fgbuster.visualization import corner_norm, plot_component
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import _rj2cmb, _jysr2rj, get_noise_realization

# Imports needed for component separation
from fgbuster import (separation_recipes, xForecast, CMB, Dust, Synchrotron, FreeFree, PowerLaw,  # sky-fitting model
                      basic_comp_sep)  # separation routine
                      
                      
                      
                      
                      
class CompSep :
    
    def __init__(self) :
        
        pass
    
    def fg_buster(nb_bands, map1, comp1) :
        
        """
        --------
        inputs :
            - nb_bands : number of sub bands -> int type
            - map1 : map of which we want to separate the components -> array type & (nb_bands, Nstokes, Npix)
            - comp1 : Component in addition to the CMB signal -> frequency nu in argument
        
        --------
        output :
            - r : Dictionary where r.s give the amplitude of different components -> r.s[ind_comp, istokes, Npix]. 
                    ind_comp means the component (0 for CMB and 1 for comp1). 
        
        """
        
        comp = [CMB(), comp1]
        
        ins = get_instrument('Qubic' + str(nb_bands) + 'bands')
        
        r = basic_comp_sep(comp, ins, map1)
        
        return r
        
    def weighted(nb_bands, map1, coverage, comp1, ifreq) :
        
        """
        --------
        inputs :
            - nb_bands : number of sub bands -> int type
            - map1 : map of which we want to separate the components -> array type & (nb_bands, Nstokes, Npix)
            - comp1 : Component in addition to the CMB signal -> frequency nu in argument
        
        --------
        output :
            - r : Dictionary where r.s give the amplitude of different components -> r.s[ind_comp, istokes, Npix]. 
                    ind_comp means the component (0 for CMB and 1 for comp1). 
        
        """
 
        cov_I, cov_Q, cov_U, c_all, c = qss.get_cov_nunu(np.transpose(map1, (0, 2, 1)), coverage, return_flat_maps = False)
        ins = get_instrument('Qubic' + str(nb_bands) + 'bands')
        comp = [CMB(), comp1]
        
        rI = weighted_comp_sep(comp, ins, np.transpose(map1, (0, 2, 1)), cov_I[ifreq])
        rQ = weighted_comp_sep(comp, ins, np.transpose(map1, (0, 2, 1)), cov_Q[ifreq])
        rU = weighted_comp_sep(comp, ins, np.transpose(map1, (0, 2, 1)), cov_U[ifreq])
        
        r = [rI, rQ, rU]
        
        return r
        
    def weighted_2_tab(X, nb_bands) :
        tab_cmb = np.zeros(((nb_bands, 3, 12*256**2)))
        tab_dust = np.zeros(((nb_bands, 3, 12*256**2)))
    
        
        for j in range(3) :
            tab_cmb[0, j, :] = X[j].s[0, :, j]
            tab_dust[0, j, :] = X[j].s[1, :, j]
        return tab_cmb, tab_dust
        
    def basic_2_tab(X, nb_bands) :
        tab_cmb = np.zeros(((nb_bands, 3, 12*256**2)))
        tab_dust = np.zeros(((nb_bands, 3, 12*256**2)))
    
        for i in range(nb_bands) :
            for j in range(3) :
                tab_cmb[i, j, :] = X.s[0, j, :]
                tab_dust[i, j, :] = X.s[1, j, :]
        return tab_cmb, tab_dust
        
    def internal_linear_combination(nb_bands, map1) :
		
        """
        
        ----------
        inputs :
            - nb_bands : number of sub bands
            - map1 : map of which we want to estimate CMB signal

        ----------
        outputs :
            - r : Dictionary for each Stokes parameter (I, Q, U) in a list. To have the amplitude, we can write r[ind_stk].s[0].

        """
		
        comp = [CMB()]
        ins = get_instrument('Qubic' + str(nb_bands) + 'bands')
        map_convert = np.transpose(map1, (0, 2, 1))
        A = []
        B = []
        for i in range(3) :
            A.append(map1[:, :, i])
            B.append(np.transpose(A[i], (0, 1)))

        r = []

        for i in range(3) :
            r.append(ilc(comp, ins, B[i]))

        return r
    
    def ilc_2_tab(X, nb_bands) :
    
        tab_cmb = np.zeros(((3, 3, 12*256**2)))
        
        for i in range(3) :
            tab_cmb[0, i, :] = X[i].s[0]
        
        return tab_cmb
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
                      
