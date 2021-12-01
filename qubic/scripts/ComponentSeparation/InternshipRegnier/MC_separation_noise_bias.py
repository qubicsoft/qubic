"""

This file use previous computed maps and make the component separation in order to save the beta estimation.

"""


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

### Specific qubic modules
from qubicpack.utilities import Qubic_DataDir
from pysimulators import FitsArray
import pysm3
import pysm3.units as u
import pysm3.utils as utils
import qubic
from qubic import QubicSkySim as qss
from qubic import fibtools as ft
from qubic import camb_interface as qc
from qubic import SpectroImLib as si
from qubic import NamasterLib as nam
from qubic import mcmc
from pysimulators.interfaces.healpy import HealpixConvolutionGaussianOperator

### FGBuster functions module
from fgbuster import get_instrument, get_sky, get_observation, ilc, basic_comp_sep, harmonic_ilc, weighted_comp_sep, multi_res_comp_sep, harmonic_ilc_alm  # Predefined instrumental and sky-creation configurations
from fgbuster.visualization import corner_norm, plot_component
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import _rj2cmb, _jysr2rj, get_noise_realization

# Imports needed for component separation
from fgbuster import (separation_recipes, xForecast, CMB, Dust, Synchrotron, FreeFree, PowerLaw,  # sky-fitting model
                      basic_comp_sep)  # separation routine

import ComponentSeparation

reload(qss)
reload(ft)

plt.rc('figure', figsize=(16, 10))
plt.rc('font', size=15)
plt.rcParams['image.cmap'] = 'jet'

## Some initializations, to be replaced with specific path, or to modify in bash
os.environ['QUBIC_DATADIR'] = '/home/mathias/Bureau/qubic/qubic'
os.environ['QUBIC_DICT'] = '/home/mathias/Bureau/qubic/qubic/dicts'
global_dir = Qubic_DataDir(datafile='instrument.py', datadir=os.environ['QUBIC_DATADIR'])

### Qubic dictionaries for 150GHz and 220Ghz
config_150, config_220 = 'FI-150', 'FI-220'
dictfilename150 = global_dir + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_150)
dictfilename220 = global_dir + '/doc/FastSimulator/FastSimDemo_{}.dict'.format(config_220)
d150, d220 = qubic.qubicdict.qubicDict(), qubic.qubicdict.qubicDict()
d150.read_from_file(dictfilename150)
d220.read_from_file(dictfilename220)
qub_dic = {'150': d150, '220': d220}
center = qubic.equ2gal(d220['RA_center'], d220['DEC_center'])

### Read some stuff

# Read dictionary chosen
d150['focal_length'] = 0.3
d150['nside'] = 256
d220['focal_length'] = 0.3
d220['nside'] = 256

#Define the number of reconstruction bands:
nbands = 3
d150['nf_recon'] = nbands
d150['nf_sub'] = nbands
d220['nf_recon'] = nbands
d220['nf_sub'] = nbands

npix = 12 * d150['nside'] ** 2
Nf = int(d150['nf_sub'])
band = d150['filter_nu'] / 1e9
filter_relative_bandwidth = d150['filter_relative_bandwidth']
npix = 12 * d220['nside'] ** 2
Nf = int(d220['nf_sub'])
band = d220['filter_nu'] / 1e9
filter_relative_bandwidth = d220['filter_relative_bandwidth']
a, nus_edge, nus_in, df, e, Nbbands_in = qubic.compute_freq(band, Nf, filter_relative_bandwidth)

def give_me_freqs_fwhm(dic, Nb) :
    band = dic['filter_nu'] / 1e9
    filter_relative_bandwidth = dic['filter_relative_bandwidth']
    a, nus_edge, nus_in, df, e, Nbbands_in = qubic.compute_freq(band, Nb, filter_relative_bandwidth)
    return nus_in, dic['synthbeam_peak150_fwhm'] * 150 / nus_in


def open_picklefile(directory, name, var) :

    """
    Open a pickle file from saved data
    """

    tab = pickle.load(open(directory + name, 'rb'))
    variable = tab[var]
    return variable

cov150 = open_picklefile('/home/mathias/Bureau/', 'Coverage_dtheta_15_pointing_3000.pkl', 'coverage')

def give_me_mask_apo(okpix = None, lmin = 2, lmax = 512, delta_ell = 16) :
    mask = np.zeros(12 * d150['nside']**2)
    mask[okpix] = 1
    # Namaster object
    Namaster = nam.Namaster(mask, lmin=2, lmax=512, delta_ell=16)
    apo = Namaster.mask_apo
    return apo

# Define okpix
okpix_inside = (cov150 > (0.5*np.max(cov150)))
okpix = (cov150 > (0.1*np.max(cov150)))

apo = give_me_mask_apo(okpix = okpix)

freqs, fwhmdegs = give_me_freqs_fwhm(d150, nbands)
fwhm_final = np.max(fwhmdegs) + 1e-8

nsim = 50

noise = [4e3, 4e2, 1e2, 2e1, 8e0, 4e0, 3e0]
beta_150 = np.zeros((len(noise), nsim))
beta_220 = np.zeros((len(noise), nsim))
beta_combined = np.zeros((len(noise), nsim))
T_150 = np.zeros((len(noise), nsim))
T_220 = np.zeros((len(noise), nsim))
T_combined = np.zeros((len(noise), nsim))
nu0 = 150.

for i, j in enumerate(noise) :
    for k in range(nsim) :
        print(i, j, k)
        dir = '/home/mathias/Bureau/FG-Buster/pickle_file_fgbuster/Maps+noise/'
        all_maps = np.zeros(((2*nbands, 3, 12*256**2)))
        maps_150 = open_picklefile(dir + '150GHz/3bands/whitenoise/{}years/'.format(int(j)), 'maps_150_noise_3bands_spa_nu_corr_False_nyears_{}_{}.pkl'.format(int(j), k+1), 'maps')
        maps_220 = open_picklefile(dir + '220GHz/3bands/whitenoise/{}years/'.format(int(j)), 'maps_220_noise_3bands_spa_nu_corr_False_nyears_{}_{}.pkl'.format(int(j), k+1), 'maps')

        freqs, fwhmdegs = give_me_freqs_fwhm(d150, nbands)
        fwhm_final = np.max(fwhmdegs) + 1e-8

        R_150 = ComponentSeparation.CompSep(d150).fg_buster(maps_150, [CMB(), Dust(nu0)], freq=freqs, fwhmdeg=fwhmdegs, target = fwhm_final, okpix = okpix_inside, Ny = int(j))

        freqs, fwhmdegs = give_me_freqs_fwhm(d220, nbands)
        fwhm_final = np.max(fwhmdegs) + 1e-8

        R_220 = ComponentSeparation.CompSep(d220).fg_buster(maps_220, [CMB(), Dust(nu0)], freq=freqs, fwhmdeg=fwhmdegs, target = fwhm_final, okpix = okpix_inside, Ny = int(j))

        all_maps[:nbands] = maps_150
        all_maps[nbands:] = maps_220

        freqs150, fwhmdegs150 = give_me_freqs_fwhm(d150, nbands)
        freqs220, fwhmdegs220 = give_me_freqs_fwhm(d220, nbands)

        freqs = list(freqs150) + list(freqs220)
        FWHM = list(fwhmdegs150) + list(fwhmdegs220)

        target_fwhm = np.max(FWHM) + 1e-8

        R = ComponentSeparation.CompSep(d220).fg_buster(all_maps, [CMB(), Dust(nu0)], freq=freqs, fwhmdeg=FWHM, target = target_fwhm, okpix = okpix_inside, Ny = int(j))

        print(" \n ========= Estimation of beta and T ========== \n ")

        print(R_150.x[0], R_150.x[1])
        print(R_220.x[0], R_220.x[1])
        print(R.x[0], R.x[1])

        print()

        beta_150[i, k] = R_150.x[0]
        beta_220[i, k] = R_220.x[0]
        beta_combined[i, k] = R.x[0]

        T_150[i, k] = R_150.x[1]
        T_220[i, k] = R_220.x[1]
        T_combined[i, k] = R.x[1]

        del maps_150
        del maps_220
        del all_maps
        del R_150
        del R_220
        del R

dir = '/home/mathias/Bureau/FG-Buster/pickle_file_fgbuster/noise_bias/'
data_dict = {'beta_150' : beta_150, 'beta_220' : beta_220, 'beta_combined' : beta_combined, 'T_150' : T_150, 'T_220' : T_220, 'T_combined' : T_combined}
files = open(dir + '{}bands/estimation_beta_T_3bands_50reals_1.pkl'.format(nbands), 'wb')
pickle.dump(data_dict, files)
files.close()
