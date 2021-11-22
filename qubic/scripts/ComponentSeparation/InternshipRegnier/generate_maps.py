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
from astropy.io import fits

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


### Some initializations, to be replaced with specific path, or to modify in bash
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

dic = d220

# Read dictionary chosen
dic['focal_length'] = 0.3
dic['nside'] = 256

#Define the number of reconstruction bands:
nbands = 3
dic['nf_recon'] = nbands
dic['nf_sub'] = nbands

npix = 12 * dic['nside'] ** 2
Nf = int(dic['nf_sub'])
band = dic['filter_nu'] / 1e9
filter_relative_bandwidth = dic['filter_relative_bandwidth']
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

def MC_generate_maps(nsim1, nsim2, dic, Nb, iib, nunu_corr, spa_corr, noise) :

    """
    Run a Monte-Carlo simulation

    Inputs :
    ----------
        - nsim : Number of simulations
        - dic : Dictionary from QUBIC soft
        - Nb : Number of sub-bands
        - iib : Boolean argument -> Integration into band
        - nunu_corr : Boolean argument -> Frequencies correlation
        - spa_corr : Boolean argument -> Spatials correlation
        - noise : int argment -> noise level

    Outputs :
    ----------
        - beta_d : list -> Estimation of beta_d spectral parameter
        - tab_seed : list -> List of using seed
    """

    beta_d = []
    T = []
    tab_seed = []

    verbose = False
    Stkp = 'IQU'

    freqs, fwhmdegs = give_me_freqs_fwhm(dic, Nb)
    fwhm_final = np.max(fwhmdegs) + 1e-8

    all_maps = np.zeros((((nsim2-nsim1, Nb, 3, 12*256**2))))
    all_N = np.zeros((((nsim2-nsim1, Nb, 3, 12*256**2))))

    for i in range(nsim1, nsim2) :
        seed = i+1

        print()
        print("=============== MAPS GENERATION : {} ===============".format(i+1))

        dic['nf_recon'] = Nb
        dic['nf_sub'] = 4*Nb
        freqs, fwhmdegs = give_me_freqs_fwhm(dic, 4*Nb)

        sky_config = {'dust': 'd0', 'cmb': seed}
        Qubic_sky_150 = qss.Qubic_sky(sky_config, dic)

        CMBdust, _, N, _ = Qubic_sky_150.get_partial_sky_maps_withnoise(coverage=cov150,
                                       Nyears=noise, verbose=verbose, FWHMdeg=fwhmdegs, seed = seed,
                                       spatial_noise=spa_corr,
                                       nunu_correlation=nunu_corr,
                                       integrate_into_band=iib, noise_profile = False
                                            )
        CMBdust = np.transpose(CMBdust, (0, 2, 1))*apo
        N = np.transpose(N, (0, 2, 1))*apo

        dir = '/home/mathias/Bureau/FG-Buster/pickle_file_fgbuster/Maps+noise/220GHz/{}bands/whitenoise/'.format(Nb)
        data_dict = {'maps' : CMBdust, 'noise' : N, 'seed' : seed}
        files = open(dir + 'maps_220_noise_{}bands_spa_nu_corr_{}_nyears_{}_{}.pkl'.format(Nb, nunu_corr, int(noise), i+1), 'wb')
        pickle.dump(data_dict, files)
        files.close()

        del CMBdust
        del N

    return None

nsim1 = 20
nsim2 = 50
Nb = 3
spa_corr = False
nunu_corr = False
Nyears = [4e3, 4e2, 1e2, 2e1, 8e0, 4e0, 3e0]

for i, j in enumerate(Nyears) :
    print("Noise : ", j)
    MC_generate_maps(nsim1, nsim2, dic, Nb, True, nunu_corr, spa_corr, j)
